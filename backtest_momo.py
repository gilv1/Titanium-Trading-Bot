"""
MoMo Small-Cap Backtester — "3 Balas de Oro" Strategy.

Tests the momentum gap-up strategy on 1-minute intraday data for small-cap
stocks.  Supports an arbitrary number of CSV files dropped in
``data/historical/momo/`` (named ``TICKER_YYYY-MM-DD.csv``).

Usage:
    python backtest_momo.py                         # run backtest
    python backtest_momo.py --generate-sample       # create 50 synthetic CSVs
    python backtest_momo.py --generate-sample --n 100   # create N synthetic CSVs
    python backtest_momo.py --download-yahoo        # download real data from Yahoo Finance
    python backtest_momo.py --download-ibkr         # download real data from IBKR
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from analysis.technical import calculate_ema, calculate_vwap
from config import settings
from core.brain import AIBrain, TradeOutcome
from core.news_correlator import NewsCorrelator
from core.risk_manager import RiskManager
from core.reto_tracker import RetoTracker
from core.sympathy_detector import SympathyDetector

logger = logging.getLogger(__name__)

MOMO_DIR = Path("data/historical/momo")
RESULTS_FILE = Path("journal/momo_backtest_results.json")

# PDT limits
PDT_MAX_TRADES_PER_WINDOW = 3
PDT_ROLLING_DAYS = 5

# Trading hours (24-hour, Eastern)
SESSION_START = (9, 30)
SESSION_END = (12, 0)

# Signal detection thresholds
VWAP_TOLERANCE_PCT = 0.005   # within 0.5% of VWAP
EMA_PULLBACK_TOL_PCT = 0.003  # within 0.3% of EMA9
TRAIL_STOP_PCT = 0.03         # trail stop 3% below running high


# ──────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────


@dataclass
class MomoTrade:
    ticker: str
    date: str
    direction: str
    entry: float
    stop: float
    target1: float  # prior HOD — sell 50% here
    entry_bar: int
    exit_bar: int
    pnl: float
    won: bool
    setup_type: str


@dataclass
class MomoStats:
    stocks_analyzed: int
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    total_pnl: float
    best_trade: Optional[MomoTrade]
    worst_trade: Optional[MomoTrade]
    trades: list[MomoTrade] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# PDT tracker
# ──────────────────────────────────────────────────────────────


class PDTTracker:
    """Track round-trip day trades to stay within 3 per 5 rolling business days."""

    def __init__(self) -> None:
        self._trade_dates: list[date] = []

    def _business_days_window(self, reference_date: date) -> list[date]:
        """Return the PDT_ROLLING_DAYS business-day window ending on reference_date."""
        window: list[date] = []
        d = reference_date
        while len(window) < PDT_ROLLING_DAYS:
            if d.weekday() < 5:  # Mon–Fri
                window.append(d)
            d -= timedelta(days=1)
        return window

    def can_trade(self, trade_date: date) -> bool:
        window = self._business_days_window(trade_date)
        count = sum(1 for td in self._trade_dates if td in window)
        return count < PDT_MAX_TRADES_PER_WINDOW

    def record_trade(self, trade_date: date) -> None:
        self._trade_dates.append(trade_date)

    def trades_remaining(self, trade_date: date) -> int:
        window = self._business_days_window(trade_date)
        count = sum(1 for td in self._trade_dates if td in window)
        return max(0, PDT_MAX_TRADES_PER_WINDOW - count)


# ──────────────────────────────────────────────────────────────
# Signal detection (3 Balas de Oro)
# ──────────────────────────────────────────────────────────────


def _detect_vwap_bounce(df: pd.DataFrame, i: int, vwap: pd.Series) -> Optional[dict]:
    """Detect VWAP bounce on bar ``i``."""
    if i < 5:
        return None
    window = df.iloc[: i + 1]
    price = float(window["close"].iloc[-1])
    v = float(vwap.iloc[-1]) if not vwap.empty else 0.0
    if v == 0:
        return None
    if abs(price - v) / v > VWAP_TOLERANCE_PCT:
        return None
    # Volume decreasing (pullback into VWAP)
    vols = window["volume"].iloc[-3:].values
    if len(vols) == 3 and not (vols[0] > vols[1]):
        return None
    if price >= v:  # long if price above VWAP
        sl = price * 0.97
        tp = price * 1.06
        return {"direction": "LONG", "entry": price, "sl": sl, "tp": tp, "setup": "VWAP_BOUNCE"}
    return None


def _detect_ema_pullback(df: pd.DataFrame, i: int, ema9: pd.Series) -> Optional[dict]:
    """Detect EMA9 pullback setup."""
    if i < 20 or ema9.empty:
        return None
    price = float(df["close"].iloc[i])
    e9 = float(ema9.iloc[-1])
    if e9 == 0:
        return None
    dist = abs(price - e9) / e9
    if dist < EMA_PULLBACK_TOL_PCT:  # within tolerance of EMA9
        # Check trend: EMA9 sloping up
        if len(ema9) > 5 and float(ema9.iloc[-1]) > float(ema9.iloc[-5]):
            sl = price * 0.97
            tp = price * 1.06
            return {"direction": "LONG", "entry": price, "sl": sl, "tp": tp, "setup": "EMA_PULLBACK"}
    return None


def _detect_consolidation_breakout(df: pd.DataFrame, i: int) -> Optional[dict]:
    """Detect consolidation breakout after initial gap-up."""
    if i < 15:
        return None
    window = df.iloc[max(0, i - 15) : i + 1]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    price = float(df["close"].iloc[i])
    vol = float(df["volume"].iloc[i])
    avg_vol = float(window["volume"].mean())
    if vol < avg_vol * 1.5:
        return None
    if price > range_high * 0.999:
        sl = range_low
        tp = price + (price - range_low) * 1.5
        return {"direction": "LONG", "entry": price, "sl": sl, "tp": tp, "setup": "CONSOLIDATION_BREAKOUT"}
    return None


# ──────────────────────────────────────────────────────────────
# Trade simulation (one file)
# ──────────────────────────────────────────────────────────────


def _simulate_file(
    df: pd.DataFrame,
    ticker: str,
    trade_date: str,
    brain: AIBrain,
    pdt: PDTTracker,
) -> list[MomoTrade]:
    """Simulate one day's worth of intraday data for a MoMo stock."""
    trades: list[MomoTrade] = []

    # Filter to session hours
    if "time" in df.columns:
        try:
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"])
            df = df[
                (df["time"].dt.hour * 60 + df["time"].dt.minute >= SESSION_START[0] * 60 + SESSION_START[1])
                & (df["time"].dt.hour * 60 + df["time"].dt.minute <= SESSION_END[0] * 60 + SESSION_END[1])
            ].reset_index(drop=True)
        except Exception:  # noqa: BLE001
            df = df.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if len(df) < 30:
        return trades

    # Check PDT availability
    try:
        parsed_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    except ValueError:
        parsed_date = date.today()

    if not pdt.can_trade(parsed_date):
        return trades

    vwap = calculate_vwap(df)
    ema9 = calculate_ema(df, 9)

    # Determine prior high (open of first bar acts as "prior HOD" proxy)
    prior_hod = float(df["high"].iloc[0]) if len(df) > 0 else 0.0

    in_trade = False
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = prior_hod  # first target = prior HOD
    entry_bar = 0
    setup_type = ""
    half_sold = False

    for i in range(10, len(df)):
        if not in_trade:
            # Try each entry type in order
            sig = (
                _detect_vwap_bounce(df, i, vwap.iloc[: i + 1] if not vwap.empty else vwap)
                or _detect_ema_pullback(df, i, ema9.iloc[: i + 1] if not ema9.empty else ema9)
                or _detect_consolidation_breakout(df, i)
            )
            if sig is None:
                continue

            # Brain evaluation
            atr_approx = float((df["high"] - df["low"]).iloc[max(0, i - 14) : i].mean()) if i >= 14 else 1.0
            decision = brain.evaluate_trade(
                setup_type=sig["setup"],
                engine="momo_backtest",
                entry=sig["entry"],
                stop=sig["sl"],
                target=sig["tp"],
                session="NY",
                atr=atr_approx,
            )
            if not decision.approved:
                continue

            in_trade = True
            entry_price = sig["entry"]
            sl_price = sig["sl"]
            tp1_price = max(prior_hod, sig["tp"])  # use the higher of prior HOD or calc TP
            entry_bar = i
            setup_type = sig["setup"]
            half_sold = False

        else:
            bar = df.iloc[i]
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])

            # Check stop
            if bar_low <= sl_price:
                pnl = (sl_price - entry_price) * (0.5 if half_sold else 1.0)
                won = pnl > 0
                trade = MomoTrade(
                    ticker=ticker,
                    date=trade_date,
                    direction="LONG",
                    entry=entry_price,
                    stop=sl_price,
                    target1=tp1_price,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    pnl=round(pnl, 4),
                    won=won,
                    setup_type=setup_type,
                )
                trades.append(trade)
                pdt.record_trade(parsed_date)
                brain.record_outcome(
                    TradeOutcome(
                        setup_type=setup_type,
                        session="NY",
                        day_of_week="Monday",
                        hour=9,
                        volatility_regime="high",
                        won=won,
                        engine="momo_backtest",
                    )
                )
                in_trade = False
                if not pdt.can_trade(parsed_date):
                    break
                continue

            # Check TP1 — sell half and trail remainder at -3%
            if not half_sold and bar_high >= tp1_price:
                half_sold = True
                # Move SL to breakeven and trail remaining 3% below current price
                sl_price = max(sl_price, entry_price)

            # If half sold and price trails 3% below current high → exit remainder
            if half_sold:
                trail_sl = bar_high * (1.0 - TRAIL_STOP_PCT)
                if bar_low <= trail_sl or bar_low <= sl_price:
                    exit_price = max(sl_price, trail_sl)
                    pnl = (tp1_price - entry_price) * 0.5 + (exit_price - entry_price) * 0.5
                    won = pnl > 0
                    trade = MomoTrade(
                        ticker=ticker,
                        date=trade_date,
                        direction="LONG",
                        entry=entry_price,
                        stop=sl_price,
                        target1=tp1_price,
                        entry_bar=entry_bar,
                        exit_bar=i,
                        pnl=round(pnl, 4),
                        won=won,
                        setup_type=setup_type,
                    )
                    trades.append(trade)
                    pdt.record_trade(parsed_date)
                    brain.record_outcome(
                        TradeOutcome(
                            setup_type=setup_type,
                            session="NY",
                            day_of_week="Monday",
                            hour=9,
                            volatility_regime="high",
                            won=won,
                            engine="momo_backtest",
                        )
                    )
                    in_trade = False
                    if not pdt.can_trade(parsed_date):
                        break

    # Close any open trade at session end
    if in_trade and len(df) > 0:
        close_price = float(df["close"].iloc[-1])
        pnl = (close_price - entry_price) * (0.5 if half_sold else 1.0)
        if half_sold:
            pnl = (tp1_price - entry_price) * 0.5 + (close_price - entry_price) * 0.5
        won = pnl > 0
        trades.append(
            MomoTrade(
                ticker=ticker,
                date=trade_date,
                direction="LONG",
                entry=entry_price,
                stop=sl_price,
                target1=tp1_price,
                entry_bar=entry_bar,
                exit_bar=len(df) - 1,
                pnl=round(pnl, 4),
                won=won,
                setup_type=setup_type,
            )
        )
        pdt.record_trade(parsed_date)

    return trades


# ──────────────────────────────────────────────────────────────
# Main backtest loop
# ──────────────────────────────────────────────────────────────


def run_backtest() -> MomoStats:
    """Run the MoMo backtest on all CSVs in ``data/historical/momo/``."""
    if not MOMO_DIR.exists():
        print(f"[momo-backtest] Directory not found: {MOMO_DIR}")
        print("  Run with --generate-sample to create synthetic test data.")
        return MomoStats(0, 0, 0.0, 0.0, 0.0, 0.0, None, None)

    csv_files = sorted(MOMO_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[momo-backtest] No CSV files in {MOMO_DIR}.")
        print("  Run with --generate-sample to create synthetic test data.")
        return MomoStats(0, 0, 0.0, 0.0, 0.0, 0.0, None, None)

    brain = AIBrain()
    pdt = PDTTracker()
    news_correlator = NewsCorrelator()
    sympathy_detector = SympathyDetector()
    all_trades: list[MomoTrade] = []
    stocks_seen: set[str] = set()

    for csv_path in csv_files:
        stem = csv_path.stem  # e.g. MULN_2026-01-15
        parts = stem.split("_", 1)
        if len(parts) == 2:
            ticker, trade_date = parts[0], parts[1]
        else:
            ticker, trade_date = stem, "2026-01-01"

        stocks_seen.add(ticker)

        try:
            df = pd.read_csv(csv_path)
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                logger.warning("Skipping %s: missing columns.", csv_path)
                continue

            # Calculate gap pct from first bar if possible
            gap_pct = 0.0
            if len(df) > 0 and "open" in df.columns:
                gap_pct = float(df["open"].iloc[0])

            # Run news correlation analysis asynchronously
            asyncio.run(
                news_correlator.analyze_gap_up(ticker, trade_date, df, gap_pct)
            )

            day_trades = _simulate_file(df, ticker, trade_date, brain, pdt)
            all_trades.extend(day_trades)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing %s: %s", csv_path, exc)

    # Learn sympathy correlations from downloaded data
    sympathy_detector.learn_from_historical_data(str(MOMO_DIR))

    # Save news patterns learned during this backtest run
    news_correlator.save_patterns()

    # Aggregate stats
    total = len(all_trades)
    wins = [t for t in all_trades if t.won]
    losses = [t for t in all_trades if not t.won]
    win_rate = len(wins) / total if total > 0 else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    total_pnl = sum(t.pnl for t in all_trades)
    best = max(all_trades, key=lambda t: t.pnl) if all_trades else None
    worst = min(all_trades, key=lambda t: t.pnl) if all_trades else None

    return MomoStats(
        stocks_analyzed=len(stocks_seen),
        total_trades=total,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_pnl=total_pnl,
        best_trade=best,
        worst_trade=worst,
        trades=all_trades,
    )


# ──────────────────────────────────────────────────────────────
# Sample data generator
# ──────────────────────────────────────────────────────────────

_SAMPLE_TICKERS = [
    "MULN", "BBIO", "NKLA", "CLOV", "AMC", "GME", "SNDL", "EXPR", "BB",
    "NOK", "KOSS", "NAKD", "WISH", "CLVS", "ATER", "PROG", "PHUN", "GFAI",
    "BRQT", "PAYA", "SBFM", "ASTR", "SPRT", "BFRI", "OPAD", "MYPS", "AEYE",
    "COMS", "DPRO", "GXII", "GSAT", "TTOO", "IDEX", "VERB", "AAME", "OCGN",
    "CTRM", "SHIP", "GNUS", "ILUS", "MMAT", "MRIN", "SENS", "ZKIN", "HYMC",
    "GFLO", "SFUN", "LKCO", "MOGO", "MNDR",
]


def generate_sample_data(n: int = 50, seed: int = 42) -> None:
    """
    Generate ``n`` synthetic small-cap gap-up day CSVs for testing.

    Each CSV simulates a realistic gap-up session:
    - Opens +10–40% above a synthetic previous close
    - High-volume first 30 minutes
    - Pullback to VWAP ≈ 45 minutes in
    - Second leg up or breakdown in PM
    """
    rng = random.Random(seed)
    MOMO_DIR.mkdir(parents=True, exist_ok=True)

    # Generate trading dates (last 60 business days)
    base_date = date(2026, 1, 2)
    biz_dates: list[date] = []
    d = base_date
    while len(biz_dates) < 80:
        if d.weekday() < 5:
            biz_dates.append(d)
        d += timedelta(days=1)

    tickers = (_SAMPLE_TICKERS * math.ceil(n / len(_SAMPLE_TICKERS)))[:n]

    created = 0
    for ticker in tickers:
        trade_date = rng.choice(biz_dates)
        filename = MOMO_DIR / f"{ticker}_{trade_date.isoformat()}.csv"
        if filename.exists():
            continue

        prev_close = rng.uniform(2.0, 18.0)
        gap_pct = rng.uniform(0.10, 0.40)
        open_price = round(prev_close * (1 + gap_pct), 2)

        rows = []
        current_price = open_price
        base_vol = rng.randint(300_000, 2_000_000)

        for minute in range(150):  # 9:30 to 12:00 = 150 mins
            hour = 9 + (30 + minute) // 60
            minute_of_hour = (30 + minute) % 60
            ts = datetime(trade_date.year, trade_date.month, trade_date.day, hour, minute_of_hour)

            # Volume profile: high at open, decay, spike at VWAP test
            if minute < 30:
                vol = int(base_vol * rng.uniform(0.8, 1.5))
            elif minute < 50:
                vol = int(base_vol * rng.uniform(0.3, 0.7))
                # Pullback to VWAP zone
                current_price = current_price * rng.uniform(0.992, 0.999)
            elif minute < 70:
                vol = int(base_vol * rng.uniform(0.5, 1.2))
                # Second leg
                current_price = current_price * rng.uniform(0.999, 1.012)
            else:
                vol = int(base_vol * rng.uniform(0.1, 0.4))
                current_price = current_price * rng.uniform(0.996, 1.003)

            bar_range = current_price * rng.uniform(0.005, 0.025)
            open_b = round(current_price * rng.uniform(0.998, 1.002), 4)
            close_b = round(current_price * rng.uniform(0.997, 1.003), 4)
            high_b = round(max(open_b, close_b) + bar_range * rng.uniform(0.3, 0.7), 4)
            low_b = round(min(open_b, close_b) - bar_range * rng.uniform(0.3, 0.7), 4)

            rows.append({
                "time": ts.isoformat(),
                "open": open_b,
                "high": high_b,
                "low": low_b,
                "close": close_b,
                "volume": max(vol, 1000),
            })

            current_price = close_b

        with open(filename, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["time", "open", "high", "low", "close", "volume"])
            writer.writeheader()
            writer.writerows(rows)

        created += 1

    print(f"[momo-backtest] Generated {created} synthetic CSV files in {MOMO_DIR}/")


# ──────────────────────────────────────────────────────────────
# Yahoo Finance downloader
# ──────────────────────────────────────────────────────────────

# Known small-cap movers — stocks that frequently gap up
SMALL_CAP_UNIVERSE = [
    # AI / quantum
    "IONQ", "KULR", "RGTI", "QUBT", "AI", "BBAI", "SOUN", "GFAI",
    # Crypto-related miners
    "MARA", "RIOT", "BITF", "HUT", "CLSK",
    # Biotech
    "NVAX", "MRNA", "BNTX", "BBIO",
    # EV
    "LCID", "RIVN", "GOEV", "FSR", "NKLA",
    # Fintech
    "PLTR", "SOFI", "HOOD", "AFRM", "UPST",
    # Space
    "RKLB", "LUNR", "ASTS", "RDW", "MNTS",
    # Former meme stocks
    "OPEN", "CLOV", "WISH", "SDC",
    # Classic meme stocks
    "AMC", "GME", "KOSS", "BB", "NOK",
    # Other space/energy
    "SPCE", "ASTR", "TELL", "NEXT", "BE", "PLUG", "FCEL",
    # Genomics
    "DNA", "CRSP", "EDIT", "NTLA", "BEAM",
    # Lidar
    "LAZR", "LIDR", "AEVA", "OUST",
    # Speculative
    "HYMC", "GEVO", "REE", "WKHS",
    # Additional active small-caps
    "MULN", "SMCI", "ASTS",
]

_YAHOO_GAP_MIN_PCT = 0.10   # minimum 10% gap-up
_YAHOO_PRICE_MIN = 1.0      # minimum price $1
_YAHOO_PRICE_MAX = 20.0     # maximum price $20
_YAHOO_VOL_MIN = 100_000    # minimum volume 100K
_YAHOO_SCAN_DAYS = 60       # look back 60 trading days
_YAHOO_RECENT_DAYS = 7      # last 7 days → use 1-min bars; older → 5-min bars
_YAHOO_TOP_GAPUPS_PER_DAY = 10  # keep top N gap-ups per day


def _get_trading_days(n_days: int) -> list[date]:
    """Return the last *n_days* trading days (Mon–Fri) ending today."""
    days: list[date] = []
    d = date.today()
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return days  # most-recent first


def _save_bars_to_csv(ticker: str, trade_date: str, df: pd.DataFrame) -> None:
    """Save intraday bars to ``data/historical/momo/TICKER_YYYY-MM-DD.csv``."""
    MOMO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MOMO_DIR / f"{ticker}_{trade_date}.csv"

    # Keep only market-hours rows (9:30–16:00 ET)
    if "datetime" in df.columns or df.index.dtype == "datetime64[ns, America/New_York]":
        try:
            idx = df.index if hasattr(df.index, "hour") else pd.to_datetime(df.index)
            mask = (idx.hour * 60 + idx.minute >= 570) & (idx.hour * 60 + idx.minute < 960)
            df = df[mask]
        except Exception:  # noqa: BLE001
            pass

    if df.empty:
        return

    df = df.rename(columns=str.lower)
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    out_df = df[cols].copy()
    out_df.index.name = "datetime"
    out_df.to_csv(out_path)


def download_yahoo_data() -> None:
    """
    Download real historical gap-up intraday data from Yahoo Finance.

    - Scans SMALL_CAP_UNIVERSE for gap-ups > 10% over last 60 trading days.
    - Downloads 1-min bars for the last 7 days; 5-min bars for older days.
    - Saves each to ``data/historical/momo/TICKER_YYYY-MM-DD.csv``.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        import subprocess  # noqa: S404
        print("[momo-download] Installing yfinance...")
        subprocess.check_call(["pip", "install", "yfinance"])  # noqa: S603,S607
        import yfinance as yf  # type: ignore

    trading_days = _get_trading_days(_YAHOO_SCAN_DAYS)
    today = date.today()
    cutoff_recent = today - timedelta(days=_YAHOO_RECENT_DAYS)

    # Download daily data for the whole universe at once
    print(f"[momo-download] Scanning {len(SMALL_CAP_UNIVERSE)} tickers for gap-ups...")

    tickers_str = " ".join(SMALL_CAP_UNIVERSE)
    try:
        daily = yf.download(
            tickers_str,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[momo-download] Failed to download daily data: {exc}")
        return

    # Identify gap-up events: open vs previous day's close
    gap_events: list[tuple[str, str, float]] = []  # (ticker, date_str, gap_pct)

    close_df = daily.get("Close") if isinstance(daily.columns, pd.MultiIndex) else daily.get("Close")
    open_df = daily.get("Open") if isinstance(daily.columns, pd.MultiIndex) else daily.get("Open")
    volume_df = daily.get("Volume") if isinstance(daily.columns, pd.MultiIndex) else daily.get("Volume")

    if close_df is None or open_df is None:
        print("[momo-download] Unexpected data format from yfinance; aborting.")
        return

    for ticker in SMALL_CAP_UNIVERSE:
        if ticker not in close_df.columns:
            continue
        try:
            closes = close_df[ticker].dropna()
            opens = open_df[ticker].dropna() if ticker in open_df.columns else None
            vols = volume_df[ticker].dropna() if (volume_df is not None and ticker in volume_df.columns) else None

            if opens is None or len(closes) < 2:
                continue

            for i in range(1, len(closes)):
                bar_date = closes.index[i].date()
                if bar_date not in trading_days:
                    continue

                prev_close = float(closes.iloc[i - 1])
                open_price = float(opens.iloc[i]) if i < len(opens) else 0.0
                volume = float(vols.iloc[i]) if (vols is not None and i < len(vols)) else 0.0

                if prev_close <= 0 or open_price <= 0:
                    continue

                gap_pct = (open_price - prev_close) / prev_close
                if (
                    gap_pct >= _YAHOO_GAP_MIN_PCT
                    and _YAHOO_PRICE_MIN <= open_price <= _YAHOO_PRICE_MAX
                    and volume >= _YAHOO_VOL_MIN
                ):
                    gap_events.append((ticker, bar_date.isoformat(), gap_pct))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[momo-download] Skip %s: %s", ticker, exc)

    print(f"[momo-download] Found {len(gap_events)} gap-up events. Downloading intraday bars...")

    downloaded = 0
    failed = 0

    for ticker, date_str, gap_pct in gap_events:
        bar_date = date.fromisoformat(date_str)
        out_path = MOMO_DIR / f"{ticker}_{date_str}.csv"
        if out_path.exists():
            downloaded += 1
            continue

        # Choose interval based on data age
        if bar_date >= cutoff_recent:
            interval = "1m"
            period = "7d"
        else:
            interval = "5m"
            period = "60d"

        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, interval=interval, prepost=False)

            if hist.empty:
                failed += 1
                continue

            # Filter to just this day
            hist.index = hist.index.tz_convert("America/New_York")
            day_mask = hist.index.date == bar_date
            day_df = hist[day_mask]

            if day_df.empty:
                failed += 1
                continue

            _save_bars_to_csv(ticker, date_str, day_df)
            downloaded += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("[momo-download] Failed %s %s: %s", ticker, date_str, exc)
            failed += 1

    print()
    print("[momo-download] Yahoo Finance Download Complete")
    print(f"  Days scanned:     {_YAHOO_SCAN_DAYS}")
    print(f"  Gap-ups found:    {len(gap_events)}")
    print(f"  CSVs downloaded:  {downloaded} ({failed} failed — delisted or no data)")
    print(f"  Saved to:         {MOMO_DIR}/")


# ──────────────────────────────────────────────────────────────
# IBKR historical downloader
# ──────────────────────────────────────────────────────────────


async def _download_ibkr_bars(
    ticker: str,
    date_str: str,
    ib: "Any",  # ib_insync.IB
) -> bool:
    """
    Download 1-min intraday bars from IBKR for *ticker* on *date_str*.

    Returns True on success, False on failure.
    """
    try:
        from ib_insync import Stock  # type: ignore

        contract = Stock(ticker, "SMART", "USD")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=f"{date_str} 16:00:00",
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
        )
        if not bars:
            return False

        df = pd.DataFrame(
            {
                "datetime": [b.date for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            }
        )
        df = df.set_index("datetime")
        _save_bars_to_csv(ticker, date_str, df)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("[momo-download-ibkr] Failed %s %s: %s", ticker, date_str, exc)
        return False


async def _run_ibkr_download() -> None:
    """Connect to IBKR and download 1-min bars for all known gap-up tickers."""
    try:
        from ib_insync import IB  # type: ignore
    except ImportError:
        print("[momo-download] ib_insync is not installed. Cannot use --download-ibkr.")
        return

    ib = IB()
    try:
        await ib.connectAsync("127.0.0.1", 7497, clientId=99)
    except Exception as exc:  # noqa: BLE001
        print(f"[momo-download] Cannot connect to IBKR: {exc}")
        print("  Make sure TWS or IB Gateway is running.")
        return

    print("[momo-download] Connected to IBKR. Collecting tickers from Yahoo data...")

    # Collect ticker/date pairs from previously downloaded Yahoo CSVs
    tickers_and_dates: list[tuple[str, str]] = []
    if MOMO_DIR.exists():
        for csv_path in sorted(MOMO_DIR.glob("*.csv")):
            parts = csv_path.stem.split("_", 1)
            if len(parts) == 2:
                tickers_and_dates.append((parts[0], parts[1]))

    if not tickers_and_dates:
        # Fall back to scanning SMALL_CAP_UNIVERSE for recent trading days
        trading_days = _get_trading_days(60)
        for ticker in SMALL_CAP_UNIVERSE:
            for d in trading_days[:5]:  # last 5 days only for fallback
                tickers_and_dates.append((ticker, d.isoformat()))

    print(f"[momo-download] Downloading {len(tickers_and_dates)} ticker/day pairs from IBKR...")

    downloaded = 0
    failed = 0
    for ticker, date_str in tickers_and_dates:
        # IBKR has more accurate data — overwrite any existing Yahoo CSV
        success = await _download_ibkr_bars(ticker, date_str, ib)
        if success:
            downloaded += 1
        else:
            failed += 1
        await asyncio.sleep(0.4)  # stay within IBKR pacing limits

    ib.disconnect()
    print()
    print("[momo-download] IBKR Download Complete")
    print(f"  Pairs requested:  {len(tickers_and_dates)}")
    print(f"  CSVs downloaded:  {downloaded} ({failed} failed)")
    print(f"  Saved to:         {MOMO_DIR}/")


def download_ibkr_data() -> None:
    """Synchronous entry point for ``--download-ibkr``."""
    asyncio.run(_run_ibkr_download())


# ──────────────────────────────────────────────────────────────
# Output printer
# ──────────────────────────────────────────────────────────────


def print_results(stats: MomoStats) -> None:
    line = "─" * 50
    print(f"\n{line}")
    print("  MoMo Backtest Results")
    print(f"  Stocks analyzed:    {stats.stocks_analyzed}")
    print(f"  Total Trades:       {stats.total_trades} (3/week PDT limited)")
    print(f"  Win Rate:           {stats.win_rate:.0%}")
    print(f"  Avg Win:            +${stats.avg_win:+.2f}")
    print(f"  Avg Loss:           ${stats.avg_loss:.2f}")
    print(f"  Total P&L:          ${stats.total_pnl:+.2f}")
    if stats.best_trade:
        print(f"  Best Trade:         {stats.best_trade.ticker} +${stats.best_trade.pnl:.2f}")
    if stats.worst_trade:
        print(f"  Worst Trade:        {stats.worst_trade.ticker} -${abs(stats.worst_trade.pnl):.2f}")
    print(line)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="MoMo Small-Cap Backtester — 3 Balas de Oro")
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate synthetic small-cap intraday CSVs for testing.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of sample CSVs to generate (default: 50).",
    )
    parser.add_argument(
        "--download-yahoo",
        action="store_true",
        help="Download real historical gap-up data from Yahoo Finance (free, no IBKR needed).",
    )
    parser.add_argument(
        "--download-ibkr",
        action="store_true",
        help="Download 1-min intraday data from IBKR (requires TWS/IB Gateway running).",
    )
    args = parser.parse_args()

    if args.download_yahoo:
        download_yahoo_data()

    if args.download_ibkr:
        download_ibkr_data()

    if args.generate_sample:
        generate_sample_data(n=args.n)

    stats = run_backtest()
    print_results(stats)

    # Save JSON results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "stocks_analyzed": stats.stocks_analyzed,
                "total_trades": stats.total_trades,
                "win_rate": round(stats.win_rate, 4),
                "avg_win": round(stats.avg_win, 4),
                "avg_loss": round(stats.avg_loss, 4),
                "total_pnl": round(stats.total_pnl, 4),
                "best_trade": {
                    "ticker": stats.best_trade.ticker,
                    "pnl": stats.best_trade.pnl,
                } if stats.best_trade else None,
                "worst_trade": {
                    "ticker": stats.worst_trade.ticker,
                    "pnl": stats.worst_trade.pnl,
                } if stats.worst_trade else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            fh,
            indent=2,
        )
    print(f"[momo-backtest] Results saved to {RESULTS_FILE}")

    # Save JSON results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "stocks_analyzed": stats.stocks_analyzed,
                "total_trades": stats.total_trades,
                "win_rate": round(stats.win_rate, 4),
                "avg_win": round(stats.avg_win, 4),
                "avg_loss": round(stats.avg_loss, 4),
                "total_pnl": round(stats.total_pnl, 4),
                "best_trade": {
                    "ticker": stats.best_trade.ticker,
                    "pnl": stats.best_trade.pnl,
                } if stats.best_trade else None,
                "worst_trade": {
                    "ticker": stats.worst_trade.ticker,
                    "pnl": stats.worst_trade.pnl,
                } if stats.worst_trade else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            fh,
            indent=2,
        )
    print(f"[momo-backtest] Results saved to {RESULTS_FILE}")
