"""
MoMo Small-Cap Backtester — "3 Balas de Oro" Strategy.

Tests the momentum gap-up strategy on 1-minute intraday data for small-cap
stocks.  Supports an arbitrary number of CSV files dropped in
``data/historical/momo/`` (named ``TICKER_YYYY-MM-DD.csv``).

Usage:
    python backtest_momo.py                         # run backtest
    python backtest_momo.py --generate-sample       # create 50 synthetic CSVs
    python backtest_momo.py --generate-sample --n 100   # create N synthetic CSVs
"""

from __future__ import annotations

import argparse
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
from core.risk_manager import RiskManager
from core.reto_tracker import RetoTracker

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

            day_trades = _simulate_file(df, ticker, trade_date, brain, pdt)
            all_trades.extend(day_trades)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing %s: %s", csv_path, exc)

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
    args = parser.parse_args()

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
