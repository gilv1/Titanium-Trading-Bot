"""
MoMo Small-Cap Backtester — Definitive Overhaul (18 improvements).

Strategy rules (from whiteboard):
  - +10% gap at open
  - High Relative Volume today
  - News catalyst
  - "Obvious" setup — if you have to think about it, it's not a trade
  - Float under 10 million shares
  - Price under $20
  - Analyze Daily & Intraday charts
  - Buy Pullbacks, Dips, Breakouts

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
from typing import Any, Optional

import pandas as pd

from analysis.technical import calculate_ema, calculate_vwap
from config import settings
from core.brain import AIBrain, BrainMemory, TradeOutcome
from core.news_correlator import NewsCorrelator
from core.risk_manager import RiskManager
from core.reto_tracker import RetoTracker
from core.sympathy_detector import SympathyDetector

logger = logging.getLogger(__name__)

MOMO_DIR = Path("data/historical/momo")
RESULTS_FILE = Path("journal/momo_backtest_results.json")

# ── PDT limits ──
PDT_MAX_TRADES_PER_WINDOW = 3
PDT_ROLLING_DAYS = 5

# ── Trading hours (24-hour, Eastern) ──
SESSION_START = (9, 30)
SESSION_END = (12, 0)

# ── Signal detection tolerances ──
VWAP_TOLERANCE_PCT = 0.02     # within 2% of VWAP (small-caps are volatile)
EMA_PULLBACK_TOL_PCT = 0.015  # within 1.5% of EMA9
VWAP_STOP_BUFFER = 0.005      # stop 0.5% below VWAP

# ── Capital and position sizing (improvement #15) ──
CAPITAL = 500.0            # starting capital
RISK_PCT_PER_TRADE = 0.02  # 2% risk per trade = $10
MAX_POSITION_PCT = 0.20    # max 20% of capital in one trade

# ── ATR-based stop loss (improvement #7) ──
ATR_STOP_MULTIPLIER = 1.5   # stop = 1.5× ATR below entry
ATR_TRAIL_MULTIPLIER = 1.0  # trailing stop = 1× ATR below running high
MIN_STOP_PCT = 0.04         # minimum 4% stop
MAX_STOP_PCT = 0.08         # maximum 8% stop

# ── Scaled exit R-multiples (improvements #8, #9, #10) ──
BREAKEVEN_R = 1.0   # move stop to break-even at +1R
EXIT1_R = 1.5       # sell 50% at +1.5R
EXIT2_R = 2.5       # sell 25% at +2.5R
TRAIL_R = 2.0       # switch to ATR-trailing stop after +2R (remaining 25%)

# ── Anti-chase limits (improvement #4) ──
MAX_ABOVE_VWAP_TO_ENTER = 0.15     # skip if price >15% above VWAP
MAX_ABOVE_OPEN_TO_ENTER = 0.15     # skip if price >15% above open
MAX_CONSEC_GREEN_BEFORE_ENTRY = 3  # skip if 3+ consecutive green candles

# ── Score thresholds (improvement #5) ──
SCORE_FULL = 75   # full position
SCORE_HALF = 65   # half position; below 65 = no entry

# ── In-memory daily-trend cache (improvement #13) ──
_daily_trend_cache: dict[tuple[str, str], str] = {}


# ──────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────


@dataclass
class MomoSetup:
    """Pre-screened setup candidate before execution (used for smart PDT ranking)."""

    ticker: str
    trade_date: str
    parsed_date: date
    gap_pct: float
    rel_volume: float
    catalyst: str
    float_shares: float
    price: float
    daily_trend: str
    score: int
    obvious_passed: bool
    entry_type: str       # "Pullback", "Dip", or "Breakout"
    entry_bar: int
    entry_price: float
    raw_stop: float
    hour_of_day: int
    df: pd.DataFrame


@dataclass
class MomoTrade:
    ticker: str
    date: str
    direction: str
    entry: float
    stop: float
    target1: float        # first target (+1.5R)
    entry_bar: int
    exit_bar: int
    pnl: float            # total realized P&L (all tranches)
    won: bool
    setup_type: str
    score: int = 0
    entry_type: str = ""  # "Pullback", "Dip", "Breakout"
    catalyst: str = ""
    shares: int = 1
    hour_of_day: int = 9


@dataclass
class MomoStats:
    stocks_analyzed: int
    gap_ups_detected: int
    setups_scored: int
    passed_obvious: int
    passed_score_filter: int
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float
    total_pnl: float
    max_drawdown: float
    best_trade: Optional[MomoTrade]
    worst_trade: Optional[MomoTrade]
    trades: list[MomoTrade] = field(default_factory=list)
    wins_by_hour: dict = field(default_factory=dict)
    trades_by_hour: dict = field(default_factory=dict)
    wins_by_entry_type: dict = field(default_factory=dict)
    trades_by_entry_type: dict = field(default_factory=dict)
    wins_by_catalyst: dict = field(default_factory=dict)
    trades_by_catalyst: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# PDT tracker
# ──────────────────────────────────────────────────────────────


class PDTTracker:
    """Track round-trip day trades to stay within 3 per 5 rolling business days."""

    def __init__(self) -> None:
        self._trade_dates: list[date] = []

    def _business_days_window(self, reference_date: date) -> list[date]:
        window: list[date] = []
        d = reference_date
        while len(window) < PDT_ROLLING_DAYS:
            if d.weekday() < 5:
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
# Helper: time filter (improvement #3)
# ──────────────────────────────────────────────────────────────


def _min_score_for_time(hour: int, minute: int) -> int:
    """Return the minimum score required to enter at this time of day."""
    t = hour * 60 + minute
    if t < 10 * 60 + 15:       # 9:30–10:15  — best window
        return 60
    elif t < 11 * 60:           # 10:15–11:00
        return 70
    elif t < 11 * 60 + 30:     # 11:00–11:30
        return 80
    else:                        # 11:30+  — almost never enters
        return 90


# ──────────────────────────────────────────────────────────────
# Helper: ATR calculation
# ──────────────────────────────────────────────────────────────


def _compute_atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
    """Compute Average True Range for the bar at index *i*."""
    start = max(0, i - period)
    window = df.iloc[start : i + 1]
    if len(window) < 2:
        return float(df["close"].iloc[i]) * 0.02
    trs = []
    for j in range(1, len(window)):
        h = float(window["high"].iloc[j])
        lo = float(window["low"].iloc[j])
        pc = float(window["close"].iloc[j - 1])
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs) if trs else float(df["close"].iloc[i]) * 0.02


# ──────────────────────────────────────────────────────────────
# Helper: relative volume (improvement #2)
# ──────────────────────────────────────────────────────────────


def _calc_rel_volume(df: pd.DataFrame, first_n: int = 30) -> float:
    """Compute relative volume: average of first N bars vs. average of the rest."""
    if len(df) < first_n + 5:
        return 1.0
    early_vol = float(df["volume"].iloc[:first_n].mean())
    rest_vol = float(df["volume"].iloc[first_n:].mean())
    if rest_vol <= 0:
        return max(1.0, early_vol / 1000.0)
    return early_vol / rest_vol


# ──────────────────────────────────────────────────────────────
# Helper: daily chart trend analysis (improvement #13)
# ──────────────────────────────────────────────────────────────


def _analyze_daily_trend(ticker: str, trade_date: date) -> str:
    """
    Check daily chart trend for the last 10 trading days.

    Returns "uptrend", "downtrend", or "flat".
    Uses yfinance if available; falls back to "flat".
    """
    cache_key = (ticker, trade_date.isoformat())
    if cache_key in _daily_trend_cache:
        return _daily_trend_cache[cache_key]

    trend = "flat"
    try:
        import yfinance as yf  # type: ignore

        # Suppress yfinance's verbose logging during the download
        _yf_logger = logging.getLogger("yfinance")
        _prev_level = _yf_logger.level
        _yf_logger.setLevel(logging.CRITICAL)
        try:
            start = (trade_date - timedelta(days=30)).isoformat()
            end = (trade_date + timedelta(days=1)).isoformat()
            hist = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        finally:
            _yf_logger.setLevel(_prev_level)
        if hist.empty or len(hist) < 10:
            trend = "flat"
        else:
            closes = hist["Close"].values.astype(float).tolist()

            def _ema(data: list[float], period: int) -> list[float]:
                k = 2.0 / (period + 1)
                result = [data[0]]
                for v in data[1:]:
                    result.append(v * k + result[-1] * (1 - k))
                return result

            e5 = _ema(closes, 5)
            e10 = _ema(closes, 10)
            e20 = _ema(closes, 20)
            if e5[-1] > e10[-1] > e20[-1]:
                trend = "uptrend"
            elif e5[-1] < e10[-1] < e20[-1]:
                trend = "downtrend"
            else:
                trend = "flat"
    except Exception:  # noqa: BLE001
        trend = "flat"

    _daily_trend_cache[cache_key] = trend
    return trend


# ──────────────────────────────────────────────────────────────
# Setup quality score (improvement #5)
# ──────────────────────────────────────────────────────────────


def _compute_setup_score(
    gap_pct: float,
    rel_volume: float,
    catalyst: str,
    float_shares: float,
    price: float,
    daily_trend: str,
    sector_penalty: int = 0,
    has_offering: bool = False,
) -> tuple[int, bool]:
    """
    Compute setup quality score 0–100.

    Returns (score, should_skip).
    should_skip=True means don't trade regardless of score
    (e.g. offering detected, float > 20M, insufficient volume).
    """
    # Red flag: stock offering → immediate skip (improvement #11)
    if has_offering:
        return 0, True

    score = 0

    # Gap size scoring (10% minimum already enforced upstream)
    if 0.10 <= gap_pct <= 0.25:
        score += 20   # sweet spot
    elif 0.25 < gap_pct <= 0.50:
        score += 15   # good but risky
    elif gap_pct > 0.50:
        score += 5    # too extended, likely to reverse

    # Relative Volume scoring (MANDATORY: < 2× = no entry)
    if rel_volume >= 5.0:
        score += 20   # monster volume
    elif rel_volume >= 3.0:
        score += 15   # strong volume
    elif rel_volume >= 2.0:
        score += 5    # marginal
    else:
        return 0, True   # insufficient volume → skip trade

    # News/Catalyst scoring
    cat = catalyst or "unknown"
    if cat in ("fda_approval", "contract_announcement"):
        score += 20
    elif cat in ("earnings_beat", "partnership", "merger_acquisition"):
        score += 15
    elif cat in ("analyst_upgrade", "insider_buying"):
        score += 10
    # "unknown" / "sector_momentum" → +0

    # Float scoring (from whiteboard: float under 10M is the rule)
    if float_shares > 0:
        if float_shares < 10_000_000:
            score += 20   # low float = explosive
        elif float_shares < 20_000_000:
            score += 10   # moderate float
        else:
            return score, True   # > 20M float → hard skip (whiteboard rule)
    else:
        score += 10   # unknown float → moderate bonus; no hard skip

    # Price range scoring ($2–$20 from whiteboard)
    if 2.0 <= price <= 10.0:
        score += 15   # sweet spot
    elif 1.0 <= price < 2.0:
        score += 10   # penny stock territory
    elif 10.0 < price <= 20.0:
        score += 10   # higher price, less explosive

    # Daily chart trend (improvement #13)
    if daily_trend == "uptrend":
        score += 10
    elif daily_trend == "flat":
        score += 5
    elif daily_trend == "downtrend":
        score += 0    # dead cat bounce danger → no bonus

    # Sector penalty (improvement #12)
    score -= sector_penalty

    return max(0, score), False


# ──────────────────────────────────────────────────────────────
# "Obvious" check (improvement #6)
# ──────────────────────────────────────────────────────────────


def _obvious_check(
    gap_pct: float,
    rel_volume: float,
    has_news_catalyst: bool,
    float_shares: float,
    price: float,
) -> bool:
    """
    At least 3 of 5 criteria must be GREEN for the setup to be 'OBVIOUS'.
    If you have to think about it, it's not a trade (whiteboard rule).
    """
    passed = 0
    if gap_pct >= 0.10:
        passed += 1
    if rel_volume >= 3.0:
        passed += 1
    if has_news_catalyst:
        passed += 1
    if 0 < float_shares < 10_000_000:
        passed += 1
    if 2.0 <= price <= 20.0:
        passed += 1
    return passed >= 3


# ──────────────────────────────────────────────────────────────
# Anti-chase helpers (improvement #4)
# ──────────────────────────────────────────────────────────────


def _count_consecutive_green(df: pd.DataFrame, i: int) -> int:
    """Count consecutive green candles ending at bar *i*."""
    count = 0
    for j in range(i, -1, -1):
        o = float(df["open"].iloc[j])
        c = float(df["close"].iloc[j])
        if c > o:
            count += 1
        else:
            break
    return count


def _check_anti_chase(
    price: float,
    vwap_price: float,
    open_price: float,
    consec_green: int,
) -> bool:
    """
    Returns True if it is SAFE to enter (not chasing).
    Returns False if the entry should be skipped.
    """
    if vwap_price > 0 and (price - vwap_price) / vwap_price > MAX_ABOVE_VWAP_TO_ENTER:
        return False
    if open_price > 0 and (price - open_price) / open_price > MAX_ABOVE_OPEN_TO_ENTER:
        return False
    if consec_green >= MAX_CONSEC_GREEN_BEFORE_ENTRY:
        return False
    return True


# ──────────────────────────────────────────────────────────────
# Entry type A: Pullback to VWAP / EMA9 (improvement #14)
# ──────────────────────────────────────────────────────────────


def _detect_pullback_entry(
    df: pd.DataFrame,
    i: int,
    vwap: pd.Series,
    ema9: pd.Series,
) -> Optional[dict]:
    """
    Detect pullback to VWAP or EMA9 confirmed by 2 consecutive green candles.
    Entry at close of 2nd green candle; stop slightly below the support level.
    """
    if i < 5:
        return None

    # Need 2 consecutive green candles ending at bar i
    if float(df["close"].iloc[i]) <= float(df["open"].iloc[i]):
        return None
    if float(df["close"].iloc[i - 1]) <= float(df["open"].iloc[i - 1]):
        return None

    # Check if price was near VWAP or EMA9 within the last 5 bars (before the bounce)
    found_level = False
    sl_ref = 0.0
    lookback_start = max(0, i - 5)
    for j in range(lookback_start, i - 1):
        bar_close = float(df["close"].iloc[j])
        v = float(vwap.iloc[j]) if j < len(vwap) else 0.0
        e9 = float(ema9.iloc[j]) if j < len(ema9) else 0.0

        if v > 0 and abs(bar_close - v) / v <= VWAP_TOLERANCE_PCT:
            found_level = True
            sl_ref = v
            break
        if e9 > 0 and abs(bar_close - e9) / e9 <= EMA_PULLBACK_TOL_PCT:
            found_level = True
            sl_ref = e9
            break

    if not found_level or sl_ref <= 0:
        return None

    entry_price = float(df["close"].iloc[i])
    stop_price = sl_ref * (1.0 - VWAP_STOP_BUFFER)

    return {
        "entry": entry_price,
        "sl": stop_price,
        "setup": "PULLBACK_VWAP_EMA",
        "entry_type": "Pullback",
    }


# ──────────────────────────────────────────────────────────────
# Entry type B: Dip Buy (improvement #14)
# ──────────────────────────────────────────────────────────────


def _detect_dip_buy(df: pd.DataFrame, i: int) -> Optional[dict]:
    """
    Detect a sharp dip (>3% in 2–3 candles) followed by a quick recovery
    with a volume spike on the recovery candle.
    """
    if i < 7:
        return None

    lookback = df.iloc[max(0, i - 6) : i + 1]
    if len(lookback) < 4:
        return None

    # Locate the dip low within the window
    local_low_pos = int(lookback["low"].values.argmin())
    if local_low_pos <= 0 or local_low_pos >= len(lookback) - 1:
        return None  # dip must be in the middle, not at the edges

    pre_dip_high = float(lookback["high"].iloc[:local_low_pos].max())
    dip_low = float(lookback["low"].iloc[local_low_pos])

    if pre_dip_high <= 0:
        return None

    dip_pct = (pre_dip_high - dip_low) / pre_dip_high
    if dip_pct < 0.03:   # need > 3% dip
        return None

    # Price must have recovered at least 50% of the dip
    current_close = float(df["close"].iloc[i])
    recovery_level = dip_low + (pre_dip_high - dip_low) * 0.50
    if current_close < recovery_level:
        return None

    # Volume spike on current (recovery) candle
    avg_vol = float(df["volume"].iloc[max(0, i - 20) : i].mean())
    curr_vol = float(df["volume"].iloc[i])
    if avg_vol > 0 and curr_vol < avg_vol * 2.0:
        return None

    return {
        "entry": current_close,
        "sl": dip_low * 0.995,
        "setup": "DIP_BUY",
        "entry_type": "Dip",
    }


# ──────────────────────────────────────────────────────────────
# Entry type C: Breakout (improvement #14)
# ──────────────────────────────────────────────────────────────


def _detect_breakout_entry(df: pd.DataFrame, i: int) -> Optional[dict]:
    """
    Detect a breakout above the High of Day (HOD) or a consolidation range.
    Requires a volume surge (>2× average).
    """
    if i < 10:
        return None

    current_close = float(df["close"].iloc[i])
    current_high = float(df["high"].iloc[i])
    avg_vol = float(df["volume"].iloc[max(0, i - 20) : i].mean())
    curr_vol = float(df["volume"].iloc[i])

    if avg_vol <= 0 or curr_vol < avg_vol * 2.0:
        return None   # volume surge required

    # HOD breakout
    prev_hod = float(df["high"].iloc[:i].max())
    if current_high > prev_hod * 1.001:   # tiny buffer to avoid noise
        return {
            "entry": current_close,
            "sl": prev_hod * 0.99,
            "setup": "BREAKOUT_HOD",
            "entry_type": "Breakout",
        }

    # Consolidation breakout: 3+ tight bars (< 1% range) then price breaks above
    if i >= 15:
        lookback = df.iloc[max(0, i - 10) : i]
        tight_bars = sum(
            1
            for j in range(len(lookback))
            if (
                float(lookback["close"].iloc[j]) > 0
                and (float(lookback["high"].iloc[j]) - float(lookback["low"].iloc[j]))
                / float(lookback["close"].iloc[j])
                < 0.01
            )
        )
        if tight_bars >= 3:
            consol_high = float(lookback["high"].max())
            consol_low = float(lookback["low"].min())
            if current_close > consol_high:
                return {
                    "entry": current_close,
                    "sl": consol_low * 0.99,
                    "setup": "BREAKOUT_CONSOL",
                    "entry_type": "Breakout",
                }

    return None


# ──────────────────────────────────────────────────────────────
# Pre-scan: find setup candidates from one day's CSV
# ──────────────────────────────────────────────────────────────


def _pre_scan_file(
    df: pd.DataFrame,
    ticker: str,
    trade_date: str,
    gap_pct: float,
    news_correlator: NewsCorrelator,
    float_shares: float = 0.0,
    sector_penalty: int = 0,
) -> list[MomoSetup]:
    """
    Scan one day's intraday data for MoMo setup candidates.

    Returns a list with at most one MomoSetup per day (the first valid signal).
    """
    try:
        parsed_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    except ValueError:
        parsed_date = date.today()

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

    if len(df) < 15 or gap_pct < 0.10:
        return []

    open_price = float(df["open"].iloc[0])
    if not (1.0 <= open_price <= 20.0):
        return []

    # Relative volume
    rel_volume = _calc_rel_volume(df)

    # Catalyst (best-effort; falls back to "unknown" when offline)
    catalyst = "unknown"
    has_offering = False
    try:
        # Suppress yfinance verbose logging during news lookup
        _yf_logger = logging.getLogger("yfinance")
        _prev_yf = _yf_logger.level
        _yf_logger.setLevel(logging.CRITICAL)
        try:
            catalyst = news_correlator.classify_ticker_news(ticker, gap_pct)
        finally:
            _yf_logger.setLevel(_prev_yf)
        has_offering = catalyst == "stock_offering"
    except Exception:  # noqa: BLE001
        pass

    has_news = catalyst not in ("unknown", "sector_momentum", "")

    # Daily trend
    daily_trend = _analyze_daily_trend(ticker, parsed_date)

    # Quality score
    score, should_skip = _compute_setup_score(
        gap_pct=gap_pct,
        rel_volume=rel_volume,
        catalyst=catalyst,
        float_shares=float_shares,
        price=open_price,
        daily_trend=daily_trend,
        sector_penalty=sector_penalty,
        has_offering=has_offering,
    )

    if should_skip:
        return []

    # "Obvious" check (3 of 5)
    obvious = _obvious_check(gap_pct, rel_volume, has_news, float_shares, open_price)
    if not obvious:
        return []

    # Find first valid entry signal in the session
    vwap = calculate_vwap(df)
    ema9 = calculate_ema(df, 9)

    for i in range(5, len(df)):
        # Determine bar time
        if "time" in df.columns:
            try:
                t = pd.to_datetime(df["time"].iloc[i])
                hour, minute = int(t.hour), int(t.minute)
            except Exception:  # noqa: BLE001
                hour, minute = 9, 30
        else:
            total_minutes = SESSION_START[0] * 60 + SESSION_START[1] + i
            hour = total_minutes // 60
            minute = total_minutes % 60

        # Time-based score minimum (improvement #3)
        min_score = _min_score_for_time(hour, minute)
        if score < min_score:
            if hour * 60 + minute >= 12 * 60:
                break   # past noon — stop looking
            continue

        # Volume filter on entry bar (improvement #2)
        avg_vol = float(df["volume"].iloc[max(0, i - 20) : i].mean())
        bar_vol = float(df["volume"].iloc[i])
        if avg_vol > 0 and bar_vol < avg_vol * 2.0:
            continue

        # Anti-chase protection (improvement #4)
        price = float(df["close"].iloc[i])
        vwap_val = float(vwap.iloc[i]) if i < len(vwap) and not vwap.empty else 0.0
        consec_green = _count_consecutive_green(df, i)
        if not _check_anti_chase(price, vwap_val, open_price, consec_green):
            continue

        # Red flag: 3+ consecutive red candles — wait, don't enter (improvement #11)
        consec_red = 0
        for j in range(i, max(-1, i - 4), -1):
            if float(df["close"].iloc[j]) < float(df["open"].iloc[j]):
                consec_red += 1
            else:
                break
        if consec_red >= 3:
            continue

        # Red flag: decreasing volume over last 3 bars (improvement #11)
        if i >= 3:
            recent_vols = df["volume"].iloc[i - 3 : i + 1].values
            if len(recent_vols) == 4 and all(
                recent_vols[k] < recent_vols[k - 1] for k in range(1, 4)
            ):
                continue

        # Try entry types: Pullback → Dip → Breakout (improvement #14)
        vwap_slice = vwap.iloc[: i + 1] if not vwap.empty else vwap
        ema9_slice = ema9.iloc[: i + 1] if not ema9.empty else ema9

        sig = (
            _detect_pullback_entry(df, i, vwap_slice, ema9_slice)
            or _detect_dip_buy(df, i)
            or _detect_breakout_entry(df, i)
        )

        if sig is None:
            continue

        return [
            MomoSetup(
                ticker=ticker,
                trade_date=trade_date,
                parsed_date=parsed_date,
                gap_pct=gap_pct,
                rel_volume=rel_volume,
                catalyst=catalyst,
                float_shares=float_shares,
                price=price,
                daily_trend=daily_trend,
                score=score,
                obvious_passed=obvious,
                entry_type=sig["entry_type"],
                entry_bar=i,
                entry_price=sig["entry"],
                raw_stop=sig["sl"],
                hour_of_day=hour,
                df=df,
            )
        ]

    return []   # no valid entry found


# ──────────────────────────────────────────────────────────────
# Smart PDT selection (improvement #16)
# ──────────────────────────────────────────────────────────────


def _select_top_setups_smart_pdt(setups: list[MomoSetup]) -> list[MomoSetup]:
    """
    Rank ALL setups by score and take the TOP 3 per calendar week.

    This ensures we never waste a PDT bullet on a mediocre setup when
    a better one arrives later in the same week.
    """
    # Group by calendar week (Monday start)
    groups: dict[date, list[MomoSetup]] = {}
    for setup in setups:
        week_start = setup.parsed_date - timedelta(days=setup.parsed_date.weekday())
        groups.setdefault(week_start, []).append(setup)

    selected: list[MomoSetup] = []
    for week_start in sorted(groups.keys()):
        week_setups = sorted(groups[week_start], key=lambda s: s.score, reverse=True)
        selected.extend(week_setups[:PDT_MAX_TRADES_PER_WINDOW])
    return selected


# ──────────────────────────────────────────────────────────────
# Trade simulator (improvements #7–#10, #15)
# ──────────────────────────────────────────────────────────────


def _simulate_trade(setup: MomoSetup, brain: AIBrain) -> Optional[MomoTrade]:
    """
    Simulate a single MoMo trade using:
    - ATR-based dynamic stop loss (4%–8% bounds) or VWAP stop (tighter wins)
    - Break-even move at +1R
    - Scaled exits: 50% at +1.5R, 25% at +2.5R, 25% with ATR trailing stop
    - Risk-based position sizing (2% of $500 capital)
    - Brain approval check

    Returns MomoTrade on completion, None if brain rejects or data is invalid.
    """
    df = setup.df
    i = setup.entry_bar
    entry_price = setup.entry_price

    if i >= len(df) or entry_price <= 0:
        return None

    # ── ATR-based dynamic stop loss (improvement #7) ──
    atr = _compute_atr(df, i)
    vwap = calculate_vwap(df)
    vwap_val = float(vwap.iloc[i]) if i < len(vwap) and not vwap.empty else 0.0

    atr_stop = entry_price - ATR_STOP_MULTIPLIER * atr
    vwap_stop = vwap_val * (1.0 - VWAP_STOP_BUFFER) if vwap_val > 0 else 0.0

    # Use the tighter (higher) stop
    stop_price = max(atr_stop, vwap_stop)

    # Enforce min/max stop distance
    if entry_price > 0:
        stop_pct = (entry_price - stop_price) / entry_price
        stop_pct = max(MIN_STOP_PCT, min(MAX_STOP_PCT, stop_pct))
        stop_price = entry_price * (1.0 - stop_pct)

    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        risk_per_share = entry_price * MIN_STOP_PCT
        stop_price = entry_price - risk_per_share

    # ── R-multiple target levels ──
    be_level = entry_price + BREAKEVEN_R * risk_per_share    # +1R  → move SL to BE
    tp1_level = entry_price + EXIT1_R * risk_per_share       # +1.5R → sell 50%
    tp2_level = entry_price + EXIT2_R * risk_per_share       # +2.5R → sell 25%
    trail_level = entry_price + TRAIL_R * risk_per_share     # +2R  → start trailing

    # ── Brain evaluation ──
    decision = brain.evaluate_trade(
        setup_type=setup.entry_type or "MOMO",
        engine="momo_backtest",
        entry=entry_price,
        stop=stop_price,
        target=tp1_level,
        session="NY",
        atr=atr,
    )
    if not decision.approved:
        return None

    # ── Risk-based position sizing (improvement #15) ──
    risk_per_trade = RISK_PCT_PER_TRADE * CAPITAL   # $10
    shares_full = int(risk_per_trade / risk_per_share)
    max_shares = int(MAX_POSITION_PCT * CAPITAL / entry_price)
    shares = min(max(shares_full, 1), max(max_shares, 1))

    # Half size for borderline setups (improvement #5)
    if setup.score < SCORE_FULL:
        shares = max(1, shares // 2)

    # ── Simulate bar by bar ──
    remaining = shares
    realized_pnl = 0.0
    sl = stop_price
    trail_high = entry_price
    trail_mode = False
    be_moved = False
    exit1_done = False
    exit2_done = False
    exit_bar = len(df) - 1

    for j in range(i + 1, len(df)):
        bar = df.iloc[j]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        # Update trailing high
        if bar_high > trail_high:
            trail_high = bar_high

        # Break-even move at +1R (improvement #8)
        if not be_moved and bar_high >= be_level:
            sl = max(sl, entry_price)
            be_moved = True

        # Switch to ATR trailing stop at +2R (improvement #10)
        if bar_high >= trail_level:
            trail_mode = True
        if trail_mode:
            current_atr = _compute_atr(df, j)
            trail_sl = trail_high - ATR_TRAIL_MULTIPLIER * current_atr
            sl = max(sl, trail_sl)   # trail only moves up

        # Scaled exit 1: sell 50% at +1.5R (improvement #9)
        if not exit1_done and bar_high >= tp1_level:
            exit1_shares = max(1, shares // 2)
            realized_pnl += (tp1_level - entry_price) * exit1_shares
            remaining = max(0, remaining - exit1_shares)
            exit1_done = True
            if remaining == 0:
                exit_bar = j
                break

        # Scaled exit 2: sell 25% at +2.5R (improvement #9)
        if not exit2_done and exit1_done and bar_high >= tp2_level:
            exit2_shares = max(1, shares // 4)
            realized_pnl += (tp2_level - entry_price) * exit2_shares
            remaining = max(0, remaining - exit2_shares)
            exit2_done = True
            if remaining == 0:
                exit_bar = j
                break

        # Stop loss hit
        if bar_low <= sl and remaining > 0:
            realized_pnl += (sl - entry_price) * remaining
            remaining = 0
            exit_bar = j
            break

    # Close any remaining position at session end
    if remaining > 0:
        close_px = float(df["close"].iloc[-1])
        realized_pnl += (close_px - entry_price) * remaining
        remaining = 0
        exit_bar = len(df) - 1

    won = realized_pnl > 0

    return MomoTrade(
        ticker=setup.ticker,
        date=setup.trade_date,
        direction="LONG",
        entry=entry_price,
        stop=stop_price,
        target1=tp1_level,
        entry_bar=i,
        exit_bar=exit_bar,
        pnl=round(realized_pnl, 4),
        won=won,
        setup_type=setup.entry_type,
        score=setup.score,
        entry_type=setup.entry_type,
        catalyst=setup.catalyst,
        shares=shares,
        hour_of_day=setup.hour_of_day,
    )


# ──────────────────────────────────────────────────────────────
# Statistics aggregation (improvement #17)
# ──────────────────────────────────────────────────────────────


def _aggregate_stats(
    stocks_analyzed: int,
    gap_ups_detected: int,
    setups_scored: int,
    passed_obvious: int,
    passed_score_filter: int,
    all_trades: list[MomoTrade],
) -> MomoStats:
    """Compute all backtest statistics from a list of completed trades."""
    total = len(all_trades)
    wins = [t for t in all_trades if t.won]
    losses = [t for t in all_trades if not t.won]

    win_rate = len(wins) / total if total > 0 else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    total_pnl = sum(t.pnl for t in all_trades)

    gross_wins = sum(t.pnl for t in wins)
    gross_losses = abs(sum(t.pnl for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    expectancy = (
        win_rate * avg_win + (1.0 - win_rate) * avg_loss if total > 0 else 0.0
    )

    # Max drawdown
    running_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in all_trades:
        running_pnl += t.pnl
        if running_pnl > peak:
            peak = running_pnl
        dd = peak - running_pnl
        if dd > max_dd:
            max_dd = dd

    # ── Breakdown by hour ──
    hour_labels = ["9:30-10:15", "10:15-11:00", "11:00+"]
    wins_by_hour: dict[str, int] = {k: 0 for k in hour_labels}
    trades_by_hour: dict[str, int] = {k: 0 for k in hour_labels}

    def _hour_label(hour: int) -> str:
        t = hour * 60
        if t < 10 * 60 + 15:
            return "9:30-10:15"
        elif t < 11 * 60:
            return "10:15-11:00"
        return "11:00+"

    for t in all_trades:
        lbl = _hour_label(t.hour_of_day)
        trades_by_hour[lbl] = trades_by_hour.get(lbl, 0) + 1
        if t.won:
            wins_by_hour[lbl] = wins_by_hour.get(lbl, 0) + 1

    # ── Breakdown by entry type ──
    wins_by_entry: dict[str, int] = {}
    trades_by_entry: dict[str, int] = {}
    for t in all_trades:
        et = t.entry_type or t.setup_type
        trades_by_entry[et] = trades_by_entry.get(et, 0) + 1
        if t.won:
            wins_by_entry[et] = wins_by_entry.get(et, 0) + 1

    # ── Breakdown by catalyst ──
    wins_by_cat: dict[str, int] = {}
    trades_by_cat: dict[str, int] = {}
    for t in all_trades:
        cat = t.catalyst or "unknown"
        if cat in ("fda_approval", "contract_announcement"):
            cat_label = "FDA/Contract"
        elif cat in ("earnings_beat", "partnership", "merger_acquisition"):
            cat_label = "Earnings/Partnership"
        elif cat in ("analyst_upgrade", "insider_buying"):
            cat_label = "Analyst/Insider"
        else:
            cat_label = "Unknown"
        trades_by_cat[cat_label] = trades_by_cat.get(cat_label, 0) + 1
        if t.won:
            wins_by_cat[cat_label] = wins_by_cat.get(cat_label, 0) + 1

    best = max(all_trades, key=lambda t: t.pnl) if all_trades else None
    worst = min(all_trades, key=lambda t: t.pnl) if all_trades else None

    return MomoStats(
        stocks_analyzed=stocks_analyzed,
        gap_ups_detected=gap_ups_detected,
        setups_scored=setups_scored,
        passed_obvious=passed_obvious,
        passed_score_filter=passed_score_filter,
        total_trades=total,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        expectancy=expectancy,
        total_pnl=total_pnl,
        max_drawdown=max_dd,
        best_trade=best,
        worst_trade=worst,
        trades=all_trades,
        wins_by_hour=wins_by_hour,
        trades_by_hour=trades_by_hour,
        wins_by_entry_type=wins_by_entry,
        trades_by_entry_type=trades_by_entry,
        wins_by_catalyst=wins_by_cat,
        trades_by_catalyst=trades_by_cat,
    )


def _empty_stats() -> MomoStats:
    return MomoStats(
        stocks_analyzed=0,
        gap_ups_detected=0,
        setups_scored=0,
        passed_obvious=0,
        passed_score_filter=0,
        total_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        win_loss_ratio=0.0,
        expectancy=0.0,
        total_pnl=0.0,
        max_drawdown=0.0,
        best_trade=None,
        worst_trade=None,
    )


# ──────────────────────────────────────────────────────────────
# Main backtest loop (two-phase: pre-scan → smart PDT → simulate)
# ──────────────────────────────────────────────────────────────


def run_backtest() -> MomoStats:
    """Run the MoMo backtest on all CSVs in ``data/historical/momo/``."""
    if not MOMO_DIR.exists():
        print(f"[momo-backtest] Directory not found: {MOMO_DIR}")
        print("  Run with --generate-sample to create synthetic test data.")
        return _empty_stats()

    csv_files = sorted(MOMO_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[momo-backtest] No CSV files in {MOMO_DIR}.")
        print("  Run with --generate-sample to create synthetic test data.")
        return _empty_stats()

    brain = AIBrain()
    # Reset to clean-slate memory so brain approval threshold stays consistent
    # across all files and doesn't block trades after the first loss.
    brain.memory = BrainMemory()
    brain.record_outcome = lambda _outcome: None  # type: ignore[method-assign]

    news_correlator = NewsCorrelator()
    sympathy_detector = SympathyDetector()

    stocks_seen: set[str] = set()
    gap_ups_detected = 0
    setups_scored = 0
    passed_obvious = 0
    passed_score_filter = 0
    all_candidates: list[MomoSetup] = []

    # ── Phase 1: Pre-scan all CSV files, score every candidate ──
    print(f"[momo-backtest] Phase 1 — Scanning {len(csv_files)} CSV files for setups...")
    for csv_path in csv_files:
        stem = csv_path.stem
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

            # All CSVs in the momo directory were downloaded because they are
            # known gap-up events (>10%).  Use a conservative 15% default since
            # the exact prev-close is not available in the 1-min CSV.
            gap_pct = 0.15

            if gap_pct >= 0.10:
                gap_ups_detected += 1

            # Update news patterns (async, best-effort; suppress yfinance log noise)
            try:
                _yf_log = logging.getLogger("yfinance")
                _saved = _yf_log.level
                _yf_log.setLevel(logging.CRITICAL)
                try:
                    asyncio.run(
                        news_correlator.analyze_gap_up(ticker, trade_date, df, gap_pct)
                    )
                finally:
                    _yf_log.setLevel(_saved)
            except Exception:  # noqa: BLE001
                pass

            candidates = _pre_scan_file(
                df=df,
                ticker=ticker,
                trade_date=trade_date,
                gap_pct=gap_pct,
                news_correlator=news_correlator,
            )

            if candidates:
                setups_scored += 1
                for c in candidates:
                    if c.obvious_passed:
                        passed_obvious += 1
                    if c.score >= SCORE_HALF:
                        passed_score_filter += 1
                all_candidates.extend(candidates)

        except Exception as exc:  # noqa: BLE001
            logger.error("Error pre-scanning %s: %s", csv_path, exc)

    print(
        f"[momo-backtest] Phase 1 done: {len(all_candidates)} candidates "
        f"({passed_score_filter} passed score filter)"
    )

    # ── Phase 2: Smart PDT — rank by score, take top 3 per week ──
    selected = _select_top_setups_smart_pdt(all_candidates)
    print(f"[momo-backtest] Phase 2 — Smart PDT selected {len(selected)} trades to simulate")

    # ── Phase 3: Simulate selected trades ──
    all_trades: list[MomoTrade] = []
    for setup in selected:
        try:
            trade = _simulate_trade(setup, brain)
            if trade is not None:
                all_trades.append(trade)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Error simulating trade %s %s: %s", setup.ticker, setup.trade_date, exc
            )

    # Learn sympathy correlations from historical data
    sympathy_detector.learn_from_historical_data(str(MOMO_DIR))
    news_correlator.save_patterns()

    return _aggregate_stats(
        stocks_analyzed=len(stocks_seen),
        gap_ups_detected=gap_ups_detected,
        setups_scored=setups_scored,
        passed_obvious=passed_obvious,
        passed_score_filter=passed_score_filter,
        all_trades=all_trades,
    )


# ──────────────────────────────────────────────────────────────
# Sample data generator
# ──────────────────────────────────────────────────────────────

_SAMPLE_TICKERS = [
    # Kept only non-delisted liquid tickers for sample generation
    "BBIO", "CLOV", "AMC", "GME", "SNDL", "BB",
    "NOK", "KOSS", "ATER", "PHUN", "GFAI",
    "BRQT", "SBFM", "BFRI", "OPAD", "MYPS", "AEYE",
    "COMS", "DPRO", "GSAT", "TTOO", "AAME", "OCGN",
    "CTRM", "SHIP", "ILUS", "SENS", "ZKIN", "HYMC",
    "MNDR", "IONQ", "RGTI", "QUBT", "SOUN", "MARA",
    "RIOT", "BITF", "HUT", "CLSK", "NVAX", "BBIO",
    "VKTX", "LCID", "RIVN", "SOFI", "HOOD", "RKLB",
    "LUNR", "ASTS",
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

            # ── Realistic MoMo volume & price profile ──
            if minute < 30:
                # Opening surge: very high volume (2–4× base), price drifts up
                vol = int(base_vol * rng.uniform(2.0, 4.0))
                current_price = current_price * rng.uniform(0.998, 1.008)
            elif minute < 50:
                # Pullback to VWAP zone: low volume, price declines
                vol = int(base_vol * rng.uniform(0.15, 0.40))
                current_price = current_price * rng.uniform(0.990, 0.997)
            elif minute < 70:
                # Second leg: occasional volume spike on bounce/breakout
                if rng.random() < 0.35:
                    # Volume spike bar — this is where entry signals fire
                    vol = int(base_vol * rng.uniform(1.5, 3.5))
                else:
                    vol = int(base_vol * rng.uniform(0.35, 0.65))
                current_price = current_price * rng.uniform(0.999, 1.015)
            else:
                # Afternoon fade: low volume, price stabilizes
                vol = int(base_vol * rng.uniform(0.05, 0.25))
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
# (improvement #18: removed all Yahoo/IBKR download failures)
# Removed Yahoo failures: TCON, HYZN, EAST, GRCL, FFIE, VERB, DWAC, LBPH, MKFG,
#   KRON, BIGB, OBLG, AKRO, VORB, NBEV, NILE, JAN, MINM, PTRA, YGTY, ATNF, NOVV,
#   APDN, ASTR, ELMS
# Removed IBKR "No security definition": ADTX, ASTR, CLVS, DVAX, EXPR, GFLO, GNUS,
#   GXII, IDEX, LKCO, MMAT, MOGO, MRIN, MULN, NAKD, NKLA, PAYA, PROG, SFUN, SPRT,
#   VERB, WISH
SMALL_CAP_UNIVERSE = [
    # ── AI / Quantum Computing ──
    "IONQ", "RGTI", "QUBT", "QBTS", "SOUN", "BBAI", "AI", "GFAI",
    "INOD", "PRST", "KARO",

    # ── Crypto-Related ──
    "MARA", "RIOT", "BITF", "HUT", "CLSK", "COIN", "CIFR", "IREN",
    "BTBT", "BTDR", "CORZ", "WULF",

    # ── Biotech / Pharma ──
    "NVAX", "MRNA", "BNTX", "BBIO", "DNA", "VKTX", "VERA", "NUVB",
    "RXRX", "SMMT", "NRIX", "APLT", "RVMD", "TGTX", "IMVT",
    "KRYS", "VXRT", "OCGN", "SAVA", "PRAX", "ARQT",
    "MNKD", "IBRX", "MDGL", "CPRX", "CORT", "AGIO",

    # ── EV / Clean Energy ──
    "LCID", "RIVN", "GOEV", "NIO", "XPEV", "LI",
    "QS", "CHPT", "BLNK", "EVGO", "VFS",
    "PLUG", "FCEL", "BE", "ENPH", "SEDG", "RUN",

    # ── Fintech / Digital Finance ──
    "SOFI", "HOOD", "AFRM", "UPST", "OPEN", "CLOV",
    "NU", "PSFE", "DAVE", "LMND", "ROOT",

    # ── Space / Defense ──
    "RKLB", "LUNR", "ASTS", "RDW", "MNTS", "SPCE",
    "BKSY", "PL",

    # ── Meme / High-Volatility ──
    "AMC", "GME", "KOSS", "BB", "NOK", "CENN",
    "CVNA", "DJT", "PHUN",

    # ── Semiconductors / Tech ──
    "SMCI", "KULR", "WIMI", "SILO", "GCTS",
    "AMSC", "ACMR", "CEVA", "POWI",

    # ── Genomics / CRISPR ──
    "CRSP", "EDIT", "NTLA", "BEAM",

    # ── Lidar / Autonomous ──
    "LIDR", "AEVA", "OUST", "INVZ",

    # ── Cannabis ──
    "TLRY", "CGC", "ACB", "SNDL", "GRNH",

    # ── Speculative / Micro-Cap Movers (verified working) ──
    "HYMC", "GEVO", "REE", "WKHS",
    "ZKIN", "CXAI", "TPST", "BEEM",
    "NNOX", "GROM", "BIOR", "CNTB",
    "AEHL", "BFRI",
    "BOXL", "BYRN", "CING", "CISO", "CLNN",
    "CNET", "CNTX", "DATS", "EFTR",
    "FAMI", "FNGR", "FTFT", "GFOO", "GIPR",
    "GTEC", "HCTI", "HOUR", "HTOO",
    "ILAG", "IMPP", "INDO", "INTG", "ISPC",
    "KALA", "KAVL", "LGHL",
    "LIQT", "MBOT", "MEGL", "MGAM", "MIGI",
    "MLGO", "MNMD", "MVST",
    "NATH", "NDRA", "NEON",
    "NRGV", "OCEA", "PALI",
    "PCSA", "PIK", "PPBT", "PRCH", "PROC",
    "QNRX", "RVPH", "SBFM", "SES",
    "SLNO", "SNTI", "SPCB", "STSS", "TACT",
    "TNON", "TOP", "TRVN", "UCAR",
    "VCNX", "VNET", "VNRX", "VRM", "VYGR",
    "WRAP", "XELA", "XNET", "ZENV",
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
        # Convert "2026-03-07" → "20260307 16:00:00 US/Eastern" (IBKR required format)
        ibkr_date = date_str.replace("-", "")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=f"{ibkr_date} 16:00:00 US/Eastern",
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
    """Print comprehensive backtest results (improvement #17)."""
    line = "═" * 51
    print(f"\n{line}")
    print("  MoMo Backtest Results — DETAILED")
    print(line)
    print(f"  Stocks analyzed:        {stats.stocks_analyzed}")
    print(f"  Gap-ups detected:       {stats.gap_ups_detected}")
    print(f"  Setups scored:          {stats.setups_scored}")
    print(f"  Passed 'Obvious' check: {stats.passed_obvious}")
    print(f"  Passed score filter:    {stats.passed_score_filter}")
    print()
    print(f"  Total Trades:           {stats.total_trades} (3/week PDT limited)")
    print(f"  Win Rate:               {stats.win_rate:.0%}")

    if stats.profit_factor == float("inf"):
        pf_str = "∞ (no losses)"
    else:
        pf_str = f"{stats.profit_factor:.2f}"
    print(f"  Profit Factor:          {pf_str}")
    print(f"  Avg Win:                +${stats.avg_win:.2f}")
    print(f"  Avg Loss:               ${stats.avg_loss:.2f}")

    if stats.win_loss_ratio == float("inf"):
        wl_str = "∞"
    else:
        wl_str = f"{stats.win_loss_ratio:.2f}"
    print(f"  Win/Loss Ratio:         {wl_str}")
    print(f"  Expectancy:             ${stats.expectancy:+.2f}/trade")
    print(f"  Total P&L:              ${stats.total_pnl:+.2f}")
    print(f"  Max Drawdown:           -${stats.max_drawdown:.2f}")
    print()

    if stats.best_trade:
        bt = stats.best_trade
        print(
            f"  Best Trade:             {bt.ticker} +${bt.pnl:.2f}"
            f" ({bt.entry_type or bt.setup_type}, score {bt.score})"
        )
    if stats.worst_trade:
        wt = stats.worst_trade
        print(
            f"  Worst Trade:            {wt.ticker} ${wt.pnl:.2f}"
            f" ({wt.entry_type or wt.setup_type}, score {wt.score})"
        )

    print()
    print("  ── Win Rate by Hour ──")
    for lbl in ("9:30-10:15", "10:15-11:00", "11:00+"):
        n = stats.trades_by_hour.get(lbl, 0)
        w = stats.wins_by_hour.get(lbl, 0)
        wr = f"{w/n:.0%}" if n > 0 else "N/A"
        print(f"  {lbl:<20} {wr} ({n} trades)")

    print()
    print("  ── Win Rate by Entry Type ──")
    for et, n in sorted(stats.trades_by_entry_type.items()):
        w = stats.wins_by_entry_type.get(et, 0)
        wr = f"{w/n:.0%}" if n > 0 else "N/A"
        print(f"  {et:<22} {wr} ({n} trades)")

    print()
    print("  ── Win Rate by Catalyst ──")
    for cat, n in sorted(stats.trades_by_catalyst.items()):
        w = stats.wins_by_catalyst.get(cat, 0)
        wr = f"{w/n:.0%}" if n > 0 else "N/A"
        print(f"  {cat:<22} {wr} ({n} trades)")

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
                "gap_ups_detected": stats.gap_ups_detected,
                "setups_scored": stats.setups_scored,
                "passed_obvious": stats.passed_obvious,
                "passed_score_filter": stats.passed_score_filter,
                "total_trades": stats.total_trades,
                "win_rate": round(stats.win_rate, 4),
                "profit_factor": (
                    None if stats.profit_factor == float("inf")
                    else round(stats.profit_factor, 4)
                ),
                "avg_win": round(stats.avg_win, 4),
                "avg_loss": round(stats.avg_loss, 4),
                "win_loss_ratio": (
                    None if stats.win_loss_ratio == float("inf")
                    else round(stats.win_loss_ratio, 4)
                ),
                "expectancy": round(stats.expectancy, 4),
                "total_pnl": round(stats.total_pnl, 4),
                "max_drawdown": round(stats.max_drawdown, 4),
                "best_trade": {
                    "ticker": stats.best_trade.ticker,
                    "pnl": stats.best_trade.pnl,
                    "entry_type": stats.best_trade.entry_type,
                    "score": stats.best_trade.score,
                } if stats.best_trade else None,
                "worst_trade": {
                    "ticker": stats.worst_trade.ticker,
                    "pnl": stats.worst_trade.pnl,
                    "entry_type": stats.worst_trade.entry_type,
                    "score": stats.worst_trade.score,
                } if stats.worst_trade else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            fh,
            indent=2,
        )
    print(f"[momo-backtest] Results saved to {RESULTS_FILE}")
