"""
Pattern Detection for Titanium Warrior v3.

Detects trade setups on 1-minute OHLCV DataFrames and returns Signal objects.
All detectors return None when the pattern is not present.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from engines.base_engine import Signal

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# VWAP Bounce
# ──────────────────────────────────────────────────────────────


def detect_vwap_bounce(
    df: pd.DataFrame,
    vwap_series: pd.Series,
    rsi_series: Optional[pd.Series] = None,
    tolerance_pct: float = 0.001,
) -> Optional[Signal]:
    """
    Detect a VWAP Bounce setup.

    Conditions (long):
      - Last close within ``tolerance_pct`` of VWAP.
      - RSI between 35–45 (if provided).
      - Volume decreasing over last 3 bars (pullback into VWAP).

    Conditions (short):
      - RSI between 55–65 (if provided).
    """
    if df.empty or len(df) < 5:
        return None
    if vwap_series.empty:
        return None

    last = df.iloc[-1]
    vwap = float(vwap_series.iloc[-1])
    price = float(last["close"])
    if vwap == 0:
        return None

    distance_pct = abs(price - vwap) / vwap
    if distance_pct > tolerance_pct * 10:  # not near VWAP
        return None

    # Volume decreasing on last 3 bars
    vols = df["volume"].iloc[-3:].values
    vol_decreasing = vols[0] > vols[1] > vols[2] if len(vols) == 3 else True

    rsi_val = None
    if rsi_series is not None and not rsi_series.empty:
        rsi_val = float(rsi_series.iloc[-1])

    # Long setup
    if price <= vwap * (1 + tolerance_pct):
        rsi_ok = rsi_val is None or 30 <= rsi_val <= 50
        if vol_decreasing and rsi_ok:
            sl = price - (price * 0.003)  # 0.3 % below entry
            tp = price + (price - sl) * 2  # 1:2 R:R
            return Signal(
                direction="LONG",
                confidence=70,
                entry_price=price,
                stop_price=sl,
                target_price=tp,
                setup_type="VWAP_BOUNCE",
                reasoning=f"VWAP bounce LONG: price={price:.2f} vwap={vwap:.2f} rsi={rsi_val}",
            )

    # Short setup
    if price >= vwap * (1 - tolerance_pct):
        rsi_ok = rsi_val is None or 50 <= rsi_val <= 70
        if vol_decreasing and rsi_ok:
            sl = price + (price * 0.003)
            tp = price - (sl - price) * 2
            return Signal(
                direction="SHORT",
                confidence=70,
                entry_price=price,
                stop_price=sl,
                target_price=tp,
                setup_type="VWAP_BOUNCE",
                reasoning=f"VWAP bounce SHORT: price={price:.2f} vwap={vwap:.2f} rsi={rsi_val}",
            )

    return None


# ──────────────────────────────────────────────────────────────
# Opening Range Breakout (ORB)
# ──────────────────────────────────────────────────────────────


def detect_orb(
    df: pd.DataFrame,
    session_start_time: Optional[datetime] = None,
    orb_minutes: int = 15,
    volume_multiplier: float = 1.5,
) -> Optional[Signal]:
    """
    Detect an Opening Range Breakout.

    Conditions:
      - First ``orb_minutes`` of session define the range (high/low).
      - Current bar closes outside the range.
      - Volume > ``volume_multiplier`` × 14-bar average.
    """
    if df.empty or len(df) < orb_minutes + 2:
        return None

    # Use first N bars as the ORB range
    orb_df = df.iloc[:orb_minutes]
    orb_high = float(orb_df["high"].max())
    orb_low = float(orb_df["low"].min())
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return None

    last = df.iloc[-1]
    current_close = float(last["close"])
    current_vol = float(last["volume"])
    avg_vol = float(df["volume"].iloc[-14:].mean()) if len(df) >= 14 else float(df["volume"].mean())

    # EMA 9 alignment check
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema9_last = float(ema9.iloc[-1])

    high_breakout = current_close > orb_high and current_vol > avg_vol * volume_multiplier
    low_breakdown = current_close < orb_low and current_vol > avg_vol * volume_multiplier

    if high_breakout and current_close > ema9_last:
        sl = orb_high - orb_range * 0.5
        tp = current_close + orb_range
        return Signal(
            direction="LONG",
            confidence=75,
            entry_price=current_close,
            stop_price=sl,
            target_price=tp,
            setup_type="ORB",
            reasoning=f"ORB breakout LONG: close={current_close:.2f} orb_high={orb_high:.2f} vol_ratio={current_vol/avg_vol:.1f}x",
        )

    if low_breakdown and current_close < ema9_last:
        sl = orb_low + orb_range * 0.5
        tp = current_close - orb_range
        return Signal(
            direction="SHORT",
            confidence=75,
            entry_price=current_close,
            stop_price=sl,
            target_price=tp,
            setup_type="ORB",
            reasoning=f"ORB breakdown SHORT: close={current_close:.2f} orb_low={orb_low:.2f} vol_ratio={current_vol/avg_vol:.1f}x",
        )

    return None


# ──────────────────────────────────────────────────────────────
# EMA 9/21 Pullback
# ──────────────────────────────────────────────────────────────


def detect_ema_pullback(
    df: pd.DataFrame,
    ema9: pd.Series,
    ema21: pd.Series,
    macd_hist: Optional[pd.Series] = None,
) -> Optional[Signal]:
    """
    Detect an EMA 9/21 Pullback setup.

    Conditions (long):
      - EMA 9 > EMA 21 (uptrend).
      - Price pulls back to within ``tolerance`` of EMA 9.
      - MACD histogram positive (if provided).
    """
    if df.empty or ema9.empty or ema21.empty:
        return None

    price = float(df["close"].iloc[-1])
    e9 = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])

    if e9 == 0:
        return None

    tolerance = abs(price - e9) / e9

    # Long
    if e9 > e21:
        macd_ok = macd_hist is None or float(macd_hist.iloc[-1]) > 0
        if tolerance < 0.002 and macd_ok:
            sl = e9 - (e9 - e21)
            tp = price + (price - sl) * 2
            return Signal(
                direction="LONG",
                confidence=72,
                entry_price=price,
                stop_price=sl,
                target_price=tp,
                setup_type="EMA_PULLBACK",
                reasoning=f"EMA pullback LONG: price={price:.2f} ema9={e9:.2f} ema21={e21:.2f}",
            )

    # Short
    if e9 < e21:
        macd_ok = macd_hist is None or float(macd_hist.iloc[-1]) < 0
        if tolerance < 0.002 and macd_ok:
            sl = e9 + (e21 - e9)
            tp = price - (sl - price) * 2
            return Signal(
                direction="SHORT",
                confidence=72,
                entry_price=price,
                stop_price=sl,
                target_price=tp,
                setup_type="EMA_PULLBACK",
                reasoning=f"EMA pullback SHORT: price={price:.2f} ema9={e9:.2f} ema21={e21:.2f}",
            )

    return None


# ──────────────────────────────────────────────────────────────
# Liquidity Grab & Reversal
# ──────────────────────────────────────────────────────────────


def detect_liquidity_grab(
    df: pd.DataFrame,
    levels: list[float],
    spike_pts: float = 3.0,
    reversal_bars: int = 2,
) -> Optional[Signal]:
    """
    Detect a Liquidity Grab & Reversal.

    Price breaks an S/R level by ``spike_pts`` and then reverses with
    explosive volume within ``reversal_bars`` bars.
    """
    if df.empty or len(df) < reversal_bars + 2:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["close"])
    prev_low = float(prev["low"])
    prev_high = float(prev["high"])
    vol_avg = float(df["volume"].iloc[-10:].mean()) if len(df) >= 10 else float(df["volume"].mean())
    vol_current = float(last["volume"])
    explosive = vol_current > vol_avg * 2.0

    # Bullish liquidity grab: sharp spike below support then close above
    if float(last["low"]) < prev_low - spike_pts and price > prev_low and explosive:
        sl = float(last["low"]) - 1.0
        tp = price + (price - sl) * 2
        return Signal(
            direction="LONG",
            confidence=80,
            entry_price=price,
            stop_price=sl,
            target_price=tp,
            setup_type="LIQUIDITY_GRAB",
            reasoning=f"Liquidity grab LONG: spike to {last['low']:.2f} then reversal close={price:.2f}",
        )

    # Bearish liquidity grab: spike above resistance then close below
    if float(last["high"]) > prev_high + spike_pts and price < prev_high and explosive:
        sl = float(last["high"]) + 1.0
        tp = price - (sl - price) * 2
        return Signal(
            direction="SHORT",
            confidence=80,
            entry_price=price,
            stop_price=sl,
            target_price=tp,
            setup_type="LIQUIDITY_GRAB",
            reasoning=f"Liquidity grab SHORT: spike to {last['high']:.2f} then reversal close={price:.2f}",
        )

    return None


# ──────────────────────────────────────────────────────────────
# Range Breakout
# ──────────────────────────────────────────────────────────────


def detect_breakout(
    df: pd.DataFrame,
    range_high: float,
    range_low: float,
    volume_avg: float,
    volume_surge_multiplier: float = 2.0,
) -> Optional[Signal]:
    """
    Detect a consolidation breakout.

    Parameters
    ----------
    df                      : 1-min OHLCV
    range_high / range_low  : Consolidation range boundaries
    volume_avg              : Average volume baseline
    volume_surge_multiplier : Required volume ratio for breakout confirmation
    """
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    volume = float(last["volume"])
    range_size = range_high - range_low
    if range_size <= 0:
        return None

    if price > range_high and volume > volume_avg * volume_surge_multiplier:
        sl = range_high - range_size * 0.3
        tp = range_high + range_size * 1.5
        return Signal(
            direction="LONG",
            confidence=78,
            entry_price=price,
            stop_price=sl,
            target_price=tp,
            setup_type="RANGE_BREAKOUT",
            reasoning=f"Breakout LONG: close={price:.2f} above range_high={range_high:.2f}",
        )

    if price < range_low and volume > volume_avg * volume_surge_multiplier:
        sl = range_low + range_size * 0.3
        tp = range_low - range_size * 1.5
        return Signal(
            direction="SHORT",
            confidence=78,
            entry_price=price,
            stop_price=sl,
            target_price=tp,
            setup_type="RANGE_BREAKOUT",
            reasoning=f"Breakdown SHORT: close={price:.2f} below range_low={range_low:.2f}",
        )

    return None


# ──────────────────────────────────────────────────────────────
# Candlestick patterns
# ──────────────────────────────────────────────────────────────


def detect_pin_bar(candle: pd.Series) -> bool:
    """
    Return True if the candle is a pin bar (long wick, small body).

    A pin bar has a wick at least 2× the body size.
    """
    body = abs(float(candle["close"]) - float(candle["open"]))
    total_range = float(candle["high"]) - float(candle["low"])
    if total_range == 0:
        return False
    # wick = total range minus body
    wick = total_range - body
    return wick >= body * 2.0


def detect_engulfing(candle: pd.Series, prev_candle: pd.Series) -> bool:
    """
    Return True if ``candle`` engulfs ``prev_candle``.

    Engulfing: current candle's body completely contains the prior bar's body.
    """
    cur_open = float(candle["open"])
    cur_close = float(candle["close"])
    prev_open = float(prev_candle["open"])
    prev_close = float(prev_candle["close"])

    cur_body_lo = min(cur_open, cur_close)
    cur_body_hi = max(cur_open, cur_close)
    prev_body_lo = min(prev_open, prev_close)
    prev_body_hi = max(prev_open, prev_close)

    return cur_body_lo < prev_body_lo and cur_body_hi > prev_body_hi


# ──────────────────────────────────────────────────────────────
# Higher-Timeframe Trend Confirmation
# ──────────────────────────────────────────────────────────────


def check_higher_timeframe_trend(df: pd.DataFrame, direction: str) -> bool:
    """
    Check if the 5-minute resampled trend agrees with the proposed trade direction.

    Uses the last 50 bars, groups them into 5-bar (5-min equivalent) chunks,
    and checks whether EMA9 > EMA21 (LONG) or EMA9 < EMA21 (SHORT) on that
    higher-timeframe view.

    Returns True if the higher-timeframe trend is aligned (or there is
    insufficient data to make a determination).
    """
    if len(df) < 30:
        return True

    recent = df.tail(50).copy().reset_index(drop=True)
    groups = recent.groupby(recent.index // 5).agg({"close": "last"})

    if len(groups) < 5:
        return True

    ema9 = groups["close"].ewm(span=9, adjust=False).mean()
    ema21 = groups["close"].ewm(span=21, adjust=False).mean()

    if direction == "LONG":
        return float(ema9.iloc[-1]) > float(ema21.iloc[-1])
    return float(ema9.iloc[-1]) < float(ema21.iloc[-1])
