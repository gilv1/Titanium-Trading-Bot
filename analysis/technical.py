"""
Technical Analysis indicators for Titanium Warrior v3.

All functions operate on a pandas DataFrame with columns:
    open, high, low, close, volume

Returns pandas Series or tuples of Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# VWAP
# ──────────────────────────────────────────────────────────────


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate intraday VWAP (Volume Weighted Average Price).

    Resets at the start of each day if the DataFrame spans multiple days.

    Parameters
    ----------
    df : DataFrame with columns high, low, close, volume

    Returns
    -------
    pd.Series of VWAP values aligned to df.index
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tpv = typical_price * df["volume"]
    cumulative_tpv = tpv.cumsum()
    cumulative_volume = df["volume"].cumsum()
    vwap = cumulative_tpv / cumulative_volume.replace(0, np.nan)
    return vwap.rename("vwap")


# ──────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Parameters
    ----------
    df     : DataFrame with a ``close`` column
    period : EMA period (e.g. 9, 21)
    """
    return df["close"].ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


# ──────────────────────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────────────────────


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder smoothing).

    Parameters
    ----------
    df     : DataFrame with ``close``
    period : Look-back window (default 14)
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi")


# ──────────────────────────────────────────────────────────────
# MACD
# ──────────────────────────────────────────────────────────────


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, and histogram.

    Returns
    -------
    (macd_line, signal_line, histogram)
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).rename("macd")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("macd_signal")
    histogram = (macd_line - signal_line).rename("macd_hist")
    return macd_line, signal_line, histogram


# ──────────────────────────────────────────────────────────────
# ATR
# ──────────────────────────────────────────────────────────────


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    Parameters
    ----------
    df     : DataFrame with high, low, close
    period : Look-back window (default 14)
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().rename("atr")


# ──────────────────────────────────────────────────────────────
# Relative Volume
# ──────────────────────────────────────────────────────────────


def calculate_rvol(current_volume: float, avg_volume_14d: float) -> float:
    """
    Relative Volume = current_volume / avg_volume_14d.

    Returns 0.0 if avg_volume_14d is zero to avoid division by zero.
    """
    if avg_volume_14d <= 0:
        return 0.0
    return current_volume / avg_volume_14d


# ──────────────────────────────────────────────────────────────
# Bollinger Bands
# ──────────────────────────────────────────────────────────────


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns
    -------
    (upper_band, middle_band, lower_band)
    """
    middle = df["close"].rolling(window=period).mean().rename("bb_mid")
    rolling_std = df["close"].rolling(window=period).std()
    upper = (middle + std * rolling_std).rename("bb_upper")
    lower = (middle - std * rolling_std).rename("bb_lower")
    return upper, middle, lower


# ──────────────────────────────────────────────────────────────
# Volume Delta (buy vs sell pressure estimate)
# ──────────────────────────────────────────────────────────────


def estimate_volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    Estimate buy/sell volume imbalance per bar.

    Positive values indicate buying pressure; negative = selling.

    Method: if close > open → treat as buy volume; else sell volume.
    Returns a signed series (positive = net buying).
    """
    body_pct = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    return (body_pct * df["volume"]).rename("volume_delta")
