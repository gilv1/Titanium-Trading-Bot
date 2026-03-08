"""
Backtesting Framework for Titanium Warrior v3.

Loads historical 1-minute CSV data and simulates the same pattern detection
and AI Brain logic used in live trading.

Usage:
    python backtest.py
    python backtest.py --analyze-events

Input CSV format (per file):
    time,open,high,low,close,volume

Saves results to ``journal/backtest_results.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis.patterns import (
    check_higher_timeframe_trend,
    detect_ema_pullback,
    detect_liquidity_grab,
    detect_orb,
    detect_vwap_bounce,
)
from analysis.technical import calculate_atr, calculate_ema, calculate_rsi, calculate_vwap
from config import settings
from core.brain import AIBrain
from core.reto_tracker import RetoTracker, TradeResult as RetoTradeResult
from engines.base_engine import Signal

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/historical")
RESULTS_FILE = Path("journal/backtest_results.json")

# Contract multipliers: points → dollars
CONTRACT_MULTIPLIERS: dict[str, float] = {
    "MES": 5,
    "ES": 50,
    "MNQ": 2,
    "NQ": 20,
}

# Trailing stop progression thresholds (in points).
# When profit reaches TRAIL_BREAKEVEN_PTS, move SL to entry+1.
# When profit reaches TRAIL_TIGHT_PTS, trail SL at price - TRAIL_TIGHT_OFFSET_PTS.
TRAIL_BREAKEVEN_PTS: float = 10.0
TRAIL_BREAKEVEN_OFFSET_PTS: float = 1.0
TRAIL_TIGHT_PTS: float = 15.0
TRAIL_TIGHT_OFFSET_PTS: float = 5.0


# ──────────────────────────────────────────────────────────────
# Back-test result dataclass
# ──────────────────────────────────────────────────────────────


@dataclass
class BacktestStats:
    ticker: str
    total_return_pct: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    starting_capital: float
    ending_capital: float


# ──────────────────────────────────────────────────────────────
# Core simulation
# ──────────────────────────────────────────────────────────────


def simulate(df: pd.DataFrame, ticker: str, brain: AIBrain, reto: RetoTracker) -> BacktestStats:
    """
    Simulate trades on a 1-minute OHLCV DataFrame.

    Walk-forward: for each bar evaluate signals and simulate fills.
    """
    capital = reto.capital
    start_capital = capital
    pnl_series: list[float] = []
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    contract_mult = CONTRACT_MULTIPLIERS.get(ticker.upper(), 1)
    is_futures_no_orb = ticker.upper() in ("MES", "ES")

    window = 50  # minimum bars needed before trading

    for i in range(window, len(df)):
        window_df = df.iloc[: i + 1].copy()
        vwap = calculate_vwap(window_df)
        ema9 = calculate_ema(window_df, 9)
        ema21 = calculate_ema(window_df, 21)
        rsi = calculate_rsi(window_df)
        atr_series = calculate_atr(window_df)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        # Detect signal — skip ORB for MES/ES futures
        if is_futures_no_orb:
            signal: Signal | None = (
                detect_vwap_bounce(window_df, vwap, rsi_series=rsi)
                or detect_ema_pullback(window_df, ema9, ema21)
                or detect_liquidity_grab(window_df, [])
            )
        else:
            signal = (
                detect_vwap_bounce(window_df, vwap, rsi_series=rsi)
                or detect_orb(window_df)
                or detect_ema_pullback(window_df, ema9, ema21)
                or detect_liquidity_grab(window_df, [])
            )

        if signal is None:
            continue

        # Multi-timeframe confirmation: filter signals that disagree with the
        # higher-timeframe trend
        if not check_higher_timeframe_trend(window_df, signal.direction):
            signal = None
            continue

        # AI Brain evaluation
        decision = brain.evaluate_trade(
            setup_type=signal.setup_type,
            engine="backtest",
            entry=signal.entry_price,
            stop=signal.stop_price,
            target=signal.target_price,
            session="NY",
            atr=atr,
        )
        if not decision.approved:
            continue

        # Simulate trade with trailing stop over next 10 bars
        if i + 1 >= len(df):
            break

        trade_pnl = 0.0
        hit = False
        trailing_stop = signal.stop_price

        for j in range(i + 1, min(i + 11, len(df))):
            bar = df.iloc[j]

            if signal.direction == "LONG":
                # Check stop hit
                if float(bar["low"]) <= trailing_stop:
                    trade_pnl = trailing_stop - signal.entry_price
                    hit = True
                    break
                # Check target hit
                if float(bar["high"]) >= signal.target_price:
                    trade_pnl = signal.target_price - signal.entry_price
                    hit = True
                    break
                # Trail stop upward
                current_high = float(bar["high"])
                if current_high - signal.entry_price >= TRAIL_BREAKEVEN_PTS:
                    trailing_stop = max(trailing_stop, signal.entry_price + TRAIL_BREAKEVEN_OFFSET_PTS)
                if current_high - signal.entry_price >= TRAIL_TIGHT_PTS:
                    trailing_stop = max(trailing_stop, current_high - TRAIL_TIGHT_OFFSET_PTS)
            else:  # SHORT
                # Check stop hit
                if float(bar["high"]) >= trailing_stop:
                    trade_pnl = signal.entry_price - trailing_stop
                    hit = True
                    break
                # Check target hit
                if float(bar["low"]) <= signal.target_price:
                    trade_pnl = signal.entry_price - signal.target_price
                    hit = True
                    break
                # Trail stop downward
                current_low = float(bar["low"])
                if signal.entry_price - current_low >= TRAIL_BREAKEVEN_PTS:
                    trailing_stop = min(trailing_stop, signal.entry_price - TRAIL_BREAKEVEN_OFFSET_PTS)
                if signal.entry_price - current_low >= TRAIL_TIGHT_PTS:
                    trailing_stop = min(trailing_stop, current_low + TRAIL_TIGHT_OFFSET_PTS)

        if not hit:
            last_bar = df.iloc[min(i + 10, len(df) - 1)]
            if signal.direction == "LONG":
                trade_pnl = float(last_bar["close"]) - signal.entry_price
            else:
                trade_pnl = signal.entry_price - float(last_bar["close"])

        won = trade_pnl > 0

        # Apply contract multiplier (points → dollars)
        trade_pnl *= contract_mult

        # Apply brain size multiplier
        trade_pnl *= decision.size_multiplier

        capital += trade_pnl
        pnl_series.append(capital)

        if won:
            wins += 1
            gross_profit += trade_pnl
        else:
            losses += 1
            gross_loss += abs(trade_pnl)

        # Brain self-learning
        from core.brain import TradeOutcome
        brain.record_outcome(
            TradeOutcome(
                setup_type=signal.setup_type,
                session="NY",
                day_of_week="Monday",
                hour=9,
                volatility_regime="medium",
                won=won,
                engine="backtest",
            )
        )

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_return_pct = ((capital - start_capital) / start_capital * 100) if start_capital > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    max_dd = 0.0
    if pnl_series:
        peak = pnl_series[0]
        for v in pnl_series:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

    # Sharpe ratio (simplified, daily returns)
    sharpe = 0.0
    if len(pnl_series) > 1:
        daily_returns = pd.Series(pnl_series).pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * (252**0.5))

    return BacktestStats(
        ticker=ticker,
        total_return_pct=round(total_return_pct, 2),
        num_trades=total_trades,
        win_rate=round(win_rate, 4),
        max_drawdown=round(max_dd * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        profit_factor=round(profit_factor, 3),
        starting_capital=round(start_capital, 2),
        ending_capital=round(capital, 2),
    )


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


def run() -> None:
    """
    Run backtests for all CSV files found in ``data/historical/``.

    Each file should be named ``<TICKER>.csv`` with columns:
        time,open,high,low,close,volume
    """
    brain = AIBrain()
    reto = RetoTracker(initial_capital=settings.INITIAL_CAPITAL)

    if not DATA_DIR.exists():
        logger.warning("No historical data directory found at %s. Create it and add CSV files.", DATA_DIR)
        print(f"[backtest] No data directory found at '{DATA_DIR}'.")
        print("           Create it and add files like 'data/historical/MNQ.csv'")
        return

    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s.", DATA_DIR)
        print(f"[backtest] No CSV files in '{DATA_DIR}'.")
        return

    all_results: list[dict] = []

    for csv_path in sorted(csv_files):
        ticker = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                logger.warning("Skipping %s: missing columns %s", csv_path, required - set(df.columns))
                continue

            logger.info("Backtesting %s (%d bars)…", ticker, len(df))
            stats = simulate(df, ticker, brain, reto)

            print(
                f"\n{'─'*50}\n"
                f"  Ticker:          {stats.ticker}\n"
                f"  Total Return:    {stats.total_return_pct:+.2f}%\n"
                f"  Trades:          {stats.num_trades}\n"
                f"  Win Rate:        {stats.win_rate:.0%}\n"
                f"  Max Drawdown:    {stats.max_drawdown:.2f}%\n"
                f"  Sharpe Ratio:    {stats.sharpe_ratio:.3f}\n"
                f"  Profit Factor:   {stats.profit_factor:.3f}\n"
                f"  Starting Cap:    ${stats.starting_capital:.2f}\n"
                f"  Ending Cap:      ${stats.ending_capital:.2f}\n"
            )

            all_results.append(
                {
                    "ticker": stats.ticker,
                    "total_return_pct": stats.total_return_pct,
                    "num_trades": stats.num_trades,
                    "win_rate": stats.win_rate,
                    "max_drawdown": stats.max_drawdown,
                    "sharpe_ratio": stats.sharpe_ratio,
                    "profit_factor": stats.profit_factor,
                    "starting_capital": stats.starting_capital,
                    "ending_capital": stats.ending_capital,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Error backtesting %s: %s", ticker, exc, exc_info=True)

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[backtest] Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="Titanium Warrior Backtester")
    parser.add_argument(
        "--analyze-events",
        action="store_true",
        help="Run the Event Impact Analyzer instead of the normal backtest.",
    )
    args = parser.parse_args()

    if args.analyze_events:
        from core.event_analyzer import EventAnalyzer
        analyzer = EventAnalyzer()
        analyzer.run()
    else:
        run()
