"""
Trade Journal for Titanium Warrior v3.

Logs every trade to both CSV and JSON formats.
Provides summary statistics (daily, weekly, win rate, total P&L).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from engines.base_engine import TradeResult

logger = logging.getLogger(__name__)

JOURNAL_DIR = Path("journal")
CSV_FILE = JOURNAL_DIR / "trades.csv"
JSON_FILE = JOURNAL_DIR / "trades.json"

CSV_FIELDS = [
    "timestamp",
    "engine",
    "ticker",
    "direction",
    "entry_price",
    "exit_price",
    "stop_loss",
    "take_profit",
    "quantity",
    "pnl_dollars",
    "pnl_percent",
    "duration_seconds",
    "setup_type",
    "session",
    "ai_score",
    "phase",
    "capital_after",
    "notes",
    "won",
]


@dataclass
class DailySummary:
    """Aggregate statistics for a single trading day."""

    date: date
    total_pnl: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    engines: dict[str, Any]


@dataclass
class WeeklySummary:
    """Aggregate statistics for a trading week."""

    week_start: date
    week_end: date
    total_pnl: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    max_drawdown: float


class TradeJournal:
    """
    Persistent trade journal backed by CSV and JSON files.

    All append operations keep both files in sync.
    """

    def __init__(self) -> None:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()
        self._trades: list[dict[str, Any]] = self._load_json()

    # ──────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────

    def _ensure_csv_header(self) -> None:
        if not CSV_FILE.exists():
            with open(CSV_FILE, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()

    def _load_json(self) -> list[dict[str, Any]]:
        if JSON_FILE.exists():
            try:
                with open(JSON_FILE, encoding="utf-8") as fh:
                    data = json.load(fh)
                    return data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_json(self) -> None:
        with open(JSON_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._trades, fh, indent=2, default=str)

    # ──────────────────────────────────────────────────────────
    # Log a trade
    # ──────────────────────────────────────────────────────────

    def log_trade(self, result: TradeResult) -> None:
        """Append a closed trade to both CSV and JSON."""
        row = {
            "timestamp": result.exit_time.isoformat(),
            "engine": result.engine,
            "ticker": result.ticker,
            "direction": result.direction,
            "entry_price": result.entry_price,
            "exit_price": result.exit_price,
            "stop_loss": result.stop_loss,
            "take_profit": result.take_profit,
            "quantity": result.quantity,
            "pnl_dollars": round(result.pnl, 4),
            "pnl_percent": round(result.pnl_pct, 4),
            "duration_seconds": round(result.duration_seconds, 1),
            "setup_type": result.setup_type,
            "session": result.session,
            "ai_score": result.ai_score,
            "phase": result.phase,
            "capital_after": result.capital_after,
            "notes": result.notes,
            "won": result.won,
        }

        # Append to CSV
        with open(CSV_FILE, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writerow(row)

        # Append to in-memory list and persist JSON
        self._trades.append(row)
        self._save_json()
        logger.info("Trade logged: %s %s pnl=%.2f", result.engine, result.ticker, result.pnl)

    # ──────────────────────────────────────────────────────────
    # Summaries
    # ──────────────────────────────────────────────────────────

    def get_daily_summary(self, target_date: date | None = None) -> DailySummary:
        """Return aggregated stats for ``target_date`` (defaults to today)."""
        if target_date is None:
            target_date = date.today()

        day_trades = [
            t for t in self._trades
            if t.get("timestamp", "").startswith(target_date.isoformat())
        ]

        engine_stats: dict[str, dict[str, Any]] = {}
        total_pnl = 0.0
        wins = 0
        losses = 0

        for t in day_trades:
            eng = t.get("engine", "unknown")
            if eng not in engine_stats:
                engine_stats[eng] = {"pnl": 0.0, "trades": 0, "wins": 0}
            pnl = float(t.get("pnl_dollars", 0))
            won = bool(t.get("won", False))
            total_pnl += pnl
            engine_stats[eng]["pnl"] += pnl
            engine_stats[eng]["trades"] += 1
            if won:
                wins += 1
                engine_stats[eng]["wins"] += 1
            else:
                losses += 1

        total = wins + losses
        return DailySummary(
            date=target_date,
            total_pnl=round(total_pnl, 4),
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=wins / total if total > 0 else 0.0,
            engines=engine_stats,
        )

    def get_weekly_summary(self, week_start: date | None = None) -> WeeklySummary:
        """Return aggregated stats for the week beginning ``week_start``."""
        if week_start is None:
            today = date.today()
            week_start = today - timedelta(days=today.weekday())

        week_end = week_start + timedelta(days=6)
        week_trades = [
            t for t in self._trades
            if week_start.isoformat() <= t.get("timestamp", "")[:10] <= week_end.isoformat()
        ]

        total_pnl = 0.0
        wins = 0
        losses = 0
        cumulative_pnl: list[float] = []
        running = 0.0

        for t in week_trades:
            pnl = float(t.get("pnl_dollars", 0))
            total_pnl += pnl
            running += pnl
            cumulative_pnl.append(running)
            if bool(t.get("won", False)):
                wins += 1
            else:
                losses += 1

        # Max drawdown: largest peak-to-trough in cumulative P&L
        max_dd = 0.0
        if cumulative_pnl:
            peak = cumulative_pnl[0]
            for v in cumulative_pnl:
                if v > peak:
                    peak = v
                dd = peak - v
                if dd > max_dd:
                    max_dd = dd

        total = wins + losses
        return WeeklySummary(
            week_start=week_start,
            week_end=week_end,
            total_pnl=round(total_pnl, 4),
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=wins / total if total > 0 else 0.0,
            max_drawdown=round(max_dd, 4),
        )

    # ──────────────────────────────────────────────────────────
    # Statistics helpers
    # ──────────────────────────────────────────────────────────

    def get_win_rate(
        self,
        engine: str | None = None,
        setup: str | None = None,
        session: str | None = None,
    ) -> float:
        """
        Return win rate filtered by optional engine, setup type, and session.

        Returns 0.0 when there are no matching trades.
        """
        trades = self._trades
        if engine:
            trades = [t for t in trades if t.get("engine") == engine]
        if setup:
            trades = [t for t in trades if t.get("setup_type") == setup]
        if session:
            trades = [t for t in trades if t.get("session") == session]

        if not trades:
            return 0.0
        wins = sum(1 for t in trades if bool(t.get("won", False)))
        return wins / len(trades)

    def get_total_pnl(self, period_days: int | None = None) -> float:
        """
        Return total realised P&L.

        Parameters
        ----------
        period_days : optional, number of days to look back (None = all time)
        """
        trades = self._trades
        if period_days is not None:
            cutoff = (date.today() - timedelta(days=period_days)).isoformat()
            trades = [t for t in trades if t.get("timestamp", "")[:10] >= cutoff]
        return round(sum(float(t.get("pnl_dollars", 0)) for t in trades), 4)
