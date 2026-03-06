"""
Reto Tracker — Compound Interest & Auto-Scale for Titanium Warrior v3.

Tracks total capital across all engines, determines trading phase (1–4),
calculates appropriate position sizes, and triggers milestone alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from config import settings
from config.settings import MILESTONE_ALERTS, PHASES, PhaseConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Summary of a single closed trade passed to the Reto Tracker."""

    engine: str
    pnl: float          # realised P&L in dollars (positive = profit)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DailyPnL:
    """Intraday P&L snapshot."""

    date: date
    starting_capital: float
    current_capital: float

    @property
    def pnl(self) -> float:
        return self.current_capital - self.starting_capital

    @property
    def pnl_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return (self.pnl / self.starting_capital) * 100


class RetoTracker:
    """
    Compound interest tracker that auto-scales position sizes as capital grows.

    Responsibilities:
      - Track total capital.
      - Determine current phase based on capital thresholds.
      - Calculate contracts / max position size per engine per phase.
      - Apply daily drawdown protection (drop >8% in a day → previous-phase sizing).
      - Fire capital-milestone alerts.
    """

    def __init__(self, initial_capital: float | None = None) -> None:
        self._capital: float = initial_capital if initial_capital is not None else settings.INITIAL_CAPITAL
        self._today_start_capital: float = self._capital
        self._today_date: date = date.today()
        self._drawdown_override: bool = False  # True when daily DD protection is active
        self._triggered_milestones: set[float] = set()

    # ──────────────────────────────────────────────────────────
    # Capital management
    # ──────────────────────────────────────────────────────────

    @property
    def capital(self) -> float:
        return self._capital

    def update_capital(self, trade_result: TradeResult) -> list[str]:
        """
        Apply trade P&L and recalculate buying power.

        Returns a list of milestone alert messages that were just triggered
        (empty list if none).
        """
        self._capital += trade_result.pnl
        self._capital = max(self._capital, 0.0)  # floor at zero
        self._refresh_daily_reset()

        # Check drawdown protection
        daily_dd_pct = abs(self.get_daily_pnl().pnl_pct) if self.get_daily_pnl().pnl < 0 else 0.0
        if daily_dd_pct > 8.0:
            if not self._drawdown_override:
                logger.warning(
                    "Daily drawdown %.1f%% exceeded 8%%; activating previous-phase sizing protection.",
                    daily_dd_pct,
                )
            self._drawdown_override = True
        else:
            self._drawdown_override = False

        logger.info(
            "Capital updated: pnl=%.2f total=%.2f phase=%d",
            trade_result.pnl,
            self._capital,
            self.get_phase(),
        )

        return self.check_milestones()

    def _refresh_daily_reset(self) -> None:
        """Reset daily tracking counter at the start of each new trading day."""
        today = date.today()
        if today != self._today_date:
            self._today_date = today
            self._today_start_capital = self._capital
            self._drawdown_override = False

    # ──────────────────────────────────────────────────────────
    # Phase logic
    # ──────────────────────────────────────────────────────────

    def get_phase(self) -> int:
        """Determine current phase (1–4) based on capital."""
        for phase_num in sorted(PHASES.keys(), reverse=True):
            if self._capital >= PHASES[phase_num].min_capital:
                return phase_num
        return 1

    def _effective_phase(self) -> int:
        """
        Return the phase used for position sizing.

        If drawdown protection is active, use one phase lower to reduce risk.
        """
        actual = self.get_phase()
        if self._drawdown_override and actual > 1:
            return actual - 1
        return actual

    def _phase_config(self) -> PhaseConfig:
        return PHASES[self._effective_phase()]

    # ──────────────────────────────────────────────────────────
    # Position sizing
    # ──────────────────────────────────────────────────────────

    def get_contracts(self, engine: str) -> int:
        """
        Return the number of futures contracts for the given engine.

        Only meaningful for the futures engine.
        """
        if engine == "futures":
            return self._phase_config().futures_contracts
        return 1

    def get_position_size(self, engine: str) -> float:
        """
        Return the maximum dollar amount to risk for a given engine.

        For futures this is expressed as max contracts; for others as $ amount.
        """
        cfg = self._phase_config()
        match engine:
            case "futures":
                return float(cfg.futures_contracts)
            case "options":
                return cfg.options_max_capital
            case "momo":
                return cfg.momo_max_capital
            case "crypto":
                # 30 % of capital allocated to crypto (configurable via CRYPTO_ALLOCATION env var)
                return self._capital * (settings.CRYPTO_ALLOCATION / 100)
            case _:
                return 0.0

    def get_futures_instrument(self) -> str:
        """Return "MNQ" or "NQ" based on current phase."""
        return self._phase_config().futures_instrument

    # ──────────────────────────────────────────────────────────
    # Daily P&L
    # ──────────────────────────────────────────────────────────

    def get_daily_pnl(self) -> DailyPnL:
        """Return a DailyPnL object for the current trading day."""
        self._refresh_daily_reset()
        return DailyPnL(
            date=self._today_date,
            starting_capital=self._today_start_capital,
            current_capital=self._capital,
        )

    # ──────────────────────────────────────────────────────────
    # Milestones
    # ──────────────────────────────────────────────────────────

    def check_milestones(self) -> list[str]:
        """
        Check if any capital milestone has been crossed.

        Returns a list of alert messages for milestones newly triggered.
        """
        alerts: list[str] = []
        for milestone in MILESTONE_ALERTS:
            threshold = milestone.capital_threshold
            if self._capital >= threshold and threshold not in self._triggered_milestones:
                self._triggered_milestones.add(threshold)
                alerts.append(milestone.message)
                logger.info("Capital milestone reached: $%.0f — %s", threshold, milestone.message)
        return alerts

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Return a dict snapshot of the tracker state for logging/Telegram."""
        daily = self.get_daily_pnl()
        cfg = self._phase_config()
        return {
            "capital": round(self._capital, 2),
            "phase": self.get_phase(),
            "effective_phase": self._effective_phase(),
            "drawdown_override": self._drawdown_override,
            "daily_pnl": round(daily.pnl, 2),
            "daily_pnl_pct": round(daily.pnl_pct, 2),
            "futures_instrument": cfg.futures_instrument,
            "futures_contracts": cfg.futures_contracts,
            "sessions": cfg.sessions,
        }
