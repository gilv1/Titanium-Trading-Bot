"""
Global Risk Manager for Titanium Warrior v3.

Responsibilities:
  - Cap total daily risk at 8 % of capital.
  - Cap per-engine daily risk (futures 4 %, options/momo/crypto 2 % each).
  - Kill switch: if capital drops >12 % in a day → shutdown all engines for 24 h.
  - Pause an engine for 4 h after 3 consecutive losses.
  - Enforce maximum 3 simultaneous open positions.
  - Correlation guard (NQ long + BTC long blocked).
  - PDT compliance tracking for momo engine.
  - Per-engine trade-count limits per day.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import DefaultDict

from config import settings
from config.settings import ENGINE_DAILY_RISK_PCT

logger = logging.getLogger(__name__)

ENGINES = ("futures", "options", "momo", "crypto")


@dataclass
class TradeRecord:
    """Minimal record for risk-tracking purposes."""

    engine: str
    pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    won: bool = True
    direction: str = "LONG"   # "LONG" or "SHORT"
    ticker: str = ""


class RiskManager:
    """
    Global risk veto layer.

    All engines MUST call ``can_trade()`` before placing any order.
    Engines report outcomes via ``register_trade()``.
    """

    def __init__(self, reto_tracker: object | None = None) -> None:
        """
        Parameters
        ----------
        reto_tracker : RetoTracker | None
            Optional reference used to read live capital / daily-drawdown figures.
        """
        self._reto_tracker = reto_tracker

        # Per-engine trade list (current day)
        self._daily_trades: DefaultDict[str, list[TradeRecord]] = defaultdict(list)
        self._today: date = date.today()

        # Consecutive loss counter per engine
        self._consecutive_losses: DefaultDict[str, int] = defaultdict(int)
        # Pause expiry per engine (datetime when pause ends)
        self._pause_until: dict[str, datetime] = {}

        # Kill switch state
        self._kill_switch_active: bool = False
        self._kill_switch_until: datetime | None = None

        # Open-position tracking: {engine: [(ticker, direction)]}
        self._open_positions: DefaultDict[str, list[tuple[str, str]]] = defaultdict(list)

        # PDT tracking: rolling deque of day-trade timestamps for momo
        self._momo_day_trades: deque[date] = deque()

    # ──────────────────────────────────────────────────────────
    # Daily reset
    # ──────────────────────────────────────────────────────────

    def _maybe_reset_daily(self) -> None:
        today = date.today()
        if today != self._today:
            logger.info("Risk manager: new trading day — resetting daily counters.")
            self._today = today
            self._daily_trades.clear()
            self._consecutive_losses.clear()
            # Kill switch expires on day reset
            if self._kill_switch_active and self._kill_switch_until and datetime.utcnow() > self._kill_switch_until:
                self._kill_switch_active = False
                self._kill_switch_until = None

    # ──────────────────────────────────────────────────────────
    # Kill switch
    # ──────────────────────────────────────────────────────────

    def check_kill_switch(self) -> bool:
        """Return True if the global kill switch is active (all engines halted)."""
        self._maybe_reset_daily()
        if not self._kill_switch_active:
            return False
        if self._kill_switch_until and datetime.utcnow() > self._kill_switch_until:
            self._kill_switch_active = False
            logger.info("Kill switch expired — engines may resume.")
            return False
        return True

    def _activate_kill_switch(self) -> None:
        self._kill_switch_active = True
        self._kill_switch_until = datetime.utcnow() + timedelta(hours=24)
        logger.critical("KILL SWITCH ACTIVATED — all engines halted for 24 h.")

    # ──────────────────────────────────────────────────────────
    # Daily P&L / drawdown helpers
    # ──────────────────────────────────────────────────────────

    def _daily_pnl_pct(self, engine: str | None = None) -> float:
        """Return the negative draw as a positive percentage (loss = positive number)."""
        if self._reto_tracker is not None:
            daily = self._reto_tracker.get_daily_pnl()  # type: ignore[attr-defined]
            capital = daily.starting_capital
            if capital == 0:
                return 0.0
            pnl = daily.pnl
        else:
            # Approximate from trade records when no tracker available
            trades = self._daily_trades[engine] if engine else [t for ts in self._daily_trades.values() for t in ts]
            pnl = sum(t.pnl for t in trades)
            capital = settings.INITIAL_CAPITAL

        if pnl < 0 and capital > 0:
            return abs(pnl / capital) * 100
        return 0.0

    # ──────────────────────────────────────────────────────────
    # can_trade
    # ──────────────────────────────────────────────────────────

    def can_trade(self, engine: str) -> bool:  # noqa: PLR0911
        """
        Primary veto gate called by every engine before placing a trade.

        Returns False (with a logged reason) if ANY risk rule is violated.
        """
        self._maybe_reset_daily()

        # 1. Kill switch
        if self.check_kill_switch():
            logger.warning("[%s] BLOCKED: kill switch active.", engine)
            return False

        # 2. Engine pause (consecutive losses)
        pause_until = self._pause_until.get(engine)
        if pause_until and datetime.utcnow() < pause_until:
            logger.warning("[%s] BLOCKED: paused until %s.", engine, pause_until.isoformat())
            return False

        # 3. Global daily drawdown → kill switch check
        global_dd = self._daily_pnl_pct()
        if global_dd >= settings.KILL_SWITCH_PCT:
            self._activate_kill_switch()
            return False

        # 4. Global max daily risk
        if global_dd >= settings.MAX_DAILY_RISK_PCT:
            logger.warning("[%s] BLOCKED: global daily risk %.1f%% ≥ %.1f%%.", engine, global_dd, settings.MAX_DAILY_RISK_PCT)
            return False

        # 5. Per-engine daily risk cap
        engine_max = ENGINE_DAILY_RISK_PCT.get(engine, 2.0)
        engine_dd = self._daily_pnl_pct(engine)
        if engine_dd >= engine_max:
            logger.warning("[%s] BLOCKED: engine daily risk %.1f%% ≥ %.1f%%.", engine, engine_dd, engine_max)
            return False

        # 6. Max simultaneous open positions
        total_open = sum(len(v) for v in self._open_positions.values())
        if total_open >= settings.MAX_SIMULTANEOUS_POSITIONS:
            logger.warning("[%s] BLOCKED: max simultaneous positions (%d) reached.", engine, settings.MAX_SIMULTANEOUS_POSITIONS)
            return False

        # 7. Per-day trade count limit
        from config.settings import PHASES
        # Determine phase via reto_tracker if available
        phase = 1
        if self._reto_tracker is not None:
            phase = self._reto_tracker.get_phase()  # type: ignore[attr-defined]
        phase_cfg = PHASES.get(phase, PHASES[1])
        engine_trades_today = len(self._daily_trades[engine])
        if engine_trades_today >= phase_cfg.max_trades_per_day:
            logger.warning("[%s] BLOCKED: max trades/day (%d) reached.", engine, phase_cfg.max_trades_per_day)
            return False

        # 8. PDT compliance for momo
        if engine == "momo" and not self.is_pdt_compliant():
            logger.warning("[momo] BLOCKED: PDT limit reached.")
            return False

        return True

    # ──────────────────────────────────────────────────────────
    # register_trade
    # ──────────────────────────────────────────────────────────

    def register_trade(
        self,
        engine: str,
        pnl: float,
        won: bool,
        direction: str = "LONG",
        ticker: str = "",
    ) -> None:
        """
        Record the result of a completed trade.

        Updates consecutive-loss counters and triggers engine pause when needed.
        Also triggers global kill switch if capital drawdown threshold is met.
        """
        self._maybe_reset_daily()
        record = TradeRecord(
            engine=engine,
            pnl=pnl,
            won=won,
            direction=direction,
            ticker=ticker,
        )
        self._daily_trades[engine].append(record)

        if won:
            self._consecutive_losses[engine] = 0
        else:
            self._consecutive_losses[engine] += 1
            if self._consecutive_losses[engine] >= settings.MAX_CONSECUTIVE_LOSSES:
                pause_until = datetime.utcnow() + timedelta(hours=settings.CONSECUTIVE_LOSS_PAUSE_HOURS)
                self._pause_until[engine] = pause_until
                logger.warning(
                    "[%s] %d consecutive losses → engine paused until %s.",
                    engine,
                    self._consecutive_losses[engine],
                    pause_until.isoformat(),
                )

        # PDT tracking for momo
        if engine == "momo":
            self._momo_day_trades.append(date.today())
            self._prune_pdt_window()

        # Global kill switch check
        self.check_kill_switch()

    # ──────────────────────────────────────────────────────────
    # Open position tracking
    # ──────────────────────────────────────────────────────────

    def open_position(self, engine: str, ticker: str, direction: str) -> None:
        """Register an open position."""
        self._open_positions[engine].append((ticker, direction))
        logger.debug("[%s] Position opened: %s %s (total open=%d)", engine, ticker, direction, self.get_open_position_count())

    def close_position(self, engine: str, ticker: str) -> None:
        """Remove a position from the open-position tracker."""
        positions = self._open_positions[engine]
        self._open_positions[engine] = [(t, d) for t, d in positions if t != ticker]

    def get_open_position_count(self) -> int:
        return sum(len(v) for v in self._open_positions.values())

    def get_remaining_bullets(self) -> int:
        """Return how many more trades are allowed today across ALL engines."""
        self._maybe_reset_daily()
        phase = 1
        if self._reto_tracker is not None:
            phase = self._reto_tracker.get_phase()  # type: ignore[attr-defined]
        from config.settings import PHASES
        max_per_day = PHASES.get(phase, PHASES[1]).max_trades_per_day * len(ENGINES)
        used = sum(len(v) for v in self._daily_trades.values())
        return max(0, max_per_day - used)

    # ──────────────────────────────────────────────────────────
    # Correlation guard
    # ──────────────────────────────────────────────────────────

    def has_correlation_conflict(self, engine: str, direction: str) -> bool:
        """
        Return True if placing a trade would create a correlated position.

        NQ long + BTC long are considered ~0.7 correlated → block.
        """
        if direction != "LONG":
            return False  # only guard same-direction correlation

        correlated_pairs: list[tuple[str, str]] = [
            ("futures", "crypto"),
            ("crypto", "futures"),
        ]
        for own_engine, other_engine in correlated_pairs:
            if engine == own_engine:
                # Check if the other engine has an open LONG position
                for _, d in self._open_positions.get(other_engine, []):
                    if d == "LONG":
                        return True
        return False

    # ──────────────────────────────────────────────────────────
    # PDT Tracking
    # ──────────────────────────────────────────────────────────

    def _prune_pdt_window(self) -> None:
        """Remove day-trade records older than 5 business days from the deque."""
        cutoff = _business_days_ago(settings.PDT_ROLLING_DAYS)
        while self._momo_day_trades and self._momo_day_trades[0] < cutoff:
            self._momo_day_trades.popleft()

    def is_pdt_compliant(self) -> bool:
        """Return True if the momo engine has not hit the 3-trade PDT limit."""
        self._prune_pdt_window()
        return len(self._momo_day_trades) < settings.PDT_MAX_DAY_TRADES

    def get_pdt_trades_remaining(self) -> int:
        self._prune_pdt_window()
        return max(0, settings.PDT_MAX_DAY_TRADES - len(self._momo_day_trades))


# ──────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────

def _business_days_ago(n: int) -> date:
    """Return the date N business days before today (skipping weekends)."""
    d = date.today()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            count += 1
    return d
