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
  - Adaptive profit protection (tier-based score/size restrictions + trailing floor).
  - Persist kill switch and daily state to disk so restarts don't bypass risk limits.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, DefaultDict

from config import settings
from config.settings import ENGINE_DAILY_RISK_PCT

if TYPE_CHECKING:
    from notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

ENGINES = ("futures", "options", "momo", "crypto")

# Path used to persist kill-switch / daily state across restarts
_RISK_STATE_PATH = os.path.join("data", "risk_state.json")

# ── Dynamic profit protection tiers (dollar-based) ────────────────────────────
# Each row: (min_pnl_usd, retention_pct, min_brain_score, size_multiplier)
# Tiers are evaluated from highest to lowest threshold.
_PROFIT_LOCK_TIERS: list[tuple[float, float, int, float]] = [
    (250.0, 0.75, 85, 0.50),
    (150.0, 0.70, 80, 0.75),
    (100.0, 0.65, 75, 1.00),
    ( 50.0, 0.60, 70, 1.00),
    ( 25.0, 0.50, 70, 1.00),
]


@dataclass
class TradeRecord:
    """Minimal record for risk-tracking purposes."""

    engine: str
    pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    won: bool = True
    direction: str = "LONG"   # "LONG" or "SHORT"
    ticker: str = ""


@dataclass
class ProfitProtectionEvent:
    """Events emitted by update_daily_pnl() that the engine should act on."""

    tier_entered: int | None = None   # 1, 2, or 3 when a new tier is first entered
    floor_activated: bool = False     # trailing floor just became active
    floor_hit: bool = False           # P&L has dropped to / below the floor
    current_pnl: float = 0.0
    floor_value: float = 0.0
    min_score: int = 65
    size_multiplier: float = 1.0


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

        # ── Adaptive profit protection state ──────────────────
        self._daily_pnl_gain: float = 0.0       # latest reported positive P&L
        self._max_daily_pnl_gain: float = 0.0   # peak positive P&L for the day
        self._last_profit_tier: int = 0          # last tier notified (0 = below all thresholds)
        self._floor_active: bool = False         # trailing floor has been activated
        self._daily_profit_locked: bool = False  # P&L has dropped to/below floor (locks for the day)

        # Load persisted state (kill switch survives restarts)
        self._load_state()

    # ──────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load persisted risk state from disk (kill switch, daily counters)."""
        try:
            if not os.path.exists(_RISK_STATE_PATH):
                return
            with open(_RISK_STATE_PATH, encoding="utf-8") as fh:
                state = json.load(fh)

            # Only restore state for the current calendar day
            saved_date_str = state.get("today_date", "")
            today_str = date.today().isoformat()
            if saved_date_str != today_str:
                logger.info("Risk state file is from a previous day (%s) — ignoring.", saved_date_str)
                return

            # Kill switch
            if state.get("kill_switch_active", False):
                until_str = state.get("kill_switch_until", "")
                if until_str:
                    until = datetime.fromisoformat(until_str)
                    if datetime.utcnow() < until:
                        self._kill_switch_active = True
                        self._kill_switch_until = until
                        logger.warning(
                            "Kill switch restored from disk — still active until %s.",
                            until.isoformat(),
                        )
                    else:
                        logger.info("Persisted kill switch found but already expired — ignoring.")

            # Per-engine daily trade counts (as dummy records so the count is preserved)
            for engine, count in state.get("daily_trades_count", {}).items():
                for _ in range(int(count)):
                    self._daily_trades[engine].append(
                        TradeRecord(engine=engine, pnl=0.0, won=True)
                    )

            # Consecutive losses
            for engine, count in state.get("consecutive_losses", {}).items():
                self._consecutive_losses[engine] = int(count)

            # Engine pauses
            for engine, until_str in state.get("paused_until", {}).items():
                if until_str:
                    until = datetime.fromisoformat(until_str)
                    if datetime.utcnow() < until:
                        self._pause_until[engine] = until
                        logger.info("Engine %s pause restored from disk (until %s).", engine, until_str)

            # Profit protection state
            self._max_daily_pnl_gain = float(state.get("daily_peak_pnl", 0.0))
            self._floor_active = bool(state.get("floor_active", False))
            self._daily_profit_locked = bool(state.get("daily_profit_locked", False))
            if self._daily_profit_locked:
                logger.warning("Daily profit lock restored from disk — trading blocked for today.")

            logger.info("Risk state loaded from %s.", _RISK_STATE_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load risk state from %s: %s", _RISK_STATE_PATH, exc)

    def _save_state(self) -> None:
        """Persist current risk state to disk so a restart preserves kill switch etc."""
        try:
            os.makedirs(os.path.dirname(_RISK_STATE_PATH), exist_ok=True)
            state = {
                "kill_switch_active": self._kill_switch_active,
                "kill_switch_until": self._kill_switch_until.isoformat() if self._kill_switch_until else None,
                "daily_trades_count": {k: len(v) for k, v in self._daily_trades.items()},
                "consecutive_losses": dict(self._consecutive_losses),
                "paused_until": {
                    k: v.isoformat() for k, v in self._pause_until.items()
                },
                "today_date": self._today.isoformat(),
                # Profit protection state — persists across restarts within the same day
                "daily_peak_pnl": self._max_daily_pnl_gain,
                "floor_active": self._floor_active,
                "daily_profit_locked": self._daily_profit_locked,
            }
            with open(_RISK_STATE_PATH, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save risk state to %s: %s", _RISK_STATE_PATH, exc)

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
            # Reset profit protection state
            self._daily_pnl_gain = 0.0
            self._max_daily_pnl_gain = 0.0
            self._last_profit_tier = 0
            self._floor_active = False
            self._daily_profit_locked = False
            self._save_state()

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
            self._save_state()
            logger.info("Kill switch expired — engines may resume.")
            return False
        return True

    def _activate_kill_switch(self) -> None:
        self._kill_switch_active = True
        self._kill_switch_until = datetime.utcnow() + timedelta(hours=24)
        self._save_state()
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

        # 1. Daily profit lock — checked FIRST before any other rule
        if self._daily_profit_locked:
            logger.warning("[%s] BLOCKED: daily profit locked — protecting gains.", engine)
            return False

        # 2. Kill switch
        if self.check_kill_switch():
            logger.warning("[%s] BLOCKED: kill switch active.", engine)
            return False

        # 3. Engine pause (consecutive losses)
        pause_until = self._pause_until.get(engine)
        if pause_until and datetime.utcnow() < pause_until:
            logger.warning("[%s] BLOCKED: paused until %s.", engine, pause_until.isoformat())
            return False

        # 4. Global daily drawdown → kill switch check
        global_dd = self._daily_pnl_pct()
        if global_dd >= settings.KILL_SWITCH_PCT:
            self._activate_kill_switch()
            return False

        # 5. Global max daily risk
        if global_dd >= settings.MAX_DAILY_RISK_PCT:
            logger.warning("[%s] BLOCKED: global daily risk %.1f%% ≥ %.1f%%.", engine, global_dd, settings.MAX_DAILY_RISK_PCT)
            return False

        # 6. Per-engine daily risk cap
        engine_max = ENGINE_DAILY_RISK_PCT.get(engine, 2.0)
        engine_dd = self._daily_pnl_pct(engine)
        if engine_dd >= engine_max:
            logger.warning("[%s] BLOCKED: engine daily risk %.1f%% ≥ %.1f%%.", engine, engine_dd, engine_max)
            return False

        # 7. Max simultaneous open positions
        total_open = sum(len(v) for v in self._open_positions.values())
        if total_open >= settings.MAX_SIMULTANEOUS_POSITIONS:
            logger.warning("[%s] BLOCKED: max simultaneous positions (%d) reached.", engine, settings.MAX_SIMULTANEOUS_POSITIONS)
            return False

        # 8. Per-day trade count limit
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

        # 9. PDT compliance for momo
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

        # Persist updated state to disk
        self._save_state()

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

    def sync_open_positions(self, engine: str, actual_tickers: list[str]) -> None:
        """
        Reconcile the risk manager's open-position list for an engine with
        the engine's own authoritative list.

        Call this at the end of every monitor cycle so that the risk manager
        stays in sync even if ``close_position`` was skipped (e.g. due to a
        temporary IBKR disconnect).
        """
        actual_set = set(actual_tickers)
        old_list = self._open_positions[engine]
        new_list = [(t, d) for t, d in old_list if t in actual_set]
        removed = len(old_list) - len(new_list)
        if removed:
            logger.info(
                "[%s] Position sync: removed %d stale position(s) from risk tracker.",
                engine,
                removed,
            )
        self._open_positions[engine] = new_list

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

    # ──────────────────────────────────────────────────────────
    # Adaptive Profit Protection
    # ──────────────────────────────────────────────────────────

    def _get_capital(self) -> float:
        """Return today's starting capital for profit-protection percentage calculations."""
        if self._reto_tracker is not None:
            try:
                daily = self._reto_tracker.get_daily_pnl()  # type: ignore[attr-defined]
                if daily.starting_capital > 0:
                    return daily.starting_capital
            except Exception:  # noqa: BLE001
                pass
        return settings.INITIAL_CAPITAL

    def _get_retention_pct(self, pnl: float) -> float:
        """Return the dynamic floor retention percentage for the given P&L level.

        Retention percentage increases with P&L to protect larger gains more aggressively:
          $250+:    75% retained
          $150–249: 70% retained
          $100–149: 65% retained
          $50–99:   60% retained
          $25–49:   50% retained
          < $25:     0% (floor not active)
        """
        for threshold, retention, _, _ in _PROFIT_LOCK_TIERS:
            if pnl >= threshold:
                return retention
        return 0.0

    def get_profit_tier(self) -> int:
        """
        Return the current profit protection tier (0–4) based on today's P&L
        in dollar terms using the dynamic tier thresholds.

        Tier 0: < $25          (normal)
        Tier 1: $25–$99        (min score 70, mult 1.0)
        Tier 2: $100–$149      (min score 75, mult 1.0)
        Tier 3: $150–$249      (min score 80, mult 0.75)
        Tier 4: $250+          (min score 85, mult 0.50)
        """
        pnl = self._daily_pnl_gain
        if pnl >= 250.0:
            return 4
        if pnl >= 150.0:
            return 3
        if pnl >= 100.0:
            return 2
        if pnl >= 25.0:
            return 1
        return 0

    def get_min_score_for_tier(self) -> int:
        """Return the minimum brain score required for the current profit tier."""
        pnl = self._daily_pnl_gain
        for threshold, _, min_score, _ in _PROFIT_LOCK_TIERS:
            if pnl >= threshold:
                return min_score
        return settings.PROFIT_TIER_0_MIN_SCORE  # below all thresholds: normal baseline

    def get_size_multiplier_for_tier(self) -> float:
        """Return the position size multiplier for the current profit tier."""
        pnl = self._daily_pnl_gain
        for threshold, _, _, size_mult in _PROFIT_LOCK_TIERS:
            if pnl >= threshold:
                return size_mult
        return settings.PROFIT_TIER_0_SIZE_MULT  # below all thresholds: full size

    def is_profit_floor_hit(self) -> bool:
        """Return True if the trailing profit floor has been breached (trading locked for the day)."""
        return self._daily_profit_locked

    def update_daily_pnl(self, pnl: float) -> ProfitProtectionEvent:
        """
        Update today's P&L figure and return a ProfitProtectionEvent describing
        any state changes (tier entry, floor activation, floor breach).

        Parameters
        ----------
        pnl : float
            Current total daily P&L in dollars (positive = gain, negative = loss).
        """
        self._maybe_reset_daily()
        self._daily_pnl_gain = pnl

        # Track running peak (never decreases)
        if pnl > self._max_daily_pnl_gain:
            self._max_daily_pnl_gain = pnl

        event = ProfitProtectionEvent(
            current_pnl=pnl,
            min_score=self.get_min_score_for_tier(),
            size_multiplier=self.get_size_multiplier_for_tier(),
        )

        capital = self._get_capital()

        # ── Tier change detection ──────────────────────────────
        new_tier = self.get_profit_tier()
        if new_tier > self._last_profit_tier:
            event.tier_entered = new_tier
            self._last_profit_tier = new_tier
            logger.info(
                "[risk] Profit tier %d entered — P&L=+$%.2f (%.1f%% of $%.2f). "
                "Min score=%d, size mult=%.0f%%.",
                new_tier,
                pnl,
                (pnl / capital * 100) if capital > 0 else 0,
                capital,
                event.min_score,
                event.size_multiplier * 100,
            )

        # ── Trailing profit floor ──────────────────────────────
        # Floor activates once P&L reaches PROFIT_FLOOR_ACTIVATION_USD ($25 by default).
        # The retention percentage scales dynamically with the peak P&L level.
        if self._max_daily_pnl_gain >= settings.PROFIT_FLOOR_ACTIVATION_USD:
            retention_pct = self._get_retention_pct(self._max_daily_pnl_gain)
            floor_value = self._max_daily_pnl_gain * retention_pct
            event.floor_value = floor_value

            if not self._floor_active:
                self._floor_active = True
                event.floor_activated = True
                logger.info(
                    "[risk] Trailing profit floor activated — peak=+$%.2f, retention=%.0f%%, floor=+$%.2f.",
                    self._max_daily_pnl_gain,
                    retention_pct * 100,
                    floor_value,
                )

            # Detect floor breach — once locked, stays locked for the rest of the day
            if not self._daily_profit_locked and pnl <= floor_value:
                self._daily_profit_locked = True
                event.floor_hit = True
                self._save_state()
                logger.warning(
                    "[risk] Daily profit protected at $%.2f. Trading stopped for the day. "
                    "(P&L=+$%.2f dropped to/below floor=+$%.2f)",
                    floor_value,
                    pnl,
                    floor_value,
                )

        return event


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
