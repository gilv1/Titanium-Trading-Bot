"""
Base Engine — Abstract base class for all Titanium Warrior v3 trading engines.

Defines the common interface and shared dataclasses:
  - Setup
  - Signal
  - Position
  - TradeResult
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.brain import AIBrain
    from core.connection import ConnectionManager
    from core.reto_tracker import RetoTracker
    from core.risk_manager import RiskManager
    from notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Shared dataclasses
# ──────────────────────────────────────────────────────────────


@dataclass
class Signal:
    """A detected trade signal from the pattern layer."""

    direction: str               # "LONG" or "SHORT"
    confidence: int              # 0–100
    entry_price: float
    stop_price: float
    target_price: float
    setup_type: str
    reasoning: str = ""
    ticker: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Setup:
    """A potential trade setup ready for evaluation."""

    signal: Signal
    engine: str
    session: str
    atr: float = 0.0
    trend_aligned: bool = True


@dataclass
class Position:
    """An open position."""

    engine: str
    ticker: str
    direction: str           # "LONG" or "SHORT"
    entry_price: float
    stop_price: float
    target_price: float
    quantity: float
    entry_time: datetime = field(default_factory=datetime.utcnow)
    trade_id: str = ""


@dataclass
class TradeResult:
    """The result of a closed trade."""

    engine: str
    ticker: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_seconds: float
    setup_type: str
    session: str
    ai_score: int
    phase: int
    capital_after: float
    won: bool
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""


# ──────────────────────────────────────────────────────────────
# Abstract base engine
# ──────────────────────────────────────────────────────────────


class BaseEngine(ABC):
    """
    Abstract base class for all trading engines.

    Subclasses must implement:
      - ``get_engine_name()``
      - ``is_active_session()``
      - ``scan_for_setups()``
      - ``execute_trade()``
      - ``monitor_position()``

    The ``run_loop()`` orchestrates the scan → evaluate → trade cycle.
    """

    def __init__(
        self,
        connection_manager: "ConnectionManager",
        brain: "AIBrain",
        reto_tracker: "RetoTracker",
        risk_manager: "RiskManager",
        telegram: "TelegramNotifier | None" = None,
        loop_interval: float = 60.0,
    ) -> None:
        self._connection = connection_manager
        self._brain = brain
        self._reto = reto_tracker
        self._risk = risk_manager
        self._telegram = telegram
        self._loop_interval = loop_interval
        self._running = False
        self._open_positions: list[Position] = []
        self._trade_history: list[TradeResult] = []

    # ──────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the engine loop."""
        self._running = True
        logger.info("[%s] Engine started.", self.get_engine_name())
        await self.run_loop()

    async def stop(self) -> None:
        """Signal the engine to stop after the current iteration."""
        self._running = False
        logger.info("[%s] Engine stop requested.", self.get_engine_name())

    # ──────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Main trading loop: scan → filter → evaluate → trade → monitor."""
        while self._running:
            try:
                if not self.is_active_session():
                    await asyncio.sleep(self._loop_interval)
                    continue

                # Scan for potential setups
                setups = await self.scan_for_setups()

                for setup in setups:
                    if not self._risk.can_trade(self.get_engine_name()):
                        break

                    # Check correlation conflict
                    correlation_conflict = self._risk.has_correlation_conflict(
                        self.get_engine_name(), setup.signal.direction
                    )

                    # Let AI brain evaluate
                    daily_dd = self._reto.get_daily_pnl().pnl_pct if self._reto.get_daily_pnl().pnl < 0 else 0.0
                    decision = self._brain.evaluate_trade(
                        setup_type=setup.signal.setup_type,
                        engine=self.get_engine_name(),
                        entry=setup.signal.entry_price,
                        stop=setup.signal.stop_price,
                        target=setup.signal.target_price,
                        session=setup.session,
                        atr=setup.atr,
                        daily_drawdown_pct=abs(daily_dd),
                        open_positions=self._risk.get_open_position_count(),
                        trend_aligned=setup.trend_aligned,
                        correlation_conflict=correlation_conflict,
                    )

                    if not decision.approved:
                        logger.debug(
                            "[%s] Trade rejected by brain (score=%d): %s",
                            self.get_engine_name(),
                            decision.score,
                            decision.reasoning,
                        )
                        continue

                    # Execute trade
                    result = await self.execute_trade(setup, decision.size_multiplier, decision.score)
                    if result is not None:
                        self._trade_history.append(result)
                        self._risk.register_trade(
                            engine=self.get_engine_name(),
                            pnl=result.pnl,
                            won=result.won,
                            direction=result.direction,
                            ticker=result.ticker,
                        )
                        # Self-learning update
                        from core.brain import TradeOutcome
                        from zoneinfo import ZoneInfo
                        from config import settings as cfg

                        now = datetime.now(tz=ZoneInfo(cfg.TIMEZONE))
                        self._brain.record_outcome(
                            TradeOutcome(
                                setup_type=setup.signal.setup_type,
                                session=setup.session,
                                day_of_week=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][now.weekday()],
                                hour=now.hour,
                                volatility_regime="medium" if setup.atr == 0 else ("low" if setup.atr < 5 else ("high" if setup.atr > 20 else "medium")),
                                won=result.won,
                                engine=self.get_engine_name(),
                            )
                        )

                        # Telegram notification (non-blocking)
                        if self._telegram:
                            asyncio.create_task(self._telegram.send_trade_exit(result))

                # Monitor existing positions
                for position in list(self._open_positions):
                    await self.monitor_position(position)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] Unhandled error in run_loop: %s", self.get_engine_name(), exc, exc_info=True)

            await asyncio.sleep(self._loop_interval)

    # ──────────────────────────────────────────────────────────
    # Abstract interface
    # ──────────────────────────────────────────────────────────

    @abstractmethod
    def get_engine_name(self) -> str:
        """Return the unique name of this engine (e.g. 'futures')."""

    @abstractmethod
    def is_active_session(self) -> bool:
        """Return True if the current time is within an active trading session."""

    @abstractmethod
    async def scan_for_setups(self) -> list[Setup]:
        """Scan market data and return a list of potential trade setups."""

    @abstractmethod
    async def execute_trade(
        self,
        setup: Setup,
        size_multiplier: float,
        ai_score: int,
    ) -> TradeResult | None:
        """Place the order for a given setup and return the TradeResult once closed."""

    @abstractmethod
    async def monitor_position(self, position: Position) -> None:
        """Check position status and manage trailing stops / partial exits."""
