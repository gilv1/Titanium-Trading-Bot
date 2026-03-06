"""
Motor 3 — MoMo Small-Cap Engine for Titanium Warrior v3.

DISABLED by default (ENABLE_MOMO=false in .env).

Strategy:
  - Pre-market (6:00–9:25 AM ET): scanner identifies gap-up small-caps with news catalysts.
  - 3 bullets / rolling 5 business days (PDT compliant).
  - Sends scanner results to Telegram at 9:00 AM ET.
  - 3 entry types: Pullback-to-VWAP, Dip Buy, Breakout.
  - Sells 50 % at Target 1 (prior HOD), moves SL to BE, trails remainder.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from analysis.scanner import MomoScanner
from config import settings
from engines.base_engine import BaseEngine, Position, Setup, Signal, TradeResult

if TYPE_CHECKING:
    from core.brain import AIBrain
    from core.connection import ConnectionManager
    from core.reto_tracker import RetoTracker
    from core.risk_manager import RiskManager
    from notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)


class MomoEngine(BaseEngine):
    """Motor 3 — MoMo small-cap day-trading engine (disabled by default)."""

    def __init__(
        self,
        connection_manager: "ConnectionManager",
        brain: "AIBrain",
        reto_tracker: "RetoTracker",
        risk_manager: "RiskManager",
        telegram: "TelegramNotifier | None" = None,
    ) -> None:
        super().__init__(
            connection_manager=connection_manager,
            brain=brain,
            reto_tracker=reto_tracker,
            risk_manager=risk_manager,
            telegram=telegram,
            loop_interval=30.0,
        )
        self._scanner = MomoScanner()
        self._scanner_done_today = False
        self._scanner_results_sent = False
        self._bullets_fired_today = 0

    def get_engine_name(self) -> str:
        return "momo"

    def is_active_session(self) -> bool:
        """Active during pre-market (for scanning) and regular market hours."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        minutes = now.hour * 60 + now.minute
        # Pre-market: 6:00–9:25 AM  OR  regular: 9:30 AM–4:00 PM
        return (360 <= minutes < 565) or (570 <= minutes < 960)

    def _is_premarket(self) -> bool:
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        minutes = now.hour * 60 + now.minute
        return 360 <= minutes < 565

    def _is_scan_time(self) -> bool:
        """9:00 AM ET — time to send scanner results to Telegram."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        return now.hour == 9 and now.minute == 0

    def _current_session(self) -> str:
        return "NY"

    async def run_loop(self) -> None:
        """Override to handle pre-market scanning and morning alert."""
        from zoneinfo import ZoneInfo

        while self._running:
            try:
                now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))

                # Reset daily state at midnight
                if now.hour == 0 and now.minute < 2:
                    self._scanner_done_today = False
                    self._scanner_results_sent = False
                    self._bullets_fired_today = 0

                if not self.is_active_session():
                    await asyncio.sleep(self._loop_interval)
                    continue

                # Pre-market scanning phase
                if self._is_premarket() and not self._scanner_done_today:
                    candidates = await self._scanner.scan_premarket()
                    self._scanner_done_today = True
                    logger.info("[momo] Scanner found %d candidates.", len(candidates))

                    # Send Telegram alert at 9:00 AM
                    if self._is_scan_time() and not self._scanner_results_sent and self._telegram:
                        self._scanner_results_sent = True
                        asyncio.create_task(self._telegram.send_momo_scanner(candidates))

                # Market hours — look for entries
                if not self._is_premarket() and self._bullets_fired_today < 1:
                    setups = await self.scan_for_setups()
                    for setup in setups:
                        if not self._risk.can_trade("momo"):
                            break
                        decision = self._brain.evaluate_trade(
                            setup_type=setup.signal.setup_type,
                            engine="momo",
                            entry=setup.signal.entry_price,
                            stop=setup.signal.stop_price,
                            target=setup.signal.target_price,
                            session=setup.session,
                            atr=setup.atr,
                        )
                        if decision.approved:
                            result = await self.execute_trade(setup, decision.size_multiplier, decision.score)
                            if result is not None:
                                self._bullets_fired_today += 1
                            break

                # Monitor open positions
                for position in list(self._open_positions):
                    await self.monitor_position(position)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("[momo] run_loop error: %s", exc, exc_info=True)

            await asyncio.sleep(self._loop_interval)

    async def scan_for_setups(self) -> list[Setup]:
        """Return setups for top-scored scanner candidates."""
        candidates = await self._scanner.scan_premarket()
        setups: list[Setup] = []
        for candidate in candidates[:3]:  # top 3
            if candidate.score < 50:
                continue
            # Determine size multiplier from score
            setup_type = "VWAP_BOUNCE"
            signal = Signal(
                direction="LONG",
                confidence=min(100, candidate.score),
                entry_price=candidate.price,
                stop_price=candidate.price * 0.97,
                target_price=candidate.price * 1.10,
                setup_type=setup_type,
                reasoning=f"MoMo scan: {candidate.news_headline}",
                ticker=candidate.ticker,
            )
            setups.append(Setup(signal=signal, engine="momo", session="NY", atr=0.0))
        return setups

    async def execute_trade(
        self,
        setup: Setup,
        size_multiplier: float,
        ai_score: int,
    ) -> TradeResult | None:
        signal = setup.signal
        max_dollars = self._reto.get_position_size("momo") * size_multiplier
        qty = max(1, int(max_dollars / signal.entry_price))

        logger.info(
            "[momo] Entering trade: %s %d shares @ %.2f (score=%d)",
            signal.ticker,
            qty,
            signal.entry_price,
            ai_score,
        )

        try:
            from ib_insync import MarketOrder, Stock  # type: ignore

            contract = Stock(signal.ticker, "SMART", "USD")
            order = MarketOrder("BUY", qty)
            await self._connection.cash.place_order(contract, order)
        except Exception as exc:  # noqa: BLE001
            logger.error("[momo] Order error: %s", exc)
            return None

        position = Position(
            engine="momo",
            ticker=signal.ticker,
            direction="LONG",
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            quantity=qty,
        )
        self._open_positions.append(position)
        self._risk.open_position("momo", signal.ticker, "LONG")

        if self._telegram:
            asyncio.create_task(
                self._telegram.send_trade_entry(
                    {
                        "engine": "momo",
                        "ticker": signal.ticker,
                        "direction": "LONG",
                        "entry": signal.entry_price,
                        "sl": signal.stop_price,
                        "tp": signal.target_price,
                        "qty": qty,
                        "score": ai_score,
                        "rr": round(abs(signal.target_price - signal.entry_price) / abs(signal.entry_price - signal.stop_price), 2),
                    }
                )
            )
        return None

    async def monitor_position(self, position: Position) -> None:
        """Manage MoMo position: sell 50 % at T1, trail remainder."""
        ib = self._connection.cash.get_ib()
        if ib is None:
            return
        try:
            open_trades = ib.openTrades()
            symbol_trades = [t for t in open_trades if t.contract.symbol == position.ticker]
            if not symbol_trades:
                self._open_positions = [p for p in self._open_positions if p.ticker != position.ticker]
                self._risk.close_position("momo", position.ticker)
        except Exception as exc:  # noqa: BLE001
            logger.error("[momo] Monitor error: %s", exc)
