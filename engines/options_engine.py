"""
Motor 2 — Options Engine (0DTE SPY/QQQ) for Titanium Warrior v3.

DISABLED by default (ENABLE_OPTIONS=false in .env).

Buys 0DTE calls or puts on SPY and QQQ based on VWAP Bounce / ORB setups.
Uses the IBKR Cash account.

Risk rules:
  - Stop loss: −30 % of premium paid.
  - Take profit: +80 % to +150 %.
  - Theta guard: close if no movement within 3–5 minutes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date
from typing import TYPE_CHECKING, Any

import pandas as pd

from analysis.patterns import detect_orb, detect_vwap_bounce
from analysis.technical import calculate_rsi, calculate_vwap
from config import settings
from engines.base_engine import BaseEngine, Position, Setup, Signal, TradeResult

if TYPE_CHECKING:
    from core.brain import AIBrain
    from core.connection import ConnectionManager
    from core.reto_tracker import RetoTracker
    from core.risk_manager import RiskManager
    from notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

TICKERS = ("SPY", "QQQ")


class OptionsEngine(BaseEngine):
    """Motor 2 — 0DTE options trading engine (disabled by default)."""

    def __init__(
        self,
        connection_manager: "ConnectionManager",
        brain: "AIBrain",
        reto_tracker: "RetoTracker",
        risk_manager: "RiskManager",
        telegram: "TelegramNotifier | None" = None,
        journal: "Any | None" = None,
    ) -> None:
        super().__init__(
            connection_manager=connection_manager,
            brain=brain,
            reto_tracker=reto_tracker,
            risk_manager=risk_manager,
            telegram=telegram,
            loop_interval=60.0,
            journal=journal,
        )

    def get_engine_name(self) -> str:
        return "options"

    def is_active_session(self) -> bool:
        """Active 9:30 AM – 4:15 PM ET only."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        sess = settings.SESSIONS["Options"]
        minutes = now.hour * 60 + now.minute
        start = sess.start_hour * 60 + sess.start_minute
        end = sess.end_hour * 60 + sess.end_minute
        return start <= minutes < end

    def _current_session(self) -> str:
        return "NY"

    async def _get_underlying_price(self, ticker: str) -> float:
        """Fetch last price for a stock via IBKR."""
        try:
            from ib_insync import Stock  # type: ignore
        except ImportError:
            return 0.0
        ib = self._connection.cash.get_ib()
        if ib is None:
            return 0.0
        try:
            contract = Stock(ticker, "SMART", "USD")
            ticker_data = ib.reqMktData(contract, "", False, False)
            await asyncio.sleep(2)
            return float(ticker_data.last or ticker_data.close or 0)
        except Exception as exc:  # noqa: BLE001
            logger.error("[options] Error fetching price for %s: %s", ticker, exc)
            return 0.0

    async def _find_atm_strike(self, ticker: str, price: float, right: str) -> object | None:
        """Find ATM option contract expiring today."""
        try:
            from ib_insync import Option  # type: ignore
        except ImportError:
            return None
        # Round to nearest $1 strike
        strike = round(price)
        today = date.today().strftime("%Y%m%d")
        return Option(ticker, today, strike, right, "SMART")

    async def scan_for_setups(self) -> list[Setup]:
        setups: list[Setup] = []
        for ticker in TICKERS:
            price = await self._get_underlying_price(ticker)
            if price <= 0:
                continue
            # Would need bars to run pattern detection
            # For now return placeholder – full impl requires live data
        return setups

    async def execute_trade(
        self,
        setup: Setup,
        size_multiplier: float,
        ai_score: int,
    ) -> TradeResult | None:
        """Buy 0DTE call or put, manage with -30%/+80% rules."""
        signal = setup.signal
        right = "C" if signal.direction == "LONG" else "P"
        price = signal.entry_price

        contract = await self._find_atm_strike(signal.ticker, price, right)
        if contract is None:
            return None

        max_dollars = self._reto.get_position_size("options") * size_multiplier
        premium = price * 100  # approx cost per contract
        qty = max(1, int(max_dollars / premium))

        try:
            from ib_insync import MarketOrder  # type: ignore

            order = MarketOrder("BUY", qty)
            await self._connection.cash.place_order(contract, order)
        except Exception as exc:  # noqa: BLE001
            logger.error("[options] Order error: %s", exc)
            return None

        position = Position(
            engine="options",
            ticker=signal.ticker,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_price=signal.entry_price * 0.70,
            target_price=signal.entry_price * 1.80,
            quantity=qty,
        )
        self._open_positions.append(position)
        self._risk.open_position("options", signal.ticker, signal.direction)

        if self._telegram:
            asyncio.create_task(
                self._telegram.send_trade_entry(
                    {
                        "engine": "options",
                        "ticker": signal.ticker,
                        "direction": signal.direction,
                        "entry": signal.entry_price,
                        "sl": position.stop_price,
                        "tp": position.target_price,
                        "qty": qty,
                        "score": ai_score,
                        "rr": 2.5,
                    }
                )
            )
        return None

    async def monitor_position(self, position: Position) -> None:
        """Check premium P&L and apply theta guard."""
        # Full implementation would poll option premium and close on rules
        pass
