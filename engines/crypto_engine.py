"""
Motor 4 — Crypto Engine (BTC / ETH via IBKR Paxos) for Titanium Warrior v3.

Trades BTC and ETH through IBKR (not Bybit/Binance) to keep everything on one platform.
Operates 24/7 with session tracking (Asia, Europe, US).

Setups: VWAP Bounce, EMA 9/21 Pullback, Range Breakout.

Risk:
  - SL: 1.5–2 % of crypto position value.
  - TP: 3–5 % (min R:R 1:2).
  - Correlation guard: if already long NQ, don't go long BTC.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from analysis.patterns import detect_ema_pullback, detect_vwap_bounce
from analysis.technical import calculate_atr, calculate_ema, calculate_vwap
from config import settings
from engines.base_engine import BaseEngine, Position, Setup, Signal, TradeResult

if TYPE_CHECKING:
    from core.brain import AIBrain
    from core.connection import ConnectionManager
    from core.reto_tracker import RetoTracker
    from core.risk_manager import RiskManager
    from notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

CRYPTO_PAIRS = ("BTC", "ETH")

# Minimum order sizes for IBKR Paxos crypto
CRYPTO_MIN_QTY = {
    "BTC": 0.0001,   # ~$6-7 at current prices
    "ETH": 0.01,     # ~$20 at current prices
}

CRYPTO_MIN_ORDER_USD = 10.0  # Don't place orders worth less than $10


class CryptoEngine(BaseEngine):
    """Motor 4 — BTC/ETH crypto trading engine via IBKR Paxos."""

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
            loop_interval=60.0,
        )
        # Guard against duplicate bracket orders per ticker within the same loop cycle
        self._pending_tickers: set[str] = set()

    def get_engine_name(self) -> str:
        return "crypto"

    def is_active_session(self) -> bool:
        """Crypto is 24/7 — always active."""
        return True

    def _current_session(self) -> str:
        """Classify current time into a crypto session."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        minutes = now.hour * 60 + now.minute
        # Asia: 8 PM – 2 AM ET
        if minutes >= 20 * 60 or minutes < 2 * 60:
            return "Crypto_Asia"
        # Europe: 3 AM – 8 AM ET
        if 3 * 60 <= minutes < 8 * 60:
            return "Crypto_Europe"
        # US: 9:30 AM – 4 PM ET
        if 9 * 60 + 30 <= minutes < 16 * 60:
            return "Crypto_US"
        return "Crypto_Asia"

    def _get_contract(self, symbol: str) -> object | None:
        """Build an ib_insync Crypto contract."""
        try:
            from ib_insync import Crypto  # type: ignore
        except ImportError:
            return None
        return Crypto(symbol, "PAXOS", "USD")

    async def _fetch_bars(self, contract: object) -> pd.DataFrame:
        ib = self._connection.margin.get_ib()
        if ib is None or not self._connection.margin.is_connected():
            return pd.DataFrame()

        for attempt in range(2):  # Retry once on Error 162
            try:
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr="1 D",
                    barSizeSetting="1 min",
                    whatToShow="MIDPOINT",
                    useRTH=False,
                )
                if not bars:
                    return pd.DataFrame()
                return pd.DataFrame(
                    {
                        "time": [b.date for b in bars],
                        "open": [b.open for b in bars],
                        "high": [b.high for b in bars],
                        "low": [b.low for b in bars],
                        "close": [b.close for b in bars],
                        "volume": [b.volume for b in bars],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                if "different IP address" in error_msg or "162" in error_msg:
                    if attempt == 0:
                        logger.warning(
                            "[crypto] IP conflict on %s data fetch; retrying in 5s...",
                            getattr(contract, 'symbol', '?'),
                        )
                        await asyncio.sleep(5)
                        continue
                    else:
                        logger.error(
                            "[crypto] IP conflict persists for %s; skipping this scan cycle.",
                            getattr(contract, 'symbol', '?'),
                        )
                        return pd.DataFrame()
                logger.error("[crypto] Error fetching bars for %s: %s", getattr(contract, 'symbol', '?'), exc)
                return pd.DataFrame()
        return pd.DataFrame()

    async def scan_for_setups(self) -> list[Setup]:
        setups: list[Setup] = []
        session = self._current_session()

        for symbol in CRYPTO_PAIRS:
            try:
                contract = self._get_contract(symbol)
                if contract is None:
                    continue

                df = await self._fetch_bars(contract)
                if df.empty or len(df) < 30:
                    continue

                vwap = calculate_vwap(df)
                ema9 = calculate_ema(df, 9)
                ema21 = calculate_ema(df, 21)
                atr_series = calculate_atr(df)
                atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

                # VWAP Bounce
                sig = detect_vwap_bounce(df, vwap)
                if sig:
                    sig.ticker = symbol
                    setups.append(Setup(signal=sig, engine="crypto", session=session, atr=atr))

                # EMA Pullback
                sig = detect_ema_pullback(df, ema9, ema21)
                if sig:
                    sig.ticker = symbol
                    setups.append(Setup(signal=sig, engine="crypto", session=session, atr=atr))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[crypto] Error scanning %s: %s", symbol, exc)
                continue

        return setups

    async def _get_current_price(self, symbol: str) -> float:
        """Fetch the current market price for a crypto symbol."""
        contract = self._get_contract(symbol)
        if contract is None:
            return 0.0
        ib = self._connection.margin.get_ib()
        if ib is None:
            return 0.0
        try:
            ticker = ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(2)
            price = float(ticker.last or ticker.close or ticker.midpoint() or 0)
            ib.cancelMktData(contract)
            return price
        except Exception as exc:  # noqa: BLE001
            logger.error("[crypto] Error fetching current price for %s: %s", symbol, exc)
            return 0.0

    async def execute_trade(
        self,
        setup: Setup,
        size_multiplier: float,
        ai_score: int,
    ) -> TradeResult | None:
        signal = setup.signal
        direction = signal.direction

        # Guard: refuse to place a new order for a ticker that already has one pending
        if signal.ticker in self._pending_tickers:
            logger.warning("[crypto] Order already pending for %s — skipping duplicate bracket order.", signal.ticker)
            return None

        # Check correlation guard before placing
        if self._risk.has_correlation_conflict("crypto", direction):
            logger.warning("[crypto] Correlation conflict with futures; skipping %s %s.", direction, signal.ticker)
            return None

        max_dollars = self._reto.get_position_size("crypto") * size_multiplier
        entry = round(signal.entry_price, 2)
        sl_pct = 0.015  # 1.5 %
        tp_pct = 0.035  # 3.5 %
        sl = round(entry * (1 - sl_pct), 2) if direction == "LONG" else round(entry * (1 + sl_pct), 2)
        tp = round(entry * (1 + tp_pct), 2) if direction == "LONG" else round(entry * (1 - tp_pct), 2)
        qty = round(max_dollars / entry, 6)  # BTC/ETH fractional

        # Validate minimum quantity
        min_qty = CRYPTO_MIN_QTY.get(signal.ticker, 0.001)
        if qty < min_qty:
            logger.warning(
                "[crypto] Calculated qty %.6f for %s is below minimum %.6f (capital=$%.2f). Skipping.",
                qty, signal.ticker, min_qty, max_dollars,
            )
            return None

        # Validate minimum USD value
        order_value = qty * entry
        if order_value < CRYPTO_MIN_ORDER_USD:
            logger.warning(
                "[crypto] Order value $%.2f for %s is below minimum $%.2f. Skipping.",
                order_value, signal.ticker, CRYPTO_MIN_ORDER_USD,
            )
            return None

        contract = self._get_contract(signal.ticker)
        if contract is None:
            return None

        action = "BUY" if direction == "LONG" else "SELL"
        try:
            from ib_insync import Order  # type: ignore

            ib = self._connection.margin.get_ib()
            if ib is None:
                return None
            bracket = ib.bracketOrder(action, qty, entry, tp, sl)
            for order in bracket:
                order.tif = 'GTC'
            entry_order, tp_order, sl_order = bracket

            # Validate entry price is reasonable for this symbol before placing
            current_price = await self._get_current_price(signal.ticker)
            if current_price > 0:
                deviation = abs(entry - current_price) / current_price
                if deviation > 0.05:  # more than 5% deviation from current price
                    logger.error(
                        "[crypto] Entry price %.2f deviates >5%% from current price %.2f for %s. Skipping.",
                        entry,
                        current_price,
                        signal.ticker,
                    )
                    return None

            self._pending_tickers.add(signal.ticker)  # set before placing to prevent duplicates
            entry_trade = await self._connection.margin.place_order(contract, entry_order)
            await self._connection.margin.place_order(contract, tp_order)
            await self._connection.margin.place_order(contract, sl_order)
        except Exception as exc:  # noqa: BLE001
            logger.error("[crypto] Order error: %s", exc)
            self._pending_tickers.discard(signal.ticker)  # clear so next cycle can retry
            return None

        position = Position(
            engine="crypto",
            ticker=signal.ticker,
            direction=direction,
            entry_price=entry,
            stop_price=sl,
            target_price=tp,
            quantity=qty,
        )
        self._open_positions.append(position)
        self._risk.open_position("crypto", signal.ticker, direction)

        if self._telegram:
            asyncio.create_task(
                self._telegram.send_trade_entry(
                    {
                        "engine": "crypto",
                        "ticker": signal.ticker,
                        "direction": direction,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "qty": qty,
                        "score": ai_score,
                        "rr": round(abs(tp - entry) / abs(entry - sl), 2) if abs(entry - sl) > 0 else 0,
                    }
                )
            )
        return None

    async def monitor_position(self, position: Position) -> None:
        ib = self._connection.margin.get_ib()
        if ib is None:
            return
        try:
            open_trades = ib.openTrades()
            symbol_trades = [t for t in open_trades if t.contract.symbol == position.ticker]
            if not symbol_trades:
                self._open_positions = [p for p in self._open_positions if p.ticker != position.ticker]
                self._risk.close_position("crypto", position.ticker)
                self._pending_tickers.discard(position.ticker)  # allow new orders for this ticker
        except Exception as exc:  # noqa: BLE001
            logger.error("[crypto] Monitor error: %s", exc)
