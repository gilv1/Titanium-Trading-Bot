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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd

from analysis.patterns import detect_ema_pullback, detect_vwap_bounce
from analysis.technical import calculate_atr, calculate_ema, calculate_vwap
from config import settings
from core.reto_tracker import TradeResult as RetoTradeResult
from engines.base_engine import BaseEngine, Position, Setup, Signal, TradeResult

if TYPE_CHECKING:
    from core.brain import AIBrain
    from core.connection import ConnectionManager
    from core.reto_tracker import RetoTracker
    from core.risk_manager import RiskManager
    from journal.trade_journal import TradeJournal
    from notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

CRYPTO_PAIRS = ("BTC", "ETH")

# Minimum order sizes for IBKR Paxos crypto
CRYPTO_MIN_QTY = {
    "BTC": 0.0001,   # ~$6-7 at current prices
    "ETH": 0.01,     # ~$20 at current prices
}

CRYPTO_MIN_ORDER_USD = 10.0  # Don't place orders worth less than $10

# Decimal precision for IBKR Paxos crypto quantities (6 decimal places = 0.000001 BTC minimum step)
CRYPTO_QUANTITY_PRECISION = Decimal('0.000001')

# Tick sizes for IBKR Paxos crypto (Warning 110: price must conform to minimum price variation)
CRYPTO_TICK_SIZE = {
    "BTC": 0.25,   # BTC prices must be in $0.25 increments
    "ETH": 0.01,   # ETH prices in $0.01 increments
}

# How long (seconds) to hold a position before closing at market if TP/SL not yet hit
CRYPTO_MAX_POSITION_AGE_SECS = 30 * 60  # 30 minutes


def _round_to_tick(price: float, symbol: str) -> float:
    """Round price to the nearest valid tick increment for the given crypto symbol."""
    tick = CRYPTO_TICK_SIZE.get(symbol, 0.01)
    return round(round(price / tick) * tick, 2)


@dataclass
class CryptoPosition:
    """Tracks an open software-managed crypto position (TP/SL handled in-process)."""

    symbol: str
    action: str          # 'BUY' or 'SELL' (entry side)
    qty: float
    entry_price: float
    tp_price: float
    sl_price: float
    entry_time: datetime
    order_id: int


class CryptoEngine(BaseEngine):
    """Motor 4 — BTC/ETH crypto trading engine via IBKR Paxos."""

    def __init__(
        self,
        connection_manager: "ConnectionManager",
        brain: "AIBrain",
        reto_tracker: "RetoTracker",
        risk_manager: "RiskManager",
        telegram: "TelegramNotifier | None" = None,
        journal: "TradeJournal | None" = None,
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
        # Guard against duplicate orders per ticker within the same loop cycle
        self._pending_tickers: set[str] = set()
        # Tracks the active software-managed position (TP/SL monitored in-process)
        self._active_position: CryptoPosition | None = None

        # Startup warmup: skip scanning for the first 120 seconds to allow IBKR
        # data feeds to stabilise and any existing positions to be detected.
        self._startup_time: datetime = datetime.utcnow()
        self._warmup_complete: bool = False

    def get_engine_name(self) -> str:
        return "crypto"

    def is_active_session(self) -> bool:
        """Crypto is 24/7 — always active."""
        return True

    def _current_session(self) -> str:
        """Classify current time into a crypto session."""
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

    def _get_effective_allocation(self) -> float:
        """Return crypto allocation: 70% off-hours, 30% during market hours.

        During regular market hours (9:30 AM–4:00 PM ET, Mon–Fri), the futures
        and momo engines are active so crypto uses its standard 30% allocation.
        Outside those hours capital from idle engines is redirected to crypto (70%).
        """
        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        minutes = now.hour * 60 + now.minute
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Weekend: always 70%
        if weekday >= 5:
            return 0.70

        # Weekday market hours (9:30 AM–4:00 PM ET): 30%
        if 570 <= minutes < 960:
            return 0.30

        # Weekday off-hours: 70%
        return 0.70

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
        # Bug 3: skip scanning during startup warmup (first 120 s)
        if not self._warmup_complete:
            elapsed = (datetime.utcnow() - self._startup_time).total_seconds()
            if elapsed < 120.0:
                logger.debug("[crypto] Startup warmup in progress (%.0f/120 s) — skipping scan.", elapsed)
                return []
            self._warmup_complete = True
            logger.info("[crypto] Startup warmup complete — beginning normal scan.")

        # Pre-RTH cutoff: stop opening new crypto trades at 8:30 AM ET on weekdays
        # to free capital for the RTH open (9:30 AM), which is the most profitable session.
        _now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        _minutes = _now.hour * 60 + _now.minute
        _weekday = _now.weekday()
        if _weekday < 5 and 510 <= _minutes < 570:  # 8:30–9:30 AM Mon–Fri
            logger.info("[crypto] Pre-RTH cutoff: no new trades after 8:30 AM ET.")
            return []

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

                # Bug 4: skip flat markets — ATR < 0.05 % of price
                last_price = float(df["close"].iloc[-1])
                if last_price > 0 and atr / last_price < 0.0005:
                    logger.debug(
                        "[crypto] %s: ATR too low (%.4f%% of price) — skipping flat market.",
                        symbol,
                        (atr / last_price) * 100,
                    )
                    continue

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

        # Guard: refuse to place a new order while a position is already active
        if signal.ticker in self._pending_tickers:
            logger.warning("[crypto] Order already pending for %s — skipping.", signal.ticker)
            return None

        # One position at a time across all crypto pairs
        if self._active_position is not None:
            logger.debug("[crypto] Active position in %s — skipping new entry for %s.",
                         self._active_position.symbol, signal.ticker)
            return None

        # Check correlation guard before placing
        if self._risk.has_correlation_conflict("crypto", direction):
            logger.warning("[crypto] Correlation conflict with futures; skipping %s %s.", direction, signal.ticker)
            return None

        # Dynamic allocation based on time of day
        allocation = self._get_effective_allocation()
        max_dollars = self._reto.capital * allocation * size_multiplier
        sl_pct = 0.015  # 1.5 %
        tp_pct = 0.035  # 3.5 %
        # Round entry, tp, sl to valid IBKR Paxos tick increments
        entry = _round_to_tick(signal.entry_price, signal.ticker)
        sl = (
            _round_to_tick(entry * (1 - sl_pct), signal.ticker)
            if direction == "LONG"
            else _round_to_tick(entry * (1 + sl_pct), signal.ticker)
        )
        tp = (
            _round_to_tick(entry * (1 + tp_pct), signal.ticker)
            if direction == "LONG"
            else _round_to_tick(entry * (1 - tp_pct), signal.ticker)
        )
        qty = round(max_dollars / entry, 6)  # BTC/ETH fractional

        logger.info(
            "[crypto] Sizing: capital=$%.2f × alloc=%.0f%% × mult=%.2f = $%.2f → qty=%.6f %s @ $%.2f",
            self._reto.capital, allocation * 100, size_multiplier, max_dollars, qty, signal.ticker, entry,
        )

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
            from ib_insync import LimitOrder  # type: ignore

            ib = self._connection.margin.get_ib()
            if ib is None:
                return None

            # Convert qty to Decimal for IBKR Paxos (required for fractional crypto quantities)
            qty_decimal = Decimal(str(qty)).quantize(CRYPTO_QUANTITY_PRECISION, rounding=ROUND_DOWN)
            if qty_decimal <= 0:
                logger.warning(
                    "[crypto] Qty rounded to zero for %s after Decimal conversion. Skipping.",
                    signal.ticker,
                )
                return None

            # Use the current market price as IOC limit to maximise fill probability.
            # IOC (Immediate-or-Cancel) is the ONLY supported TIF for Market/Limit orders
            # on IBKR Paxos — bracket/stop orders are NOT supported (Error 387).
            current_price = await self._get_current_price(signal.ticker)
            if current_price <= 0:
                logger.warning("[crypto] Could not fetch current price for %s — skipping IOC entry.", signal.ticker)
                return None

            # Reject if signal price is stale (>5% from current market)
            deviation = abs(entry - current_price) / current_price
            if deviation > 0.05:
                logger.error(
                    "[crypto] Entry price %.2f deviates >5%% from current price %.2f for %s. Skipping.",
                    entry, current_price, signal.ticker,
                )
                return None

            # Price the IOC limit at current market to ensure immediate fill
            ioc_limit = _round_to_tick(current_price, signal.ticker)

            entry_order = LimitOrder(
                action=action,
                totalQuantity=float(qty_decimal),
                lmtPrice=ioc_limit,
                tif='IOC',
            )

            logger.info(
                "[crypto] Placing LimitOrder IOC %s %s qty=%.6f @ %.2f (TP=%.2f SL=%.2f)",
                action, signal.ticker, float(qty_decimal), ioc_limit, tp, sl,
            )

            self._pending_tickers.add(signal.ticker)  # prevent duplicates before async place
            entry_trade = await self._connection.margin.place_order(contract, entry_order)
            if entry_trade is None:
                self._pending_tickers.discard(signal.ticker)
                return None

            # Wait briefly for IOC to settle
            await asyncio.sleep(2)

            order_status = entry_trade.orderStatus
            status = order_status.status if order_status is not None else ""
            if status != "Filled":
                logger.info(
                    "[crypto] Entry not filled (IOC expired): %s %s @ %.2f — status=%s",
                    action, signal.ticker, ioc_limit, status,
                )
                self._pending_tickers.discard(signal.ticker)
                return None

            fill_price = _round_to_tick(
                float(order_status.avgFillPrice) if order_status.avgFillPrice else ioc_limit,
                signal.ticker,
            )
            fill_qty = float(qty_decimal)

        except Exception as exc:  # noqa: BLE001
            logger.error("[crypto] Order error: %s", exc)
            self._pending_tickers.discard(signal.ticker)
            return None

        # Entry confirmed — set up software-managed position
        now_utc = datetime.now(timezone.utc)
        self._active_position = CryptoPosition(
            symbol=signal.ticker,
            action=action,
            qty=fill_qty,
            entry_price=fill_price,
            tp_price=tp,
            sl_price=sl,
            entry_time=now_utc,
            order_id=entry_trade.order.orderId,
        )

        position = Position(
            engine="crypto",
            ticker=signal.ticker,
            direction=direction,
            entry_price=fill_price,
            stop_price=sl,
            target_price=tp,
            quantity=fill_qty,
            entry_time=now_utc,
        )
        self._open_positions.append(position)
        self._risk.open_position("crypto", signal.ticker, direction)

        logger.info(
            "[crypto] Entry FILLED: %s %s qty=%.6f @ %.2f | TP=%.2f SL=%.2f",
            action, signal.ticker, fill_qty, fill_price, tp, sl,
        )

        if self._telegram:
            asyncio.create_task(
                self._telegram.send_trade_entry(
                    {
                        "engine": "crypto",
                        "ticker": signal.ticker,
                        "direction": direction,
                        "entry": fill_price,
                        "sl": sl,
                        "tp": tp,
                        "qty": fill_qty,
                        "score": ai_score,
                        "rr": round(abs(tp - fill_price) / abs(fill_price - sl), 2) if abs(fill_price - sl) > 0 else 0,
                    }
                )
            )
        return None

    async def _close_position(self, contract: Any, position: Position, reason: str, current_price: float) -> None:
        """Close the active crypto position with a Market IOC (SL) or Limit IOC (TP) order."""
        close_action = "SELL" if position.direction == "LONG" else "BUY"
        qty_decimal = Decimal(str(position.quantity)).quantize(CRYPTO_QUANTITY_PRECISION, rounding=ROUND_DOWN)
        if qty_decimal <= 0:
            logger.error("[crypto] Cannot close %s: qty rounds to zero.", position.ticker)
            return

        try:
            if reason == "SL_HIT":
                from ib_insync import MarketOrder  # type: ignore
                order: Any = MarketOrder(
                    action=close_action,
                    totalQuantity=float(qty_decimal),
                    tif='IOC',
                )
                logger.warning(
                    "[crypto] 🛑 STOP LOSS HIT: %s @ %.2f (SL=%.2f) — closing with Market IOC",
                    position.ticker, current_price, position.stop_price,
                )
            else:
                from ib_insync import LimitOrder  # type: ignore
                close_price = _round_to_tick(current_price, position.ticker)
                order = LimitOrder(
                    action=close_action,
                    totalQuantity=float(qty_decimal),
                    lmtPrice=close_price,
                    tif='IOC',
                )
                logger.info(
                    "[crypto] 🎯 TAKE PROFIT HIT: %s @ %.2f (TP=%.2f) — closing with Limit IOC",
                    position.ticker, current_price, position.target_price,
                )

            trade = await self._connection.margin.place_order(contract, order)
            if trade is None:
                logger.error("[crypto] Close order placement failed for %s.", position.ticker)
                return

            await asyncio.sleep(2)

            order_status = trade.orderStatus
            status = order_status.status if order_status is not None else ""
            if status != "Filled":
                logger.error(
                    "[crypto] Exit order not filled! Status=%s — will retry next cycle", status
                )
                return  # Don't clear position; retry next scan cycle

            fill_price = float(order_status.avgFillPrice) if order_status.avgFillPrice else current_price

        except Exception as exc:  # noqa: BLE001
            logger.error("[crypto] Close order error for %s: %s", position.ticker, exc)
            return

        # Position closed — record result
        if position.direction == "LONG":
            pnl = (fill_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - fill_price) * position.quantity
        won = pnl > 0

        reto_result = RetoTradeResult(engine="crypto", pnl=pnl)
        milestones = self._reto.update_capital(reto_result)

        duration = (datetime.now(timezone.utc) - position.entry_time).total_seconds()
        result = TradeResult(
            engine="crypto",
            ticker=position.ticker,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=fill_price,
            stop_loss=position.stop_price,
            take_profit=position.target_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=(
                (pnl / (position.entry_price * position.quantity)) * 100
                if position.entry_price > 0
                else 0.0
            ),
            duration_seconds=duration,
            setup_type="",
            session=self._current_session(),
            ai_score=0,
            phase=self._reto.get_phase(),
            capital_after=self._reto.capital,
            won=won,
        )
        self._trade_history.append(result)

        # Log to trade journal (daily summary)
        if self._journal is not None:
            try:
                self._journal.log_trade(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[crypto] Journal log error: %s", exc)

        self._risk.register_trade(
            engine="crypto",
            pnl=pnl,
            won=won,
            direction=position.direction,
            ticker=position.ticker,
        )

        # Clean up position state
        self._open_positions = [p for p in self._open_positions if p.ticker != position.ticker]
        self._risk.close_position("crypto", position.ticker)
        self._pending_tickers.discard(position.ticker)
        self._active_position = None

        logger.info(
            "[crypto] Position closed: %s %s | Entry=%.2f Exit=%.2f | P&L=$%.2f | Reason=%s",
            position.ticker, "WIN" if won else "LOSS",
            position.entry_price, fill_price, pnl, reason,
        )

        if self._telegram:
            asyncio.create_task(self._telegram.send_trade_exit(result))

        for msg in milestones:
            if self._telegram:
                asyncio.create_task(self._telegram.send_milestone_alert(msg))

    async def _force_close_position(self, ib: Any, position: Position) -> None:
        """Cancel any open orders and flatten the position with a Market IOC order.

        Used at 9:00 AM ET to free capital before the RTH open.
        """
        contract = self._get_contract(position.ticker)
        if contract is None:
            return
        try:
            # Cancel all open orders for this symbol
            open_trades = ib.openTrades()
            for trade in open_trades:
                if trade.contract.symbol == position.ticker:
                    try:
                        ib.cancelOrder(trade.order)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[crypto] Could not cancel order %s for %s: %s",
                            trade.order.orderId, position.ticker, exc,
                        )

            await asyncio.sleep(1)  # brief pause for cancels to propagate

            from ib_insync import MarketOrder  # type: ignore

            close_action = "SELL" if position.direction == "LONG" else "BUY"
            qty_decimal = Decimal(str(position.quantity)).quantize(
                CRYPTO_QUANTITY_PRECISION, rounding=ROUND_DOWN
            )
            if qty_decimal <= 0:
                logger.error("[crypto] Cannot force-close %s: qty rounds to zero.", position.ticker)
                return
            # Paxos requires tif='IOC' for Market orders
            market_order = MarketOrder(
                action=close_action,
                totalQuantity=float(qty_decimal),
                tif='IOC',
            )
            await self._connection.margin.place_order(contract, market_order)
            logger.warning(
                "[crypto] Market IOC close order placed for %s qty=%.6f.",
                position.ticker, float(qty_decimal),
            )

            # Remove the position from tracking immediately
            self._open_positions = [p for p in self._open_positions if p.ticker != position.ticker]
            self._risk.close_position("crypto", position.ticker)
            self._pending_tickers.discard(position.ticker)
            self._active_position = None
        except Exception as exc:  # noqa: BLE001
            logger.error("[crypto] Force-close error for %s: %s", position.ticker, exc)

    async def monitor_position(self, position: Position) -> None:
        ib = self._connection.margin.get_ib()
        if ib is None:
            return
        try:
            # Force-close overnight crypto positions at 9:00 AM ET to free capital for RTH
            _now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
            if _now.weekday() < 5 and _now.hour == 9 and _now.minute < 5:
                logger.warning(
                    "[crypto] Force-closing %s position before RTH open.", position.ticker
                )
                await self._force_close_position(ib, position)
                return

            # Time-based stop: close position if it has been open > 30 minutes
            age_secs = (datetime.now(timezone.utc) - position.entry_time).total_seconds()
            if age_secs > CRYPTO_MAX_POSITION_AGE_SECS:
                logger.warning(
                    "[crypto] Position %s open for %.0f min — closing at market (time-based stop).",
                    position.ticker, age_secs / 60,
                )
                contract = self._get_contract(position.ticker)
                current_price = await self._get_current_price(position.ticker)
                if contract is not None:
                    await self._close_position(contract, position, "TIME_STOP", current_price)
                return

            # Fetch current price for TP/SL comparison
            current_price = await self._get_current_price(position.ticker)
            if current_price <= 0:
                logger.debug("[crypto] Could not fetch price for %s — skipping TP/SL check.", position.ticker)
                return

            exit_reason: str | None = None

            if position.direction == "LONG":
                if current_price >= position.target_price:
                    exit_reason = "TP_HIT"
                elif current_price <= position.stop_price:
                    exit_reason = "SL_HIT"
            else:  # SHORT
                if current_price <= position.target_price:
                    exit_reason = "TP_HIT"
                elif current_price >= position.stop_price:
                    exit_reason = "SL_HIT"

            if exit_reason:
                contract = self._get_contract(position.ticker)
                if contract is not None:
                    await self._close_position(contract, position, exit_reason, current_price)

        except Exception as exc:  # noqa: BLE001
            logger.error("[crypto] Monitor error: %s", exc)
