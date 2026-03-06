"""
Motor 3 — MoMo Small-Cap Engine for Titanium Warrior v3.

DISABLED by default (ENABLE_MOMO=false in .env).

Strategy:
  - Pre-market (6:00–9:25 AM ET): scanner identifies gap-up small-caps with news catalysts.
  - 3 bullets / rolling 5 business days (PDT compliant, tracked via RiskManager).
  - Sends scanner results to Telegram at 9:00 AM ET.
  - Execution window: 9:30 AM – 12:00 PM ET only (MoMo moves happen in first 2.5 hours).
  - 3 entry types: Pullback-to-VWAP, Dip Buy (EMA pullback), Breakout.
  - Score-based sizing: >80 → 100%, 65-80 → 75%, 50-65 → 50%, <50 → skip.
  - Sells 50 % at Target 1 (prior HOD), moves SL to BE, trails remainder.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from analysis.patterns import detect_breakout, detect_ema_pullback, detect_vwap_bounce
from analysis.scanner import MomoScanner
from analysis.technical import calculate_ema, calculate_vwap
from config import settings
from engines.base_engine import BaseEngine, Position, Setup, Signal, TradeResult

if TYPE_CHECKING:
    from core.brain import AIBrain
    from core.connection import ConnectionManager
    from core.reto_tracker import RetoTracker
    from core.risk_manager import RiskManager
    from notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# Score-based sizing thresholds
_SCORE_FULL = 80   # score > 80  → 100 % of allocated capital
_SCORE_MID = 65    # score 65–80 → 75 %
_SCORE_MIN = 50    # score 50–65 → 50 %; score < 50 → no trade

# Trailing stop distance on remainder after partial exit
_TRAIL_PCT = 0.03  # 3 %

# Intraday condition thresholds
_VWAP_MIN_DISTANCE = 0.98     # price must be within 2 % below VWAP to qualify
_VWAP_BOUNCE_TOLERANCE = 0.003  # tolerance for detect_vwap_bounce (0.3 %)
_VWAP_SL_BUFFER = 0.99        # SL placed 1 % below VWAP
_FALLBACK_SL_PCT = 0.97       # fallback SL: 3 % below entry when no bars available
_FALLBACK_TP_PCT = 1.10       # fallback TP: 10 % above entry when no bars available
_HOD_MIN_BUFFER = 1.01        # price must be at least 1 % below HOD for HOD to be used as TP
_RECONNECT_MAX_DELAY = 60.0   # maximum reconnect delay cap (seconds)


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
        # Pass IB connection to scanner so it can use IBKR data
        ib = connection_manager.cash.get_ib() if connection_manager.cash.is_connected() else None
        self._scanner = MomoScanner(ib=ib)
        self._scanner_done_today = False
        self._scanner_results_sent = False
        # Track which tickers have had their 50 % partial exit placed
        self._partial_exits: set[str] = set()
        # Highest price seen per ticker for trailing stop calculation
        self._highest_price: dict[str, float] = {}

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

    def _is_execution_window(self) -> bool:
        """9:30 AM – MOMO_EXECUTION_END_HOUR ET — MoMo execution window."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        minutes = now.hour * 60 + now.minute
        end_minute = settings.MOMO_EXECUTION_END_HOUR * 60
        return 570 <= minutes < end_minute  # 9:30 AM to noon

    def _is_scan_time(self) -> bool:
        """9:00 AM ET — time to send scanner results to Telegram."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        return now.hour == 9 and now.minute == 0

    def _current_session(self) -> str:
        return "NY"

    def _size_multiplier_from_score(self, score: int) -> float:
        """Return position size multiplier based on candidate score."""
        if score > _SCORE_FULL:
            return 1.0
        if score >= _SCORE_MID:
            return 0.75
        if score >= _SCORE_MIN:
            return 0.50
        return 0.0  # no trade

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

                if not self.is_active_session():
                    await asyncio.sleep(self._loop_interval)
                    continue

                # Refresh scanner IB connection if it became available
                if self._scanner._ib is None and self._connection.cash.is_connected():  # noqa: SLF001
                    self._scanner.set_ib_connection(self._connection.cash.get_ib())

                # Pre-market scanning phase
                if self._is_premarket() and not self._scanner_done_today:
                    candidates = await self._scanner.scan_premarket()
                    self._scanner_done_today = True
                    logger.info("[momo] Scanner found %d candidates.", len(candidates))

                    # Send Telegram alert at 9:00 AM
                    if self._is_scan_time() and not self._scanner_results_sent and self._telegram:
                        self._scanner_results_sent = True
                        asyncio.create_task(self._telegram.send_momo_scanner(candidates))

                # Execution window — look for entries (9:30 AM – noon only)
                if self._is_execution_window() and self._risk.get_pdt_trades_remaining() > 0:
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
                                # Register bullet in PDT tracker
                                self._risk.register_trade(
                                    engine="momo",
                                    pnl=0.0,
                                    won=True,
                                    direction="LONG",
                                    ticker=setup.signal.ticker,
                                )
                            break

                # Monitor open positions until 4 PM (regardless of execution window)
                for position in list(self._open_positions):
                    await self.monitor_position(position)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("[momo] run_loop error: %s", exc, exc_info=True)

            await asyncio.sleep(self._loop_interval)

    async def _fetch_intraday_bars(self, ticker: str) -> pd.DataFrame:
        """Fetch today's intraday 1-min bars for a stock via IBKR cash connection."""
        ib = self._connection.cash.get_ib()
        if ib is None or not self._connection.cash.is_connected():
            return pd.DataFrame()
        try:
            from ib_insync import Stock  # type: ignore

            contract = Stock(ticker, "SMART", "USD")
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
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
            logger.warning("[momo] Error fetching bars for %s: %s", ticker, exc)
            return pd.DataFrame()

    async def _get_current_price(self, ticker: str) -> float:
        """Fetch the current market price for a stock."""
        ib = self._connection.cash.get_ib()
        if ib is None:
            return 0.0
        try:
            from ib_insync import Stock  # type: ignore

            contract = Stock(ticker, "SMART", "USD")
            ticker_data = ib.reqMktData(contract, "", False, False)
            await asyncio.sleep(1)
            price = float(ticker_data.last or ticker_data.close or 0)
            ib.cancelMktData(contract)
            return price
        except Exception as exc:  # noqa: BLE001
            logger.warning("[momo] Error fetching price for %s: %s", ticker, exc)
            return 0.0

    async def scan_for_setups(self) -> list[Setup]:
        """Return setups for top-scored scanner candidates using intraday analysis."""
        candidates = await self._scanner.scan_premarket()
        setups: list[Setup] = []

        for candidate in candidates[:5]:  # top 5 by score
            if candidate.score < _SCORE_MIN:
                continue

            size_mult = self._size_multiplier_from_score(candidate.score)
            if size_mult == 0.0:
                continue

            # Fetch intraday 1-min bars
            df = await self._fetch_intraday_bars(candidate.ticker)
            if df.empty or len(df) < 15:
                # Fall back: use scanner candidate price with conservative SL/TP
                entry = candidate.price
                sl = round(entry * _FALLBACK_SL_PCT, 2)
                tp = round(entry * _FALLBACK_TP_PCT, 2)
                sig = Signal(
                    direction="LONG",
                    confidence=candidate.score,
                    entry_price=entry,
                    stop_price=sl,
                    target_price=tp,
                    setup_type="VWAP_BOUNCE",
                    reasoning=f"MoMo fallback: {candidate.news_headline}",
                    ticker=candidate.ticker,
                )
                setups.append(Setup(signal=sig, engine="momo", session="NY", atr=0.0))
                continue

            # Compute indicators
            vwap = calculate_vwap(df)
            ema9 = calculate_ema(df, 9)
            ema21 = calculate_ema(df, 21)
            price = float(df["close"].iloc[-1])
            vwap_last = float(vwap.iloc[-1]) if not vwap.empty else 0.0
            hod = float(df["high"].max())

            # Check: price should not be too far below VWAP (basic bullish filter)
            if vwap_last > 0 and price < vwap_last * _VWAP_MIN_DISTANCE:
                logger.debug(
                    "[momo] %s: price %.2f too far below VWAP %.2f; skipping.",
                    candidate.ticker, price, vwap_last,
                )
                continue

            # Check: green volume > red volume (buying pressure confirmed)
            recent_df = df.iloc[-10:]
            green_vol = float(recent_df[recent_df["close"] >= recent_df["open"]]["volume"].sum())
            red_vol = float(recent_df[recent_df["close"] < recent_df["open"]]["volume"].sum())
            if green_vol < red_vol:
                logger.debug("[momo] %s: red volume dominates; skipping.", candidate.ticker)
                continue

            sig = None

            # --- Entry type 1: Pullback to VWAP/EMA9 (preferred, safest) ---
            vwap_sig = detect_vwap_bounce(df, vwap, tolerance_pct=_VWAP_BOUNCE_TOLERANCE)
            if vwap_sig:
                vwap_sig.ticker = candidate.ticker
                vwap_sig.confidence = candidate.score
                # SL just below VWAP, TP at prior HOD
                vwap_sig.stop_price = (
                    round(vwap_last * _VWAP_SL_BUFFER, 2) if vwap_last > 0
                    else round(price * _FALLBACK_SL_PCT, 2)
                )
                vwap_sig.target_price = self._calculate_target_price(price, hod)
                sig = vwap_sig

            # --- Entry type 2: EMA 9 Pullback / Dip Buy ---
            if sig is None:
                ema_sig = detect_ema_pullback(df, ema9, ema21)
                if ema_sig:
                    ema_sig.ticker = candidate.ticker
                    ema_sig.confidence = candidate.score
                    # TP at prior HOD (partial recovery target)
                    ema_sig.target_price = self._calculate_target_price(price, hod)
                    sig = ema_sig

            # --- Entry type 3: Consolidation Breakout ---
            if sig is None and len(df) >= 20:
                # First 15 bars define the consolidation range
                range_high = float(df["high"].iloc[:15].max())
                range_low = float(df["low"].iloc[:15].min())
                vol_avg = float(df["volume"].mean())
                breakout_sig = detect_breakout(df, range_high, range_low, vol_avg)
                if breakout_sig:
                    breakout_sig.ticker = candidate.ticker
                    breakout_sig.confidence = candidate.score
                    # TP = range extension above range_high
                    breakout_sig.target_price = self._calculate_target_price(price, hod)
                    sig = breakout_sig

            if sig is None:
                logger.debug("[momo] %s: no entry pattern detected.", candidate.ticker)
                continue

            setups.append(Setup(signal=sig, engine="momo", session="NY", atr=0.0))

        return setups

    @staticmethod
    def _calculate_target_price(price: float, hod: float) -> float:
        """Return TP: prior HOD when meaningful, else +10 % above entry."""
        if hod > price * _HOD_MIN_BUFFER:
            return round(hod, 2)
        return round(price * _FALLBACK_TP_PCT, 2)

    async def execute_trade(
        self,
        setup: Setup,
        size_multiplier: float,
        ai_score: int,
    ) -> TradeResult | None:
        signal = setup.signal

        # Apply score-based size multiplier (overrides brain's multiplier for momo)
        score_mult = self._size_multiplier_from_score(signal.confidence)
        effective_mult = score_mult if score_mult > 0.0 else size_multiplier

        max_dollars = self._reto.get_position_size("momo") * effective_mult
        qty = max(1, int(max_dollars / signal.entry_price))

        logger.info(
            "[momo] Entering trade: %s %d shares @ %.2f (score=%d)",
            signal.ticker,
            qty,
            signal.entry_price,
            ai_score,
        )

        try:
            from ib_insync import Stock  # type: ignore

            contract = Stock(signal.ticker, "SMART", "USD")
            ib = self._connection.cash.get_ib()
            if ib is None:
                return None

            bracket = ib.bracketOrder(
                "BUY",
                qty,
                signal.entry_price,
                signal.target_price,
                signal.stop_price,
            )
            entry_order, tp_order, sl_order = bracket
            for order in bracket:
                order.tif = "DAY"

            entry_trade = await self._connection.cash.place_order(contract, entry_order)
            await self._connection.cash.place_order(contract, tp_order)
            await self._connection.cash.place_order(contract, sl_order)
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
        self._highest_price[signal.ticker] = signal.entry_price

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
                        "rr": round(
                            abs(signal.target_price - signal.entry_price)
                            / abs(signal.entry_price - signal.stop_price),
                            2,
                        ) if abs(signal.entry_price - signal.stop_price) > 0 else 0,
                    }
                )
            )
        return None

    async def monitor_position(self, position: Position) -> None:
        """Manage MoMo position: sell 50 % at T1, move SL to BE, trail remainder."""
        ib = self._connection.cash.get_ib()
        if ib is None:
            return
        try:
            open_trades = ib.openTrades()
            symbol_trades = [t for t in open_trades if t.contract.symbol == position.ticker]
            if not symbol_trades:
                # All orders closed — position is fully exited
                self._open_positions = [p for p in self._open_positions if p.ticker != position.ticker]
                self._risk.close_position("momo", position.ticker)
                self._partial_exits.discard(position.ticker)
                self._highest_price.pop(position.ticker, None)
                return

            # Fetch current price for position management decisions
            current_price = await self._get_current_price(position.ticker)
            if current_price <= 0:
                return

            # Update highest price seen (used for trailing stop calculation)
            prev_high = self._highest_price.get(position.ticker, position.entry_price)
            if current_price > prev_high:
                self._highest_price[position.ticker] = current_price

            # --- Partial exit: sell 50 % when Target 1 (prior HOD) is reached ---
            if (
                position.ticker not in self._partial_exits
                and current_price >= position.target_price
            ):
                partial_qty = max(1, int(position.quantity * 0.5))
                try:
                    from ib_insync import Stock, MarketOrder  # type: ignore

                    contract = Stock(position.ticker, "SMART", "USD")
                    sell_order = MarketOrder("SELL", partial_qty)
                    await self._connection.cash.place_order(contract, sell_order)
                    self._partial_exits.add(position.ticker)
                    # Move stop to breakeven on remaining shares
                    position.stop_price = position.entry_price
                    logger.info(
                        "[momo] Partial exit: sold %d/%d shares of %s at T1 (%.2f). SL → breakeven.",
                        partial_qty, position.quantity, position.ticker, current_price,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("[momo] Partial exit error for %s: %s", position.ticker, exc)

            # --- Trailing stop on remainder after partial exit ---
            elif position.ticker in self._partial_exits:
                trail_stop = round(self._highest_price[position.ticker] * (1 - _TRAIL_PCT), 2)
                if trail_stop > position.stop_price:
                    position.stop_price = trail_stop
                    logger.debug(
                        "[momo] Trailing stop updated to %.2f for %s.",
                        trail_stop, position.ticker,
                    )

                # If trailing stop is hit, exit remaining position
                if current_price <= position.stop_price:
                    partial_qty = max(1, int(position.quantity * 0.5))
                    remaining_qty = max(1, position.quantity - partial_qty)
                    try:
                        from ib_insync import Stock, MarketOrder  # type: ignore

                        contract = Stock(position.ticker, "SMART", "USD")
                        sell_order = MarketOrder("SELL", remaining_qty)
                        await self._connection.cash.place_order(contract, sell_order)
                        self._open_positions = [
                            p for p in self._open_positions if p.ticker != position.ticker
                        ]
                        self._risk.close_position("momo", position.ticker)
                        self._partial_exits.discard(position.ticker)
                        self._highest_price.pop(position.ticker, None)
                        logger.info(
                            "[momo] Trailing stop hit for %s at %.2f; exited remaining %d shares.",
                            position.ticker, current_price, remaining_qty,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error("[momo] Trailing stop exit error for %s: %s", position.ticker, exc)

        except Exception as exc:  # noqa: BLE001
            logger.error("[momo] Monitor error: %s", exc)
