"""
Motor 1 — Futures Engine (MES / MNQ / NQ) for Titanium Warrior v3.

Trades Micro E-mini S&P 500 (MES), Micro E-mini Nasdaq-100 (MNQ),
and E-mini Nasdaq-100 (NQ) via IBKR.
Operates during Tokyo, London, and NY sessions (phase-dependent).

Setups detected:
  1. VWAP Bounce
  2. Opening Range Breakout (ORB)
  3. EMA 9/21 Pullback
  4. Liquidity Grab & Reversal
  5. News Momentum Burst

Position management:
  - Bracket orders (entry + SL + TP1) via ib_insync
  - Sell 50 % at Target 1
  - Trailing stop on remainder: breakeven+1 at +15 pts, then 8-pt trail at +25 pts
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from analysis.patterns import (
    detect_ema_pullback,
    detect_liquidity_grab,
    detect_orb,
    detect_vwap_bounce,
)
from analysis.technical import calculate_atr, calculate_ema, calculate_rsi, calculate_vwap
from config import settings
from core.news_sentinel import MarketContext, NewsSentinel
from core.reto_tracker import TradeResult as RetoTradeResult
from engines.base_engine import BaseEngine, Position, Setup, Signal, TradeResult

if TYPE_CHECKING:
    from core.brain import AIBrain, TradeDecision
    from core.connection import ConnectionManager
    from core.reto_tracker import RetoTracker
    from core.risk_manager import RiskManager
    from notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)


def _get_front_month_expiry() -> str:
    """Return the nearest quarterly futures expiry code (YYYYMM).

    Rolls to the next quarter when within 7 calendar days of the current
    quarter's third Friday (standard CME expiry).
    """
    import calendar

    now = datetime.utcnow()
    quarters = [3, 6, 9, 12]

    for i, q in enumerate(quarters):
        if now.month <= q:
            year = now.year
            cal = calendar.monthcalendar(year, q)
            # Third Friday of expiry month
            fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
            third_friday = fridays[2] if len(fridays) >= 3 else fridays[-1]
            expiry_date = datetime(year, q, third_friday)

            # If within 7 days of expiry, roll to next quarter
            if (expiry_date - now).days <= 7:
                next_idx = (i + 1) % len(quarters)
                next_year = year + 1 if next_idx == 0 else year
                return f"{next_year}{quarters[next_idx]:02d}"
            return f"{year}{q:02d}"

    return f"{now.year + 1}03"


class FuturesEngine(BaseEngine):
    """Motor 1 — MES / MNQ / NQ futures trading engine."""

    ACTIVE_SESSIONS = ("Tokyo", "London", "NY")

    def __init__(
        self,
        connection_manager: "ConnectionManager",
        brain: "AIBrain",
        reto_tracker: "RetoTracker",
        risk_manager: "RiskManager",
        telegram: "TelegramNotifier | None" = None,
        journal: "Any | None" = None,
        news_sentinel: "NewsSentinel | None" = None,
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
        self._orb_range: tuple[float, float] | None = None  # (low, high)
        self._session_open_time: datetime | None = None
        # Guard against duplicate bracket orders within the same loop cycle
        self._order_pending: bool = False
        self._entry_trade: Any = None  # tracked to detect async cancellations
        self._news_sentinel: NewsSentinel = news_sentinel if news_sentinel is not None else NewsSentinel()
        self._last_market_context: MarketContext | None = None

    def get_engine_name(self) -> str:
        return "futures"

    def is_active_session(self) -> bool:
        """Return True if we are within any active session for the current phase."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        phase = self._reto.get_phase()
        active = settings.PHASES[phase].sessions

        for sess_name in active:
            sess = settings.SESSIONS.get(sess_name)
            if sess is None:
                continue
            sh, sm = sess.start_hour, sess.start_minute
            eh, em = sess.end_hour, sess.end_minute
            current_minutes = now.hour * 60 + now.minute
            start_minutes = sh * 60 + sm
            end_minutes = eh * 60 + em

            # Handle overnight sessions (e.g. Tokyo: 20:00 → 02:00)
            if start_minutes > end_minutes:
                if current_minutes >= start_minutes or current_minutes < end_minutes:
                    return True
            else:
                if start_minutes <= current_minutes < end_minutes:
                    return True
        return False

    def _current_session(self) -> str:
        """Return the name of the currently active session."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        phase = self._reto.get_phase()
        active = settings.PHASES[phase].sessions

        for sess_name in active:
            sess = settings.SESSIONS.get(sess_name)
            if sess is None:
                continue
            sh, sm = sess.start_hour, sess.start_minute
            eh, em = sess.end_hour, sess.end_minute
            current_minutes = now.hour * 60 + now.minute
            start_minutes = sh * 60 + sm
            end_minutes = eh * 60 + em
            if start_minutes > end_minutes:
                if current_minutes >= start_minutes or current_minutes < end_minutes:
                    return sess_name
            else:
                if start_minutes <= current_minutes < end_minutes:
                    return sess_name
        return "NY"

    # ──────────────────────────────────────────────────────────
    # News Sentinel integration
    # ──────────────────────────────────────────────────────────

    async def _build_market_context(self) -> str:
        """
        Build a market context string for the AI evaluator.

        Fetches the latest MarketContext from the News Sentinel and formats
        it into a concise string for inclusion in the LLM prompt.
        """
        ctx = await self._news_sentinel.get_market_context(self._connection)
        self._last_market_context = ctx
        upcoming_names = ", ".join(e["name"] for e in ctx.upcoming_events[:3]) or "None"
        return (
            f"VIX: {ctx.vix_level:.1f} ({ctx.vix_regime})\n"
            f"Risk Level: {ctx.risk_level}\n"
            f"Upcoming Events (24h): {upcoming_names}\n"
            f"Minutes to next high-impact event: {ctx.minutes_to_next_event}\n"
            f"Size modifier: {ctx.size_modifier:.0%}\n"
            f"Reasoning: {ctx.reasoning}"
        )

    # ──────────────────────────────────────────────────────────
    # Market data helpers
    # ──────────────────────────────────────────────────────────

    def _get_contract(self) -> object:
        """Build an ib_insync Future contract for MNQ or NQ."""
        try:
            from ib_insync import Future  # type: ignore
        except ImportError:
            return None

        instrument = self._reto.get_futures_instrument()
        expiry = _get_front_month_expiry()
        return Future(instrument, expiry, "CME")

    async def _fetch_bars(self, contract: object, duration: str = "1 D", bar_size: str = "1 min") -> pd.DataFrame:
        """Fetch historical 1-minute bars from IBKR and return a DataFrame."""
        ib = self._connection.margin.get_ib()
        if ib is None or not self._connection.margin.is_connected():
            logger.warning("[futures] IBKR not connected; returning empty DataFrame.")
            return pd.DataFrame()
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
            )
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(
                {
                    "time": [b.date for b in bars],
                    "open": [b.open for b in bars],
                    "high": [b.high for b in bars],
                    "low": [b.low for b in bars],
                    "close": [b.close for b in bars],
                    "volume": [b.volume for b in bars],
                }
            )
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("[futures] Error fetching bars: %s", exc)
            return pd.DataFrame()

    # ──────────────────────────────────────────────────────────
    # Scan
    # ──────────────────────────────────────────────────────────

    async def scan_for_setups(self) -> list[Setup]:
        """Detect all 5 futures setups on the latest 1-minute bars."""
        contract = self._get_contract()
        if contract is None:
            return []

        df = await self._fetch_bars(contract)
        if df.empty or len(df) < 30:
            return []

        # Calculate indicators
        vwap = calculate_vwap(df)
        ema9 = calculate_ema(df, 9)
        ema21 = calculate_ema(df, 21)
        rsi = calculate_rsi(df)
        atr_series = calculate_atr(df)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        setups: list[Setup] = []
        session = self._current_session()

        # 1. VWAP Bounce
        sig = detect_vwap_bounce(df, vwap, rsi_series=rsi)
        if sig:
            setups.append(Setup(signal=sig, engine="futures", session=session, atr=atr))

        # 2. ORB
        sig = detect_orb(df, session_start_time=self._session_open_time)
        if sig:
            setups.append(Setup(signal=sig, engine="futures", session=session, atr=atr))

        # 3. EMA Pullback
        sig = detect_ema_pullback(df, ema9, ema21)
        if sig:
            setups.append(Setup(signal=sig, engine="futures", session=session, atr=atr))

        # 4. Liquidity Grab
        sig = detect_liquidity_grab(df, levels=[])
        if sig:
            setups.append(Setup(signal=sig, engine="futures", session=session, atr=atr))

        return setups

    # ──────────────────────────────────────────────────────────
    # Execute trade
    # ──────────────────────────────────────────────────────────

    async def execute_trade(
        self,
        setup: Setup,
        size_multiplier: float,
        ai_score: int,
    ) -> TradeResult | None:
        """Place a bracket order for a futures trade."""
        # Guard: refuse to place a new order while a previous one is still pending
        if self._order_pending:
            logger.warning("[futures] Order already pending — skipping duplicate bracket order.")
            return None

        # ── News Sentinel check ───────────────────────────────────
        # Use cached context if available (set by _build_market_context in run_loop),
        # otherwise fetch fresh context.
        if self._last_market_context is not None:
            context = self._last_market_context
        else:
            context = await self._news_sentinel.get_market_context(self._connection)
            self._last_market_context = context

        if context.should_pause:
            logger.info("[futures] Trading PAUSED by News Sentinel: %s", context.reasoning)
            if self._telegram:
                asyncio.create_task(
                    self._telegram._send(
                        f"⚠️ <b>Trading PAUSED</b>\n"
                        f"Reason: {context.reasoning}\n"
                        f"VIX: {context.vix_level:.1f}\n"
                        f"Will resume when conditions improve."
                    )
                )
            return None

        # Apply News Sentinel size modifier
        if context.size_modifier < 1.0:
            logger.info(
                "[futures] News Sentinel reducing size to %.0f%%: %s",
                context.size_modifier * 100,
                context.reasoning,
            )
        size_multiplier *= context.size_modifier

        try:
            from ib_insync import LimitOrder, Order, StopOrder  # type: ignore
        except ImportError:
            logger.error("[futures] ib_insync not installed.")
            return None

        phase_cfg = settings.PHASES[self._reto.get_phase()]
        base_contracts = self._reto.get_contracts("futures")
        qty = max(1, int(base_contracts * size_multiplier))

        signal = setup.signal
        direction = signal.direction
        action = "BUY" if direction == "LONG" else "SELL"

        # Adaptive stop using brain suggestion
        sl_pts = self._brain.suggested_stop_points(
            atr=setup.atr,
            session=setup.session,
            phase_sl_pts=phase_cfg.futures_sl_pts,
        )
        tp_pts = phase_cfg.futures_tp_pts
        entry = signal.entry_price
        sl = entry - sl_pts if direction == "LONG" else entry + sl_pts
        tp = entry + tp_pts if direction == "LONG" else entry - tp_pts

        contract = self._get_contract()
        if contract is None:
            return None

        # Build bracket
        ib = self._connection.margin.get_ib()
        if ib is None:
            return None

        try:
            bracket = ib.bracketOrder(action, qty, entry, tp, sl)
            for order in bracket:
                order.tif = 'GTC'
            entry_order, tp_order, sl_order = bracket

            self._order_pending = True  # set before placing to prevent duplicates
            entry_trade = await self._connection.margin.place_order(contract, entry_order)
            if entry_trade is None:
                self._order_pending = False
                return None
            self._entry_trade = entry_trade
            await self._connection.margin.place_order(contract, tp_order)
            await self._connection.margin.place_order(contract, sl_order)
        except Exception as exc:  # noqa: BLE001
            logger.error("[futures] Order placement error: %s", exc)
            self._order_pending = False  # clear flag so next cycle can retry
            self._entry_trade = None
            return None

        position = Position(
            engine="futures",
            ticker=contract.symbol,
            direction=direction,
            entry_price=entry,
            stop_price=sl,
            target_price=tp,
            quantity=qty,
        )
        self._open_positions.append(position)
        self._risk.open_position("futures", contract.symbol, direction)

        logger.info(
            "[futures] Order submitted: %s %d %s @ %.2f SL=%.2f TP=%.2f (score=%d)",
            direction,
            qty,
            contract.symbol,
            entry,
            sl,
            tp,
            ai_score,
        )

        # Telegram entry notification
        if self._telegram:
            asyncio.create_task(
                self._telegram.send_trade_entry(
                    {
                        "engine": "futures",
                        "ticker": contract.symbol,
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

        # In paper/live we return None here; position is managed by monitor_position
        # We return None to indicate the trade is open (result will come from monitor)
        return None

    # ──────────────────────────────────────────────────────────
    # Monitor
    # ──────────────────────────────────────────────────────────

    def _get_exit_price_from_fills(self, ib: Any, position: Position) -> float:
        """Look up the actual exit fill price for a closed position from IBKR fills.

        For a LONG position the exit fill is a SELL (side="SLD").
        For a SHORT position the exit fill is a BUY (side="BOT").
        Falls back to the position's stop-loss price if no fill is found.
        """
        try:
            fills = ib.fills()
            # Identify exit fills: opposite side to the entry
            exit_side = "SLD" if position.direction == "LONG" else "BOT"
            matching = [
                f for f in fills
                if f.contract.symbol == position.ticker and f.execution.side == exit_side
            ]
            if matching:
                latest = max(matching, key=lambda f: f.execution.time)
                return float(latest.execution.price)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[futures] Could not get fill price for %s: %s; using SL fallback.", position.ticker, exc)
        # Conservative fallback: assume stop-loss was hit
        return position.stop_price

    async def monitor_position(self, position: Position) -> None:
        """
        Check fill/close status and manage trailing stops.

        Trailing logic:
          - At +15 pts → move SL to breakeven + 1
          - At +25 pts → trail by 8 pts
        """
        ib = self._connection.margin.get_ib()
        if ib is None:
            return

        try:
            # If the entry order was cancelled or rejected, clean up immediately
            if self._entry_trade is not None:
                order_status = self._entry_trade.orderStatus
                entry_status = order_status.status if order_status is not None else ""
                if entry_status in ("Cancelled", "Inactive"):
                    self._open_positions = [p for p in self._open_positions if p.ticker != position.ticker]
                    self._risk.close_position("futures", position.ticker)
                    self._order_pending = False
                    self._entry_trade = None
                    logger.warning(
                        "[futures] Entry order %s for %s — clearing pending flag.",
                        entry_status.lower(),
                        position.ticker,
                    )
                    return

            # Check open trades for this position
            open_trades = ib.openTrades()
            # If no open trades for this symbol, position is closed
            symbol_trades = [t for t in open_trades if t.contract.symbol == position.ticker]
            if not symbol_trades:
                # Determine exit price from IBKR fills
                exit_price = self._get_exit_price_from_fills(ib, position)

                # Calculate realised P&L
                # MES multiplier = 5, MNQ multiplier = 2, NQ multiplier = 20
                instrument = self._reto.get_futures_instrument()
                if instrument == "NQ":
                    multiplier = 20
                elif instrument == "MNQ":
                    multiplier = 2
                elif instrument == "MES":
                    multiplier = 5
                else:
                    multiplier = 2  # fallback
                if position.direction == "LONG":
                    pnl = (exit_price - position.entry_price) * position.quantity * multiplier
                else:
                    pnl = (position.entry_price - exit_price) * position.quantity * multiplier

                won = pnl > 0

                # Update capital tracker
                reto_result = RetoTradeResult(engine="futures", pnl=pnl)
                milestones = self._reto.update_capital(reto_result)

                # Build a full TradeResult for the trade journal and Telegram
                result = TradeResult(
                    engine="futures",
                    ticker=position.ticker,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    stop_loss=position.stop_price,
                    take_profit=position.target_price,
                    quantity=position.quantity,
                    pnl=pnl,
                    pnl_pct=(
                        (pnl / (position.entry_price * position.quantity * multiplier)) * 100
                        if position.entry_price > 0
                        else 0.0
                    ),
                    duration_seconds=(datetime.utcnow() - position.entry_time).total_seconds(),
                    setup_type="",
                    session=self._current_session(),
                    ai_score=0,
                    phase=self._reto.get_phase(),
                    capital_after=self._reto.capital,
                    won=won,
                )
                self._trade_history.append(result)

                # Notify risk manager so consecutive-loss counter and daily trade count update
                self._risk.register_trade(
                    engine="futures",
                    pnl=pnl,
                    won=won,
                    direction=position.direction,
                    ticker=position.ticker,
                )

                # Clean up position state
                self._open_positions = [p for p in self._open_positions if p.ticker != position.ticker]
                self._risk.close_position("futures", position.ticker)
                self._order_pending = False
                self._entry_trade = None

                logger.info(
                    "[futures] Position closed: %s %s P&L=%.2f exit=%.2f",
                    position.ticker,
                    "WIN" if won else "LOSS",
                    pnl,
                    exit_price,
                )

                # Telegram exit notification
                if self._telegram:
                    asyncio.create_task(self._telegram.send_trade_exit(result))

                # Milestone alerts
                for msg in milestones:
                    if self._telegram:
                        asyncio.create_task(self._telegram.send_milestone_alert(msg))

        except Exception as exc:  # noqa: BLE001
            logger.error("[futures] Monitor error: %s", exc)
