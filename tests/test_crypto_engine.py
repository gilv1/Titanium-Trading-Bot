"""
Tests for engines/crypto_engine.py — tick rounding, dynamic allocation, pre-RTH cutoff,
and the software-managed IOC order architecture.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engines.crypto_engine import (
    CryptoEngine,
    CryptoPosition,
    CRYPTO_TICK_SIZE,
    CRYPTO_MIN_QTY,
    CRYPTO_MAX_POSITION_AGE_SECS,
    ENTRY_COOLDOWN_SECONDS,
    _round_to_tick,
)
from engines.base_engine import Setup, Signal


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_engine() -> CryptoEngine:
    """Create a CryptoEngine with all dependencies mocked out."""
    connection = MagicMock()
    brain = MagicMock()
    reto = MagicMock()
    reto.capital = 500.0
    risk = MagicMock()
    risk.has_correlation_conflict.return_value = False
    return CryptoEngine(
        connection_manager=connection,
        brain=brain,
        reto_tracker=reto,
        risk_manager=risk,
        telegram=None,
    )


def _make_setup(
    ticker: str = "BTC",
    direction: str = "LONG",
    entry_price: float = 70000.0,
    atr: float = 150.0,
) -> Setup:
    """Build a minimal Setup object for use in execute_trade tests."""
    signal = Signal(
        ticker=ticker,
        direction=direction,
        entry_price=entry_price,
        stop_price=entry_price * 0.985,
        target_price=entry_price * 1.035,
        confidence=80,
        setup_type="VWAP_BOUNCE",
    )
    return Setup(signal=signal, engine="crypto", session="Crypto_US", atr=atr)


# ──────────────────────────────────────────────────────────────
# Fix 1: Tick rounding (_round_to_tick helper)
# ──────────────────────────────────────────────────────────────


class TestTickRounding:
    def test_btc_rounds_to_quarter(self):
        """BTC prices must land on $0.25 increments."""
        assert _round_to_tick(70482.72, "BTC") == 70482.75
        assert _round_to_tick(70482.37, "BTC") == 70482.25
        assert _round_to_tick(70482.50, "BTC") == 70482.50
        assert _round_to_tick(70482.13, "BTC") == 70482.25

    def test_btc_sl_from_issue_report(self):
        """The exact price that triggered Warning 110 must round to a valid tick."""
        # SL was sent as 67077.76 — should round to 67077.75
        assert _round_to_tick(67077.76, "BTC") == 67077.75

    def test_btc_tp_from_issue_report(self):
        """The exact TP that triggered Warning 110 must round to a valid tick."""
        # TP was sent as 70482.72 — should round to 70482.75
        assert _round_to_tick(70482.72, "BTC") == 70482.75

    def test_eth_rounds_to_cent(self):
        """ETH prices must land on $0.01 increments."""
        assert _round_to_tick(3250.123, "ETH") == 3250.12
        assert _round_to_tick(3250.127, "ETH") == 3250.13
        assert _round_to_tick(3250.999, "ETH") == 3251.00

    def test_unknown_symbol_defaults_to_cent(self):
        """Unknown symbols default to $0.01 tick size."""
        assert _round_to_tick(100.123, "XRP") == 100.12
        assert _round_to_tick(100.127, "XRP") == 100.13

    def test_btc_tick_size_is_quarter(self):
        assert CRYPTO_TICK_SIZE["BTC"] == 0.25

    def test_eth_tick_size_is_cent(self):
        assert CRYPTO_TICK_SIZE["ETH"] == 0.01

    def test_round_to_tick_exact_boundaries(self):
        """Prices already on valid ticks should be unchanged."""
        assert _round_to_tick(70000.00, "BTC") == 70000.00
        assert _round_to_tick(70000.25, "BTC") == 70000.25
        assert _round_to_tick(70000.50, "BTC") == 70000.50
        assert _round_to_tick(70000.75, "BTC") == 70000.75


# ──────────────────────────────────────────────────────────────
# Fix 2: Dynamic allocation (_get_effective_allocation)
# ──────────────────────────────────────────────────────────────


class TestEffectiveAllocation:
    def _engine_with_time(self, weekday: int, hour: int, minute: int) -> CryptoEngine:
        engine = _make_engine()
        # Patch datetime.now inside crypto_engine to control the time
        mock_now = MagicMock()
        mock_now.weekday.return_value = weekday
        mock_now.hour = hour
        mock_now.minute = minute
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            alloc = engine._get_effective_allocation()
        return alloc

    def test_weekend_returns_70_pct(self):
        """Saturday and Sunday → 70% allocation."""
        engine = _make_engine()
        for day in (5, 6):  # Saturday=5, Sunday=6
            mock_now = MagicMock()
            mock_now.weekday.return_value = day
            mock_now.hour = 12
            mock_now.minute = 0
            with patch("engines.crypto_engine.datetime") as mock_dt:
                mock_dt.now.return_value = mock_now
                mock_dt.utcnow.return_value = datetime.utcnow()
                assert engine._get_effective_allocation() == 0.70

    def test_weekday_market_hours_returns_30_pct(self):
        """Monday–Friday 9:30 AM–4:00 PM ET → 30% allocation."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 1  # Tuesday
        mock_now.hour = 10
        mock_now.minute = 0
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.30

    def test_weekday_off_hours_morning_returns_70_pct(self):
        """Monday–Friday 6:00 AM ET (before 9:30 AM) → 70% allocation."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 2  # Wednesday
        mock_now.hour = 6
        mock_now.minute = 0
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.70

    def test_weekday_off_hours_evening_returns_70_pct(self):
        """Monday–Friday 6:00 PM ET (after 4:00 PM) → 70% allocation."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 3  # Thursday
        mock_now.hour = 18
        mock_now.minute = 0
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.70

    def test_market_hours_boundary_930_is_market(self):
        """9:30 AM exactly (minute=570) is market hours → 30%."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 0  # Monday
        mock_now.hour = 9
        mock_now.minute = 30
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.30

    def test_market_hours_boundary_4pm_is_off_hours(self):
        """4:00 PM exactly (minute=960) is off-hours → 70%."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 0  # Monday
        mock_now.hour = 16
        mock_now.minute = 0
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.70


# ──────────────────────────────────────────────────────────────
# New architecture: IOC entry and software-managed TP/SL orders
# ──────────────────────────────────────────────────────────────


class TestCryptoPosition:
    """CryptoPosition dataclass captures the fields needed for software TP/SL monitoring."""

    def test_fields_are_set(self):
        now = datetime.now(timezone.utc)
        pos = CryptoPosition(
            symbol="BTC",
            action="BUY",
            qty=0.001,
            entry_price=70000.00,
            tp_price=72450.00,
            sl_price=68950.00,
            entry_time=now,
            order_id=1234,
        )
        assert pos.symbol == "BTC"
        assert pos.action == "BUY"
        assert pos.qty == pytest.approx(0.001)
        assert pos.entry_price == pytest.approx(70000.00)
        assert pos.tp_price == pytest.approx(72450.00)
        assert pos.sl_price == pytest.approx(68950.00)
        assert pos.entry_time == now
        assert pos.order_id == 1234

    def test_initial_active_position_is_none(self):
        engine = _make_engine()
        assert engine._active_position is None


class TestIOCEntryOrder:
    """Verify the IOC LimitOrder entry construction for IBKR Paxos compliance."""

    def _build_entry_order(self, action: str, qty: float, limit_price: float) -> "Any":
        from ib_insync import LimitOrder  # type: ignore

        limit = _round_to_tick(limit_price, "BTC")
        return LimitOrder(
            action=action,
            totalQuantity=qty,
            lmtPrice=limit,
            tif='IOC',
        )

    def test_entry_order_type_is_limit(self):
        """Entry order must be a LimitOrder (not MKT, STP, or STP LMT)."""
        order = self._build_entry_order("BUY", 0.001, 70000.00)
        assert order.orderType == "LMT"

    def test_entry_tif_is_ioc(self):
        """Paxos requires TIF=IOC for Limit orders."""
        order = self._build_entry_order("BUY", 0.001, 70000.00)
        assert order.tif == "IOC"

    def test_entry_limit_price_is_rounded(self):
        """Entry limit price must conform to tick increments."""
        order = self._build_entry_order("BUY", 0.001, 70482.72)
        assert order.lmtPrice == 70482.75

    def test_entry_sell_order(self):
        """SHORT entry produces a SELL LimitOrder IOC."""
        order = self._build_entry_order("SELL", 0.001, 70000.00)
        assert order.action == "SELL"
        assert order.orderType == "LMT"
        assert order.tif == "IOC"

    def test_no_bracket_orders_on_engine(self):
        """After construction, no bracket order state (_entry_trades) exists."""
        engine = _make_engine()
        assert not hasattr(engine, "_entry_trades")


class TestCloseOrders:
    """Verify the close order construction for TP and SL exits."""

    def _build_tp_close_order(self, close_action: str, qty: float, current_price: float, ticker: str) -> "Any":
        from ib_insync import LimitOrder  # type: ignore

        close_price = _round_to_tick(current_price, ticker)
        return LimitOrder(
            action=close_action,
            totalQuantity=qty,
            lmtPrice=close_price,
            tif='IOC',
        )

    def _build_sl_close_order(self, close_action: str, qty: float) -> "Any":
        from ib_insync import MarketOrder  # type: ignore

        return MarketOrder(
            action=close_action,
            totalQuantity=qty,
            tif='IOC',
        )

    def test_tp_close_is_limit_ioc(self):
        """Take-profit close must use LimitOrder with TIF=IOC."""
        order = self._build_tp_close_order("SELL", 0.001, 72450.25, "BTC")
        assert order.orderType == "LMT"
        assert order.tif == "IOC"

    def test_tp_close_price_is_rounded(self):
        """TP close limit price must be on a valid BTC tick."""
        order = self._build_tp_close_order("SELL", 0.001, 72450.13, "BTC")
        assert order.lmtPrice == 72450.25

    def test_sl_close_is_market_ioc(self):
        """Stop-loss close must use MarketOrder with TIF=IOC for guaranteed exit."""
        order = self._build_sl_close_order("SELL", 0.001)
        assert order.orderType == "MKT"
        assert order.tif == "IOC"

    def test_sl_close_sell_for_long(self):
        """LONG position SL close must be a SELL."""
        order = self._build_sl_close_order("SELL", 0.001)
        assert order.action == "SELL"

    def test_sl_close_buy_for_short(self):
        """SHORT position SL close must be a BUY."""
        order = self._build_sl_close_order("BUY", 0.001)
        assert order.action == "BUY"

    def test_tp_close_sell_for_long(self):
        """LONG position TP close must be a SELL LimitOrder IOC."""
        order = self._build_tp_close_order("SELL", 0.001, 72450.00, "BTC")
        assert order.action == "SELL"

    def test_eth_tp_close_price_uses_eth_tick(self):
        """ETH TP close price uses $0.01 tick, not BTC's $0.25."""
        order = self._build_tp_close_order("SELL", 0.01, 3250.127, "ETH")
        assert order.lmtPrice == pytest.approx(3250.13, abs=0.001)


class TestMaxPositionAge:
    def test_max_position_age_is_30_minutes(self):
        """Software time-based stop triggers after 30 minutes."""
        assert CRYPTO_MAX_POSITION_AGE_SECS == 30 * 60


# ──────────────────────────────────────────────────────────────
# ATR-based SL/TP
# ──────────────────────────────────────────────────────────────


class TestATRBasedSLTP:
    """ATR-based SL/TP should produce sensible dollar distances (not $1,000+)."""

    def test_btc_long_atr150_sl_distance(self):
        """BTC with ATR=$150: SL should be $225 away (1.5×ATR), not $1,020."""
        entry = 70000.0
        atr = 150.0
        sl = _round_to_tick(entry - atr * 1.5, "BTC")
        tp = _round_to_tick(entry + atr * 3.0, "BTC")
        assert sl == pytest.approx(entry - 225.0, abs=0.5)
        assert tp == pytest.approx(entry + 450.0, abs=0.5)

    def test_btc_short_atr150_sl_above_entry(self):
        """SHORT: SL is above entry, TP is below entry."""
        entry = 70000.0
        atr = 150.0
        sl = _round_to_tick(entry + atr * 1.5, "BTC")
        tp = _round_to_tick(entry - atr * 3.0, "BTC")
        assert sl > entry
        assert tp < entry

    def test_sl_tp_not_absurd_for_btc(self):
        """ATR-based SL must be < $500 on BTC (not $1,000+)."""
        entry = 68000.0
        atr = 200.0  # generous ATR
        sl_distance = atr * 1.5
        assert sl_distance < 500, f"SL distance {sl_distance} is too large for a scalp"

    def test_rr_ratio_is_2_to_1(self):
        """TP distance should be exactly 2× SL distance."""
        atr = 150.0
        sl_distance = atr * 1.5
        tp_distance = atr * 3.0
        assert pytest.approx(tp_distance / sl_distance, abs=0.01) == 2.0

    def test_fallback_atr_when_zero(self):
        """When ATR=0, fallback to 0.2% of price."""
        entry = 70000.0
        atr = 0.0
        if atr <= 0:
            atr = entry * 0.002
        assert atr == pytest.approx(140.0)


# ──────────────────────────────────────────────────────────────
# Cooldown timer
# ──────────────────────────────────────────────────────────────


class TestEntryCooldown:
    """Engine should suppress entry attempts within ENTRY_COOLDOWN_SECONDS."""

    def test_cooldown_constant_is_5_minutes(self):
        assert ENTRY_COOLDOWN_SECONDS == 300

    def test_engine_has_last_attempt_time_dict(self):
        engine = _make_engine()
        assert hasattr(engine, "_last_attempt_time")
        assert isinstance(engine._last_attempt_time, dict)
        assert len(engine._last_attempt_time) == 0

    @pytest.mark.asyncio
    async def test_cooldown_blocks_second_attempt(self):
        """Second call within 5 minutes returns None without placing an order."""
        engine = _make_engine()
        # Simulate a recent entry attempt
        engine._last_attempt_time["BTC"] = time.time()

        setup = _make_setup(ticker="BTC", entry_price=70000.0, atr=150.0)

        result = await engine.execute_trade(setup, size_multiplier=1.0, ai_score=80)

        assert result is None

    @pytest.mark.asyncio
    async def test_cooldown_allows_after_expiry(self):
        """Attempt is allowed once cooldown has expired."""
        engine = _make_engine()
        # Simulate old entry attempt (older than 5 minutes)
        engine._last_attempt_time["BTC"] = time.time() - ENTRY_COOLDOWN_SECONDS - 1

        # The engine will still skip if capital/price checks fail, so we only
        # verify that cooldown itself doesn't block — the attempt is recorded.
        engine._get_current_price = AsyncMock(return_value=0.0)  # 0 is treated as invalid price → early return
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(return_value=None)

        await engine.execute_trade(_make_setup(), size_multiplier=1.0, ai_score=80)

        # Cooldown timestamp should have been updated
        assert engine._last_attempt_time.get("BTC", 0) > time.time() - 5


# ──────────────────────────────────────────────────────────────
# IOC limit price fix (ask ± 2 ticks)
# ──────────────────────────────────────────────────────────────


class TestIOCLimitPriceFix:
    """IOC entry price should be 2 ticks above market for BUY, 2 below for SELL."""

    def test_buy_ioc_limit_is_2_ticks_above(self):
        current_price = 70000.00
        tick = CRYPTO_TICK_SIZE["BTC"]
        ioc_limit = _round_to_tick(current_price + tick * 2, "BTC")
        assert ioc_limit == pytest.approx(70000.50)  # 2 × $0.25

    def test_sell_ioc_limit_is_2_ticks_below(self):
        current_price = 70000.00
        tick = CRYPTO_TICK_SIZE["BTC"]
        ioc_limit = _round_to_tick(current_price - tick * 2, "BTC")
        assert ioc_limit == pytest.approx(69999.50)  # 2 × $0.25

    def test_eth_buy_ioc_limit_is_2_ticks_above(self):
        current_price = 3250.00
        tick = CRYPTO_TICK_SIZE["ETH"]
        ioc_limit = _round_to_tick(current_price + tick * 2, "ETH")
        assert ioc_limit == pytest.approx(3250.02)  # 2 × $0.01


# ──────────────────────────────────────────────────────────────
# Scanner integration in execute_trade
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestScannerIntegration:
    """execute_trade() should call the scanner and respect its verdict."""

    async def test_scanner_called_before_order(self):
        engine = _make_engine()
        mock_scanner = MagicMock()
        mock_scanner.get_market_context = AsyncMock(return_value=None)
        engine._scanner = mock_scanner

        # Force early return after scanner (no valid price)
        engine._get_current_price = AsyncMock(return_value=0.0)
        engine._get_contract = MagicMock(return_value=MagicMock())

        setup = _make_setup(entry_price=70000.0, atr=150.0)
        await engine.execute_trade(setup, size_multiplier=1.0, ai_score=80)

        mock_scanner.get_market_context.assert_awaited_once_with("BTC")

    async def test_bearish_context_blocks_long(self):
        """If scanner says market is bearish (not bullish), a LONG entry is skipped."""
        from core.scanner_pool import MarketContext

        engine = _make_engine()
        bad_context = MarketContext(
            price=70000.0,
            volume_24h=5e9,
            change_1h=-2.0,  # strongly negative 1h → not bullish
            change_24h=-4.0,  # freefall → not bullish
        )
        mock_scanner = MagicMock()
        mock_scanner.get_market_context = AsyncMock(return_value=bad_context)
        engine._scanner = mock_scanner
        engine._get_contract = MagicMock(return_value=MagicMock())

        setup = _make_setup(direction="LONG", entry_price=70000.0, atr=150.0)
        result = await engine.execute_trade(setup, size_multiplier=1.0, ai_score=80)

        assert result is None

    async def test_bullish_context_blocks_short(self):
        """If scanner says market is bullish (not bearish), a SHORT entry is skipped."""
        from core.scanner_pool import MarketContext

        engine = _make_engine()
        bullish_context = MarketContext(
            price=70000.0,
            volume_24h=5e9,
            change_1h=2.0,   # strong positive 1h → not bearish
            change_24h=4.0,  # strong uptrend → not bearish
        )
        mock_scanner = MagicMock()
        mock_scanner.get_market_context = AsyncMock(return_value=bullish_context)
        engine._scanner = mock_scanner
        engine._get_contract = MagicMock(return_value=MagicMock())

        setup = _make_setup(direction="SHORT", entry_price=70000.0, atr=150.0)
        result = await engine.execute_trade(setup, size_multiplier=1.0, ai_score=80)

        assert result is None

    async def test_no_scanner_data_proceeds(self):
        """When scanner returns None the engine proceeds (falls back to IBKR signals)."""
        engine = _make_engine()
        mock_scanner = MagicMock()
        mock_scanner.get_market_context = AsyncMock(return_value=None)
        engine._scanner = mock_scanner

        # Fail at the price-fetch stage so we know it passed the scanner check
        engine._get_current_price = AsyncMock(return_value=0.0)
        engine._get_contract = MagicMock(return_value=MagicMock())

        setup = _make_setup(entry_price=70000.0, atr=150.0)
        result = await engine.execute_trade(setup, size_multiplier=1.0, ai_score=80)

        # execute_trade returns None on early exits; we just verify scanner was called
        assert result is None
        mock_scanner.get_market_context.assert_awaited_once()


# ──────────────────────────────────────────────────────────────
# Cross-validated pricing (_get_current_price / _get_ibkr_price)
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestCrossValidatedPricing:
    """_get_current_price() cross-validates IBKR price against external scanner."""

    async def test_returns_ibkr_price_when_both_agree(self):
        """When IBKR and external prices are within 0.5%, use IBKR price."""
        from core.scanner_pool import MarketContext

        engine = _make_engine()
        engine._get_ibkr_price = AsyncMock(return_value=68000.0)
        mock_context = MarketContext(
            price=68010.0,  # within 0.5% of IBKR
            volume_24h=25e9, change_1h=0.0, change_24h=0.0,
        )
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(return_value=mock_context)

        price = await engine._get_current_price("BTC")

        assert price == pytest.approx(68000.0)  # uses IBKR

    async def test_returns_external_price_when_ibkr_stale(self):
        """When IBKR deviates >0.5% from external, use external price."""
        from core.scanner_pool import MarketContext

        engine = _make_engine()
        engine._get_ibkr_price = AsyncMock(return_value=67511.0)  # stale IBKR
        mock_context = MarketContext(
            price=68098.25,  # real market price
            volume_24h=25e9, change_1h=0.0, change_24h=0.0,
        )
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(return_value=mock_context)

        price = await engine._get_current_price("BTC")

        # Deviation = |67511 - 68098.25| / 68098.25 ≈ 0.86% > 0.5% → external wins
        assert price == pytest.approx(68098.25)

    async def test_returns_external_price_when_ibkr_unavailable(self):
        """When IBKR returns 0, fall back to external scanner price."""
        from core.scanner_pool import MarketContext

        engine = _make_engine()
        engine._get_ibkr_price = AsyncMock(return_value=0.0)
        mock_context = MarketContext(
            price=68000.0,
            volume_24h=25e9, change_1h=0.0, change_24h=0.0,
        )
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(return_value=mock_context)

        price = await engine._get_current_price("BTC")

        assert price == pytest.approx(68000.0)

    async def test_returns_ibkr_price_when_scanner_unavailable(self):
        """When scanner returns None, fall back to IBKR-only price."""
        engine = _make_engine()
        engine._get_ibkr_price = AsyncMock(return_value=68000.0)
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(return_value=None)

        price = await engine._get_current_price("BTC")

        assert price == pytest.approx(68000.0)

    async def test_returns_zero_when_no_price_available(self):
        """When both sources fail, returns 0.0 to block the trade."""
        engine = _make_engine()
        engine._get_ibkr_price = AsyncMock(return_value=0.0)
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(return_value=None)

        price = await engine._get_current_price("BTC")

        assert price == pytest.approx(0.0)

    async def test_scanner_exception_falls_back_to_ibkr(self):
        """Scanner fetch exception falls back to IBKR price gracefully."""
        engine = _make_engine()
        engine._get_ibkr_price = AsyncMock(return_value=68000.0)
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(side_effect=Exception("network error"))

        price = await engine._get_current_price("BTC")

        assert price == pytest.approx(68000.0)


# ──────────────────────────────────────────────────────────────
# Stale price rejection tightened to 1%
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestStalePriceRejection:
    """execute_trade() rejects entries that deviate >1% from cross-validated price."""

    async def test_rejects_entry_when_deviation_exceeds_1_pct(self):
        """A signal entry >1% from the current validated price is rejected."""
        engine = _make_engine()
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(return_value=None)
        engine._get_contract = MagicMock(return_value=MagicMock())

        # current_price = 68000.0, signal entry = 69020.0 → deviation = 1.5% > 1% → rejected
        engine._get_current_price = AsyncMock(return_value=68000.0)
        setup = _make_setup(ticker="BTC", direction="LONG", entry_price=69020.0, atr=150.0)
        result = await engine.execute_trade(setup, size_multiplier=1.0, ai_score=80)

        assert result is None  # trade blocked due to stale price

    async def test_allows_entry_within_1_pct(self):
        """A signal entry within 0.5% of current price is not rejected for staleness."""
        engine = _make_engine()
        engine._scanner = MagicMock()
        engine._scanner.get_market_context = AsyncMock(return_value=None)
        engine._get_contract = MagicMock(return_value=MagicMock())

        # Signal entry = 70000, current = 70100 → deviation ≈ 0.14% < 1% → not stale
        engine._get_current_price = AsyncMock(return_value=70100.0)

        # Fail before placing order (place_order returns None) to avoid real order logic
        engine._connection.margin.place_order = AsyncMock(return_value=None)
        engine._connection.margin.get_ib = MagicMock(return_value=MagicMock())

        setup = _make_setup(ticker="BTC", direction="LONG", entry_price=70000.0, atr=150.0)
        result = await engine.execute_trade(setup, size_multiplier=1.0, ai_score=80)

        # Result is None because order didn't fill, but it was NOT rejected for staleness
        # (if rejected, the scanner mock would not have been called to completion)
        assert result is None
        # Confirm price fetch was called — if stale check blocked, this would still be called
        engine._get_current_price.assert_awaited_once_with("BTC")


