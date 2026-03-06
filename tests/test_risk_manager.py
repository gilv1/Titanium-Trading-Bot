"""
Tests for core/risk_manager.py — daily limits, kill switch, PDT, correlation guard.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from core.risk_manager import RiskManager, _business_days_ago


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def risk() -> RiskManager:
    """Return a fresh RiskManager with no reto_tracker."""
    return RiskManager(reto_tracker=None)


@pytest.fixture
def risk_with_reto() -> RiskManager:
    """Return a RiskManager wired to a mock reto_tracker."""
    mock_reto = MagicMock()
    mock_reto.get_phase.return_value = 1
    mock_daily = MagicMock()
    mock_daily.pnl = 0.0
    mock_daily.pnl_pct = 0.0
    mock_daily.starting_capital = 500.0
    mock_reto.get_daily_pnl.return_value = mock_daily
    return RiskManager(reto_tracker=mock_reto)


# ──────────────────────────────────────────────────────────────
# can_trade basic cases
# ──────────────────────────────────────────────────────────────


class TestCanTrade:
    def test_allows_trade_when_clean(self, risk):
        assert risk.can_trade("futures") is True

    def test_blocks_after_kill_switch(self, risk):
        risk._activate_kill_switch()
        assert risk.can_trade("futures") is False

    def test_blocks_when_max_positions_reached(self, risk):
        # Open 3 positions (default max)
        risk.open_position("futures", "MNQ", "LONG")
        risk.open_position("crypto", "BTC", "LONG")
        risk.open_position("momo", "NVDA", "LONG")
        assert risk.can_trade("options") is False

    def test_allows_trade_below_max_positions(self, risk):
        risk.open_position("futures", "MNQ", "LONG")
        assert risk.can_trade("crypto") is True


# ──────────────────────────────────────────────────────────────
# Consecutive losses → engine pause
# ──────────────────────────────────────────────────────────────


class TestConsecutiveLosses:
    def test_three_losses_pause_engine(self, risk):
        for _ in range(3):
            risk.register_trade("futures", pnl=-100, won=False)
        # Engine should now be paused
        assert risk.can_trade("futures") is False

    def test_win_resets_consecutive_counter(self, risk):
        risk.register_trade("futures", pnl=-100, won=False)
        risk.register_trade("futures", pnl=-100, won=False)
        risk.register_trade("futures", pnl=200, won=True)
        # Only 2 losses before a win → counter reset
        assert risk._consecutive_losses["futures"] == 0


# ──────────────────────────────────────────────────────────────
# Kill switch
# ──────────────────────────────────────────────────────────────


class TestKillSwitch:
    def test_kill_switch_not_active_by_default(self, risk):
        assert risk.check_kill_switch() is False

    def test_kill_switch_activates_and_blocks_all(self, risk):
        risk._activate_kill_switch()
        assert risk.check_kill_switch() is True
        for engine in ("futures", "options", "momo", "crypto"):
            assert risk.can_trade(engine) is False

    def test_kill_switch_expires(self, risk):
        risk._activate_kill_switch()
        # Backdate the expiry
        risk._kill_switch_until = datetime.utcnow() - timedelta(seconds=1)
        assert risk.check_kill_switch() is False


# ──────────────────────────────────────────────────────────────
# Correlation guard
# ──────────────────────────────────────────────────────────────


class TestCorrelationGuard:
    def test_blocks_long_crypto_when_futures_long(self, risk):
        risk.open_position("futures", "NQ", "LONG")
        assert risk.has_correlation_conflict("crypto", "LONG") is True

    def test_allows_short_crypto_when_futures_long(self, risk):
        risk.open_position("futures", "NQ", "LONG")
        assert risk.has_correlation_conflict("crypto", "SHORT") is False

    def test_no_conflict_when_no_open_positions(self, risk):
        assert risk.has_correlation_conflict("crypto", "LONG") is False

    def test_blocks_long_futures_when_crypto_long(self, risk):
        risk.open_position("crypto", "BTC", "LONG")
        assert risk.has_correlation_conflict("futures", "LONG") is True


# ──────────────────────────────────────────────────────────────
# PDT tracking
# ──────────────────────────────────────────────────────────────


class TestPDTTracking:
    def test_compliant_by_default(self, risk):
        assert risk.is_pdt_compliant() is True

    def test_blocks_after_three_trades(self, risk):
        # Register 3 momo day trades
        for _ in range(3):
            risk.register_trade("momo", pnl=100, won=True)
        assert risk.is_pdt_compliant() is False

    def test_allows_after_window_expires(self, risk):
        from config import settings

        # Fill the rolling window with trades from 6 business days ago
        # (older than the 5-day window)
        cutoff = _business_days_ago(settings.PDT_ROLLING_DAYS + 1)
        for _ in range(3):
            risk._momo_day_trades.append(cutoff)
        risk._prune_pdt_window()
        assert risk.is_pdt_compliant() is True

    def test_get_remaining_pdt_trades(self, risk):
        risk.register_trade("momo", pnl=50, won=True)
        assert risk.get_pdt_trades_remaining() == 2


# ──────────────────────────────────────────────────────────────
# Open/close position tracking
# ──────────────────────────────────────────────────────────────


class TestPositionTracking:
    def test_open_increments_count(self, risk):
        risk.open_position("futures", "MNQ", "LONG")
        assert risk.get_open_position_count() == 1

    def test_close_decrements_count(self, risk):
        risk.open_position("futures", "MNQ", "LONG")
        risk.close_position("futures", "MNQ")
        assert risk.get_open_position_count() == 0


# ──────────────────────────────────────────────────────────────
# Remaining bullets
# ──────────────────────────────────────────────────────────────


class TestRemainingBullets:
    def test_full_bullets_at_start(self, risk_with_reto):
        # Phase 1: 6 trades/day × 4 engines = 24 total
        bullets = risk_with_reto.get_remaining_bullets()
        assert bullets == 24

    def test_decrements_after_trades(self, risk_with_reto):
        risk_with_reto.register_trade("futures", pnl=100, won=True)
        assert risk_with_reto.get_remaining_bullets() == 23
