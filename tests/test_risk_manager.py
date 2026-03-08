"""
Tests for core/risk_manager.py — daily limits, kill switch, PDT, correlation guard.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.risk_manager import DynamicTrailingLock, RiskManager, _PROFIT_LOCK_TIERS, _business_days_ago


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

    def test_profit_lock_checked_before_kill_switch(self, risk):
        """Daily profit lock must be checked first — even before the kill switch."""
        risk._daily_profit_locked = True
        # do NOT activate kill switch — profit lock alone should block
        assert risk.can_trade("futures") is False
        assert risk.can_trade("crypto") is False

    def test_profit_lock_blocks_all_engines(self, risk):
        risk._daily_profit_locked = True
        for engine in ("futures", "options", "momo", "crypto"):
            assert risk.can_trade(engine) is False


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
        # Phase 1: 4 trades/day × 4 engines = 16 total
        bullets = risk_with_reto.get_remaining_bullets()
        assert bullets == 16

    def test_decrements_after_trades(self, risk_with_reto):
        risk_with_reto.register_trade("futures", pnl=100, won=True)
        assert risk_with_reto.get_remaining_bullets() == 15


# ──────────────────────────────────────────────────────────────
# Kill switch persistence
# ──────────────────────────────────────────────────────────────


class TestKillSwitchPersistence:
    def test_kill_switch_persisted_and_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            # Patch the module-level path so both instances use the temp file
            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk1 = RiskManager(reto_tracker=None)
                risk1._activate_kill_switch()
                assert risk1._kill_switch_active is True

                # A second instance (simulating a restart) should pick up the state
                risk2 = RiskManager(reto_tracker=None)
                assert risk2._kill_switch_active is True
                assert risk2.check_kill_switch() is True

    def test_expired_kill_switch_not_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk1 = RiskManager(reto_tracker=None)
                risk1._activate_kill_switch()
                # Backdate the expiry so it is already expired
                risk1._kill_switch_until = datetime.utcnow() - timedelta(seconds=1)
                risk1._save_state()

                # Restart should NOT restore an expired kill switch
                risk2 = RiskManager(reto_tracker=None)
                assert risk2._kill_switch_active is False

    def test_consecutive_losses_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk1 = RiskManager(reto_tracker=None)
                risk1.register_trade("futures", pnl=-100, won=False)
                risk1.register_trade("futures", pnl=-100, won=False)
                assert risk1._consecutive_losses["futures"] == 2

                # Restart: consecutive loss count is restored
                risk2 = RiskManager(reto_tracker=None)
                assert risk2._consecutive_losses["futures"] == 2

    def test_state_file_from_previous_day_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            # Write a state file with yesterday's date and an active kill switch
            yesterday = (date.today() - timedelta(days=1)).isoformat()
            stale_state = {
                "kill_switch_active": True,
                "kill_switch_until": (datetime.utcnow() + timedelta(hours=23)).isoformat(),
                "daily_trades_count": {},
                "consecutive_losses": {},
                "paused_until": {},
                "today_date": yesterday,
            }
            os.makedirs(tmp, exist_ok=True)
            with open(state_path, "w") as fh:
                json.dump(stale_state, fh)

            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk = RiskManager(reto_tracker=None)
                # State from a different day must be ignored
                assert risk._kill_switch_active is False


# ──────────────────────────────────────────────────────────────
# Profit floor — USD threshold (Bug 5)
# ──────────────────────────────────────────────────────────────


class TestProfitFloorUSDThreshold:
    def test_floor_activates_at_usd_threshold_before_pct_threshold(self):
        """PROFIT_FLOOR_ACTIVATION_USD=$25 activates floor at +$25."""
        mock_reto = MagicMock()
        mock_reto.get_phase.return_value = 1
        mock_daily = MagicMock()
        mock_daily.starting_capital = 500.0
        mock_reto.get_daily_pnl.return_value = mock_daily
        risk = RiskManager(reto_tracker=mock_reto)

        event = risk.update_daily_pnl(26.0)  # just above the $25 USD threshold
        assert event.floor_activated is True

    def test_floor_not_active_below_usd_threshold(self):
        mock_reto = MagicMock()
        mock_reto.get_phase.return_value = 1
        mock_daily = MagicMock()
        mock_daily.starting_capital = 500.0
        mock_reto.get_daily_pnl.return_value = mock_daily
        risk = RiskManager(reto_tracker=mock_reto)

        event = risk.update_daily_pnl(10.0)  # below both thresholds
        assert event.floor_activated is False
        assert risk._floor_active is False


# ──────────────────────────────────────────────────────────────
# Dynamic Trailing Profit Lock (Fix 5)
# ──────────────────────────────────────────────────────────────


class TestDynamicTrailingProfitLock:
    """Tests for the new dollar-based dynamic trailing profit lock system."""

    def test_retention_pct_scales_with_pnl(self, risk):
        """Retention percentage increases as P&L grows."""
        assert risk._get_retention_pct(250.0) == 0.75
        assert risk._get_retention_pct(200.0) == 0.70
        assert risk._get_retention_pct(120.0) == 0.65
        assert risk._get_retention_pct(75.0) == 0.60
        assert risk._get_retention_pct(30.0) == 0.50
        assert risk._get_retention_pct(10.0) == 0.0  # below activation threshold

    def test_floor_uses_dynamic_retention_at_250(self, risk):
        """At $268 peak, retention is 75%, floor = $268 × 0.75 = $201."""
        risk.update_daily_pnl(268.0)
        floor_value = 268.0 * 0.75
        # P&L drops below floor → lock triggered
        event = risk.update_daily_pnl(190.0)
        assert event.floor_hit is True
        assert risk._daily_profit_locked is True

    def test_profit_lock_stops_trading_permanently(self, risk):
        """Once floor is hit, trading stays locked for the rest of the day."""
        risk.update_daily_pnl(268.0)
        risk.update_daily_pnl(190.0)  # hits floor ($268 × 0.75 = $201 > $190)
        assert risk._daily_profit_locked is True

        # Even if P&L "recovers", lock must remain
        risk.update_daily_pnl(300.0)
        assert risk._daily_profit_locked is True
        assert risk.can_trade("futures") is False

    def test_floor_not_triggered_above_floor_value(self, risk):
        """P&L above floor level should NOT trigger the lock."""
        risk.update_daily_pnl(268.0)   # peak
        event = risk.update_daily_pnl(210.0)  # above 268 × 0.75 = 201
        assert event.floor_hit is False
        assert risk._daily_profit_locked is False

    def test_profit_lock_persists_across_restart(self):
        """Daily profit lock is saved to disk and restored on restart."""
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk1 = RiskManager(reto_tracker=None)
                risk1.update_daily_pnl(268.0)
                risk1.update_daily_pnl(190.0)  # hits floor
                assert risk1._daily_profit_locked is True

                # Simulate restart
                risk2 = RiskManager(reto_tracker=None)
                assert risk2._daily_profit_locked is True
                assert risk2.can_trade("futures") is False

    def test_profit_lock_resets_on_new_day(self, risk):
        """Daily profit lock must reset at the start of each new trading day."""
        risk._daily_profit_locked = True
        risk._floor_active = True
        risk._max_daily_pnl_gain = 268.0
        risk._daily_pnl_gain = 190.0

        # Simulate a new day
        risk._today = date.today() - timedelta(days=1)
        risk._maybe_reset_daily()

        assert risk._daily_profit_locked is False
        assert risk._floor_active is False
        assert risk._max_daily_pnl_gain == 0.0

    def test_tier_min_scores_use_dollar_thresholds(self, risk):
        """Min score increases at dollar thresholds, not percentage thresholds."""
        from config import settings

        # Below $25: normal baseline
        risk._daily_pnl_gain = 20.0
        assert risk.get_min_score_for_tier() == settings.PROFIT_TIER_0_MIN_SCORE

        # At $25: tier 1 (min_score = 70)
        risk._daily_pnl_gain = 25.0
        assert risk.get_min_score_for_tier() == 70

        # At $100: tier 2 (min_score = 75)
        risk._daily_pnl_gain = 100.0
        assert risk.get_min_score_for_tier() == 75

        # At $150: tier 3 (min_score = 80)
        risk._daily_pnl_gain = 150.0
        assert risk.get_min_score_for_tier() == 80

        # At $250: tier 4 (min_score = 85)
        risk._daily_pnl_gain = 250.0
        assert risk.get_min_score_for_tier() == 85

    def test_tier_size_multipliers_use_dollar_thresholds(self, risk):
        """Size multipliers tighten at dollar thresholds."""
        # Below $25: full size
        risk._daily_pnl_gain = 20.0
        assert risk.get_size_multiplier_for_tier() == 1.0

        # At $25: still full size (1.0)
        risk._daily_pnl_gain = 30.0
        assert risk.get_size_multiplier_for_tier() == 1.0

        # At $150: reduced to 0.75
        risk._daily_pnl_gain = 160.0
        assert risk.get_size_multiplier_for_tier() == 0.75

        # At $250: reduced to 0.50
        risk._daily_pnl_gain = 260.0
        assert risk.get_size_multiplier_for_tier() == 0.50

    def test_profit_tier_numbers(self, risk):
        """get_profit_tier() returns correct tier numbers for dollar ranges."""
        risk._daily_pnl_gain = 20.0
        assert risk.get_profit_tier() == 0

        risk._daily_pnl_gain = 30.0
        assert risk.get_profit_tier() == 1

        risk._daily_pnl_gain = 110.0
        assert risk.get_profit_tier() == 2

        risk._daily_pnl_gain = 160.0
        assert risk.get_profit_tier() == 3

        risk._daily_pnl_gain = 260.0
        assert risk.get_profit_tier() == 4

    def test_example_from_problem_statement(self, risk):
        """Reproduce the live trading example: peak $268, drop to $190 → lock at $201."""
        risk.update_daily_pnl(268.0)  # peak; floor = 268 × 0.75 = $201
        event = risk.update_daily_pnl(190.0)  # 190 < 201 → lock
        assert event.floor_hit is True
        assert risk._daily_profit_locked is True
        assert risk.can_trade("futures") is False
        assert risk.can_trade("crypto") is False



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
        # Phase 1: 4 trades/day × 4 engines = 16 total
        bullets = risk_with_reto.get_remaining_bullets()
        assert bullets == 16

    def test_decrements_after_trades(self, risk_with_reto):
        risk_with_reto.register_trade("futures", pnl=100, won=True)
        assert risk_with_reto.get_remaining_bullets() == 15


# ──────────────────────────────────────────────────────────────
# Kill switch persistence
# ──────────────────────────────────────────────────────────────


class TestKillSwitchPersistence:
    def test_kill_switch_persisted_and_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            # Patch the module-level path so both instances use the temp file
            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk1 = RiskManager(reto_tracker=None)
                risk1._activate_kill_switch()
                assert risk1._kill_switch_active is True

                # A second instance (simulating a restart) should pick up the state
                risk2 = RiskManager(reto_tracker=None)
                assert risk2._kill_switch_active is True
                assert risk2.check_kill_switch() is True

    def test_expired_kill_switch_not_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk1 = RiskManager(reto_tracker=None)
                risk1._activate_kill_switch()
                # Backdate the expiry so it is already expired
                risk1._kill_switch_until = datetime.utcnow() - timedelta(seconds=1)
                risk1._save_state()

                # Restart should NOT restore an expired kill switch
                risk2 = RiskManager(reto_tracker=None)
                assert risk2._kill_switch_active is False

    def test_consecutive_losses_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk1 = RiskManager(reto_tracker=None)
                risk1.register_trade("futures", pnl=-100, won=False)
                risk1.register_trade("futures", pnl=-100, won=False)
                assert risk1._consecutive_losses["futures"] == 2

                # Restart: consecutive loss count is restored
                risk2 = RiskManager(reto_tracker=None)
                assert risk2._consecutive_losses["futures"] == 2

    def test_state_file_from_previous_day_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            import core.risk_manager as rm_module
            state_path = os.path.join(tmp, "risk_state.json")

            # Write a state file with yesterday's date and an active kill switch
            yesterday = (date.today() - timedelta(days=1)).isoformat()
            stale_state = {
                "kill_switch_active": True,
                "kill_switch_until": (datetime.utcnow() + timedelta(hours=23)).isoformat(),
                "daily_trades_count": {},
                "consecutive_losses": {},
                "paused_until": {},
                "today_date": yesterday,
            }
            os.makedirs(tmp, exist_ok=True)
            with open(state_path, "w") as fh:
                json.dump(stale_state, fh)

            with patch.object(rm_module, "_RISK_STATE_PATH", state_path):
                risk = RiskManager(reto_tracker=None)
                # State from a different day must be ignored
                assert risk._kill_switch_active is False



# ──────────────────────────────────────────────────────────────
# DynamicTrailingLock — standalone 30%-drop rule
# ──────────────────────────────────────────────────────────────


class TestDynamicTrailingLock:
    """Tests for the standalone DynamicTrailingLock class (30% drop-from-peak rule)."""

    def test_not_locked_at_start(self):
        lock = DynamicTrailingLock()
        assert lock.is_locked is False
        assert lock.peak_pnl == 0.0
        assert lock.locked_amount == 0.0

    def test_no_lock_below_min_peak(self):
        """Lock must NOT activate when peak is below MIN_PEAK_TO_ACTIVATE ($25)."""
        lock = DynamicTrailingLock()
        # Peak only $20 — drop of any size must not lock
        lock.update(20.0)
        assert lock.update(0.0) is False
        assert lock.is_locked is False

    def test_peak_only_rises(self):
        """Peak should increase on new highs but never decrease."""
        lock = DynamicTrailingLock()
        lock.update(100.0)
        assert lock.peak_pnl == 100.0
        # Drop by less than 30% so the lock does NOT trigger
        lock.update(80.0)   # 20% drop — below threshold
        assert lock.peak_pnl == 100.0  # did not decrease
        assert lock.is_locked is False
        lock.update(200.0)
        assert lock.peak_pnl == 200.0  # updated to new high

    def test_continues_below_30pct_drop(self):
        """A 29% drop from peak must NOT trigger the lock (< 30% threshold)."""
        lock = DynamicTrailingLock()
        lock.update(268.0)   # peak = $268
        # Drop = $78 = 29.1% of $268 — just below the 30% threshold
        result = lock.update(190.0)
        assert result is False
        assert lock.is_locked is False

    def test_locks_at_30pct_drop(self):
        """A 30% drop from peak must trigger the lock (>= 30% threshold)."""
        lock = DynamicTrailingLock()
        lock.update(100.0)   # peak = $100; threshold = $70
        result = lock.update(70.0)  # exact 30% drop
        assert result is True
        assert lock.is_locked is True
        assert lock.locked_amount == 70.0

    def test_locks_above_30pct_drop(self):
        """A 56% drop (from user's actual data) must trigger the lock."""
        lock = DynamicTrailingLock()
        lock.update(268.0)   # peak = $268
        lock.update(190.0)   # 29% drop — OK
        # Drop = $151 (56.3% of peak $268) → LOCK
        result = lock.update(117.0)
        assert result is True
        assert lock.is_locked is True
        assert lock.locked_amount == 117.0
        assert lock.peak_pnl == 268.0

    def test_lock_is_permanent_for_the_day(self):
        """Once locked, update() must always return True even if P&L recovers."""
        lock = DynamicTrailingLock()
        lock.update(100.0)
        lock.update(70.0)  # triggers lock
        assert lock.is_locked is True
        # Simulated P&L recovery
        assert lock.update(300.0) is True
        assert lock.is_locked is True

    def test_no_cap_on_gains(self):
        """Continuous gains must NEVER trigger the lock."""
        lock = DynamicTrailingLock()
        for pnl in [50, 100, 200, 300, 400, 500, 1000]:
            assert lock.update(float(pnl)) is False
        assert lock.is_locked is False
        assert lock.peak_pnl == 1000.0

    def test_reset_clears_all_state(self):
        """reset() must completely clear peak, lock, and locked_amount."""
        lock = DynamicTrailingLock()
        lock.update(268.0)
        lock.update(117.0)  # triggers lock
        assert lock.is_locked is True

        lock.reset()
        assert lock.is_locked is False
        assert lock.peak_pnl == 0.0
        assert lock.locked_amount == 0.0
        # After reset, pnl updates are accepted again
        assert lock.update(50.0) is False

    def test_get_trade_restrictions_normal(self):
        """Below 20% P&L: normal trading restrictions."""
        lock = DynamicTrailingLock()
        r = lock.get_trade_restrictions(current_pnl=80.0, capital=500.0)  # 16%
        assert r["min_score"] == 65
        assert r["size_mult"] == 1.0

    def test_get_trade_restrictions_at_20pct(self):
        """At 20%+ P&L: require score 75, full size."""
        lock = DynamicTrailingLock()
        r = lock.get_trade_restrictions(current_pnl=100.0, capital=500.0)  # 20%
        assert r["min_score"] == 75
        assert r["size_mult"] == 1.0

    def test_get_trade_restrictions_at_30pct(self):
        """At 30%+ P&L: require score 80, 75% size."""
        lock = DynamicTrailingLock()
        r = lock.get_trade_restrictions(current_pnl=150.0, capital=500.0)  # 30%
        assert r["min_score"] == 80
        assert r["size_mult"] == 0.75

    def test_get_trade_restrictions_at_50pct(self):
        """At 50%+ P&L: require score 85, 50% size (elite trades only)."""
        lock = DynamicTrailingLock()
        r = lock.get_trade_restrictions(current_pnl=250.0, capital=500.0)  # 50%
        assert r["min_score"] == 85
        assert r["size_mult"] == 0.50

    def test_user_example_from_problem_statement(self):
        """Reproduce the live trading example exactly as described in the problem statement.

        Peak +$268 at 12:37 PM.
        - 12:40: +$190 → drop 29% → should CONTINUE (< 30%)
        - 12:45: +$117 → drop 56% → should STOP (> 30%)
        """
        lock = DynamicTrailingLock()

        # Morning gains, no cap
        for pnl in [50.26, 58.02, 96.78, 141.04, 179.80, 218.56, 268.32]:
            assert lock.update(pnl) is False, f"Should continue at +${pnl}"

        assert lock.peak_pnl == pytest.approx(268.32)

        # 12:40 — 29% drop — should continue
        assert lock.update(190.0) is False

        # 12:45 — 56% drop — should stop
        assert lock.update(117.0) is True
        assert lock.is_locked is True
        assert lock.locked_amount == 117.0


# ──────────────────────────────────────────────────────────────
# Settings — new MNQ / MoMo configuration values
# ──────────────────────────────────────────────────────────────


class TestNewSettings:
    """Verify the new settings added for MNQ and MoMo integration."""

    def test_futures_ticker_defaults_to_mnq(self):
        from config import settings
        assert settings.FUTURES_TICKER == "MNQ"

    def test_futures_multiplier_defaults_to_2(self):
        from config import settings
        assert settings.FUTURES_MULTIPLIER == 2

    def test_enable_momo_defaults_to_true(self):
        from config import settings
        assert settings.ENABLE_MOMO is True

    def test_momo_allocation_is_set(self):
        from config import settings
        assert settings.MOMO_ALLOCATION > 0

    def test_phase1_uses_mnq(self):
        """Phase 1 should use MNQ as the futures instrument."""
        from config import settings
        assert settings.PHASES[1].futures_instrument == "MNQ"

    def test_phase4_uses_nq(self):
        """Phase 4 should upgrade to full NQ."""
        from config import settings
        assert settings.PHASES[4].futures_instrument == "NQ"
