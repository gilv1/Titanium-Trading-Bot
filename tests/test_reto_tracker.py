"""
Tests for core/reto_tracker.py — phase transitions, auto-scaling, drawdown protection.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from core.reto_tracker import DailyPnL, RetoTracker, TradeResult


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def tracker() -> RetoTracker:
    """Return a RetoTracker starting at $500 (Phase 1)."""
    return RetoTracker(initial_capital=500.0)


# ──────────────────────────────────────────────────────────────
# Phase detection
# ──────────────────────────────────────────────────────────────


class TestPhaseDetection:
    def test_phase_1_at_500(self, tracker):
        assert tracker.get_phase() == 1

    def test_phase_1_at_1499(self):
        t = RetoTracker(initial_capital=1499.0)
        assert t.get_phase() == 1

    def test_phase_2_at_1500(self):
        t = RetoTracker(initial_capital=3000.0)
        assert t.get_phase() == 2

    def test_phase_2_at_4499(self):
        t = RetoTracker(initial_capital=4999.0)
        assert t.get_phase() == 2

    def test_phase_3_at_4500(self):
        t = RetoTracker(initial_capital=5000.0)
        assert t.get_phase() == 3

    def test_phase_4_at_9000(self):
        t = RetoTracker(initial_capital=9000.0)
        assert t.get_phase() == 4

    def test_phase_4_at_15000(self):
        t = RetoTracker(initial_capital=15000.0)
        assert t.get_phase() == 4


# ──────────────────────────────────────────────────────────────
# Phase transitions via update_capital
# ──────────────────────────────────────────────────────────────


class TestPhaseTransitions:
    def test_phase_1_to_2(self, tracker):
        """Gain $2,500 → cross Phase 2 threshold."""
        result = TradeResult(engine="futures", pnl=2500.0)
        tracker.update_capital(result)
        assert tracker.get_phase() == 2

    def test_phase_2_to_3(self):
        t = RetoTracker(initial_capital=3000.0)
        result = TradeResult(engine="futures", pnl=2000.0)
        t.update_capital(result)
        assert t.get_phase() == 3

    def test_phase_3_to_4(self):
        t = RetoTracker(initial_capital=5000.0)
        result = TradeResult(engine="futures", pnl=4000.0)
        t.update_capital(result)
        assert t.get_phase() == 4

    def test_capital_floored_at_zero(self, tracker):
        result = TradeResult(engine="futures", pnl=-10000.0)
        tracker.update_capital(result)
        assert tracker.capital == 0.0


# ──────────────────────────────────────────────────────────────
# Auto-scaling: position sizes per phase
# ──────────────────────────────────────────────────────────────


class TestPositionSizing:
    def test_phase_1_futures_contracts(self, tracker):
        assert tracker.get_contracts("futures") == 1

    def test_phase_1_futures_instrument(self, tracker):
        assert tracker.get_futures_instrument() == "MES"

    def test_phase_2_futures_contracts(self):
        t = RetoTracker(initial_capital=3000.0)
        assert t.get_contracts("futures") == 2

    def test_phase_3_futures_instrument(self):
        t = RetoTracker(initial_capital=5000.0)
        assert t.get_futures_instrument() == "MNQ"

    def test_phase_4_contracts(self):
        t = RetoTracker(initial_capital=9000.0)
        assert t.get_contracts("futures") == 1

    def test_phase_1_momo_max(self, tracker):
        assert tracker.get_position_size("momo") == 100.0

    def test_phase_2_momo_max(self):
        t = RetoTracker(initial_capital=3000.0)
        assert t.get_position_size("momo") == 250.0

    def test_phase_1_options_max(self, tracker):
        assert tracker.get_position_size("options") == 50.0

    def test_crypto_position_proportional(self, tracker):
        """Crypto allocation should be 30 % of capital."""
        size = tracker.get_position_size("crypto")
        assert abs(size - 500.0 * 0.30) < 0.01


# ──────────────────────────────────────────────────────────────
# Drawdown protection
# ──────────────────────────────────────────────────────────────


class TestDrawdownProtection:
    def test_drawdown_above_8pct_activates_override(self):
        t = RetoTracker(initial_capital=1000.0)
        # Force today's starting capital to 1000
        t._today_start_capital = 1000.0
        # Lose $90 = 9%
        result = TradeResult(engine="futures", pnl=-90.0)
        t.update_capital(result)
        assert t._drawdown_override is True

    def test_drawdown_override_uses_previous_phase(self):
        t = RetoTracker(initial_capital=3100.0)  # Phase 2
        t._today_start_capital = 3100.0
        result = TradeResult(engine="futures", pnl=-300.0)  # ~9.7% loss
        t.update_capital(result)
        # Should operate at Phase 1 sizing when override active
        assert t._effective_phase() == 1

    def test_no_override_below_8pct(self):
        t = RetoTracker(initial_capital=1000.0)
        t._today_start_capital = 1000.0
        result = TradeResult(engine="futures", pnl=-50.0)  # 5%
        t.update_capital(result)
        assert t._drawdown_override is False


# ──────────────────────────────────────────────────────────────
# Capital milestones
# ──────────────────────────────────────────────────────────────


class TestMilestones:
    def test_milestone_2500_triggered(self):
        t = RetoTracker(initial_capital=2400.0)
        result = TradeResult(engine="futures", pnl=200.0)
        alerts = t.update_capital(result)
        assert any("2,500" in a or "$2,500" in a for a in alerts)

    def test_milestone_only_triggers_once(self):
        t = RetoTracker(initial_capital=2400.0)
        t.update_capital(TradeResult(engine="futures", pnl=200.0))
        alerts2 = t.update_capital(TradeResult(engine="futures", pnl=1.0))
        # Second update should not re-fire the same milestone
        assert not any("2,500" in a for a in alerts2)

    def test_no_milestone_when_capital_low(self, tracker):
        result = TradeResult(engine="futures", pnl=100.0)
        alerts = tracker.update_capital(result)
        assert alerts == []


# ──────────────────────────────────────────────────────────────
# Daily P&L
# ──────────────────────────────────────────────────────────────


class TestDailyPnL:
    def test_daily_pnl_starts_at_zero(self, tracker):
        daily = tracker.get_daily_pnl()
        assert daily.pnl == 0.0

    def test_daily_pnl_reflects_trade(self, tracker):
        tracker.update_capital(TradeResult(engine="futures", pnl=50.0))
        daily = tracker.get_daily_pnl()
        assert daily.pnl == 50.0
