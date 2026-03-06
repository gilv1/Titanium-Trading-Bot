"""
Tests for core/brain.py — AI Brain scoring and self-learning.
"""

from __future__ import annotations

import pytest

from core.brain import AIBrain, BrainMemory, TradeDecision, TradeOutcome, _win_rate


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_brain(tmp_path, monkeypatch) -> AIBrain:
    """Return an AIBrain with a fresh (empty) memory file in a tmp directory."""
    monkeypatch.chdir(tmp_path)
    # Ensure data dir exists
    (tmp_path / "data").mkdir()
    brain = AIBrain()
    return brain


# ──────────────────────────────────────────────────────────────
# _win_rate helper
# ──────────────────────────────────────────────────────────────


class TestWinRate:
    def test_no_trades_returns_fifty_percent(self):
        assert _win_rate({"wins": 0, "losses": 0}) == 0.5

    def test_all_wins(self):
        assert _win_rate({"wins": 10, "losses": 0}) == 1.0

    def test_all_losses(self):
        assert _win_rate({"wins": 0, "losses": 10}) == 0.0

    def test_mixed(self):
        assert _win_rate({"wins": 3, "losses": 1}) == 0.75


# ──────────────────────────────────────────────────────────────
# TradeDecision scoring
# ──────────────────────────────────────────────────────────────


class TestEvaluateTrade:
    def test_high_score_approves_full_size(self, fresh_brain):
        decision = fresh_brain.evaluate_trade(
            setup_type="VWAP_BOUNCE",
            engine="futures",
            entry=15000,
            stop=14990,
            target=15020,
            session="NY",
            atr=8.0,
            daily_drawdown_pct=0.0,
            open_positions=0,
            trend_aligned=True,
            correlation_conflict=False,
        )
        assert decision.approved is True
        assert decision.size_multiplier == 1.0
        assert decision.score > 75

    def test_correlation_conflict_zeros_correlation_score(self, fresh_brain):
        decision = fresh_brain.evaluate_trade(
            setup_type="VWAP_BOUNCE",
            engine="crypto",
            entry=50000,
            stop=49000,
            target=52000,
            session="NY",
            atr=500.0,
            daily_drawdown_pct=0.0,
            open_positions=0,
            trend_aligned=True,
            correlation_conflict=True,
        )
        # correlation component contributes 0 → score should be lower
        assert "CONFLICT" in decision.reasoning

    def test_max_positions_blocks_correlation(self, fresh_brain):
        decision = fresh_brain.evaluate_trade(
            setup_type="ORB",
            engine="futures",
            entry=15000,
            stop=14990,
            target=15020,
            session="NY",
            atr=8.0,
            daily_drawdown_pct=0.0,
            open_positions=3,  # at max
            trend_aligned=True,
        )
        # Correlation score = 0 because max positions reached
        assert "max positions" in decision.reasoning

    def test_high_drawdown_lowers_score(self, fresh_brain):
        decision_no_dd = fresh_brain.evaluate_trade(
            setup_type="VWAP_BOUNCE",
            engine="futures",
            entry=15000,
            stop=14990,
            target=15020,
            session="NY",
            daily_drawdown_pct=0.0,
            open_positions=0,
            trend_aligned=True,
        )
        decision_with_dd = fresh_brain.evaluate_trade(
            setup_type="VWAP_BOUNCE",
            engine="futures",
            entry=15000,
            stop=14990,
            target=15020,
            session="NY",
            daily_drawdown_pct=9.0,
            open_positions=0,
            trend_aligned=True,
        )
        assert decision_no_dd.score > decision_with_dd.score

    def test_trend_not_aligned_reduces_score(self, fresh_brain):
        aligned = fresh_brain.evaluate_trade(
            setup_type="EMA_PULLBACK",
            engine="futures",
            entry=15000,
            stop=14990,
            target=15020,
            session="NY",
            trend_aligned=True,
        )
        not_aligned = fresh_brain.evaluate_trade(
            setup_type="EMA_PULLBACK",
            engine="futures",
            entry=15000,
            stop=14990,
            target=15020,
            session="NY",
            trend_aligned=False,
        )
        assert aligned.score > not_aligned.score


# ──────────────────────────────────────────────────────────────
# Self-learning updates
# ──────────────────────────────────────────────────────────────


class TestRecordOutcome:
    def test_win_increments_wins(self, fresh_brain):
        fresh_brain.record_outcome(
            TradeOutcome(
                setup_type="VWAP_BOUNCE",
                session="NY",
                day_of_week="Monday",
                hour=10,
                volatility_regime="medium",
                won=True,
                engine="futures",
            )
        )
        assert fresh_brain.memory.setup_stats["VWAP_BOUNCE"]["wins"] == 1
        assert fresh_brain.memory.setup_stats["VWAP_BOUNCE"]["losses"] == 0

    def test_loss_increments_losses(self, fresh_brain):
        fresh_brain.record_outcome(
            TradeOutcome(
                setup_type="ORB",
                session="London",
                day_of_week="Tuesday",
                hour=4,
                volatility_regime="low",
                won=False,
                engine="futures",
            )
        )
        assert fresh_brain.memory.setup_stats["ORB"]["losses"] == 1

    def test_three_consecutive_losses_apply_penalty(self, fresh_brain):
        for _ in range(3):
            fresh_brain.record_outcome(
                TradeOutcome(
                    setup_type="LIQUIDITY_GRAB",
                    session="NY",
                    day_of_week="Wednesday",
                    hour=11,
                    volatility_regime="high",
                    won=False,
                    engine="futures",
                )
            )
        assert fresh_brain.memory.confidence_penalty["LIQUIDITY_GRAB"] > 0

    def test_win_resets_consecutive_losses_and_penalty(self, fresh_brain):
        # 3 losses to set penalty
        for _ in range(3):
            fresh_brain.record_outcome(
                TradeOutcome(
                    setup_type="NEWS_BURST",
                    session="NY",
                    day_of_week="Thursday",
                    hour=10,
                    volatility_regime="high",
                    won=False,
                    engine="futures",
                )
            )
        assert fresh_brain.memory.confidence_penalty["NEWS_BURST"] > 0

        # Single win resets penalty
        fresh_brain.record_outcome(
            TradeOutcome(
                setup_type="NEWS_BURST",
                session="NY",
                day_of_week="Thursday",
                hour=10,
                volatility_regime="high",
                won=True,
                engine="futures",
            )
        )
        assert fresh_brain.memory.confidence_penalty["NEWS_BURST"] == 0
        assert fresh_brain.memory.consecutive_losses["NEWS_BURST"] == 0


# ──────────────────────────────────────────────────────────────
# Memory persistence
# ──────────────────────────────────────────────────────────────


class TestMemoryPersistence:
    def test_save_and_reload(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        brain = AIBrain()
        brain.record_outcome(
            TradeOutcome(
                setup_type="VWAP_BOUNCE",
                session="NY",
                day_of_week="Monday",
                hour=10,
                volatility_regime="medium",
                won=True,
                engine="futures",
            )
        )

        # Reload
        brain2 = AIBrain()
        assert brain2.memory.setup_stats["VWAP_BOUNCE"]["wins"] == 1


# ──────────────────────────────────────────────────────────────
# Suggested stop
# ──────────────────────────────────────────────────────────────


class TestSuggestedStop:
    def test_zero_atr_returns_phase_default(self, fresh_brain):
        result = fresh_brain.suggested_stop_points(atr=0, session="NY", phase_sl_pts=10)
        assert result == 10

    def test_high_atr_clamped(self, fresh_brain):
        result = fresh_brain.suggested_stop_points(atr=100, session="NY", phase_sl_pts=10)
        assert result <= 13  # 130 % of phase default

    def test_low_atr_clamped(self, fresh_brain):
        result = fresh_brain.suggested_stop_points(atr=1, session="NY", phase_sl_pts=10)
        assert result >= 8  # floor
