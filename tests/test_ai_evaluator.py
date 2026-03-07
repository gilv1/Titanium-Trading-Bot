"""Tests for the AI Evaluator module."""

from __future__ import annotations

import asyncio
import json

import pytest

from core.ai_evaluator import AIEvaluator, AIEvaluation


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

_TRADE_KWARGS = dict(
    setup_type="VWAP_BOUNCE",
    engine="futures",
    direction="LONG",
    entry=5850.0,
    stop=5840.0,
    target=5870.0,
    session="NY",
    atr=8.0,
    brain_score=80,
    brain_reasoning="Strong VWAP support",
    brain_memory={},
    daily_pnl=0.0,
    daily_pnl_pct=0.0,
    instrument="MES",
    open_positions=0,
)


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setattr("config.settings.AI_EVALUATOR_ENABLED", True)
    monkeypatch.setattr("config.settings.GROQ_API_KEY", "test-key")
    monkeypatch.setattr("config.settings.GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("config.settings.AI_EVALUATOR_TIMEOUT", 2.0)
    monkeypatch.setattr("config.settings.GROQ_MODEL", "llama-3.3-70b-versatile")
    return AIEvaluator()


# ──────────────────────────────────────────────────────────────
# Disabled evaluator
# ──────────────────────────────────────────────────────────────


class TestAIEvaluatorDisabled:
    def test_returns_approved_when_disabled(self, monkeypatch):
        monkeypatch.setattr("config.settings.AI_EVALUATOR_ENABLED", False)
        ev = AIEvaluator()
        result = asyncio.get_event_loop().run_until_complete(
            ev.evaluate_trade(**_TRADE_KWARGS)
        )
        assert result.approved is True
        assert result.source == "brain_only"
        assert result.response_time_ms == 0

    def test_reasoning_mentions_disabled(self, monkeypatch):
        monkeypatch.setattr("config.settings.AI_EVALUATOR_ENABLED", False)
        ev = AIEvaluator()
        result = asyncio.get_event_loop().run_until_complete(
            ev.evaluate_trade(**_TRADE_KWARGS)
        )
        assert "disabled" in result.reasoning.lower()


# ──────────────────────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────────────────────


class TestPromptBuilding:
    def test_prompt_includes_setup_type(self, evaluator):
        prompt = evaluator._build_prompt(**_TRADE_KWARGS)
        assert "VWAP_BOUNCE" in prompt

    def test_prompt_includes_instrument(self, evaluator):
        prompt = evaluator._build_prompt(**_TRADE_KWARGS)
        assert "MES" in prompt

    def test_prompt_includes_direction(self, evaluator):
        prompt = evaluator._build_prompt(**_TRADE_KWARGS)
        assert "LONG" in prompt

    def test_prompt_includes_brain_score(self, evaluator):
        prompt = evaluator._build_prompt(**_TRADE_KWARGS)
        assert "80" in prompt

    def test_prompt_includes_rr_ratio(self, evaluator):
        prompt = evaluator._build_prompt(**_TRADE_KWARGS)
        assert "Risk:Reward" in prompt

    def test_prompt_with_brain_memory_stats(self, evaluator):
        memory = {
            "setup_stats": {
                "VWAP_BOUNCE": {"wins": 7, "losses": 3},
            },
            "session_stats": {
                "NY": {"wins": 10, "losses": 5},
            },
            "consecutive_losses": {},
        }
        kwargs = {**_TRADE_KWARGS, "brain_memory": memory}
        prompt = evaluator._build_prompt(**kwargs)
        assert "VWAP_BOUNCE" in prompt
        assert "70%" in prompt  # 7/10 = 70%
        assert "NY" in prompt

    def test_prompt_warns_consecutive_losses(self, evaluator):
        memory = {"consecutive_losses": {"EMA_PULLBACK": 3}}
        kwargs = {**_TRADE_KWARGS, "brain_memory": memory}
        prompt = evaluator._build_prompt(**kwargs)
        assert "EMA_PULLBACK" in prompt
        assert "consecutive losses" in prompt.lower()

    def test_prompt_no_memory_shows_fresh_start(self, evaluator):
        prompt = evaluator._build_prompt(**_TRADE_KWARGS)
        assert "fresh start" in prompt.lower()


# ──────────────────────────────────────────────────────────────
# Graceful degradation — both APIs fail
# ──────────────────────────────────────────────────────────────


class TestGracefulDegradation:
    def test_proceeds_when_no_api_keys(self, monkeypatch):
        """When no API keys are set, trade proceeds with brain score."""
        monkeypatch.setattr("config.settings.AI_EVALUATOR_ENABLED", True)
        monkeypatch.setattr("config.settings.GROQ_API_KEY", "")
        monkeypatch.setattr("config.settings.GEMINI_API_KEY", "")
        monkeypatch.setattr("config.settings.AI_EVALUATOR_TIMEOUT", 2.0)
        monkeypatch.setattr("config.settings.GROQ_MODEL", "llama-3.3-70b-versatile")
        ev = AIEvaluator()
        result = asyncio.get_event_loop().run_until_complete(
            ev.evaluate_trade(**_TRADE_KWARGS)
        )
        assert result.approved is True
        assert result.source == "brain_only"

    def test_brain_only_source_on_degradation(self, monkeypatch):
        monkeypatch.setattr("config.settings.AI_EVALUATOR_ENABLED", True)
        monkeypatch.setattr("config.settings.GROQ_API_KEY", "")
        monkeypatch.setattr("config.settings.GEMINI_API_KEY", "")
        monkeypatch.setattr("config.settings.AI_EVALUATOR_TIMEOUT", 2.0)
        monkeypatch.setattr("config.settings.GROQ_MODEL", "llama-3.3-70b-versatile")
        ev = AIEvaluator()
        result = asyncio.get_event_loop().run_until_complete(
            ev.evaluate_trade(**_TRADE_KWARGS)
        )
        assert result.source == "brain_only"


# ──────────────────────────────────────────────────────────────
# Statistics tracking
# ──────────────────────────────────────────────────────────────


class TestStats:
    def test_initial_stats_are_zero(self, evaluator):
        stats = evaluator.get_stats()
        assert stats["total_calls"] == 0
        assert stats["approvals"] == 0
        assert stats["rejections"] == 0
        assert stats["approval_rate"] == "N/A"


# ──────────────────────────────────────────────────────────────
# AIEvaluation dataclass
# ──────────────────────────────────────────────────────────────


class TestAIEvaluationDataclass:
    def test_approved_evaluation(self):
        ev = AIEvaluation(approved=True, reasoning="Good setup", source="groq", response_time_ms=45.2)
        assert ev.approved is True
        assert ev.source == "groq"
        assert ev.response_time_ms == 45.2

    def test_rejected_evaluation(self):
        ev = AIEvaluation(approved=False, reasoning="Friday afternoon", source="gemini", response_time_ms=120.0)
        assert ev.approved is False
        assert ev.source == "gemini"
