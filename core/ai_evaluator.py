"""
AI Evaluator — Real-time trade evaluation using Groq (primary) and Gemini (fallback).

This module adds a second layer of intelligence on top of the statistical AI Brain.
Before executing any trade that passes the Brain's score threshold, the evaluator
sends the trade context + brain memory to a real LLM for contextual analysis.

The LLM can catch things the statistical brain cannot:
  - Macro events (FOMC, earnings, geopolitical)
  - Time-of-day nuances (Friday afternoon reversals)
  - Pattern fatigue (too many trades in a short window)
  - Contextual risk (already at +30% for the day, protect profits)

If the AI is unavailable or times out, the trade proceeds based on Brain score alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class AIEvaluation:
    """Result from the AI evaluator."""

    approved: bool
    reasoning: str
    source: str  # "groq", "gemini", "brain_only" (fallback)
    response_time_ms: float


class AIEvaluator:
    """
    Evaluates trades using Groq LLM (primary) with Gemini fallback.

    If both AI services fail or are disabled, returns approval based
    on Brain score alone (graceful degradation).
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=settings.AI_EVALUATOR_TIMEOUT)
        self._enabled = settings.AI_EVALUATOR_ENABLED
        self._call_count = 0
        self._approve_count = 0
        self._reject_count = 0

    async def evaluate_trade(
        self,
        setup_type: str,
        engine: str,
        direction: str,
        entry: float,
        stop: float,
        target: float,
        session: str,
        atr: float,
        brain_score: int,
        brain_reasoning: str,
        brain_memory: dict[str, Any],
        daily_pnl: float,
        daily_pnl_pct: float,
        instrument: str,
        open_positions: int,
        market_context: str = "",
    ) -> AIEvaluation:
        """
        Send trade context to Groq for evaluation.
        Falls back to Gemini, then to brain-only if both fail.
        """
        if not self._enabled:
            return AIEvaluation(
                approved=True,
                reasoning="AI evaluator disabled — using brain score only",
                source="brain_only",
                response_time_ms=0,
            )

        # Build the prompt with full context
        prompt = self._build_prompt(
            setup_type=setup_type,
            engine=engine,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            session=session,
            atr=atr,
            brain_score=brain_score,
            brain_reasoning=brain_reasoning,
            brain_memory=brain_memory,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            instrument=instrument,
            open_positions=open_positions,
            market_context=market_context,
        )

        # Try Groq first
        start = asyncio.get_event_loop().time()
        result = await self._call_groq(prompt)
        if result is not None:
            elapsed = (asyncio.get_event_loop().time() - start) * 1000
            self._call_count += 1
            if result["approved"]:
                self._approve_count += 1
            else:
                self._reject_count += 1
            logger.info(
                "[ai-eval] Groq %s trade (%dms): %s",
                "APPROVED" if result["approved"] else "REJECTED",
                int(elapsed),
                result["reasoning"],
            )
            return AIEvaluation(
                approved=result["approved"],
                reasoning=result["reasoning"],
                source="groq",
                response_time_ms=elapsed,
            )

        # Fallback to Gemini
        start = asyncio.get_event_loop().time()
        result = await self._call_gemini(prompt)
        if result is not None:
            elapsed = (asyncio.get_event_loop().time() - start) * 1000
            self._call_count += 1
            if result["approved"]:
                self._approve_count += 1
            else:
                self._reject_count += 1
            logger.info(
                "[ai-eval] Gemini %s trade (%dms): %s",
                "APPROVED" if result["approved"] else "REJECTED",
                int(elapsed),
                result["reasoning"],
            )
            return AIEvaluation(
                approved=result["approved"],
                reasoning=result["reasoning"],
                source="gemini",
                response_time_ms=elapsed,
            )

        # Both failed — graceful degradation
        logger.warning("[ai-eval] Both Groq and Gemini unavailable — using brain score only")
        return AIEvaluation(
            approved=True,
            reasoning="AI services unavailable — proceeding with brain score",
            source="brain_only",
            response_time_ms=0,
        )

    def _build_prompt(self, **kwargs: Any) -> str:
        """Build the evaluation prompt with full trade context and brain memory."""

        # Extract top-level win rates from brain memory for context
        memory = kwargs.get("brain_memory", {})
        setup_stats = memory.get("setup_stats", {})
        session_stats = memory.get("session_stats", {})

        # Format win rates
        setup_summary = ""
        for setup, stats in setup_stats.items():
            total = stats.get("wins", 0) + stats.get("losses", 0)
            if total > 0:
                wr = stats["wins"] / total * 100
                setup_summary += f"  - {setup}: {wr:.0f}% win rate ({total} trades)\n"

        session_summary = ""
        for session, stats in session_stats.items():
            total = stats.get("wins", 0) + stats.get("losses", 0)
            if total > 0:
                wr = stats["wins"] / total * 100
                session_summary += f"  - {session}: {wr:.0f}% win rate ({total} trades)\n"

        consecutive = memory.get("consecutive_losses", {})
        consec_warnings = ""
        for setup, count in consecutive.items():
            if count >= 2:
                consec_warnings += f"  ⚠️ {setup} has {count} consecutive losses\n"

        now = datetime.now(timezone.utc)
        day_of_week = now.strftime("%A")
        hour = now.strftime("%H:%M UTC")

        entry = kwargs["entry"]
        stop = kwargs["stop"]
        target = kwargs["target"]
        risk_reward = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0

        return f"""You are a professional trading risk evaluator for an automated futures trading bot.

TRADE PROPOSAL:
- Instrument: {kwargs['instrument']} ({kwargs['engine']} engine)
- Direction: {kwargs['direction']}
- Entry: {entry:.2f}
- Stop Loss: {stop:.2f}
- Take Profit: {target:.2f}
- Risk:Reward: 1:{risk_reward:.1f}
- Setup: {kwargs['setup_type']}
- Session: {kwargs['session']}
- ATR: {kwargs['atr']:.1f}
- Open positions: {kwargs['open_positions']}

BRAIN ANALYSIS (statistical scoring):
- Brain Score: {kwargs['brain_score']}/100
- Reasoning: {kwargs['brain_reasoning']}

TODAY'S PERFORMANCE:
- Daily P&L: ${kwargs['daily_pnl']:.2f} ({kwargs['daily_pnl_pct']:.1f}%)
- Day: {day_of_week}
- Time: {hour}

MARKET CONTEXT:
{kwargs.get('market_context') or 'No macro data available'}

HISTORICAL PERFORMANCE (from bot's learning memory):
Setup win rates:
{setup_summary if setup_summary else '  No data yet (fresh start)'}
Session win rates:
{session_summary if session_summary else '  No data yet (fresh start)'}
{consec_warnings if consec_warnings else ''}

RULES:
1. APPROVE trades with solid setups, good R:R, and favorable conditions
2. REJECT if: Friday afternoon (risk of reversal), right before major news events, consecutive losses on this setup >= 3, daily P&L already > +30% (protect profits), or poor R:R ratio (< 1:1.5)
3. If daily P&L is > +20%, only approve A+ setups (high confidence)
4. Consider the statistical win rates from the bot's memory — patterns with <40% win rate should be rejected
5. Be concise

Respond in EXACTLY this JSON format, nothing else:
{{"approved": true/false, "reasoning": "one line explanation"}}"""

    async def _call_groq(self, prompt: str) -> dict[str, Any] | None:
        """Call Groq API with the evaluation prompt."""
        api_key = settings.GROQ_API_KEY
        if not api_key:
            return None
        try:
            resp = await self._client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a trading risk evaluator. Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 150,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Parse JSON response — handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(content)
            return {
                "approved": bool(result.get("approved", True)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ai-eval] Groq call failed: %s", exc)
            return None

    async def _call_gemini(self, prompt: str) -> dict[str, Any] | None:
        """Call Google Gemini API as fallback."""
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            return None
        try:
            resp = await self._client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 150,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(content)
            return {
                "approved": bool(result.get("approved", True)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ai-eval] Gemini call failed: %s", exc)
            return None

    def get_stats(self) -> dict[str, Any]:
        """Return evaluator usage statistics."""
        return {
            "total_calls": self._call_count,
            "approvals": self._approve_count,
            "rejections": self._reject_count,
            "approval_rate": f"{self._approve_count / self._call_count * 100:.0f}%" if self._call_count > 0 else "N/A",
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
