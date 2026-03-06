"""
AI Brain for Titanium Warrior v3 — AUTOEVOLUTIVE / SELF-LEARNING.

Central decision hub that:
  - Scores every potential trade on a 0-100 scale.
  - Learns from each closed trade (win rate per setup, session, day, hour, ATR regime).
  - Persists memory to ``data/brain_memory.json``.
  - Decays pattern confidence after 3 consecutive losses.
  - Adapts stop-loss recommendations based on session ATR.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

SETUP_TYPES = (
    "VWAP_BOUNCE",
    "ORB",
    "EMA_PULLBACK",
    "LIQUIDITY_GRAB",
    "NEWS_BURST",
    "RANGE_BREAKOUT",
    "DIP_BUY",
)

SESSIONS = ("Tokyo", "London", "NY", "Asia", "Europe", "US", "Crypto_Asia", "Crypto_Europe", "Crypto_US")
VOLATILITY_REGIMES = ("low", "medium", "high")
DAYS_OF_WEEK = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass
class TradeDecision:
    """Result of the brain's evaluation of a potential trade."""

    approved: bool
    size_multiplier: float  # 1.0 / 0.5 / 0.0
    score: int
    reasoning: str


@dataclass
class TradeOutcome:
    """Minimal record of a closed trade used for self-learning."""

    setup_type: str
    session: str
    day_of_week: str
    hour: int
    volatility_regime: str
    won: bool
    engine: str


def _default_win_rate_entry() -> dict[str, Any]:
    return {"wins": 0, "losses": 0}


class BrainMemory:
    """
    Holds all learning data for the AI Brain.

    Tracks win/loss counts keyed by:
      - setup type
      - session
      - day of week
      - hour of day
      - volatility regime
    Also tracks consecutive losses per setup to trigger confidence decay.
    """

    def __init__(self) -> None:
        # {setup_type: {wins, losses}}
        self.setup_stats: dict[str, dict[str, int]] = {s: _default_win_rate_entry() for s in SETUP_TYPES}
        # {session: {wins, losses}}
        self.session_stats: dict[str, dict[str, int]] = {s: _default_win_rate_entry() for s in SESSIONS}
        # {day: {wins, losses}}
        self.day_stats: dict[str, dict[str, int]] = {d: _default_win_rate_entry() for d in DAYS_OF_WEEK}
        # {hour_str: {wins, losses}}
        self.hour_stats: dict[str, dict[str, int]] = {str(h): _default_win_rate_entry() for h in range(24)}
        # {volatility_regime: {wins, losses}}
        self.volatility_stats: dict[str, dict[str, int]] = {v: _default_win_rate_entry() for v in VOLATILITY_REGIMES}
        # {setup_type: consecutive_losses}
        self.consecutive_losses: dict[str, int] = {s: 0 for s in SETUP_TYPES}
        # {setup_type: confidence_penalty}  — penalty subtracted from pattern score
        self.confidence_penalty: dict[str, int] = {s: 0 for s in SETUP_TYPES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_stats": self.setup_stats,
            "session_stats": self.session_stats,
            "day_stats": self.day_stats,
            "hour_stats": self.hour_stats,
            "volatility_stats": self.volatility_stats,
            "consecutive_losses": self.consecutive_losses,
            "confidence_penalty": self.confidence_penalty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainMemory":
        mem = cls()
        mem.setup_stats.update(data.get("setup_stats", {}))
        mem.session_stats.update(data.get("session_stats", {}))
        mem.day_stats.update(data.get("day_stats", {}))
        mem.hour_stats.update(data.get("hour_stats", {}))
        mem.volatility_stats.update(data.get("volatility_stats", {}))
        mem.consecutive_losses.update(data.get("consecutive_losses", {}))
        mem.confidence_penalty.update(data.get("confidence_penalty", {}))
        return mem


# ──────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────


def _win_rate(stat: dict[str, int]) -> float:
    """Return win rate 0.0–1.0 given a {wins, losses} dict.  50 % when no data."""
    total = stat["wins"] + stat["losses"]
    if total == 0:
        return 0.5
    return stat["wins"] / total


class AIBrain:
    """
    AUTOEVOLUTIVE trading decision engine.

    Score breakdown (0–100):
      - Pattern confidence       : 0–25
      - Session win-rate history : 0–20
      - ATR / volatility assess  : 0–15
      - Daily drawdown status    : 0–15
      - Correlation check        : 0–15
      - Trend alignment          : 0–10
    """

    def __init__(self) -> None:
        self.memory = BrainMemory()
        self._load_memory()

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def _load_memory(self) -> None:
        path = settings.BRAIN_MEMORY_FILE
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                self.memory = BrainMemory.from_dict(data)
                logger.info("Brain memory loaded from %s", path)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not load brain memory (%s); starting fresh.", exc)

    def save_memory(self) -> None:
        """Persist learning data to disk."""
        path = settings.BRAIN_MEMORY_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.memory.to_dict(), fh, indent=2)

    # ──────────────────────────────────────────────────────────
    # Core evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate_trade(  # noqa: PLR0913
        self,
        setup_type: str,
        engine: str,
        entry: float,
        stop: float,
        target: float,
        session: str,
        atr: float = 0.0,
        daily_drawdown_pct: float = 0.0,
        open_positions: int = 0,
        trend_aligned: bool = True,
        correlation_conflict: bool = False,
    ) -> TradeDecision:
        """
        Evaluate a potential trade and return a TradeDecision.

        Parameters
        ----------
        setup_type:          One of SETUP_TYPES.
        engine:              Engine name (futures/options/momo/crypto).
        entry:               Proposed entry price.
        stop:                Stop-loss price.
        target:              Take-profit price.
        session:             Active session name.
        atr:                 Current Average True Range value.
        daily_drawdown_pct:  Current intraday drawdown as % of capital (positive = loss).
        open_positions:      Number of currently open positions across all engines.
        trend_aligned:       Whether higher-timeframe trend agrees with trade direction.
        correlation_conflict:Whether a correlated asset already has an open position in same direction.
        """
        reasons: list[str] = []
        score = 0

        # 1. Pattern confidence (0–25)
        setup_wr = _win_rate(self.memory.setup_stats.get(setup_type, _default_win_rate_entry()))
        pattern_score = int(setup_wr * 25)
        penalty = self.memory.confidence_penalty.get(setup_type, 0)
        pattern_score = max(0, pattern_score - penalty)
        score += pattern_score
        reasons.append(f"pattern={pattern_score}/25 (wr={setup_wr:.0%}, penalty={penalty})")

        # 2. Session win-rate history (0–20)
        session_wr = _win_rate(self.memory.session_stats.get(session, _default_win_rate_entry()))
        session_score = int(session_wr * 20)
        score += session_score
        reasons.append(f"session={session_score}/20 (wr={session_wr:.0%})")

        # 3. ATR / volatility assessment (0–15)
        if atr <= 0:
            vol_score = 10  # neutral when no data
            vol_regime = "medium"
        elif atr < 5:
            vol_score = 8
            vol_regime = "low"
        elif atr <= 20:
            vol_score = 15
            vol_regime = "medium"
        else:
            vol_score = 6
            vol_regime = "high"
        score += vol_score
        reasons.append(f"volatility={vol_score}/15 ({vol_regime} ATR={atr:.1f})")

        # 4. Daily drawdown status (0–15)
        if daily_drawdown_pct <= 0:
            dd_score = 15
        elif daily_drawdown_pct < 4:
            dd_score = 12
        elif daily_drawdown_pct < 8:
            dd_score = 5
        else:
            dd_score = 0  # already at risk limit
        score += dd_score
        reasons.append(f"drawdown={dd_score}/15 (dd={daily_drawdown_pct:.1f}%)")

        # 5. Correlation check (0–15)
        if correlation_conflict:
            corr_score = 0
            reasons.append("correlation=0/15 (CONFLICT)")
        elif open_positions >= settings.MAX_SIMULTANEOUS_POSITIONS:
            corr_score = 0
            reasons.append(f"correlation=0/15 (max positions {open_positions})")
        elif open_positions > 0:
            corr_score = 8
            reasons.append(f"correlation=8/15 ({open_positions} open)")
        else:
            corr_score = 15
            reasons.append("correlation=15/15 (no conflicts)")
        score += corr_score

        # 6. Trend alignment (0–10)
        trend_score = 10 if trend_aligned else 0
        score += trend_score
        reasons.append(f"trend={trend_score}/10")

        # ── Decision ──────────────────────────────────────────
        if score > settings.BRAIN_SCORE_FULL_SIZE:
            approved = True
            size_multiplier = 1.0
        elif score > settings.BRAIN_SCORE_HALF_SIZE:
            approved = True
            size_multiplier = 0.5
        else:
            approved = False
            size_multiplier = 0.0

        return TradeDecision(
            approved=approved,
            size_multiplier=size_multiplier,
            score=score,
            reasoning=" | ".join(reasons),
        )

    # ──────────────────────────────────────────────────────────
    # Self-learning update
    # ──────────────────────────────────────────────────────────

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """
        Update win/loss stats from a closed trade.

        Called after every closed trade. Also handles confidence decay:
        if a setup loses 3 times in a row, apply a penalty of 5 pts until
        the pattern wins again (regime change signal).
        """
        key = "wins" if outcome.won else "losses"

        # Update per-dimension stats
        _bump(self.memory.setup_stats, outcome.setup_type, key)
        _bump(self.memory.session_stats, outcome.session, key)
        _bump(self.memory.day_stats, outcome.day_of_week, key)
        _bump(self.memory.hour_stats, str(outcome.hour), key)
        _bump(self.memory.volatility_stats, outcome.volatility_regime, key)

        # Consecutive loss tracking → confidence decay
        if outcome.won:
            self.memory.consecutive_losses[outcome.setup_type] = 0
            # Reset penalty after a win (regime changed / pattern valid again)
            self.memory.confidence_penalty[outcome.setup_type] = 0
        else:
            self.memory.consecutive_losses[outcome.setup_type] = (
                self.memory.consecutive_losses.get(outcome.setup_type, 0) + 1
            )
            if self.memory.consecutive_losses[outcome.setup_type] >= settings.MAX_CONSECUTIVE_LOSSES:
                self.memory.confidence_penalty[outcome.setup_type] = min(
                    self.memory.confidence_penalty.get(outcome.setup_type, 0) + 5,
                    25,  # cap at max pattern score
                )
                logger.warning(
                    "Pattern %s has %d consecutive losses; confidence penalty → %d pts",
                    outcome.setup_type,
                    self.memory.consecutive_losses[outcome.setup_type],
                    self.memory.confidence_penalty[outcome.setup_type],
                )

        self.save_memory()
        logger.debug(
            "Brain updated: setup=%s session=%s won=%s",
            outcome.setup_type,
            outcome.session,
            outcome.won,
        )

    # ──────────────────────────────────────────────────────────
    # ATR-based adaptive SL suggestion
    # ──────────────────────────────────────────────────────────

    def suggested_stop_points(
        self,
        atr: float,
        session: str,
        phase_sl_pts: int,
    ) -> int:
        """
        Suggest an adaptive stop-loss distance in index points.

        Uses session ATR: widens stop in volatile sessions, tightens otherwise.
        Always stays within [8, 20] points for MNQ and [phase_sl_pts ± 30%].
        """
        if atr <= 0:
            return phase_sl_pts
        # Scale SL with ATR: 1 ATR * 0.8 as a proxy
        atr_sl = int(atr * 0.8)
        # Clamp to ±30 % of the phase default
        lo = max(8, int(phase_sl_pts * 0.7))
        hi = int(phase_sl_pts * 1.3)
        return max(lo, min(hi, atr_sl))

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────

    def get_win_rate_for_setup(self, setup_type: str) -> float:
        return _win_rate(self.memory.setup_stats.get(setup_type, _default_win_rate_entry()))

    def get_win_rate_for_session(self, session: str) -> float:
        return _win_rate(self.memory.session_stats.get(session, _default_win_rate_entry()))

    @staticmethod
    def current_outcome_context() -> tuple[str, int]:
        """Return (day_of_week, hour) in ET for the current moment."""
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo(settings.TIMEZONE))
        return DAYS_OF_WEEK[now.weekday()], now.hour


# ──────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────

def _bump(stats: dict[str, dict[str, int]], key: str, field: str) -> None:
    """Increment a win or loss counter, initialising the entry if needed."""
    if key not in stats:
        stats[key] = _default_win_rate_entry()
    stats[key][field] = stats[key].get(field, 0) + 1
