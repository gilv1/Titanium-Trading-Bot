"""
News Sentinel — Macro Event & Market Context Layer (Layer 3).

Monitors:
- Economic calendar (FOMC, CPI, NFP, PPI, GDP, Retail Sales, etc.)
- VIX level via IBKR
- Market regime detection

Provides context to the AI Evaluator so the bot avoids trading into
high-impact macro events and sizes down during elevated volatility.

Graceful degradation: if VIX data or calendar is unavailable,
defaults to normal operation (size_modifier=1.0, should_pause=False).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.connection import ConnectionManager

logger = logging.getLogger(__name__)

# Path to the static 2026 economic calendar
_CALENDAR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "economic_calendar_2026.json")

# How long (seconds) to cache the VIX reading to avoid excessive IBKR API calls
_VIX_CACHE_TTL = 300  # 5 minutes


@dataclass
class MarketContext:
    """Snapshot of current macro market conditions."""

    risk_level: str  # "low", "medium", "high", "extreme"
    upcoming_events: list[dict[str, Any]]  # events in next 24 hours
    minutes_to_next_event: int  # minutes to nearest upcoming high/medium event; -1 if none
    vix_level: float  # -1.0 if unavailable
    vix_regime: str  # "calm", "elevated", "fear", "panic"
    size_modifier: float  # 0.0–1.0 multiplier to apply to position size
    should_pause: bool  # True = skip trading entirely this cycle
    reasoning: str  # human-readable explanation

    # Convenience: empty/neutral context for graceful degradation
    @classmethod
    def neutral(cls) -> "MarketContext":
        return cls(
            risk_level="low",
            upcoming_events=[],
            minutes_to_next_event=-1,
            vix_level=-1.0,
            vix_regime="calm",
            size_modifier=1.0,
            should_pause=False,
            reasoning="No macro data available — normal operation",
        )


class NewsSentinel:
    """
    Macro event & market context layer.

    Monitors:
    - Economic calendar (FOMC, CPI, NFP, PPI, GDP, Jobless Claims, etc.)
    - VIX level via IBKR (cached for 5 minutes)
    - Market regime detection

    Provides context to the AI Evaluator for smarter trade decisions.
    """

    # VIX regime thresholds
    VIX_EXTREME: float = 35.0   # pause all trading
    VIX_FEAR: float = 25.0      # 25% size
    VIX_ELEVATED: float = 18.0  # 50% size

    # Economic events rated by market impact
    HIGH_IMPACT_EVENTS: frozenset[str] = frozenset({
        "FOMC", "Federal Funds Rate", "Fed Interest Rate Decision",
        "CPI", "Consumer Price Index",
        "Non-Farm Payrolls", "NFP", "Nonfarm Payrolls",
        "PPI", "Producer Price Index",
        "GDP", "Gross Domestic Product",
        "PCE", "Core PCE",
        "Retail Sales",
        "ISM Manufacturing",
        "Jackson Hole",
        "Powell", "Fed Chair",
        "FOMC Rate Decision",
        "GDP Q1 Advance", "GDP Q2 Advance", "GDP Q3 Advance", "GDP Q4 Advance",
        "GDP Q1 Final", "GDP Q2 Final", "GDP Q3 Final", "GDP Q4 Final",
    })

    MEDIUM_IMPACT_EVENTS: frozenset[str] = frozenset({
        "Jobless Claims", "Initial Claims",
        "Durable Goods",
        "Housing Starts",
        "Consumer Confidence",
        "Michigan Sentiment",
        "Trade Balance",
        "Industrial Production",
    })

    def __init__(self) -> None:
        self._calendar: list[dict[str, Any]] = []
        self._calendar_tz: str = "US/Eastern"
        self._vix_cache: float = -1.0
        self._vix_cache_time: datetime | None = None
        self._load_calendar()

    # ──────────────────────────────────────────────────────────
    # Calendar loading
    # ──────────────────────────────────────────────────────────

    def _load_calendar(self) -> None:
        """Load the static economic calendar from the JSON file."""
        path = os.path.abspath(_CALENDAR_PATH)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._calendar = data.get("events", [])
            self._calendar_tz = data.get("timezone", "US/Eastern")
            logger.info("[news-sentinel] Loaded %d calendar events from %s", len(self._calendar), path)
        except FileNotFoundError:
            logger.warning("[news-sentinel] Calendar file not found at %s — using empty calendar.", path)
            self._calendar = []
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("[news-sentinel] Failed to load calendar: %s", exc)
            self._calendar = []

    # ──────────────────────────────────────────────────────────
    # Event helpers
    # ──────────────────────────────────────────────────────────

    def _event_impact(self, event_name: str) -> str:
        """Return 'high', 'medium', or 'low' based on event name matching."""
        name_upper = event_name.upper()
        for keyword in self.HIGH_IMPACT_EVENTS:
            if keyword.upper() in name_upper:
                return "high"
        for keyword in self.MEDIUM_IMPACT_EVENTS:
            if keyword.upper() in name_upper:
                return "medium"
        return "low"

    def _event_datetime_eastern(self, event: dict[str, Any]) -> datetime | None:
        """Parse an event's date+time string into an Eastern-timezone-aware datetime."""
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            tz = ZoneInfo("America/New_York")
            dt_str = f"{event['date']} {event['time']}"
            dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            return dt_naive.replace(tzinfo=tz)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[news-sentinel] Could not parse event datetime: %s", exc)
            return None

    def _now_eastern(self) -> datetime:
        """Current time in America/New_York timezone."""
        from zoneinfo import ZoneInfo
        return datetime.now(tz=ZoneInfo("America/New_York"))

    # ──────────────────────────────────────────────────────────
    # Calendar queries
    # ──────────────────────────────────────────────────────────

    def fetch_economic_calendar(self) -> list[dict[str, Any]]:
        """
        Return today's and tomorrow's economic events with impact ratings.

        Returns a list of dicts: {name, date, time, impact, minutes_away}.
        """
        now = self._now_eastern()
        tomorrow = now + timedelta(days=1)
        today_str = now.strftime("%Y-%m-%d")
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")

        relevant: list[dict[str, Any]] = []
        for event in self._calendar:
            event_date = event.get("date", "")
            if event_date not in (today_str, tomorrow_str):
                continue

            event_dt = self._event_datetime_eastern(event)
            impact = event.get("impact") or self._event_impact(event.get("name", ""))
            minutes_away: int = -1
            if event_dt is not None:
                diff = (event_dt - now).total_seconds() / 60
                minutes_away = int(diff)

            relevant.append({
                "name": event.get("name", ""),
                "date": event_date,
                "time": event.get("time", ""),
                "impact": impact,
                "minutes_away": minutes_away,
            })

        return relevant

    def _get_nearest_impactful_event(
        self,
        events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, int]:
        """
        Find the nearest high/medium impact event relative to now.

        Returns (event_dict, minutes_away).  minutes_away is negative if the
        event has already passed (within the last 60 minutes).
        """
        nearest: dict[str, Any] | None = None
        nearest_abs = 9999

        for ev in events:
            if ev["impact"] not in ("high", "medium"):
                continue
            mins = ev["minutes_away"]
            # Consider events up to 60 minutes in the past and any time in the future
            if mins < -60:
                continue
            if abs(mins) < nearest_abs:
                nearest_abs = abs(mins)
                nearest = ev

        return nearest, (nearest["minutes_away"] if nearest else -1)

    # ──────────────────────────────────────────────────────────
    # VIX
    # ──────────────────────────────────────────────────────────

    async def get_vix_level(self, connection_manager: "ConnectionManager | None" = None) -> float:
        """
        Fetch current VIX level from IBKR.

        Returns the last VIX price, or -1.0 if unavailable.
        The value is cached for 5 minutes to avoid excessive IBKR API calls.
        """
        # Serve from cache if fresh
        if (
            self._vix_cache_time is not None
            and self._vix_cache > 0
            and (datetime.utcnow() - self._vix_cache_time).total_seconds() < _VIX_CACHE_TTL
        ):
            return self._vix_cache

        if connection_manager is None:
            return -1.0

        try:
            from ib_insync import Index  # type: ignore
            ib = connection_manager.margin.get_ib()
            if ib is None or not connection_manager.margin.is_connected():
                return -1.0

            vix_contract = Index("VIX", "CBOE")
            ib.qualifyContracts(vix_contract)

            ticker = ib.reqMktData(vix_contract, genericTickList="", snapshot=True)
            await asyncio.sleep(2)  # wait for snapshot data

            vix: float | None = ticker.last if (ticker.last and ticker.last > 0) else ticker.close
            ib.cancelMktData(vix_contract)

            if vix and vix > 0:
                self._vix_cache = float(vix)
                self._vix_cache_time = datetime.utcnow()
                return self._vix_cache
        except ImportError:
            logger.debug("[news-sentinel] ib_insync not available — VIX fetch skipped.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[news-sentinel] VIX fetch failed: %s", exc)

        return -1.0

    @staticmethod
    def _vix_regime(vix: float) -> str:
        """Classify VIX into a human-readable regime label."""
        if vix < 0:
            return "unknown"
        if vix < 15:
            return "calm"
        if vix < 20:
            return "elevated"
        if vix < 30:
            return "fear"
        return "panic"

    # ──────────────────────────────────────────────────────────
    # Risk calculation
    # ──────────────────────────────────────────────────────────

    def _calculate_risk(
        self,
        minutes_to_event: int,
        event_impact: str,
        vix: float,
    ) -> tuple[str, float, bool, list[str]]:
        """
        Core risk calculation logic.

        Returns (risk_level, size_modifier, should_pause, reasons).
        """
        should_pause = False
        size_modifier = 1.0
        risk_level = "low"
        reasons: list[str] = []

        # ── VIX-based sizing ──────────────────────────────────
        if vix > self.VIX_EXTREME:
            risk_level = "extreme"
            should_pause = True
            size_modifier = 0.0
            reasons.append(f"VIX={vix:.1f} EXTREME — trading paused")
        elif vix > self.VIX_FEAR:
            risk_level = "high"
            size_modifier = 0.25
            reasons.append(f"VIX={vix:.1f} FEAR — 25% size")
        elif vix > self.VIX_ELEVATED:
            risk_level = "medium"
            size_modifier = 0.5
            reasons.append(f"VIX={vix:.1f} ELEVATED — 50% size")
        else:
            if vix > 0:
                reasons.append(f"VIX={vix:.1f} CALM — full size")
            else:
                reasons.append("VIX unavailable — full size")

        # ── Event-based pausing ───────────────────────────────
        if event_impact == "high":
            if 0 <= minutes_to_event <= 30:
                should_pause = True
                size_modifier = 0.0
                if risk_level not in ("extreme",):
                    risk_level = "extreme"
                reasons.append(f"HIGH impact event in {minutes_to_event}min — PAUSED")
            elif 30 < minutes_to_event <= 60:
                size_modifier = min(size_modifier, 0.25)
                if risk_level not in ("extreme", "high"):
                    risk_level = "high"
                reasons.append(f"HIGH impact event in {minutes_to_event}min — 25% max")
            elif -30 <= minutes_to_event < 0:
                # Event just happened; wait for volatility to settle
                size_modifier = min(size_modifier, 0.25)
                if risk_level not in ("extreme", "high"):
                    risk_level = "high"
                reasons.append(f"HIGH impact event {abs(minutes_to_event)}min ago — settling")
        elif event_impact == "medium":
            if 0 <= minutes_to_event <= 15:
                size_modifier = min(size_modifier, 0.5)
                if risk_level == "low":
                    risk_level = "medium"
                reasons.append(f"MEDIUM impact event in {minutes_to_event}min — 50% max")

        return risk_level, size_modifier, should_pause, reasons

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    async def get_market_context(
        self,
        connection_manager: "ConnectionManager | None" = None,
    ) -> MarketContext:
        """
        Build a complete market context snapshot.

        Returns risk level, upcoming events, VIX status, and sizing recommendation.
        Gracefully degrades to a neutral context if data is unavailable.
        """
        try:
            # Fetch economic events
            events = self.fetch_economic_calendar()

            # Fetch VIX (with cache)
            vix = await self.get_vix_level(connection_manager)

            # Find nearest relevant event
            nearest_event, minutes_to_next = self._get_nearest_impactful_event(events)
            event_impact = nearest_event["impact"] if nearest_event else "low"

            # Calculate risk
            risk_level, size_modifier, should_pause, reasons = self._calculate_risk(
                minutes_to_event=minutes_to_next,
                event_impact=event_impact,
                vix=vix,
            )

            # Collect upcoming events in the next 24 hours (positive minutes_away only)
            upcoming = [e for e in events if 0 <= e["minutes_away"] <= 1440]
            upcoming.sort(key=lambda e: e["minutes_away"])

            return MarketContext(
                risk_level=risk_level,
                upcoming_events=upcoming,
                minutes_to_next_event=minutes_to_next,
                vix_level=vix,
                vix_regime=self._vix_regime(vix),
                size_modifier=size_modifier,
                should_pause=should_pause,
                reasoning="; ".join(reasons) if reasons else "Normal conditions",
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("[news-sentinel] get_market_context failed: %s — returning neutral context.", exc)
            return MarketContext.neutral()
