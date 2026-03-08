"""
Tests for core/news_sentinel.py — News Sentinel macro context layer.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.news_sentinel import MarketContext, NewsSentinel


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def sentinel_with_calendar(tmp_path) -> NewsSentinel:
    """Return a NewsSentinel loaded with a small in-memory calendar."""
    calendar_data = {
        "events": [
            {"date": "2026-03-06", "time": "08:30", "name": "Non-Farm Payrolls", "impact": "high"},
            {"date": "2026-03-11", "time": "08:30", "name": "CPI", "impact": "high"},
            {"date": "2026-03-18", "time": "14:00", "name": "FOMC Rate Decision", "impact": "high"},
            {"date": "2026-03-17", "time": "08:30", "name": "Retail Sales", "impact": "high"},
        ],
        "timezone": "US/Eastern",
    }
    calendar_file = tmp_path / "economic_calendar_2026.json"
    calendar_file.write_text(json.dumps(calendar_data))

    sentinel = NewsSentinel.__new__(NewsSentinel)
    sentinel._vix_cache = -1.0
    sentinel._vix_cache_time = None
    # Load calendar from the tmp file
    sentinel._calendar = calendar_data["events"]
    sentinel._calendar_tz = "US/Eastern"
    return sentinel


@pytest.fixture
def sentinel_empty() -> NewsSentinel:
    """Return a NewsSentinel with no calendar events."""
    sentinel = NewsSentinel.__new__(NewsSentinel)
    sentinel._calendar = []
    sentinel._calendar_tz = "US/Eastern"
    sentinel._vix_cache = -1.0
    sentinel._vix_cache_time = None
    return sentinel


# ──────────────────────────────────────────────────────────────
# Calendar loading
# ──────────────────────────────────────────────────────────────


class TestCalendarLoading:
    def test_loads_real_calendar_file(self):
        """NewsSentinel should load the data/economic_calendar_2026.json file."""
        sentinel = NewsSentinel()
        assert len(sentinel._calendar) > 0

    def test_loaded_events_have_required_keys(self):
        sentinel = NewsSentinel()
        for event in sentinel._calendar[:5]:
            assert "date" in event
            assert "time" in event
            assert "name" in event
            assert "impact" in event

    def test_missing_calendar_file_gracefully_degrades(self, tmp_path, monkeypatch):
        """If the calendar file is not found, the calendar should be empty."""
        import core.news_sentinel as ns_module
        monkeypatch.setattr(ns_module, "_CALENDAR_PATH", str(tmp_path / "nonexistent.json"))
        sentinel = NewsSentinel()
        assert sentinel._calendar == []

    def test_corrupt_json_gracefully_degrades(self, tmp_path, monkeypatch):
        """If the calendar file is corrupt JSON, the calendar should be empty."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        import core.news_sentinel as ns_module
        monkeypatch.setattr(ns_module, "_CALENDAR_PATH", str(bad_file))
        sentinel = NewsSentinel()
        assert sentinel._calendar == []


# ──────────────────────────────────────────────────────────────
# Event impact classification
# ──────────────────────────────────────────────────────────────


class TestEventImpactClassification:
    def test_fomc_is_high_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("FOMC Rate Decision") == "high"

    def test_cpi_is_high_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("CPI") == "high"

    def test_nfp_is_high_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("Non-Farm Payrolls") == "high"

    def test_gdp_is_high_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("GDP Q1 Advance") == "high"

    def test_jackson_hole_is_high_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("Jackson Hole") == "high"

    def test_pce_is_high_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("Core PCE") == "high"

    def test_retail_sales_is_high_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("Retail Sales") == "high"

    def test_durable_goods_is_medium_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("Durable Goods") == "medium"

    def test_unknown_event_is_low_impact(self, sentinel_empty):
        assert sentinel_empty._event_impact("Some Random Event") == "low"


# ──────────────────────────────────────────────────────────────
# VIX regime classification
# ──────────────────────────────────────────────────────────────


class TestVixRegime:
    def test_vix_below_15_is_calm(self):
        assert NewsSentinel._vix_regime(12.5) == "calm"

    def test_vix_at_boundary_15_is_elevated(self):
        assert NewsSentinel._vix_regime(15.0) == "elevated"

    def test_vix_18_is_elevated(self):
        assert NewsSentinel._vix_regime(18.0) == "elevated"

    def test_vix_20_is_fear(self):
        assert NewsSentinel._vix_regime(20.0) == "fear"

    def test_vix_28_is_fear(self):
        assert NewsSentinel._vix_regime(28.0) == "fear"

    def test_vix_30_is_panic(self):
        assert NewsSentinel._vix_regime(30.0) == "panic"

    def test_vix_40_is_panic(self):
        assert NewsSentinel._vix_regime(40.0) == "panic"

    def test_vix_unknown_negative(self):
        assert NewsSentinel._vix_regime(-1.0) == "unknown"


# ──────────────────────────────────────────────────────────────
# Size modifier calculation
# ──────────────────────────────────────────────────────────────


class TestSizeModifier:
    def test_calm_vix_no_event_full_size(self, sentinel_empty):
        risk_level, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=-1, event_impact="low", vix=12.0
        )
        assert size_mod == 1.0
        assert should_pause is False
        assert risk_level == "low"

    def test_elevated_vix_50_percent_size(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=-1, event_impact="low", vix=20.0
        )
        assert size_mod == 0.5
        assert should_pause is False

    def test_fear_vix_25_percent_size(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=-1, event_impact="low", vix=28.0
        )
        assert size_mod == 0.25
        assert should_pause is False

    def test_extreme_vix_pauses_trading(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=-1, event_impact="low", vix=40.0
        )
        assert size_mod == 0.0
        assert should_pause is True

    def test_high_event_in_20min_pauses(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=20, event_impact="high", vix=14.0
        )
        assert size_mod == 0.0
        assert should_pause is True

    def test_high_event_in_45min_reduces_to_25pct(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=45, event_impact="high", vix=14.0
        )
        assert size_mod == 0.25
        assert should_pause is False

    def test_high_event_30min_ago_reduces_to_25pct(self, sentinel_empty):
        # minutes_to_event is negative when event has passed
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=-15, event_impact="high", vix=14.0
        )
        assert size_mod == 0.25
        assert should_pause is False

    def test_medium_event_in_10min_reduces_to_50pct(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=10, event_impact="medium", vix=14.0
        )
        assert size_mod == 0.5
        assert should_pause is False

    def test_vix_and_event_combined_takes_minimum(self, sentinel_empty):
        # VIX gives 50% but event in 45min forces 25%
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=45, event_impact="high", vix=20.0
        )
        # VIX=20 => 0.5, but high event in 45min caps at 0.25
        assert size_mod == 0.25

    def test_size_mod_zero_when_extreme_vix_and_event_imminent(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=10, event_impact="high", vix=38.0
        )
        assert size_mod == 0.0
        assert should_pause is True


# ──────────────────────────────────────────────────────────────
# Should-pause logic
# ──────────────────────────────────────────────────────────────


class TestShouldPause:
    def test_no_pause_under_normal_conditions(self, sentinel_empty):
        _, _, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=-1, event_impact="low", vix=13.0
        )
        assert should_pause is False

    def test_pause_when_vix_extreme(self, sentinel_empty):
        _, _, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=-1, event_impact="low", vix=36.0
        )
        assert should_pause is True

    def test_pause_when_high_event_imminent(self, sentinel_empty):
        _, _, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=5, event_impact="high", vix=14.0
        )
        assert should_pause is True

    def test_no_pause_when_high_event_is_far_future(self, sentinel_empty):
        _, _, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=90, event_impact="high", vix=14.0
        )
        assert should_pause is False

    def test_no_pause_for_medium_event(self, sentinel_empty):
        _, _, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=10, event_impact="medium", vix=14.0
        )
        assert should_pause is False


# ──────────────────────────────────────────────────────────────
# MarketContext.neutral()
# ──────────────────────────────────────────────────────────────


class TestMarketContextNeutral:
    def test_neutral_context_has_full_size(self):
        ctx = MarketContext.neutral()
        assert ctx.size_modifier == 1.0
        assert ctx.should_pause is False
        assert ctx.risk_level == "low"
        assert ctx.vix_regime == "calm"

    def test_neutral_context_has_no_events(self):
        ctx = MarketContext.neutral()
        assert ctx.upcoming_events == []
        assert ctx.minutes_to_next_event == -1


# ──────────────────────────────────────────────────────────────
# Event blackout window logic (proximity)
# ──────────────────────────────────────────────────────────────


class TestEventBlackoutWindow:
    def test_event_exactly_at_30min_boundary_pauses(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=0, event_impact="high", vix=14.0
        )
        assert should_pause is True
        assert size_mod == 0.0

    def test_event_at_31min_does_not_pause_but_reduces_size(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=31, event_impact="high", vix=14.0
        )
        assert should_pause is False
        assert size_mod == 0.25

    def test_event_at_61min_no_restriction(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=61, event_impact="high", vix=14.0
        )
        assert should_pause is False
        assert size_mod == 1.0  # no restriction beyond 60 min

    def test_event_31min_ago_no_restriction(self, sentinel_empty):
        _, size_mod, should_pause, _ = sentinel_empty._calculate_risk(
            minutes_to_event=-31, event_impact="high", vix=14.0
        )
        assert should_pause is False
        assert size_mod == 1.0  # beyond the 30-min settling window


# ──────────────────────────────────────────────────────────────
# get_market_context integration
# ──────────────────────────────────────────────────────────────


class TestGetMarketContext:
    def test_returns_neutral_when_no_calendar_events(self):
        """With an empty calendar and no VIX, we should get a low-risk context."""
        sentinel = NewsSentinel.__new__(NewsSentinel)
        sentinel._calendar = []
        sentinel._calendar_tz = "US/Eastern"
        sentinel._vix_cache = -1.0
        sentinel._vix_cache_time = None

        ctx = asyncio.get_event_loop().run_until_complete(
            sentinel.get_market_context(connection_manager=None)
        )
        assert ctx.should_pause is False
        assert ctx.size_modifier == 1.0
        assert ctx.risk_level == "low"

    def test_pauses_when_high_event_imminent(self):
        """Inject an event 10 minutes from now and verify the context pauses."""
        from zoneinfo import ZoneInfo

        now_eastern = datetime.now(tz=ZoneInfo("America/New_York"))
        event_time = now_eastern + timedelta(minutes=10)

        sentinel = NewsSentinel.__new__(NewsSentinel)
        sentinel._calendar = [
            {
                "date": event_time.strftime("%Y-%m-%d"),
                "time": event_time.strftime("%H:%M"),
                "name": "FOMC Rate Decision",
                "impact": "high",
            }
        ]
        sentinel._calendar_tz = "US/Eastern"
        sentinel._vix_cache = 14.0
        sentinel._vix_cache_time = datetime.utcnow()

        ctx = asyncio.get_event_loop().run_until_complete(
            sentinel.get_market_context(connection_manager=None)
        )
        assert ctx.should_pause is True
        assert ctx.size_modifier == 0.0

    def test_reduces_size_when_vix_elevated(self):
        """With elevated VIX and no imminent events, size should be reduced."""
        sentinel = NewsSentinel.__new__(NewsSentinel)
        sentinel._calendar = []
        sentinel._calendar_tz = "US/Eastern"
        sentinel._vix_cache = 22.0  # in the fear zone
        sentinel._vix_cache_time = datetime.utcnow()

        ctx = asyncio.get_event_loop().run_until_complete(
            sentinel.get_market_context(connection_manager=None)
        )
        assert ctx.should_pause is False
        assert ctx.size_modifier < 1.0
        assert ctx.vix_regime in ("fear", "elevated")

    def test_returns_neutral_on_unexpected_error(self):
        """If something goes wrong internally, should return neutral context."""
        sentinel = NewsSentinel.__new__(NewsSentinel)
        # Deliberately corrupt the calendar to trigger an exception path
        sentinel._calendar = None  # type: ignore[assignment]
        sentinel._calendar_tz = "US/Eastern"
        sentinel._vix_cache = -1.0
        sentinel._vix_cache_time = None

        ctx = asyncio.get_event_loop().run_until_complete(
            sentinel.get_market_context(connection_manager=None)
        )
        # Should gracefully return neutral rather than raise
        assert isinstance(ctx, MarketContext)
        assert ctx.should_pause is False
        assert ctx.size_modifier == 1.0
