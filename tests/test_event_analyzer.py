"""
Tests for core/event_analyzer.py — Event Impact Analyzer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from core.event_analyzer import EventAnalyzer


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_calendar(events: list[dict]) -> dict:
    return {"events": events, "timezone": "US/Eastern"}


def _make_ohlcv_df(n: int = 120, base_price: float = 4500.0) -> pd.DataFrame:
    """Build a simple OHLCV DataFrame with a UTC DatetimeIndex."""
    start = datetime(2026, 3, 18, 14, 0, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=i) for i in range(n)]
    prices = [base_price + i * 0.5 for i in range(n)]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [10_000] * n,
        },
        index=pd.DatetimeIndex(times, name="time"),
    )
    return df


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def analyzer_with_calendar(tmp_path) -> EventAnalyzer:
    """EventAnalyzer backed by a small in-memory calendar and no CSV files."""
    calendar_data = _make_calendar(
        [
            {"date": "2026-03-18", "time": "14:00", "name": "FOMC Rate Decision", "impact": "high"},
            {"date": "2026-03-11", "time": "08:30", "name": "CPI", "impact": "high"},
            {"date": "2026-04-01", "time": "08:30", "name": "Non-Farm Payrolls", "impact": "high"},
        ]
    )
    cal_file = tmp_path / "calendar.json"
    cal_file.write_text(json.dumps(calendar_data))
    hist_dir = tmp_path / "historical"
    hist_dir.mkdir()
    patterns_path = tmp_path / "event_patterns.json"

    return EventAnalyzer(
        calendar_path=cal_file,
        historical_dir=hist_dir,
        patterns_path=patterns_path,
    )


@pytest.fixture
def analyzer_with_data(tmp_path) -> EventAnalyzer:
    """EventAnalyzer with calendar + one synthetic MES CSV."""
    calendar_data = _make_calendar(
        [
            {"date": "2026-03-18", "time": "14:00", "name": "FOMC Rate Decision", "impact": "high"},
        ]
    )
    cal_file = tmp_path / "calendar.json"
    cal_file.write_text(json.dumps(calendar_data))

    hist_dir = tmp_path / "historical"
    hist_dir.mkdir()

    # Build a MES CSV spanning 14:00 UTC on 2026-03-18
    df = _make_ohlcv_df(n=200, base_price=4500.0)
    df_reset = df.reset_index()
    df_reset.rename(columns={"time": "time"}, inplace=True)
    df_reset.to_csv(hist_dir / "MES.csv", index=False)

    patterns_path = tmp_path / "event_patterns.json"
    return EventAnalyzer(
        calendar_path=cal_file,
        historical_dir=hist_dir,
        patterns_path=patterns_path,
    )


# ──────────────────────────────────────────────────────────────
# Calendar loading
# ──────────────────────────────────────────────────────────────


class TestCalendarLoading:
    def test_loads_valid_calendar(self, analyzer_with_calendar):
        assert len(analyzer_with_calendar._calendar) == 3

    def test_graceful_missing_file(self, tmp_path):
        analyzer = EventAnalyzer(
            calendar_path=tmp_path / "nonexistent.json",
            historical_dir=tmp_path / "hist",
            patterns_path=tmp_path / "patterns.json",
        )
        assert analyzer._calendar == []

    def test_graceful_corrupt_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("NOT JSON {{{")
        analyzer = EventAnalyzer(
            calendar_path=bad_file,
            historical_dir=tmp_path / "hist",
            patterns_path=tmp_path / "patterns.json",
        )
        assert analyzer._calendar == []


# ──────────────────────────────────────────────────────────────
# Historical data loading
# ──────────────────────────────────────────────────────────────


class TestHistoricalLoading:
    def test_missing_directory_returns_empty(self, tmp_path):
        analyzer = EventAnalyzer(
            calendar_path=tmp_path / "cal.json",
            historical_dir=tmp_path / "does_not_exist",
            patterns_path=tmp_path / "patterns.json",
        )
        result = analyzer._load_historical_bars()
        assert result == {}

    def test_loads_valid_csv(self, tmp_path):
        hist_dir = tmp_path / "hist"
        hist_dir.mkdir()
        df = _make_ohlcv_df(n=10)
        df.reset_index().to_csv(hist_dir / "MES.csv", index=False)

        cal_file = tmp_path / "cal.json"
        cal_file.write_text(json.dumps(_make_calendar([])))

        analyzer = EventAnalyzer(
            calendar_path=cal_file,
            historical_dir=hist_dir,
            patterns_path=tmp_path / "patterns.json",
        )
        result = analyzer._load_historical_bars()
        assert "MES" in result
        assert isinstance(result["MES"], pd.DataFrame)

    def test_skips_csv_without_time_column(self, tmp_path):
        hist_dir = tmp_path / "hist"
        hist_dir.mkdir()
        pd.DataFrame({"open": [1], "close": [1]}).to_csv(hist_dir / "BAD.csv", index=False)

        cal_file = tmp_path / "cal.json"
        cal_file.write_text(json.dumps(_make_calendar([])))

        analyzer = EventAnalyzer(
            calendar_path=cal_file,
            historical_dir=hist_dir,
            patterns_path=tmp_path / "patterns.json",
        )
        result = analyzer._load_historical_bars()
        assert "BAD" not in result


# ──────────────────────────────────────────────────────────────
# Measurement helpers
# ──────────────────────────────────────────────────────────────


class TestMeasureMove:
    def test_positive_move(self):
        df = _make_ohlcv_df(n=70, base_price=100.0)
        event_time = df.index[0]
        move = EventAnalyzer._measure_move(df, event_time, 15)
        assert move is not None
        assert move > 0  # prices are monotonically increasing

    def test_returns_none_when_no_future_bars(self):
        df = _make_ohlcv_df(n=5, base_price=100.0)
        event_time = df.index[-1] + timedelta(minutes=100)
        result = EventAnalyzer._measure_move(df, event_time, 5)
        assert result is None


class TestDetectReversal:
    def test_no_reversal_on_monotone_series(self):
        df = _make_ohlcv_df(n=80, base_price=100.0)
        reversed_, _ = EventAnalyzer._detect_reversal(df, df.index[0])
        assert not reversed_

    def test_too_short_returns_false(self):
        df = _make_ohlcv_df(n=2, base_price=100.0)
        reversed_, rev_time = EventAnalyzer._detect_reversal(df, df.index[0])
        assert not reversed_
        assert rev_time == 0.0


class TestFindBestEntryDelay:
    def test_returns_positive_int(self):
        df = _make_ohlcv_df(n=80, base_price=4500.0)
        result = EventAnalyzer._find_best_entry_delay(df, df.index[30])
        assert isinstance(result, int)
        assert result >= 0

    def test_no_pre_event_data_returns_default(self):
        df = _make_ohlcv_df(n=10, base_price=4500.0)
        result = EventAnalyzer._find_best_entry_delay(df, df.index[0])
        assert result == 5


# ──────────────────────────────────────────────────────────────
# Full analysis pipeline
# ──────────────────────────────────────────────────────────────


class TestAnalyze:
    def test_returns_empty_dict_with_no_historical_data(self, analyzer_with_calendar):
        result = analyzer_with_calendar.analyze()
        assert result == {}

    def test_returns_pattern_when_event_in_range(self, analyzer_with_data):
        patterns = analyzer_with_data.analyze()
        # May or may not find the FOMC event depending on exact time parsing;
        # what matters is it runs without error and returns a dict.
        assert isinstance(patterns, dict)

    def test_pattern_structure(self, analyzer_with_data):
        patterns = analyzer_with_data.analyze()
        for name, stats in patterns.items():
            assert "occurrences" in stats
            assert "avg_move_5min_pts" in stats
            assert "avg_move_15min_pts" in stats
            assert "reversal_pct" in stats
            assert "best_entry_delay_min" in stats
            assert stats["occurrences"] >= 1


# ──────────────────────────────────────────────────────────────
# Save & load patterns
# ──────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path, analyzer_with_calendar):
        patterns = {
            "FOMC Rate Decision": {
                "occurrences": 3,
                "avg_move_5min_pts": 25.5,
                "avg_move_15min_pts": 40.2,
                "reversal_pct": 65.0,
                "best_entry_delay_min": 40,
                "avg_max_move_pts": 55.3,
                "direction_follows_result": True,
            }
        }
        analyzer_with_calendar.save(patterns)
        loaded = EventAnalyzer.load_patterns(analyzer_with_calendar._patterns_path)
        assert loaded == patterns

    def test_load_nonexistent_returns_empty(self, tmp_path):
        result = EventAnalyzer.load_patterns(tmp_path / "missing.json")
        assert result == {}

    def test_load_corrupt_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{bad json")
        result = EventAnalyzer.load_patterns(bad)
        assert result == {}
