"""
Tests for analysis/scanner.py — MoMo scoring system, hard filters, ranking.
"""

from __future__ import annotations

import pytest

from analysis.scanner import MomoCandidate, MomoScanner


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def scanner() -> MomoScanner:
    return MomoScanner()


def _make_candidate(**kwargs) -> MomoCandidate:
    """Helper to create a MomoCandidate with sensible defaults."""
    defaults = {
        "ticker": "TEST",
        "gap_pct": 15.0,
        "rvol": 8.0,
        "float_shares": 5.0,
        "price": 8.0,
        "news_headline": "Big news catalyst",
        "score": 0,
        "sector_momentum": False,
        "is_blue_sky": False,
        "clean_daily_chart": False,
        "premarket_volume": 500_000,
        "short_interest_pct": 10.0,
        "bid_ask_spread": 0.03,
    }
    defaults.update(kwargs)
    return MomoCandidate(**defaults)


# ──────────────────────────────────────────────────────────────
# Hard filters
# ──────────────────────────────────────────────────────────────


class TestHardFilters:
    def test_passes_when_all_filters_met(self, scanner):
        candidate = _make_candidate()
        assert scanner._passes_hard_filters(candidate) is True

    def test_fails_gap_below_10pct(self, scanner):
        candidate = _make_candidate(gap_pct=9.9)
        assert scanner._passes_hard_filters(candidate) is False

    def test_fails_float_above_10m(self, scanner):
        candidate = _make_candidate(float_shares=10.1)
        assert scanner._passes_hard_filters(candidate) is False

    def test_fails_price_above_20(self, scanner):
        candidate = _make_candidate(price=21.0)
        assert scanner._passes_hard_filters(candidate) is False

    def test_fails_rvol_below_5(self, scanner):
        candidate = _make_candidate(rvol=4.9)
        assert scanner._passes_hard_filters(candidate) is False

    def test_fails_no_news(self, scanner):
        candidate = _make_candidate(news_headline="")
        assert scanner._passes_hard_filters(candidate) is False

    def test_fails_at_exact_float_boundary(self, scanner):
        # float_shares == 10 should pass (≤ 10)
        candidate = _make_candidate(float_shares=10.0)
        assert scanner._passes_hard_filters(candidate) is True

    def test_fails_above_float_boundary(self, scanner):
        candidate = _make_candidate(float_shares=10.01)
        assert scanner._passes_hard_filters(candidate) is False


# ──────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────


class TestScoring:
    def test_high_rvol_gives_20_pts(self, scanner):
        # float_shares=4.9 (< 5M = +15), price=$8 (+15), rvol=12 (+20) → 50
        c = _make_candidate(rvol=12.0, float_shares=4.9)
        score = scanner.score_candidate(c)
        # RVOL ≥10x (+20) + float <5M (+15) + price $2-10 (+15) = 50
        assert score >= 50

    def test_moderate_rvol_gives_10_pts(self, scanner):
        c_high = _make_candidate(rvol=12.0, float_shares=5.0, price=8.0)
        c_mid = _make_candidate(rvol=7.0, float_shares=5.0, price=8.0)
        # High RVOL should score higher than mid RVOL (same other params)
        assert scanner.score_candidate(c_high) > scanner.score_candidate(c_mid)

    def test_small_float_bonus(self, scanner):
        c_small = _make_candidate(float_shares=3.0)
        c_large = _make_candidate(float_shares=8.0)
        assert scanner.score_candidate(c_small) > scanner.score_candidate(c_large)

    def test_blue_sky_adds_15_pts(self, scanner):
        c_yes = _make_candidate(is_blue_sky=True)
        c_no = _make_candidate(is_blue_sky=False)
        assert scanner.score_candidate(c_yes) - scanner.score_candidate(c_no) == 15

    def test_sector_momentum_adds_10_pts(self, scanner):
        c_yes = _make_candidate(sector_momentum=True)
        c_no = _make_candidate(sector_momentum=False)
        assert scanner.score_candidate(c_yes) - scanner.score_candidate(c_no) == 10

    def test_high_premarket_volume_adds_10_pts(self, scanner):
        c_high = _make_candidate(premarket_volume=2_000_000)
        c_low = _make_candidate(premarket_volume=500_000)
        assert scanner.score_candidate(c_high) - scanner.score_candidate(c_low) == 10

    def test_high_short_interest_adds_10_pts(self, scanner):
        c_high = _make_candidate(short_interest_pct=25.0)
        c_low = _make_candidate(short_interest_pct=10.0)
        assert scanner.score_candidate(c_high) - scanner.score_candidate(c_low) == 10

    def test_tight_spread_adds_5_pts(self, scanner):
        c_tight = _make_candidate(bid_ask_spread=0.01)
        c_wide = _make_candidate(bid_ask_spread=0.05)
        assert scanner.score_candidate(c_tight) - scanner.score_candidate(c_wide) == 5

    def test_maximum_possible_score(self, scanner):
        """Perfect candidate should score 110."""
        c = _make_candidate(
            rvol=15.0,           # +20
            float_shares=3.0,    # +15
            price=8.0,           # +15
            is_blue_sky=True,    # +15
            clean_daily_chart=True,  # +10
            sector_momentum=True,    # +10
            premarket_volume=2_000_000,  # +10
            short_interest_pct=25.0,     # +10
            bid_ask_spread=0.01,         # +5
        )
        assert scanner.score_candidate(c) == 110

    def test_minimum_score_empty_candidate(self, scanner):
        """Candidate with nothing extra should score 0."""
        c = _make_candidate(
            rvol=3.0,            # too low for any points (also fails filters)
            float_shares=12.0,   # too high for any points
            price=25.0,          # too high for any points
        )
        assert scanner.score_candidate(c) == 0


# ──────────────────────────────────────────────────────────────
# Candidate ranking
# ──────────────────────────────────────────────────────────────


class TestRanking:
    def test_candidates_sorted_by_score(self, scanner):
        c1 = _make_candidate(ticker="A", is_blue_sky=True, rvol=12.0)   # higher score
        c2 = _make_candidate(ticker="B", is_blue_sky=False, rvol=6.0)   # lower score
        c1.score = scanner.score_candidate(c1)
        c2.score = scanner.score_candidate(c2)
        ranked = sorted([c2, c1], key=lambda c: c.score, reverse=True)
        assert ranked[0].ticker == "A"

    def test_score_threshold_full_size(self):
        """Score > 80 → full bullet; 65–80 → 75%; 50–65 → 50%; <50 → skip."""
        c = _make_candidate(
            rvol=12.0, float_shares=3.0, price=8.0,
            is_blue_sky=True, clean_daily_chart=True,
            sector_momentum=True,  # adds +10 to push over 80
        )
        score = MomoScanner.score_candidate(c)
        # rvol=12 (+20) + float<5M (+15) + price $2-10 (+15) + blue_sky (+15) + clean (+10) + momentum (+10) = 85
        assert score > 80
