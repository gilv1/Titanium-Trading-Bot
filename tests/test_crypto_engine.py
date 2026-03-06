"""
Tests for engines/crypto_engine.py — tick rounding, dynamic allocation, pre-RTH cutoff.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from engines.crypto_engine import CryptoEngine, CRYPTO_TICK_SIZE, _round_to_tick


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_engine() -> CryptoEngine:
    """Create a CryptoEngine with all dependencies mocked out."""
    connection = MagicMock()
    brain = MagicMock()
    reto = MagicMock()
    reto.capital = 500.0
    risk = MagicMock()
    risk.has_correlation_conflict.return_value = False
    return CryptoEngine(
        connection_manager=connection,
        brain=brain,
        reto_tracker=reto,
        risk_manager=risk,
        telegram=None,
    )


# ──────────────────────────────────────────────────────────────
# Fix 1: Tick rounding (_round_to_tick helper)
# ──────────────────────────────────────────────────────────────


class TestTickRounding:
    def test_btc_rounds_to_quarter(self):
        """BTC prices must land on $0.25 increments."""
        assert _round_to_tick(70482.72, "BTC") == 70482.75
        assert _round_to_tick(70482.37, "BTC") == 70482.25
        assert _round_to_tick(70482.50, "BTC") == 70482.50
        assert _round_to_tick(70482.13, "BTC") == 70482.25

    def test_btc_sl_from_issue_report(self):
        """The exact price that triggered Warning 110 must round to a valid tick."""
        # SL was sent as 67077.76 — should round to 67077.75
        assert _round_to_tick(67077.76, "BTC") == 67077.75

    def test_btc_tp_from_issue_report(self):
        """The exact TP that triggered Warning 110 must round to a valid tick."""
        # TP was sent as 70482.72 — should round to 70482.75
        assert _round_to_tick(70482.72, "BTC") == 70482.75

    def test_eth_rounds_to_cent(self):
        """ETH prices must land on $0.01 increments."""
        assert _round_to_tick(3250.123, "ETH") == 3250.12
        assert _round_to_tick(3250.127, "ETH") == 3250.13
        assert _round_to_tick(3250.999, "ETH") == 3251.00

    def test_unknown_symbol_defaults_to_cent(self):
        """Unknown symbols default to $0.01 tick size."""
        assert _round_to_tick(100.123, "XRP") == 100.12
        assert _round_to_tick(100.127, "XRP") == 100.13

    def test_btc_tick_size_is_quarter(self):
        assert CRYPTO_TICK_SIZE["BTC"] == 0.25

    def test_eth_tick_size_is_cent(self):
        assert CRYPTO_TICK_SIZE["ETH"] == 0.01

    def test_round_to_tick_exact_boundaries(self):
        """Prices already on valid ticks should be unchanged."""
        assert _round_to_tick(70000.00, "BTC") == 70000.00
        assert _round_to_tick(70000.25, "BTC") == 70000.25
        assert _round_to_tick(70000.50, "BTC") == 70000.50
        assert _round_to_tick(70000.75, "BTC") == 70000.75


# ──────────────────────────────────────────────────────────────
# Fix 2: Dynamic allocation (_get_effective_allocation)
# ──────────────────────────────────────────────────────────────


class TestEffectiveAllocation:
    def _engine_with_time(self, weekday: int, hour: int, minute: int) -> CryptoEngine:
        engine = _make_engine()
        # Patch datetime.now inside crypto_engine to control the time
        mock_now = MagicMock()
        mock_now.weekday.return_value = weekday
        mock_now.hour = hour
        mock_now.minute = minute
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            alloc = engine._get_effective_allocation()
        return alloc

    def test_weekend_returns_70_pct(self):
        """Saturday and Sunday → 70% allocation."""
        engine = _make_engine()
        for day in (5, 6):  # Saturday=5, Sunday=6
            mock_now = MagicMock()
            mock_now.weekday.return_value = day
            mock_now.hour = 12
            mock_now.minute = 0
            with patch("engines.crypto_engine.datetime") as mock_dt:
                mock_dt.now.return_value = mock_now
                mock_dt.utcnow.return_value = datetime.utcnow()
                assert engine._get_effective_allocation() == 0.70

    def test_weekday_market_hours_returns_30_pct(self):
        """Monday–Friday 9:30 AM–4:00 PM ET → 30% allocation."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 1  # Tuesday
        mock_now.hour = 10
        mock_now.minute = 0
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.30

    def test_weekday_off_hours_morning_returns_70_pct(self):
        """Monday–Friday 6:00 AM ET (before 9:30 AM) → 70% allocation."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 2  # Wednesday
        mock_now.hour = 6
        mock_now.minute = 0
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.70

    def test_weekday_off_hours_evening_returns_70_pct(self):
        """Monday–Friday 6:00 PM ET (after 4:00 PM) → 70% allocation."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 3  # Thursday
        mock_now.hour = 18
        mock_now.minute = 0
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.70

    def test_market_hours_boundary_930_is_market(self):
        """9:30 AM exactly (minute=570) is market hours → 30%."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 0  # Monday
        mock_now.hour = 9
        mock_now.minute = 30
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.30

    def test_market_hours_boundary_4pm_is_off_hours(self):
        """4:00 PM exactly (minute=960) is off-hours → 70%."""
        engine = _make_engine()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 0  # Monday
        mock_now.hour = 16
        mock_now.minute = 0
        with patch("engines.crypto_engine.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.utcnow.return_value = datetime.utcnow()
            assert engine._get_effective_allocation() == 0.70
