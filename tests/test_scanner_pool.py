"""
Tests for core/scanner_pool.py — ScannerPool rotation, rate-limit reset,
error handling, and MarketContext confirmation logic.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.scanner_pool import MarketContext, ScannerPool, ScannerStatus


# ──────────────────────────────────────────────────────────────
# MarketContext confirmation logic
# ──────────────────────────────────────────────────────────────


class TestMarketContextBullish:
    """MarketContext.is_bullish_context should pass only when market looks healthy for LONG."""

    def _ctx(self, change_1h=0.5, change_24h=1.0, volume_24h=5e9, rsi=None) -> MarketContext:
        return MarketContext(
            price=50000.0,
            volume_24h=volume_24h,
            change_1h=change_1h,
            change_24h=change_24h,
            rsi=rsi,
        )

    def test_healthy_market_is_bullish(self):
        assert self._ctx().is_bullish_context is True

    def test_freefall_24h_rejected(self):
        """24h drop worse than -3% → not bullish."""
        assert self._ctx(change_24h=-3.1).is_bullish_context is False

    def test_exactly_minus_3_24h_is_rejected(self):
        """Boundary: -3.0% exactly fails (> not >=)."""
        assert self._ctx(change_24h=-3.0).is_bullish_context is False

    def test_negative_1h_momentum_rejected(self):
        """1h drop worse than -1.5% → not bullish."""
        assert self._ctx(change_1h=-1.6).is_bullish_context is False

    def test_overbought_rsi_rejected(self):
        """RSI >= 72 → not bullish (overbought)."""
        assert self._ctx(rsi=72.0).is_bullish_context is False
        assert self._ctx(rsi=80.0).is_bullish_context is False

    def test_rsi_just_below_72_allowed(self):
        """RSI < 72 passes."""
        assert self._ctx(rsi=71.9).is_bullish_context is True

    def test_no_rsi_skips_rsi_check(self):
        """When RSI is None the check is simply skipped."""
        assert self._ctx(rsi=None).is_bullish_context is True

    def test_low_volume_rejected(self):
        """Volume below $1 B → dead market, not bullish."""
        assert self._ctx(volume_24h=999_999_999).is_bullish_context is False

    def test_exact_1b_volume_does_not_pass(self):
        """The volume check requires > 1_000_000_000, so exactly $1B does NOT pass."""
        assert self._ctx(volume_24h=1_000_000_000).is_bullish_context is False

    def test_above_1b_volume_passes(self):
        """Volume strictly above $1B passes."""
        assert self._ctx(volume_24h=1_000_000_001).is_bullish_context is True

    def test_multiple_failures_all_rejected(self):
        assert self._ctx(change_24h=-5.0, change_1h=-2.0, volume_24h=500_000_000).is_bullish_context is False


class TestMarketContextBearish:
    """MarketContext.is_bearish_context should pass only when market looks healthy for SHORT."""

    def _ctx(self, change_1h=-0.5, change_24h=-1.0, volume_24h=5e9, rsi=None) -> MarketContext:
        return MarketContext(
            price=50000.0,
            volume_24h=volume_24h,
            change_1h=change_1h,
            change_24h=change_24h,
            rsi=rsi,
        )

    def test_healthy_market_is_bearish(self):
        assert self._ctx().is_bearish_context is True

    def test_strong_uptrend_24h_rejected(self):
        """24h gain above 3% → not bearish."""
        assert self._ctx(change_24h=3.1).is_bearish_context is False

    def test_strong_1h_uptrend_rejected(self):
        """1h gain above 1.5% → not bearish."""
        assert self._ctx(change_1h=1.6).is_bearish_context is False

    def test_oversold_rsi_rejected(self):
        """RSI <= 28 → not bearish (oversold)."""
        assert self._ctx(rsi=28.0).is_bearish_context is False
        assert self._ctx(rsi=15.0).is_bearish_context is False

    def test_rsi_just_above_28_allowed(self):
        """RSI > 28 passes."""
        assert self._ctx(rsi=28.1).is_bearish_context is True

    def test_no_rsi_skips_check(self):
        assert self._ctx(rsi=None).is_bearish_context is True

    def test_low_volume_rejected(self):
        assert self._ctx(volume_24h=500_000_000).is_bearish_context is False


# ──────────────────────────────────────────────────────────────
# ScannerStatus / rotation logic
# ──────────────────────────────────────────────────────────────


class TestScannerPoolRotation:
    """ScannerPool._get_best_scanner() picks the best available scanner."""

    def _make_pool(self) -> ScannerPool:
        """ScannerPool without a real HTTP client."""
        pool = ScannerPool.__new__(ScannerPool)
        pool._scanners = [
            ScannerStatus("coinlore", calls_per_min=900),
            ScannerStatus("coincap", calls_per_min=450),
            ScannerStatus("coingecko", calls_per_min=25),
        ]
        pool._client = MagicMock()
        return pool

    def test_first_scanner_returned_when_fresh(self):
        pool = self._make_pool()
        scanner = pool._get_best_scanner()
        assert scanner is not None
        assert scanner.name == "coinlore"

    def test_exhausted_scanner_skipped(self):
        """When the primary scanner has used all its capacity, fall through to next."""
        pool = self._make_pool()
        pool._scanners[0].used_this_minute = 900  # coinlore exhausted
        scanner = pool._get_best_scanner()
        assert scanner is not None
        assert scanner.name == "coincap"

    def test_all_exhausted_returns_none(self):
        pool = self._make_pool()
        for s in pool._scanners:
            s.used_this_minute = s.calls_per_min
        assert pool._get_best_scanner() is None

    def test_disabled_scanner_is_skipped(self):
        """Scanner with 3+ consecutive errors and active disabled_until is skipped."""
        pool = self._make_pool()
        pool._scanners[0].consecutive_errors = 3
        pool._scanners[0].disabled_until = time.time() + 9999  # far future
        scanner = pool._get_best_scanner()
        assert scanner is not None
        assert scanner.name == "coincap"

    def test_disabled_scanner_reenabled_after_cooldown(self):
        """Scanner is re-enabled once disabled_until has passed."""
        pool = self._make_pool()
        pool._scanners[0].consecutive_errors = 3
        pool._scanners[0].disabled_until = time.time() - 1  # in the past
        scanner = pool._get_best_scanner()
        assert scanner is not None
        assert scanner.name == "coinlore"
        assert scanner.consecutive_errors == 0  # reset on re-enable

    def test_rate_limit_counter_resets_after_60s(self):
        """Counter resets when > 60 s have elapsed since last_reset."""
        pool = self._make_pool()
        coinlore = pool._scanners[0]
        coinlore.used_this_minute = 900  # exhausted
        coinlore.last_reset = time.time() - 61  # 61 seconds ago

        scanner = pool._get_best_scanner()
        # After reset, used_this_minute is 0 again → coinlore should be selected
        assert scanner is not None
        assert scanner.name == "coinlore"
        assert coinlore.used_this_minute == 0

    def test_counter_not_reset_within_60s(self):
        """Counter should NOT reset if < 60 s have elapsed."""
        pool = self._make_pool()
        coinlore = pool._scanners[0]
        coinlore.used_this_minute = 900  # exhausted
        coinlore.last_reset = time.time() - 30  # only 30 seconds ago

        scanner = pool._get_best_scanner()
        # coinlore still exhausted → falls through to coincap
        assert scanner is not None
        assert scanner.name == "coincap"


# ──────────────────────────────────────────────────────────────
# ScannerPool.get_market_context() — rotation on error
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestScannerPoolGetMarketContext:
    """Integration-style tests for get_market_context() using mocked fetch methods."""

    def _make_pool(self) -> ScannerPool:
        pool = ScannerPool.__new__(ScannerPool)
        pool._scanners = [
            ScannerStatus("coinlore", calls_per_min=900),
            ScannerStatus("coincap", calls_per_min=450),
            ScannerStatus("coingecko", calls_per_min=25),
        ]
        pool._client = MagicMock()
        return pool

    async def test_returns_context_from_primary_scanner(self):
        pool = self._make_pool()
        expected = MarketContext(price=50000.0, volume_24h=5e9, change_1h=0.5, change_24h=1.0)
        pool._fetch_coinlore = AsyncMock(return_value=expected)

        result = await pool.get_market_context("BTC")

        assert result is expected
        pool._fetch_coinlore.assert_awaited_once_with("BTC")

    async def test_rotates_to_next_on_error(self):
        """When primary scanner errors 3 times it gets disabled, then falls back."""
        pool = self._make_pool()
        expected = MarketContext(price=50000.0, volume_24h=5e9, change_1h=0.5, change_24h=1.0)
        pool._fetch_coinlore = AsyncMock(side_effect=Exception("network error"))
        pool._fetch_coincap = AsyncMock(return_value=expected)

        result = await pool.get_market_context("BTC")

        assert result is expected
        # coinlore should accumulate 3 consecutive errors and then be disabled
        assert pool._scanners[0].consecutive_errors == 3
        assert pool._scanners[0].disabled_until > time.time()

    async def test_returns_none_when_all_scanners_fail(self):
        """Returns None when every scanner raises an exception."""
        pool = self._make_pool()
        pool._fetch_coinlore = AsyncMock(side_effect=Exception("fail"))
        pool._fetch_coincap = AsyncMock(side_effect=Exception("fail"))
        pool._fetch_coingecko = AsyncMock(side_effect=Exception("fail"))

        result = await pool.get_market_context("BTC")

        assert result is None

    async def test_increments_used_counter(self):
        pool = self._make_pool()
        expected = MarketContext(price=50000.0, volume_24h=5e9, change_1h=0.0, change_24h=0.0)
        pool._fetch_coinlore = AsyncMock(return_value=expected)

        await pool.get_market_context("BTC")

        assert pool._scanners[0].used_this_minute == 1

    async def test_error_disables_scanner_for_120s(self):
        pool = self._make_pool()
        pool._fetch_coinlore = AsyncMock(side_effect=Exception("fail"))
        pool._fetch_coincap = AsyncMock(side_effect=Exception("fail"))
        pool._fetch_coingecko = AsyncMock(side_effect=Exception("fail"))

        before = time.time()
        await pool.get_market_context("BTC")
        after = time.time()

        for s in pool._scanners:
            assert s.consecutive_errors >= 1
            # disabled_until must be in the future (roughly 120s from now)
            assert s.disabled_until > before + 100
            assert s.disabled_until < after + 130

    async def test_resets_consecutive_errors_on_success(self):
        """A successful call resets the consecutive_errors counter."""
        pool = self._make_pool()
        pool._scanners[0].consecutive_errors = 2  # pre-existing errors
        expected = MarketContext(price=50000.0, volume_24h=5e9, change_1h=0.0, change_24h=0.0)
        pool._fetch_coinlore = AsyncMock(return_value=expected)

        await pool.get_market_context("BTC")

        assert pool._scanners[0].consecutive_errors == 0

    async def test_returns_none_when_all_exhausted(self):
        pool = self._make_pool()
        for s in pool._scanners:
            s.used_this_minute = s.calls_per_min

        result = await pool.get_market_context("BTC")

        assert result is None


# ──────────────────────────────────────────────────────────────
# HTTP mock tests for individual fetch methods
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFetchMethods:
    """Verify each API implementation parses responses correctly."""

    def _make_pool(self) -> ScannerPool:
        pool = ScannerPool.__new__(ScannerPool)
        pool._client = AsyncMock()
        pool._scanners = []
        return pool

    async def test_fetch_coincap_btc(self):
        pool = self._make_pool()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "priceUsd": "68000.50",
                "volumeUsd24Hr": "25000000000.0",
                "changePercent1Hr": "0.25",
                "changePercent24Hr": "1.5",
            }
        }
        pool._client.get = AsyncMock(return_value=mock_resp)

        ctx = await pool._fetch_coincap("BTC")

        assert ctx.price == pytest.approx(68000.50)
        assert ctx.volume_24h == pytest.approx(25e9)
        assert ctx.change_1h == pytest.approx(0.25)
        assert ctx.change_24h == pytest.approx(1.5)

    async def test_fetch_coincap_eth(self):
        pool = self._make_pool()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "priceUsd": "3200.00",
                "volumeUsd24Hr": "8000000000.0",
                "changePercent1Hr": None,
                "changePercent24Hr": "-0.5",
            }
        }
        pool._client.get = AsyncMock(return_value=mock_resp)

        ctx = await pool._fetch_coincap("ETH")

        assert ctx.price == pytest.approx(3200.0)
        assert ctx.change_1h == pytest.approx(0.0)  # None → 0.0

    async def test_fetch_coinlore_btc(self):
        pool = self._make_pool()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {
                "price_usd": "67500.00",
                "volume24": "22000000000",
                "percent_change_1h": "0.10",
                "percent_change_24h": "-0.80",
            }
        ]
        pool._client.get = AsyncMock(return_value=mock_resp)

        ctx = await pool._fetch_coinlore("BTC")

        assert ctx.price == pytest.approx(67500.0)
        assert ctx.volume_24h == pytest.approx(22e9)
        assert ctx.change_1h == pytest.approx(0.10)
        assert ctx.change_24h == pytest.approx(-0.80)

    async def test_fetch_coingecko_btc(self):
        pool = self._make_pool()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "bitcoin": {
                "usd": 69000.0,
                "usd_24h_vol": 30000000000.0,
                "usd_24h_change": 2.1,
            }
        }
        pool._client.get = AsyncMock(return_value=mock_resp)

        ctx = await pool._fetch_coingecko("BTC")

        assert ctx.price == pytest.approx(69000.0)
        assert ctx.volume_24h == pytest.approx(30e9)
        assert ctx.change_1h == pytest.approx(0.0)  # CoinGecko free: no 1h data
        assert ctx.change_24h == pytest.approx(2.1)

    async def test_fetch_coingecko_missing_vol_defaults_to_zero(self):
        pool = self._make_pool()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "bitcoin": {
                "usd": 69000.0,
                # usd_24h_vol missing
                "usd_24h_change": 2.1,
            }
        }
        pool._client.get = AsyncMock(return_value=mock_resp)

        ctx = await pool._fetch_coingecko("BTC")

        assert ctx.volume_24h == pytest.approx(0.0)
