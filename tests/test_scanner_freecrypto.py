"""
Tests for ScannerPool._fetch_freecrypto() — FreeCryptoAPI integration.

Covers response parsing, RSI/MACD indicator extraction, and graceful
handling of missing or malformed indicator data.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.scanner_pool import MarketContext, ScannerPool


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_pool() -> ScannerPool:
    pool = ScannerPool.__new__(ScannerPool)
    pool._client = AsyncMock()
    pool._scanners = []
    return pool


def _mock_response(data: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data
    return mock_resp


# ──────────────────────────────────────────────────────────────
# Basic parsing
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFetchFreecryptoBasic:
    """Basic price/volume/change parsing from FreeCryptoAPI response."""

    async def test_parses_price_and_volume(self):
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.50",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.5",
            "percent_change_1h": "0.3",
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.price == pytest.approx(68000.50)
        assert ctx.volume_24h == pytest.approx(25e9)
        assert ctx.change_24h == pytest.approx(1.5)
        assert ctx.change_1h == pytest.approx(0.3)

    async def test_uses_btcusdt_symbol_for_btc(self):
        """Request URL must use BTCUSDT for BTC."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
        }))

        await pool._fetch_freecrypto("BTC")

        call_url = pool._client.get.call_args[0][0]
        assert "BTCUSDT" in call_url

    async def test_uses_ethusdt_symbol_for_eth(self):
        """Request URL must use ETHUSDT for ETH."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "3200.0",
            "volume_24h": "8000000000.0",
            "percent_change_24h": "-0.5",
            "percent_change_1h": "0.0",
        }))

        await pool._fetch_freecrypto("ETH")

        call_url = pool._client.get.call_args[0][0]
        assert "ETHUSDT" in call_url

    async def test_missing_fields_default_to_zero(self):
        """Missing price/volume/change fields default to 0.0, not an exception."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({}))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.price == pytest.approx(0.0)
        assert ctx.volume_24h == pytest.approx(0.0)
        assert ctx.change_24h == pytest.approx(0.0)
        assert ctx.change_1h == pytest.approx(0.0)

    async def test_none_values_default_to_zero(self):
        """Explicit None values default to 0.0."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": None,
            "volume_24h": None,
            "percent_change_24h": None,
            "percent_change_1h": None,
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.price == pytest.approx(0.0)
        assert ctx.volume_24h == pytest.approx(0.0)

    async def test_returns_market_context_instance(self):
        """Return type must always be MarketContext."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert isinstance(ctx, MarketContext)


# ──────────────────────────────────────────────────────────────
# RSI parsing
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFetchFreecryptoRSI:
    """RSI indicator extraction from the 'indicators' sub-object."""

    async def test_parses_rsi_when_present(self):
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": {"rsi": 55.3},
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.rsi == pytest.approx(55.3)

    async def test_rsi_none_when_indicators_missing(self):
        """No 'indicators' key → rsi is None."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.rsi is None

    async def test_rsi_none_when_rsi_key_missing(self):
        """indicators present but no 'rsi' key → rsi is None."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": {"macd": {"histogram": 10.0}},
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.rsi is None

    async def test_rsi_none_when_rsi_value_is_none(self):
        """indicators.rsi = None → rsi field is None."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": {"rsi": None},
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.rsi is None


# ──────────────────────────────────────────────────────────────
# MACD parsing
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFetchFreecryptoMACD:
    """MACD signal extraction from the 'indicators.macd' sub-object."""

    async def test_positive_histogram_is_bullish(self):
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": {"macd": {"histogram": 50.5}},
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.macd_signal == "bullish"

    async def test_negative_histogram_is_bearish(self):
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": {"macd": {"histogram": -25.0}},
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.macd_signal == "bearish"

    async def test_zero_histogram_is_neutral(self):
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": {"macd": {"histogram": 0}},
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.macd_signal == "neutral"

    async def test_macd_signal_none_when_indicators_missing(self):
        """No 'indicators' key → macd_signal is None."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.macd_signal is None

    async def test_macd_signal_none_when_macd_key_missing(self):
        """indicators present but no 'macd' key → macd_signal is None."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": {"rsi": 55.0},
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.macd_signal is None

    async def test_macd_signal_none_when_macd_is_empty_dict(self):
        """indicators.macd = {} → macd_signal is None."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": {"macd": {}},
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.macd_signal is None


# ──────────────────────────────────────────────────────────────
# Full response (RSI + MACD together)
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFetchFreecryptoFullResponse:
    """Verify complete response with both RSI and MACD is parsed correctly."""

    async def test_full_response_bullish(self):
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68098.25",
            "volume_24h": "28000000000.0",
            "percent_change_24h": "2.1",
            "percent_change_1h": "0.4",
            "indicators": {
                "rsi": 61.5,
                "macd": {"histogram": 120.3},
            },
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.price == pytest.approx(68098.25)
        assert ctx.volume_24h == pytest.approx(28e9)
        assert ctx.change_24h == pytest.approx(2.1)
        assert ctx.change_1h == pytest.approx(0.4)
        assert ctx.rsi == pytest.approx(61.5)
        assert ctx.macd_signal == "bullish"

    async def test_full_response_bearish(self):
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "62000.00",
            "volume_24h": "20000000000.0",
            "percent_change_24h": "-1.8",
            "percent_change_1h": "-0.5",
            "indicators": {
                "rsi": 42.0,
                "macd": {"histogram": -80.0},
            },
        }))

        ctx = await pool._fetch_freecrypto("BTC")

        assert ctx.rsi == pytest.approx(42.0)
        assert ctx.macd_signal == "bearish"

    async def test_graceful_on_indicator_parse_error(self):
        """If indicator data is malformed, rsi/macd_signal default to None without raising."""
        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
            "indicators": "not_a_dict",  # malformed
        }))

        # Should not raise — indicators parse failure is caught gracefully
        ctx = await pool._fetch_freecrypto("BTC")

        assert isinstance(ctx, MarketContext)
        assert ctx.rsi is None
        assert ctx.macd_signal is None
        assert ctx.price == pytest.approx(68000.0)


# ──────────────────────────────────────────────────────────────
# API key header
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFetchFreecryptoAPIKey:
    """Authorization header is sent only when FREECRYPTO_API_KEY is set."""

    async def test_no_auth_header_when_key_empty(self):
        """Empty key → no Authorization header."""
        from unittest.mock import patch

        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
        }))

        with patch("core.scanner_pool.settings") as mock_settings:
            mock_settings.FREECRYPTO_API_KEY = ""
            await pool._fetch_freecrypto("BTC")

        call_kwargs = pool._client.get.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert "Authorization" not in headers

    async def test_auth_header_sent_when_key_set(self):
        """Non-empty key → Authorization: Bearer <key> header."""
        from unittest.mock import patch

        pool = _make_pool()
        pool._client.get = AsyncMock(return_value=_mock_response({
            "price": "68000.0",
            "volume_24h": "25000000000.0",
            "percent_change_24h": "1.0",
            "percent_change_1h": "0.0",
        }))

        with patch("core.scanner_pool.settings") as mock_settings:
            mock_settings.FREECRYPTO_API_KEY = "my-secret-key"
            await pool._fetch_freecrypto("BTC")

        call_kwargs = pool._client.get.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer my-secret-key"
