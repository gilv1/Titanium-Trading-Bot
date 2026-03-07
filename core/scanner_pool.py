"""
ScannerPool — Multi-API rotation system for external market confirmation.

Rotates between free crypto APIs (CoinCap, CoinLore, CoinGecko, FreeCryptoAPI)
to provide market-context validation before placing trades.  Automatically
handles rate limits, errors, and back-off without blocking the trading loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────


@dataclass
class ScannerStatus:
    """Tracks the health and rate-limit state of a single external API scanner."""

    name: str
    calls_per_min: int
    used_this_minute: int = 0
    last_reset: float = field(default_factory=time.time)
    consecutive_errors: int = 0
    disabled_until: float = 0.0  # UNIX timestamp; scanner is skipped until this time


@dataclass
class MarketContext:
    """External market data used to confirm or reject a trade signal."""

    price: float
    volume_24h: float
    change_1h: float
    change_24h: float
    rsi: float | None = None
    macd_signal: str | None = None  # "bullish", "bearish", or "neutral"

    @property
    def is_bullish_context(self) -> bool:
        """Return True if market context supports a LONG entry."""
        checks: list[bool] = []
        # Not in freefall
        checks.append(self.change_24h > -3.0)
        # 1-hour momentum not strongly negative
        checks.append(self.change_1h > -1.5)
        # RSI not overbought (only checked when available)
        if self.rsi is not None:
            checks.append(self.rsi < 72)
        # MACD must not be bearish (only checked when available)
        if self.macd_signal is not None:
            checks.append(self.macd_signal != "bearish")  # allow "bullish" and "neutral"
        # Volume is present — not a dead / illiquid market
        checks.append(self.volume_24h > 1_000_000_000)  # $1B minimum for BTC/ETH
        return all(checks)

    @property
    def is_bearish_context(self) -> bool:
        """Return True if market context supports a SHORT entry."""
        checks: list[bool] = []
        # Not in a strong uptrend
        checks.append(self.change_24h < 3.0)
        # 1-hour momentum not strongly positive
        checks.append(self.change_1h < 1.5)
        # RSI not oversold (only checked when available)
        if self.rsi is not None:
            checks.append(self.rsi > 28)
        # MACD must not be bullish (only checked when available)
        if self.macd_signal is not None:
            checks.append(self.macd_signal != "bullish")  # allow "bearish" and "neutral"
        # Volume is present
        checks.append(self.volume_24h > 1_000_000_000)
        return all(checks)


# ──────────────────────────────────────────────────────────────
# ScannerPool
# ──────────────────────────────────────────────────────────────


class ScannerPool:
    """Rotates between multiple free crypto APIs automatically.

    Priority order (highest capacity first):
        1. CoinLore     — unlimited, no key required
        2. CoinCap      — 500 req/min with free key
        3. CoinGecko    — 30 req/min with free demo key
        4. FreeCryptoAPI — ~100 req/min (using 80% capacity); provides RSI + MACD

    The pool resets per-minute counters every 60 s and temporarily disables
    any scanner that produces 3+ consecutive errors (120 s cool-down).
    """

    def __init__(self) -> None:
        self._scanners: list[ScannerStatus] = [
            ScannerStatus("coinlore", calls_per_min=900),    # effectively unlimited
            ScannerStatus("coincap", calls_per_min=450),     # 90% of 500 limit
            ScannerStatus("coingecko", calls_per_min=25),    # 83% of 30 limit
            ScannerStatus("freecrypto", calls_per_min=80),   # ~100/min actual, use 80% capacity
        ]
        self._client = httpx.AsyncClient(timeout=10.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_best_scanner(self) -> ScannerStatus | None:
        """Return the scanner with the most available capacity.

        Resets per-minute counters when > 60 s have elapsed since the last
        reset.  Skips scanners that are disabled due to consecutive errors.
        """
        now = time.time()
        for scanner in self._scanners:
            # Reset counter every 60 seconds
            if now - scanner.last_reset > 60:
                scanner.used_this_minute = 0
                scanner.last_reset = now

            # Skip if disabled due to errors
            if scanner.consecutive_errors >= 3:
                if now < scanner.disabled_until:
                    continue
                # Cool-down expired — re-enable
                scanner.consecutive_errors = 0

            if scanner.used_this_minute < scanner.calls_per_min:
                return scanner

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_market_context(self, symbol: str) -> MarketContext | None:
        """Fetch market context from the best available scanner.

        Retries with the next available scanner on failure, up to a maximum of
        ``len(scanners) × 3 + 1`` attempts (each scanner can fail up to 3 times
        before being disabled).  Returns None when all scanners are exhausted or
        unavailable so that callers can decide to proceed without external
        confirmation.
        """
        max_attempts = len(self._scanners) * 3 + 1
        for _ in range(max_attempts):
            scanner = self._get_best_scanner()
            if scanner is None:
                logger.warning("[scanner] All scanners exhausted — skipping confirmation")
                return None

            scanner.used_this_minute += 1
            try:
                if scanner.name == "coinlore":
                    result = await self._fetch_coinlore(symbol)
                elif scanner.name == "coincap":
                    result = await self._fetch_coincap(symbol)
                elif scanner.name == "coingecko":
                    result = await self._fetch_coingecko(symbol)
                elif scanner.name == "freecrypto":
                    result = await self._fetch_freecrypto(symbol)
                else:
                    logger.warning("[scanner] Unknown scanner '%s' — skipping", scanner.name)
                    return None

                # Success — reset error counter and return
                scanner.consecutive_errors = 0
                return result

            except Exception as exc:  # noqa: BLE001
                scanner.consecutive_errors += 1
                scanner.disabled_until = time.time() + 120  # disable for 2 min
                logger.warning(
                    "[scanner] %s error: %s — rotating to next",
                    scanner.name, exc,
                )
                # Continue loop to try the next available scanner

        logger.warning("[scanner] All scanners failed — skipping confirmation")
        return None

    async def close(self) -> None:
        """Close the underlying HTTP client (call on engine shutdown)."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # API implementations
    # ------------------------------------------------------------------

    async def _fetch_coincap(self, symbol: str) -> MarketContext:
        """Fetch from CoinCap API (500 req/min with free key)."""
        asset_id = "bitcoin" if symbol == "BTC" else "ethereum"
        url = f"https://api.coincap.io/v2/assets/{asset_id}"
        headers: dict[str, str] = {}
        if settings.COINCAP_API_KEY:
            headers["Authorization"] = f"Bearer {settings.COINCAP_API_KEY}"

        resp = await self._client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()["data"]
        return MarketContext(
            price=float(data["priceUsd"]),
            volume_24h=float(data["volumeUsd24Hr"]),
            change_1h=float(data.get("changePercent1Hr") or 0),
            change_24h=float(data["changePercent24Hr"]),
        )

    async def _fetch_coinlore(self, symbol: str) -> MarketContext:
        """Fetch from CoinLore API (unlimited, no key required)."""
        coin_id = "90" if symbol == "BTC" else "80"  # CoinLore IDs
        url = f"https://api.coinlore.net/api/ticker/?id={coin_id}"
        resp = await self._client.get(url)
        resp.raise_for_status()
        data = resp.json()[0]
        return MarketContext(
            price=float(data["price_usd"]),
            volume_24h=float(data["volume24"]),
            change_1h=float(data.get("percent_change_1h") or 0),
            change_24h=float(data["percent_change_24h"]),
        )

    async def _fetch_coingecko(self, symbol: str) -> MarketContext:
        """Fetch from CoinGecko API (30 req/min with free demo key)."""
        coin = "bitcoin" if symbol == "BTC" else "ethereum"
        params: dict[str, str] = {
            "ids": coin,
            "vs_currencies": "usd",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        }
        headers: dict[str, str] = {}
        if settings.COINGECKO_API_KEY:
            headers["x-cg-demo-api-key"] = settings.COINGECKO_API_KEY

        resp = await self._client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()[coin]
        return MarketContext(
            price=float(data["usd"]),
            volume_24h=float(data.get("usd_24h_vol") or 0),
            change_1h=0.0,  # CoinGecko free tier does not provide 1h change
            change_24h=float(data.get("usd_24h_change") or 0),
        )

    async def _fetch_freecrypto(self, symbol: str) -> MarketContext:
        """FreeCryptoAPI — provides RSI + MACD technical indicators."""
        coin = "BTC" if symbol == "BTC" else "ETH"
        headers: dict[str, str] = {}
        api_key = getattr(settings, "FREECRYPTO_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = await self._client.get(
            f"https://api.freecryptoapi.com/v1/getData?symbol={coin}USDT",
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        price = float(data.get("price", 0) or 0)
        volume = float(data.get("volume_24h", 0) or 0)
        change_24h = float(data.get("percent_change_24h", 0) or 0)
        change_1h = float(data.get("percent_change_1h", 0) or 0)

        # Technical indicators (FreeCryptoAPI unique feature)
        rsi: float | None = None
        macd_signal: str | None = None
        try:
            indicators = data.get("indicators", {})
            if indicators:
                rsi_val = indicators.get("rsi")
                if rsi_val is not None:
                    rsi = float(rsi_val)
                macd_data = indicators.get("macd", {})
                if macd_data:
                    macd_hist = float(macd_data.get("histogram", 0) or 0)
                    if macd_hist > 0:
                        macd_signal = "bullish"
                    elif macd_hist < 0:
                        macd_signal = "bearish"
                    else:
                        macd_signal = "neutral"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[scanner] freecrypto indicator parse error: %s", exc)
            rsi = None
            macd_signal = None

        return MarketContext(
            price=price,
            volume_24h=volume,
            change_1h=change_1h,
            change_24h=change_24h,
            rsi=rsi,
            macd_signal=macd_signal,
        )
