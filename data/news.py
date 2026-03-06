"""
Alpha Vantage News Client for Titanium Warrior v3.

Fetches news sentiment from Alpha Vantage with:
  - In-memory TTL cache (30-minute expiry) to respect rate limits.
  - Rate limiter: max 5 calls/minute (free tier).
  - Methods: get_news(), has_catalyst(), get_sentiment().
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

_CACHE_TTL = 1800  # 30 minutes in seconds
_MAX_CALLS_PER_MINUTE = 5


@dataclass
class NewsItem:
    """A single news article from Alpha Vantage."""

    title: str
    source: str
    url: str
    time_published: str
    sentiment_score: float   # -1.0 (bearish) to +1.0 (bullish)
    relevance_score: float   # 0.0 to 1.0


class _Cache:
    """Simple in-memory TTL cache."""

    def __init__(self, ttl: float = _CACHE_TTL) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        if key in self._store:
            value, expiry = self._store[key]
            if time.time() < expiry:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time() + self._ttl)


class _RateLimiter:
    """Token-bucket rate limiter for Alpha Vantage free tier (5 calls/min)."""

    def __init__(self, max_calls: int = _MAX_CALLS_PER_MINUTE) -> None:
        self._max = max_calls
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        """Block until a call slot is available."""
        while True:
            now = time.time()
            # Remove timestamps older than 60 seconds
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) < self._max:
                self._timestamps.append(now)
                return
            wait = 60.0 - (now - self._timestamps[0]) + 0.1
            logger.debug("Alpha Vantage rate limit: waiting %.1f s", wait)
            await asyncio.sleep(max(wait, 0.1))


class NewsClient:
    """
    Async client for Alpha Vantage News Sentiment API.

    Usage::

        client = NewsClient()
        items  = await client.get_news("AAPL")
        has_it = await client.has_catalyst("AAPL")
        score  = await client.get_sentiment("AAPL")
    """

    _BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        self._cache = _Cache()
        self._rate_limiter = _RateLimiter()

    async def _fetch(self, params: dict[str, str]) -> dict[str, Any]:
        """Make a rate-limited GET request and return the JSON response."""
        await self._rate_limiter.acquire()
        try:
            import aiohttp  # type: ignore

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._BASE_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    return await resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alpha Vantage request failed: %s", exc)
            return {}

    async def get_news(self, ticker: str) -> list[NewsItem]:
        """
        Return a list of recent NewsItems for ``ticker``.

        Results are cached for 30 minutes.
        """
        cache_key = f"news_{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if not settings.ALPHA_VANTAGE_API_KEY:
            logger.warning("ALPHA_VANTAGE_API_KEY not set; cannot fetch news.")
            return []

        data = await self._fetch(
            {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "apikey": settings.ALPHA_VANTAGE_API_KEY,
            }
        )

        items: list[NewsItem] = []
        for article in data.get("feed", []):
            # Find ticker-specific sentiment
            ticker_sentiment = 0.0
            ticker_relevance = 0.0
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    ticker_sentiment = float(ts.get("ticker_sentiment_score", 0))
                    ticker_relevance = float(ts.get("relevance_score", 0))
                    break

            items.append(
                NewsItem(
                    title=article.get("title", ""),
                    source=article.get("source", ""),
                    url=article.get("url", ""),
                    time_published=article.get("time_published", ""),
                    sentiment_score=ticker_sentiment,
                    relevance_score=ticker_relevance,
                )
            )

        self._cache.set(cache_key, items)
        return items

    async def has_catalyst(self, ticker: str) -> bool:
        """
        Return True if there is at least one relevant news article for ``ticker``.

        Considers an article relevant if its relevance_score > 0.3.
        """
        items = await self.get_news(ticker)
        return any(item.relevance_score > 0.3 for item in items)

    async def get_sentiment(self, ticker: str) -> float:
        """
        Return average sentiment score for ``ticker`` (−1.0 to +1.0).

        Returns 0.0 when no articles are available.
        """
        items = await self.get_news(ticker)
        relevant = [i for i in items if i.relevance_score > 0.3]
        if not relevant:
            return 0.0
        return sum(i.sentiment_score for i in relevant) / len(relevant)
