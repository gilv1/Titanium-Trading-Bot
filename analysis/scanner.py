"""
Pre-Market MoMo Scanner for Titanium Warrior v3.

Scans for gap-up small-cap stocks with news catalysts and scores them
on a 0–110 scale. Integrates with Alpha Vantage for news data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────


@dataclass
class MomoCandidate:
    """A stock candidate identified by the pre-market scanner."""

    ticker: str
    gap_pct: float          # % gap from previous close
    rvol: float             # relative volume vs 14-day average
    float_shares: float     # shares float in millions
    price: float            # current pre-market price
    news_headline: str = ""
    score: int = 0
    sector_momentum: bool = False
    is_blue_sky: bool = False        # at 52-week high
    clean_daily_chart: bool = False
    premarket_volume: float = 0.0    # pre-market volume in shares
    short_interest_pct: float = 0.0  # short interest as % of float
    bid_ask_spread: float = 0.0      # bid/ask spread in dollars


# ──────────────────────────────────────────────────────────────
# Scanner class
# ──────────────────────────────────────────────────────────────


class MomoScanner:
    """
    Pre-market scanner that finds MoMo candidates and scores them.

    Hard filters (ALL must pass):
      1. Gap ≥ +10 %
      2. Float ≤ 10 M shares
      3. Price ≤ $20
      4. Verifiable news catalyst
      5. RVOL ≥ 5×

    Scoring (0–110):
      - RVOL ≥ 10×: +20; RVOL 5–10×: +10
      - Float < 5 M: +15; Float 5–10 M: +8
      - Price $2–$10: +15; Price $10–$20: +8
      - Blue-sky breakout (52-week high): +15
      - Clean daily chart: +10
      - Sector in momentum: +10
      - Pre-market volume > 1 M: +10
      - Short interest > 20 %: +10
      - Bid/ask spread < $0.02: +5
    """

    def __init__(self) -> None:
        self._news_client: object | None = None  # lazy import

    # ──────────────────────────────────────────────────────────
    # Hard filters
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _passes_hard_filters(candidate: MomoCandidate) -> bool:
        """Return True if the candidate passes ALL hard filters."""
        if candidate.gap_pct < 10.0:
            logger.debug("[scanner] %s failed gap filter (%.1f%%)", candidate.ticker, candidate.gap_pct)
            return False
        if candidate.float_shares > 10.0:
            logger.debug("[scanner] %s failed float filter (%.1fM)", candidate.ticker, candidate.float_shares)
            return False
        if candidate.price > 20.0:
            logger.debug("[scanner] %s failed price filter ($%.2f)", candidate.ticker, candidate.price)
            return False
        if candidate.rvol < 5.0:
            logger.debug("[scanner] %s failed RVOL filter (%.1f×)", candidate.ticker, candidate.rvol)
            return False
        if not candidate.news_headline:
            logger.debug("[scanner] %s failed news filter", candidate.ticker)
            return False
        return True

    # ──────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def score_candidate(candidate: MomoCandidate) -> int:
        """Apply the 0–110 scoring model and return the total score."""
        score = 0

        # RVOL
        if candidate.rvol >= 10.0:
            score += 20
        elif candidate.rvol >= 5.0:
            score += 10

        # Float
        if candidate.float_shares < 5.0:
            score += 15
        elif candidate.float_shares <= 10.0:
            score += 8

        # Price range
        if 2.0 <= candidate.price <= 10.0:
            score += 15
        elif candidate.price <= 20.0:
            score += 8

        # Blue-sky breakout
        if candidate.is_blue_sky:
            score += 15

        # Clean daily chart
        if candidate.clean_daily_chart:
            score += 10

        # Sector momentum
        if candidate.sector_momentum:
            score += 10

        # Pre-market volume
        if candidate.premarket_volume > 1_000_000:
            score += 10

        # Short interest
        if candidate.short_interest_pct > 20.0:
            score += 10

        # Bid/ask spread
        if candidate.bid_ask_spread < 0.02:
            score += 5

        return score

    # ──────────────────────────────────────────────────────────
    # Main scan
    # ──────────────────────────────────────────────────────────

    async def scan_premarket(self) -> list[MomoCandidate]:
        """
        Run the full pre-market scan.

        Fetches gap movers from IBKR scanner, enriches with Alpha Vantage
        news data, applies hard filters, scores, and returns sorted list.
        """
        raw_candidates = await self._fetch_gap_movers()
        results: list[MomoCandidate] = []

        for candidate in raw_candidates:
            # Check news catalyst
            has_news = await self.check_news_catalyst(candidate.ticker)
            if has_news:
                from data.news import NewsClient
                try:
                    nc = NewsClient()
                    items = await nc.get_news(candidate.ticker)
                    if items:
                        candidate.news_headline = items[0].title
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[scanner] Could not fetch headline for %s: %s", candidate.ticker, exc)

            if not self._passes_hard_filters(candidate):
                continue

            candidate.score = self.score_candidate(candidate)
            results.append(candidate)

        # Sort by score descending
        results.sort(key=lambda c: c.score, reverse=True)
        logger.info("[scanner] %d candidates passed filters after scoring.", len(results))
        return results

    # ──────────────────────────────────────────────────────────
    # Data fetch helpers
    # ──────────────────────────────────────────────────────────

    async def _fetch_gap_movers(self) -> list[MomoCandidate]:
        """
        Fetch pre-market gap movers from IBKR scanner subscription.

        Falls back to an empty list if IBKR is not available.
        """
        candidates: list[MomoCandidate] = []
        try:
            # Placeholder: in a live environment use ib.reqScannerSubscription
            # For now return an empty list to avoid hard dependency at import time
            logger.debug("[scanner] IBKR scanner not connected; returning empty candidate list.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[scanner] IBKR scanner error: %s", exc)
        return candidates

    async def get_float(self, ticker: str) -> float:
        """
        Return shares float in millions for ``ticker``.

        Uses Alpha Vantage OVERVIEW endpoint when available.
        """
        if not settings.ALPHA_VANTAGE_API_KEY:
            return 0.0
        try:
            import aiohttp  # type: ignore

            url = (
                f"https://www.alphavantage.co/query"
                f"?function=OVERVIEW&symbol={ticker}&apikey={settings.ALPHA_VANTAGE_API_KEY}"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    shares_outstanding = float(data.get("SharesOutstanding", 0))
                    return shares_outstanding / 1_000_000
        except Exception as exc:  # noqa: BLE001
            logger.warning("[scanner] get_float error for %s: %s", ticker, exc)
            return 0.0

    async def get_rvol(self, ticker: str) -> float:
        """
        Calculate relative volume for ``ticker`` using Alpha Vantage daily data.
        """
        if not settings.ALPHA_VANTAGE_API_KEY:
            return 0.0
        try:
            import aiohttp  # type: ignore

            url = (
                f"https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact"
                f"&apikey={settings.ALPHA_VANTAGE_API_KEY}"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    time_series = data.get("Time Series (Daily)", {})
                    volumes = [float(v["5. volume"]) for v in list(time_series.values())[:14]]
                    if not volumes:
                        return 0.0
                    avg_vol = sum(volumes[1:]) / max(len(volumes[1:]), 1)
                    current_vol = volumes[0] if volumes else 0.0
                    return current_vol / avg_vol if avg_vol > 0 else 0.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("[scanner] get_rvol error for %s: %s", ticker, exc)
            return 0.0

    async def check_news_catalyst(self, ticker: str) -> bool:
        """Return True if there is a relevant news item for ``ticker`` today."""
        try:
            from data.news import NewsClient

            nc = NewsClient()
            return await nc.has_catalyst(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[scanner] check_news_catalyst error for %s: %s", ticker, exc)
            return False
