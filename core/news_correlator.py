"""
News-Price Correlator for MoMo Engine.

Analyzes historical gap-ups and correlates them with news catalysts
to learn which types of news create the best trading opportunities.

Uses Yahoo Finance news API and/or free news sources.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Returned by get_context_for_ticker when no pattern data is available.
NO_PATTERN_CONTEXT = "No historical pattern data available"

# Catalyst categories
CATALYST_TYPES = [
    "earnings_beat",
    "earnings_miss",
    "fda_approval",
    "fda_rejection",
    "contract_announcement",
    "partnership",
    "merger_acquisition",
    "stock_offering",
    "short_squeeze",
    "analyst_upgrade",
    "analyst_downgrade",
    "insider_buying",
    "sector_momentum",
    "crypto_correlation",
    "ev_sector",
    "ai_hype",
    "meme_momentum",
    "unknown",
]


@dataclass
class NewsCorrelation:
    """Learned pattern for a catalyst type."""

    catalyst_type: str
    occurrences: int
    avg_gap_pct: float           # average gap-up %
    avg_intraday_move_pct: float  # average move from open to HOD
    avg_reversal_pct: float      # how much it gives back from HOD
    best_entry_time_min: int     # minutes after open for best entry
    win_rate: float              # % of times buying the dip worked
    avg_hold_time_min: int       # average winning hold time
    related_tickers: list[str] = field(default_factory=list)


class NewsCorrelator:
    """Correlates news catalysts with MoMo price behaviour."""

    PATTERNS_FILE = "data/momo_news_patterns.json"

    def __init__(self) -> None:
        self.patterns: dict[str, NewsCorrelation] = {}
        self._load_patterns()

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def _load_patterns(self) -> None:
        """Load previously learned patterns from disk."""
        path = Path(self.PATTERNS_FILE)
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for cat, info in data.items():
                # Provide default for related_tickers so old files without it still load.
                info.setdefault("related_tickers", [])
                self.patterns[cat] = NewsCorrelation(**info)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[news-correlator] Failed to load patterns: %s", exc)

    def save_patterns(self) -> None:
        """Persist learned patterns to disk."""
        path = Path(self.PATTERNS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        for cat, corr in self.patterns.items():
            data[cat] = {
                "catalyst_type": corr.catalyst_type,
                "occurrences": corr.occurrences,
                "avg_gap_pct": corr.avg_gap_pct,
                "avg_intraday_move_pct": corr.avg_intraday_move_pct,
                "avg_reversal_pct": corr.avg_reversal_pct,
                "best_entry_time_min": corr.best_entry_time_min,
                "win_rate": corr.win_rate,
                "avg_hold_time_min": corr.avg_hold_time_min,
                "related_tickers": corr.related_tickers,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ──────────────────────────────────────────────────────────
    # Analysis
    # ──────────────────────────────────────────────────────────

    async def analyze_gap_up(
        self,
        ticker: str,
        date: str,
        bars_df: pd.DataFrame | None,
        gap_pct: float,
    ) -> None:
        """
        Analyse a single gap-up event and update learned patterns.

        1. Fetch news for this ticker around *date* via yfinance.
        2. Classify the catalyst type.
        3. Measure intraday behaviour (HOD, LOD, reversal, best entry).
        4. Update running-average patterns.
        """
        try:
            import yfinance as yf  # type: ignore

            stock = yf.Ticker(ticker)
            news = stock.news or []
        except Exception:  # noqa: BLE001
            news = []

        catalyst = self._classify_catalyst(ticker, news, gap_pct)

        if bars_df is not None and len(bars_df) > 0:
            open_price = float(bars_df.iloc[0]["open"])
            high_of_day = float(bars_df["high"].max())

            # Time of HOD (bar index)
            hod_idx = int(bars_df["high"].idxmax())

            # Best entry: lowest price in first 30 bars
            first_30 = bars_df.head(30)
            best_entry_time = int(first_30["low"].idxmin()) if len(first_30) > 0 else 15

            intraday_move = (
                ((high_of_day - open_price) / open_price) * 100
                if open_price > 0
                else 0.0
            )

            close_price = float(bars_df.iloc[-1]["close"])
            reversal = (
                ((high_of_day - close_price) / (high_of_day - open_price)) * 100
                if high_of_day > open_price
                else 0.0
            )

            self._update_pattern(catalyst, gap_pct, intraday_move, reversal, best_entry_time)

    # ──────────────────────────────────────────────────────────
    # Catalyst classification
    # ──────────────────────────────────────────────────────────

    def _classify_catalyst(
        self, ticker: str, news: list, gap_pct: float
    ) -> str:
        """Classify catalyst type from news headlines using keyword matching."""
        if not news:
            return "short_squeeze" if gap_pct > 30 else "unknown"

        headlines = " ".join(n.get("title", "") for n in news).lower()

        if "fda" in headlines and any(
            w in headlines for w in ("approv", "clear", "granted")
        ):
            return "fda_approval"
        if "fda" in headlines and any(
            w in headlines for w in ("reject", "fail", "complet")
        ):
            return "fda_rejection"
        if "earn" in headlines and any(
            w in headlines for w in ("beat", "surpass", "exceed", "top")
        ):
            return "earnings_beat"
        if "earn" in headlines and any(
            w in headlines for w in ("miss", "disappoint", "below")
        ):
            return "earnings_miss"
        if any(w in headlines for w in ("contract", "deal", "award")):
            return "contract_announcement"
        if any(w in headlines for w in ("partner", "collaborat")):
            return "partnership"
        if any(w in headlines for w in ("merg", "acqui", "buyout", "takeover")):
            return "merger_acquisition"
        if any(w in headlines for w in ("offer", "dilut", "shelf")):
            return "stock_offering"
        if "short" in headlines and any(w in headlines for w in ("squeeze", "cover")):
            return "short_squeeze"
        if "upgrad" in headlines or "price target" in headlines:
            return "analyst_upgrade"
        if "downgrad" in headlines in headlines:
            return "analyst_downgrade"
        if "insider" in headlines and "buy" in headlines:
            return "insider_buying"
        if any(w in headlines for w in ("artificial intelligence", " ai ")):
            return "ai_hype"
        if any(w in headlines for w in ("ev ", "electric vehicle")):
            return "ev_sector"
        if any(w in headlines for w in ("bitcoin", "crypto", "blockchain")):
            return "crypto_correlation"
        return "sector_momentum"

    # ──────────────────────────────────────────────────────────
    # Pattern update (running average)
    # ──────────────────────────────────────────────────────────

    def _update_pattern(
        self,
        catalyst: str,
        gap_pct: float,
        intraday_move: float,
        reversal: float,
        best_entry_time: int,
    ) -> None:
        """Update running averages for *catalyst*."""
        if catalyst not in self.patterns:
            self.patterns[catalyst] = NewsCorrelation(
                catalyst_type=catalyst,
                occurrences=0,
                avg_gap_pct=0.0,
                avg_intraday_move_pct=0.0,
                avg_reversal_pct=0.0,
                best_entry_time_min=15,
                win_rate=0.5,
                avg_hold_time_min=45,
                related_tickers=[],
            )

        p = self.patterns[catalyst]
        n = p.occurrences
        p.occurrences = n + 1
        p.avg_gap_pct = (p.avg_gap_pct * n + gap_pct) / (n + 1)
        p.avg_intraday_move_pct = (p.avg_intraday_move_pct * n + intraday_move) / (n + 1)
        p.avg_reversal_pct = (p.avg_reversal_pct * n + reversal) / (n + 1)
        p.best_entry_time_min = int(
            (p.best_entry_time_min * n + best_entry_time) / (n + 1)
        )

    # ──────────────────────────────────────────────────────────
    # Query helpers
    # ──────────────────────────────────────────────────────────

    def get_context_for_ticker(
        self, ticker: str, catalyst_type: str | None = None
    ) -> str:
        """Return a human-readable context string for the AI evaluator."""
        if catalyst_type and catalyst_type in self.patterns:
            p = self.patterns[catalyst_type]
            return (
                f"Catalyst: {p.catalyst_type} | "
                f"Avg gap: {p.avg_gap_pct:.0f}% | "
                f"Avg intraday move: {p.avg_intraday_move_pct:.0f}% | "
                f"Reversal: {p.avg_reversal_pct:.0f}% | "
                f"Best entry: {p.best_entry_time_min}min after open | "
                f"Based on {p.occurrences} occurrences"
            )
        return NO_PATTERN_CONTEXT

    def classify_ticker_news(self, ticker: str, gap_pct: float = 0.0) -> str:
        """
        Fetch live news for *ticker* and classify the catalyst.

        Returns one of the CATALYST_TYPES strings.
        """
        try:
            import yfinance as yf  # type: ignore

            news = yf.Ticker(ticker).news or []
        except Exception:  # noqa: BLE001
            news = []
        return self._classify_catalyst(ticker, news, gap_pct)
