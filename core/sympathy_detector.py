"""
Sympathy Play Detector for MoMo Engine.

Learns which stocks move together:
  - NVDA up  → SMCI, AMD, AVGO also up
  - TSLA up  → LCID, RIVN, GOEV also up
  - BTC up   → MARA, RIOT, BITF, CLSK also up

Combines predefined sector groups with co-occurrence learning from historical
gap-up data saved in ``data/historical/momo/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Pre-defined sector groups — known sympathy relationships
SYMPATHY_GROUPS: dict[str, list[str]] = {
    "AI_CHIPS": ["NVDA", "AMD", "SMCI", "AVGO", "MRVL", "MU"],
    "AI_SOFTWARE": ["AI", "PLTR", "BBAI", "SOUN", "GFAI"],
    "CRYPTO_MINERS": ["MARA", "RIOT", "BITF", "HUT", "CLSK", "COIN"],
    "EV": ["TSLA", "LCID", "RIVN", "GOEV", "FSR", "NKLA", "NIO", "XPEV", "LI"],
    "BIOTECH": ["NVAX", "MRNA", "BNTX", "BBIO", "DNA"],
    "SPACE": ["RKLB", "LUNR", "ASTS", "RDW", "MNTS", "SPCE"],
    "MEME": ["GME", "AMC", "KOSS", "BB", "NOK"],
    "FINTECH": ["SOFI", "HOOD", "AFRM", "UPST", "OPEN"],
    "GENOMICS": ["CRSP", "EDIT", "NTLA", "BEAM", "DNA"],
    "LIDAR": ["LAZR", "LIDR", "AEVA", "OUST"],
    "QUANTUM": ["IONQ", "RGTI", "QUBT", "QBTS"],
    "ENERGY": ["PLUG", "FCEL", "BE", "TELL", "NEXT"],
}

# Minimum learned co-occurrence score to include in sympathy list
_LEARNED_CORRELATION_THRESHOLD = 0.6


class SympathyDetector:
    """Detects and learns sympathy plays between related stocks."""

    CORRELATIONS_FILE = "data/sympathy_correlations.json"

    def __init__(self) -> None:
        self.groups: dict[str, list[str]] = {k: list(v) for k, v in SYMPATHY_GROUPS.items()}
        self.learned_correlations: dict[str, dict[str, float]] = {}
        self._load_correlations()

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def _load_correlations(self) -> None:
        """Load previously learned co-occurrence correlations from disk."""
        path = Path(self.CORRELATIONS_FILE)
        if not path.exists():
            return
        try:
            with open(path) as f:
                self.learned_correlations = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sympathy] Failed to load correlations: %s", exc)

    def save_correlations(self) -> None:
        """Persist learned correlations to disk."""
        path = Path(self.CORRELATIONS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.learned_correlations, f, indent=2)

    # ──────────────────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────────────────

    def get_sympathy_tickers(self, leader_ticker: str) -> list[str]:
        """
        Return potential sympathy plays for *leader_ticker*.

        Checks predefined sector groups first, then learned co-occurrence
        correlations above ``_LEARNED_CORRELATION_THRESHOLD``.
        """
        sympathy: list[str] = []
        upper = leader_ticker.upper()

        for tickers in self.groups.values():
            if upper in tickers:
                sympathy.extend(t for t in tickers if t != upper)

        if upper in self.learned_correlations:
            for ticker, score in self.learned_correlations[upper].items():
                if score >= _LEARNED_CORRELATION_THRESHOLD and ticker not in sympathy:
                    sympathy.append(ticker)

        return list(set(sympathy))

    def get_sector_for_ticker(self, ticker: str) -> str | None:
        """Return the sector group name for *ticker*, or None if unknown."""
        upper = ticker.upper()
        for group_name, tickers in self.groups.items():
            if upper in tickers:
                return group_name
        return None

    # ──────────────────────────────────────────────────────────
    # Learning
    # ──────────────────────────────────────────────────────────

    def learn_from_historical_data(self, momo_data_dir: str) -> None:
        """
        Scan all downloaded MoMo CSVs and learn ticker co-occurrences.

        If MULN and GOEV both gapped up on the same day, their correlation
        score is incremented.  Correlations ≥ 0.6 are used as sympathy plays.
        """
        data_dir = Path(momo_data_dir)
        if not data_dir.exists():
            return

        # Group CSV filenames by date
        date_tickers: dict[str, list[str]] = {}
        for csv_file in data_dir.glob("*.csv"):
            parts = csv_file.stem.split("_", 1)
            if len(parts) == 2:
                ticker, date = parts[0], parts[1]
                date_tickers.setdefault(date, []).append(ticker)

        if not date_tickers:
            return

        # Increment co-occurrence scores for every same-day pair
        for tickers in date_tickers.values():
            for i, t1 in enumerate(tickers):
                for t2 in tickers[i + 1 :]:
                    self.learned_correlations.setdefault(t1, {})
                    self.learned_correlations.setdefault(t2, {})
                    self.learned_correlations[t1][t2] = (
                        self.learned_correlations[t1].get(t2, 0.0) + 0.1
                    )
                    self.learned_correlations[t2][t1] = (
                        self.learned_correlations[t2].get(t1, 0.0) + 0.1
                    )

        self.save_correlations()

        unique_tickers = {t for tickers in date_tickers.values() for t in tickers}
        logger.info(
            "[sympathy] Learned correlations from %d days, %d unique tickers.",
            len(date_tickers),
            len(unique_tickers),
        )
