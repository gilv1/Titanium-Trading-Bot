"""
Event Impact Analyzer for Titanium Warrior v3.

Correlates historical price bars with economic calendar events to learn:
- Average price movement after each event type (5min, 15min, 30min, 60min)
- Reversal probability
- Best entry delay (how long to wait after event)
- Direction bias

Saves learned patterns to data/event_patterns.json.

Usage:
    python -m core.event_analyzer
    python backtest.py --analyze-events
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CALENDAR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "economic_calendar_2026.json")
_HISTORICAL_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "data", "historical"))
_PATTERNS_PATH = Path(os.path.join(os.path.dirname(__file__), "..", "data", "event_patterns.json"))


class EventAnalyzer:
    """
    Correlates economic calendar events with historical price moves.

    For each event found in the calendar, the analyzer locates the
    corresponding bar in historical CSVs and measures price movement
    at +5, +15, +30, and +60 minutes.
    """

    def __init__(
        self,
        calendar_path: str | Path | None = None,
        historical_dir: str | Path | None = None,
        patterns_path: str | Path | None = None,
    ) -> None:
        self._calendar_path = Path(calendar_path) if calendar_path else Path(_CALENDAR_PATH)
        self._historical_dir = Path(historical_dir) if historical_dir else _HISTORICAL_DIR
        self._patterns_path = Path(patterns_path) if patterns_path else _PATTERNS_PATH
        self._calendar: list[dict[str, Any]] = []
        self._load_calendar()

    # ──────────────────────────────────────────────────────────
    # Calendar loading
    # ──────────────────────────────────────────────────────────

    def _load_calendar(self) -> None:
        path = self._calendar_path.resolve()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._calendar = data.get("events", [])
            logger.info("[event-analyzer] Loaded %d calendar events.", len(self._calendar))
        except FileNotFoundError:
            logger.warning("[event-analyzer] Calendar not found at %s.", path)
            self._calendar = []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[event-analyzer] Failed to load calendar: %s", exc)
            self._calendar = []

    # ──────────────────────────────────────────────────────────
    # Historical data loading
    # ──────────────────────────────────────────────────────────

    def _load_historical_bars(self) -> dict[str, pd.DataFrame]:
        """
        Load all CSV files from the historical data directory.

        Returns a mapping of ticker → DataFrame (indexed by timestamp).
        """
        frames: dict[str, pd.DataFrame] = {}
        if not self._historical_dir.exists():
            logger.warning("[event-analyzer] Historical directory not found: %s", self._historical_dir)
            return frames

        for csv_path in sorted(self._historical_dir.glob("*.csv")):
            ticker = csv_path.stem
            try:
                df = pd.read_csv(csv_path)
                if "time" not in df.columns:
                    logger.debug("[event-analyzer] Skipping %s: no 'time' column.", csv_path)
                    continue
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df.dropna(subset=["time"]).set_index("time").sort_index()
                frames[ticker] = df
            except Exception as exc:  # noqa: BLE001
                logger.warning("[event-analyzer] Could not load %s: %s", csv_path, exc)

        logger.info("[event-analyzer] Loaded historical data for %d tickers.", len(frames))
        return frames

    # ──────────────────────────────────────────────────────────
    # Measurement helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _measure_move(df: pd.DataFrame, event_time: datetime, offset_min: int) -> float | None:
        """
        Measure price move from ``event_time`` to ``event_time + offset_min``.

        Returns the absolute point move (close[t+offset] - close[t]), or None
        if either bar is not found.
        """
        try:
            target_time = event_time + timedelta(minutes=offset_min)
            # Find nearest bar at or after target_time
            future = df[df.index >= target_time]
            base = df[df.index >= event_time]
            if future.empty or base.empty:
                return None
            base_price = float(base.iloc[0]["close"])
            target_price = float(future.iloc[0]["close"])
            return target_price - base_price
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _detect_reversal(
        df: pd.DataFrame,
        event_time: datetime,
        window_min: int = 60,
        spike_pts: float = 5.0,
    ) -> tuple[bool, float]:
        """
        Detect whether price reversed within ``window_min`` minutes after the event.

        Returns (reversed: bool, reversal_time_min: float).
        A reversal is defined as: price moved more than ``spike_pts`` in one direction
        then came back by at least 50% of the initial move.
        """
        try:
            end_time = event_time + timedelta(minutes=window_min)
            window = df[(df.index >= event_time) & (df.index <= end_time)]
            if len(window) < 3:
                return False, 0.0

            base_price = float(window.iloc[0]["close"])
            prices = window["close"].values

            # Find max deviation
            max_up = max(float(p) - base_price for p in prices)
            max_dn = base_price - min(float(p) for p in prices)

            reversal_pct_required = 0.5

            if max_up >= spike_pts:
                # Check if price came back down by 50%
                peak_idx = int(pd.Series(prices).idxmax())
                if peak_idx < len(prices) - 1:
                    post_peak_min = min(float(p) for p in prices[peak_idx:])
                    if base_price + max_up - post_peak_min >= max_up * reversal_pct_required:
                        # Estimate time of reversal
                        reversal_bar = peak_idx + int(
                            (len(prices) - peak_idx) * reversal_pct_required
                        )
                        mins = reversal_bar  # each bar ≈ 1 min
                        return True, float(mins)

            if max_dn >= spike_pts:
                trough_idx = int(pd.Series(prices).idxmin())
                if trough_idx < len(prices) - 1:
                    post_trough_max = max(float(p) for p in prices[trough_idx:])
                    if post_trough_max - (base_price - max_dn) >= max_dn * reversal_pct_required:
                        reversal_bar = trough_idx + int(
                            (len(prices) - trough_idx) * reversal_pct_required
                        )
                        return True, float(reversal_bar)

            return False, 0.0
        except Exception:  # noqa: BLE001
            return False, 0.0

    @staticmethod
    def _find_best_entry_delay(df: pd.DataFrame, event_time: datetime) -> int:
        """
        Heuristic: best entry delay is the bar at which post-event volatility
        (measured by bar range) drops below 150% of the pre-event average range.
        """
        try:
            pre = df[(df.index >= event_time - timedelta(minutes=30)) & (df.index < event_time)]
            if pre.empty:
                return 5
            avg_range = float((pre["high"] - pre["low"]).mean())
            if avg_range == 0:
                return 5

            post = df[df.index >= event_time].head(60)
            for i, (_, bar) in enumerate(post.iterrows()):
                bar_range = float(bar["high"]) - float(bar["low"])
                if bar_range <= avg_range * 1.5:
                    return max(5, i)
            return 40
        except Exception:  # noqa: BLE001
            return 5

    # ──────────────────────────────────────────────────────────
    # Core analysis
    # ──────────────────────────────────────────────────────────

    def analyze(self) -> dict[str, Any]:
        """
        Run the event impact analysis over all loaded historical data.

        Returns a dict mapping event_name → pattern statistics.
        """
        bars_by_ticker = self._load_historical_bars()
        if not bars_by_ticker:
            logger.warning("[event-analyzer] No historical data to analyse.")
            return {}

        # Merge all tickers into one combined DataFrame for cross-ticker analysis
        # Prefer MES or the first available ticker for futures events
        preferred_order = ["MES", "ES", "MNQ", "NQ"]
        combined: pd.DataFrame | None = None
        for t in preferred_order:
            if t in bars_by_ticker:
                combined = bars_by_ticker[t]
                break
        if combined is None:
            combined = next(iter(bars_by_ticker.values()))

        # Bucket results by event name
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for event in self._calendar:
            name = event.get("name", "Unknown")
            date_str = event.get("date", "")
            time_str = event.get("time", "00:00")
            if not date_str:
                continue

            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo("America/New_York")
                dt_naive = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                event_dt = dt_naive.replace(tzinfo=tz)
                # Convert to UTC for DataFrame index comparison
                event_dt_utc = event_dt.astimezone(ZoneInfo("UTC"))
            except Exception:  # noqa: BLE001
                continue

            # Check if event falls within historical data range
            if event_dt_utc < combined.index.min() or event_dt_utc > combined.index.max():
                continue

            m5 = self._measure_move(combined, event_dt_utc, 5)
            m15 = self._measure_move(combined, event_dt_utc, 15)
            m30 = self._measure_move(combined, event_dt_utc, 30)
            m60 = self._measure_move(combined, event_dt_utc, 60)

            if m5 is None and m15 is None:
                continue

            reversed_, rev_time = self._detect_reversal(combined, event_dt_utc)
            best_delay = self._find_best_entry_delay(combined, event_dt_utc)

            # Max move in 60 min
            try:
                end = event_dt_utc + timedelta(minutes=60)
                window = combined[(combined.index >= event_dt_utc) & (combined.index <= end)]
                if not window.empty:
                    base = float(window.iloc[0]["close"])
                    max_move = max(abs(float(p) - base) for p in window["close"].values)
                else:
                    max_move = 0.0
            except Exception:  # noqa: BLE001
                max_move = 0.0

            buckets[name].append(
                {
                    "move_5": m5 or 0.0,
                    "move_15": m15 or 0.0,
                    "move_30": m30 or 0.0,
                    "move_60": m60 or 0.0,
                    "reversed": reversed_,
                    "rev_time": rev_time,
                    "best_delay": best_delay,
                    "max_move": max_move,
                }
            )

        # Aggregate per event type
        patterns: dict[str, Any] = {}
        for name, records in buckets.items():
            n = len(records)
            if n == 0:
                continue
            rev_count = sum(1 for r in records if r["reversed"])
            patterns[name] = {
                "occurrences": n,
                "avg_move_5min_pts": round(sum(abs(r["move_5"]) for r in records) / n, 2),
                "avg_move_15min_pts": round(sum(abs(r["move_15"]) for r in records) / n, 2),
                "avg_move_30min_pts": round(sum(abs(r["move_30"]) for r in records) / n, 2),
                "avg_move_60min_pts": round(sum(abs(r["move_60"]) for r in records) / n, 2),
                "reversal_pct": round(rev_count / n * 100, 1),
                "avg_reversal_time_min": round(
                    sum(r["rev_time"] for r in records if r["reversed"]) / max(rev_count, 1), 1
                ),
                "best_entry_delay_min": round(sum(r["best_delay"] for r in records) / n),
                "avg_max_move_pts": round(sum(r["max_move"] for r in records) / n, 2),
                "direction_follows_result": True,  # placeholder — requires fundamental data
            }

        return patterns

    # ──────────────────────────────────────────────────────────
    # Save & load
    # ──────────────────────────────────────────────────────────

    def save(self, patterns: dict[str, Any]) -> None:
        """Save event patterns to ``data/event_patterns.json``."""
        self._patterns_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._patterns_path, "w", encoding="utf-8") as fh:
            json.dump(patterns, fh, indent=2)
        logger.info("[event-analyzer] Saved patterns for %d event types to %s.", len(patterns), self._patterns_path)
        print(f"[event-analyzer] Saved {len(patterns)} event patterns to {self._patterns_path}")

    @staticmethod
    def load_patterns(path: str | Path | None = None) -> dict[str, Any]:
        """
        Load previously saved event patterns from disk.

        Returns an empty dict if the file does not exist.
        """
        p = Path(path) if path else _PATTERNS_PATH
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[event-analyzer] Failed to load patterns: %s", exc)
            return {}

    # ──────────────────────────────────────────────────────────
    # CLI entry point
    # ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run analysis and save results."""
        print("[event-analyzer] Analysing event impact on historical bars…")
        patterns = self.analyze()
        if not patterns:
            print("[event-analyzer] No matching events found in historical data range.")
            return
        self.save(patterns)
        print("\n[event-analyzer] Summary:")
        for name, stats in sorted(patterns.items()):
            print(
                f"  {name:40s} | n={stats['occurrences']:3d} | "
                f"avg15m={stats['avg_move_15min_pts']:6.1f}pts | "
                f"rev={stats['reversal_pct']:5.1f}% | "
                f"entry_delay={stats['best_entry_delay_min']}min"
            )


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    EventAnalyzer().run()
