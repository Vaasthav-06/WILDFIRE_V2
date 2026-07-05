"""
src/api/cache.py

Prediction cache with explicit lifecycle states.

States
------
  EMPTY → WARMING → READY → STALE → REFRESHING → READY
  Any state → ERROR (on failure; serves stale data if available)

Design
------
- Only one warm-up task runs at a time (asyncio.Lock).
- Other requests wait briefly, reuse cached data, or return 503.
- Cache is keyed by date — auto-invalidates at midnight.
- KMTR is warmed first (highest priority).
"""

import asyncio
import time
from datetime import date
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import RISK_THRESHOLD_HIGH, FEATURE_COLS
from src.spread.direction import compute_spread_vectors


# ── Cache state machine ──────────────────────────────────────────────────────

class CacheState(Enum):
    EMPTY = "EMPTY"
    WARMING = "WARMING"
    READY = "READY"
    STALE = "STALE"
    REFRESHING = "REFRESHING"
    ERROR = "ERROR"


class PredictionCache:
    """Thread-safe prediction cache with explicit lifecycle states."""

    def __init__(self):
        self.state: CacheState = CacheState.EMPTY
        self.cache_date: Optional[str] = None

        # Cached data
        self.scored_24h: Optional[pd.DataFrame] = None
        self.explanations_24h: Optional[list] = None
        self.results_7day: Optional[list] = None

        # Concurrency control
        self._lock: Optional[asyncio.Lock] = None
        self._ready_event: Optional[asyncio.Event] = None

    def init_async(self):
        """Must be called from within an async context (e.g. app startup)."""
        self._lock = asyncio.Lock()
        self._ready_event = asyncio.Event()

    @property
    def is_current(self) -> bool:
        """Check if cache is for today's date."""
        return self.cache_date == date.today().isoformat()

    @property
    def has_data(self) -> bool:
        """Check if any cached data exists (even if stale)."""
        return self.scored_24h is not None

    def _transition(self, new_state: CacheState):
        old = self.state
        self.state = new_state
        logger.info(f"Cache state: {old.value} → {new_state.value}")

    async def warm(self, engine, evi_lookup):
        """
        Warm the cache with fresh predictions.
        Only one warm-up runs at a time. Subsequent callers wait.
        """
        today_str = date.today().isoformat()

        # Fast path: already current
        if self.is_current and self.state == CacheState.READY:
            return

        if self._lock is None:
            self.init_async()

        async with self._lock:
            # Re-check inside the lock
            if self.is_current and self.state in (CacheState.READY, CacheState.REFRESHING):
                return

            is_refresh = self.has_data
            self._transition(CacheState.REFRESHING if is_refresh else CacheState.WARMING)
            self._ready_event.clear()

            try:
                loop = asyncio.get_running_loop()

                def _compute():
                    t0 = time.time()
                    
                    # KMTR bounds for priority warming
                    kmtr_bounds = {
                        "lat_min": 8.35, "lat_max": 8.85,
                        "lon_min": 77.15, "lon_max": 77.60,
                        "resolution": 0.05
                    }
                    
                    grid_df = engine.get_inference_grid(area=kmtr_bounds)

                    # 24h predictions (KMTR)
                    scored = engine.predict_now(grid_df=grid_df)
                    scored = _add_spread(scored)

                    # 7-day predictions (KMTR)
                    day_results = engine.predict_7day(grid_df=grid_df)
                    day_results = [_add_spread(df) for df in day_results]

                    elapsed = round(time.time() - t0, 1)
                    logger.info(f"Cache predictions computed in {elapsed}s ({len(grid_df)} cells)")
                    return scored, day_results

                scored, day_results = await loop.run_in_executor(None, _compute)

                self.scored_24h = scored
                self.results_7day = day_results
                self.cache_date = today_str
                self._transition(CacheState.READY)
                self._ready_event.set()

                logger.info("Cache READY. Starting background explanation generation.")

                # Generate explanations in background (non-blocking)
                def _compute_explanations():
                    from src.features.engineering import build_features
                    from src.explainability.why_engine import WhyEngine
                    from src.api.dependencies import why_engine as _why

                    if _why is None:
                        return []
                    feat_df = build_features(scored.copy(), evi_lookup=evi_lookup)
                    try:
                        return _why.explain(scored, feat_df[FEATURE_COLS])
                    except Exception as e:
                        logger.warning(f"Explanation generation failed: {e}")
                        return []

                explanations = await loop.run_in_executor(None, _compute_explanations)
                self.explanations_24h = explanations
                logger.info(f"Cache explanations ready ({len(explanations)} explanations)")

            except Exception as e:
                logger.error(f"Cache warming failed: {e}")
                self._transition(CacheState.ERROR)
                self._ready_event.set()  # Unblock waiters

    async def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Wait for cache to become ready. Returns False on timeout."""
        if self.state == CacheState.READY and self.is_current:
            return True

        if self._ready_event is None:
            return False

        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return self.state == CacheState.READY
        except asyncio.TimeoutError:
            return False

    def get_24h(self, area=None) -> tuple[Optional[pd.DataFrame], Optional[list]]:
        """Get cached 24h predictions, optionally filtered by area."""
        if self.scored_24h is None:
            return None, None

        df = _filter_by_area(self.scored_24h, area)
        explanations = _filter_explanations(self.explanations_24h, df)
        return df, explanations

    def get_7day(self, area=None) -> Optional[list]:
        """Get cached 7-day predictions, optionally filtered by area."""
        if self.results_7day is None:
            return None
        return [_filter_by_area(df, area) for df in self.results_7day]

    def should_refresh(self) -> bool:
        """Check if cache needs a refresh (stale or empty)."""
        if self.state in (CacheState.EMPTY, CacheState.ERROR):
            return True
        if not self.is_current:
            return True
        return False

    def is_warming_or_refreshing(self) -> bool:
        return self.state in (CacheState.WARMING, CacheState.REFRESHING)


# ── Helper functions ─────────────────────────────────────────────────────────

def _add_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Run spread vector computation if wind components are present."""
    if "wind_u" not in df.columns or "wind_v" not in df.columns:
        logger.warning("wind_u/wind_v not available — skipping spread computation")
        return df
    return compute_spread_vectors(df, risk_threshold=RISK_THRESHOLD_HIGH)


def _filter_by_area(df: pd.DataFrame, area) -> pd.DataFrame:
    """Filter DataFrame by bounding box area. Returns copy."""
    if area is None:
        return df.copy()
    mask = (
        (df["latitude"] >= area.lat_min)
        & (df["latitude"] <= area.lat_max)
        & (df["longitude"] >= area.lon_min)
        & (df["longitude"] <= area.lon_max)
    )
    return df[mask].copy()


def _filter_explanations(explanations: Optional[list], filtered_df: pd.DataFrame) -> list:
    """Filter explanations to only include cells present in filtered_df."""
    if not explanations:
        return []
    valid_ids = set(filtered_df["cell_id"].astype(str))
    return [e for e in explanations if str(e.get("cell_id")) in valid_ids]


# ── Module-level singleton ───────────────────────────────────────────────────

prediction_cache = PredictionCache()
