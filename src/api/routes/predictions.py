"""
src/api/routes/predictions.py

Prediction and map endpoints.

Endpoints
---------
POST /predict/24h           — 24-hour risk grid
POST /predict/7day          — 7-day risk grid
GET  /predict/cell/{lat}/{lon} — Single-cell deep prediction
GET  /map/24h               — 24h GeoJSON (fast, no explanations)
GET  /map/7day              — 7-day GeoJSON snapshot pack
GET  /map/evi               — EVI vegetation GeoJSON layer
"""

import asyncio
import time

import numpy as np
import pandas as pd
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator
from loguru import logger

from src.config import RISK_THRESHOLD_HIGH, FEATURE_COLS, INDIA_BOUNDS
from src.api.dependencies import engine, why_engine, evi_lookup, require_engine
from src.api.cache import prediction_cache, _add_spread, CacheState
from src.api.json_utils import json_response
from src.inference.map_export import (
    scored_df_to_geojson,
    build_7day_geojson_pack,
    build_spread_arrow_geojson,
)

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────


class CustomAreaRequest(BaseModel):
    """Optional bounding box for region-specific predictions."""

    lat_min: float = Field(
        ..., ge=8.0, le=37.6, description="Minimum latitude (India: 8.0–37.6)"
    )
    lat_max: float = Field(..., ge=8.0, le=37.6)
    lon_min: float = Field(
        ..., ge=68.1, le=97.4, description="Minimum longitude (India: 68.1–97.4)"
    )
    lon_max: float = Field(..., ge=68.1, le=97.4)
    resolution: float = Field(
        0.1, ge=0.05, le=1.0, description="Grid resolution in degrees"
    )

    @validator("lat_max")
    def lat_max_gt_min(cls, v, values):
        if "lat_min" in values and v <= values["lat_min"]:
            raise ValueError("lat_max must be greater than lat_min")
        return v

    @validator("lon_max")
    def lon_max_gt_min(cls, v, values):
        if "lon_min" in values and v <= values["lon_min"]:
            raise ValueError("lon_max must be greater than lon_min")
        return v


class Predict24hRequest(BaseModel):
    area: Optional[CustomAreaRequest] = Field(
        None, description="Custom bounding box. If null, runs for all of India."
    )
    include_spread: bool = Field(True, description="Compute spread direction vectors")
    include_explanation: bool = Field(
        True, description="Run WHY engine for high-risk cells"
    )
    min_prob_geojson: float = Field(
        0.15, description="Minimum P(fire) to include in GeoJSON"
    )


class Predict7dayRequest(BaseModel):
    area: Optional[CustomAreaRequest] = None
    include_spread: bool = Field(True, description="Compute spread vectors")
    include_explanation: bool = Field(
        False, description="Run WHY engine (expensive for 7 days)"
    )
    min_prob_geojson: float = Field(0.15)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_grid(area: Optional[CustomAreaRequest]) -> pd.DataFrame:
    """Build India grid or custom area grid."""
    from src.api.dependencies import engine

    base_grid = engine.get_inference_grid()
    if area is None:
        return base_grid

    filtered_grid = base_grid[
        (base_grid["latitude"] >= area.lat_min)
        & (base_grid["latitude"] <= area.lat_max)
        & (base_grid["longitude"] >= area.lon_min)
        & (base_grid["longitude"] <= area.lon_max)
    ].copy()

    return filtered_grid


def _add_explanations(scored_df, feature_df):
    """Run WHY engine for flagged cells."""
    from src.api.dependencies import why_engine

    if why_engine is None:
        return []
    try:
        return why_engine.explain(scored_df, feature_df)
    except Exception as e:
        logger.warning(f"WHY engine failed: {e}")
        return []


async def _ensure_cache():
    """Trigger cache warming if needed. Non-blocking."""
    from src.api.dependencies import engine as _eng, evi_lookup as _evi

    if _eng is None:
        return
    if prediction_cache.should_refresh() and not prediction_cache.is_warming_or_refreshing():
        asyncio.create_task(prediction_cache.warm(_eng, _evi))


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/predict/24h", tags=["Predictions"])
async def predict_24h(request: Predict24hRequest):
    """
    Generate 24-hour wildfire risk predictions.

    Returns a JSON response with:
    - summary: aggregate statistics
    - predictions: per-cell results (sorted by risk descending)
    - geojson_grid: GeoJSON FeatureCollection for the risk map
    - geojson_arrows: GeoJSON LineString overlay for spread arrows
    - explanations: WHY engine output for high-risk cells
    """
    require_engine()
    t0 = time.time()
    from src.api.dependencies import engine as _eng, evi_lookup as _evi

    await _ensure_cache()

    is_large_region = request.area is None

    # Wait for cache if it's warming and the request is for a large region
    if prediction_cache.is_warming_or_refreshing() and is_large_region:
        ready = await prediction_cache.wait_for_ready(timeout=10.0)
        if not ready:
            raise HTTPException(
                status_code=503,
                detail="Cache is currently warming up, please try again in a few moments.",
            )

    # Try cached path
    if prediction_cache.state == CacheState.READY and prediction_cache.is_current:
        scored_df, explanations = prediction_cache.get_24h(request.area)
        if scored_df is not None:
            explanations = explanations or []
        else:
            scored_df = _compute_fresh_24h(request, _eng, _evi)
            explanations = []
    else:
        scored_df = _compute_fresh_24h(request, _eng, _evi)
        explanations = []

        if request.include_explanation:
            from src.features.engineering import build_features

            feature_df = build_features(scored_df.copy(), evi_lookup=_evi)
            explanations = _add_explanations(scored_df, feature_df[FEATURE_COLS])

    # GeoJSON
    today_str = date.today().isoformat()
    geojson_grid = scored_df_to_geojson(
        scored_df=scored_df,
        explanations=explanations,
        resolution=getattr(request.area, "resolution", 0.1),
        min_prob=request.min_prob_geojson,
        forecast_date=today_str,
        forecast_day=None,
    )
    geojson_arrows = build_spread_arrow_geojson(
        scored_df, risk_threshold=RISK_THRESHOLD_HIGH
    )

    # Summary statistics
    high_risk = scored_df[scored_df["fire_prob"] >= RISK_THRESHOLD_HIGH]
    elapsed = round(time.time() - t0, 2)
    tier_counts = scored_df["risk_tier"].value_counts().to_dict()

    # Ensure spread columns exist
    if "spread_bearing_deg" not in scored_df.columns:
        scored_df["spread_bearing_deg"] = None
    if "spread_intensity" not in scored_df.columns:
        scored_df["spread_intensity"] = None

    response = {
        "status": "success",
        "elapsed_s": elapsed,
        "forecast_date": today_str,
        "n_cells": len(scored_df),
        "summary": {
            "mean_fire_prob": round(float(scored_df["fire_prob"].mean()), 4),
            "max_fire_prob": round(float(scored_df["fire_prob"].max()), 4),
            "n_high_risk": int((scored_df["fire_prob"] >= RISK_THRESHOLD_HIGH).sum()),
            "n_extreme": int((scored_df["fire_prob"] >= 0.65).sum()),
            "tier_counts": tier_counts,
        },
        "predictions": scored_df[
            [
                "cell_id", "latitude", "longitude",
                "fire_prob", "fire_pred", "risk_tier", "model_std",
                "fri", "fri_tier",
                "temp", "humidity", "wind", "vpd",
                "spread_bearing_deg", "spread_intensity",
            ]
        ]
        .sort_values("fire_prob", ascending=False)
        .head(500)
        .replace({np.nan: None})
        .to_dict(orient="records"),
        "geojson_grid": geojson_grid,
        "geojson_arrows": geojson_arrows,
        "explanations": explanations,
        "n_explanations": len(explanations),
    }

    logger.info(
        f"POST /predict/24h complete in {elapsed}s | {high_risk.shape[0]} high-risk cells"
    )
    return json_response(response)


def _compute_fresh_24h(request, _eng, _evi):
    """Compute 24h predictions without cache."""
    grid_df = _build_grid(request.area)
    logger.info(f"POST /predict/24h (uncached) | {len(grid_df):,} cells")
    scored_df = _eng.predict_now(grid_df=grid_df)
    if request.include_spread:
        scored_df = _add_spread(scored_df)
    if "spread_bearing_deg" not in scored_df.columns:
        scored_df["spread_bearing_deg"] = None
    if "spread_intensity" not in scored_df.columns:
        scored_df["spread_intensity"] = None
    return scored_df


@router.post("/predict/7day", tags=["Predictions"])
async def predict_7day(request: Predict7dayRequest):
    """Generate 7-day daily wildfire risk predictions."""
    require_engine()
    t0 = time.time()
    from src.api.dependencies import engine as _eng, evi_lookup as _evi

    await _ensure_cache()

    is_large_region = request.area is None

    if prediction_cache.is_warming_or_refreshing() and is_large_region:
        ready = await prediction_cache.wait_for_ready(timeout=10.0)
        if not ready:
            raise HTTPException(
                status_code=503,
                detail="Cache is currently warming up, please try again in a few moments.",
            )

    # Try cached path
    if prediction_cache.state == CacheState.READY and prediction_cache.is_current:
        day_results = prediction_cache.get_7day(request.area)
        if day_results is None:
            day_results = _compute_fresh_7day(request, _eng)
        n_cells = len(day_results[0])
        explanations_by_day = [None] * 7
    else:
        day_results = _compute_fresh_7day(request, _eng)
        n_cells = len(day_results[0])
        explanations_by_day = [None] * 7

        if request.include_explanation:
            from src.features.engineering import build_features

            explanations_by_day = []
            for day_df in day_results:
                feat_df = build_features(day_df.copy(), evi_lookup=_evi)
                expl = _add_explanations(day_df, feat_df[FEATURE_COLS])
                explanations_by_day.append(expl)

    # Build GeoJSON pack
    resolution = getattr(request.area, "resolution", 0.1)
    geojson_snapshots = build_7day_geojson_pack(
        day_results=day_results,
        explanations_by_day=(
            explanations_by_day if request.include_explanation else None
        ),
        resolution=resolution,
        min_prob=request.min_prob_geojson,
    )

    # Daily summaries
    daily_summaries = []
    for day_idx, df_day in enumerate(day_results):
        daily_summaries.append(
            {
                "day": day_idx + 1,
                "forecast_date": (
                    str(df_day["forecast_date"].iloc[0])
                    if "forecast_date" in df_day.columns
                    else None
                ),
                "mean_fire_prob": round(float(df_day["fire_prob"].mean()), 4),
                "max_fire_prob": round(float(df_day["fire_prob"].max()), 4),
                "n_high_risk": int(
                    (df_day["fire_prob"] >= RISK_THRESHOLD_HIGH).sum()
                ),
                "n_extreme": int((df_day["fire_prob"] >= 0.65).sum()),
                "tier_counts": df_day["risk_tier"].value_counts().to_dict(),
            }
        )

    elapsed = round(time.time() - t0, 2)
    response = {
        "status": "success",
        "elapsed_s": elapsed,
        "n_cells": n_cells,
        "n_forecast_days": len(day_results),
        "daily_summaries": daily_summaries,
        "geojson_snapshots": geojson_snapshots,
    }

    logger.info(
        f"POST /predict/7day complete in {elapsed}s | "
        f"Peak risk day: {max(daily_summaries, key=lambda d: d['n_high_risk'])['day']}"
    )
    return json_response(response)


def _compute_fresh_7day(request, _eng):
    """Compute 7-day predictions without cache."""
    grid_df = _build_grid(request.area)
    logger.info(f"POST /predict/7day (uncached) | {len(grid_df):,} cells")
    day_results = _eng.predict_7day(grid_df=grid_df)
    if request.include_spread:
        day_results = [_add_spread(df) for df in day_results]
    return day_results


@router.get("/predict/cell/{lat}/{lon}", tags=["Predictions"])
async def predict_cell(
    lat: float,
    lon: float,
    forecast_days: int = Query(
        1, ge=1, le=7, description="1 = next 24h, 2–7 = specific forecast day"
    ),
):
    """
    Single-cell deep prediction with full WHY explanation.

    Returns fire probability, risk tier, top reasons, mitigating factors,
    SHAP values, spread direction, and model confidence.
    """
    require_engine()
    from src.api.dependencies import engine as _eng, evi_lookup as _evi, why_engine as _why
    from src.features.engineering import build_features

    # Validate bounds
    if not (INDIA_BOUNDS["lat_min"] <= lat <= INDIA_BOUNDS["lat_max"]):
        raise HTTPException(400, f"lat={lat} outside India bounds")
    if not (INDIA_BOUNDS["lon_min"] <= lon <= INDIA_BOUNDS["lon_max"]):
        raise HTTPException(400, f"lon={lon} outside India bounds")

    logger.info(f"GET /predict/cell/{lat}/{lon} | forecast_days={forecast_days}")
    t0 = time.time()

    # Build a single-cell grid
    cell_grid = pd.DataFrame(
        [
            {
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "cell_id": f"{round(lat, 4)}_{round(lon, 4)}",
            }
        ]
    )

    # Fetch and score
    if forecast_days == 1:
        scored = _eng.predict_now(grid_df=cell_grid)
    else:
        day_results = _eng.predict_7day(grid_df=cell_grid)
        scored = day_results[min(forecast_days - 1, 6)]

    # Feature engineering for SHAP
    feature_df = build_features(scored.copy(), evi_lookup=_evi)

    # Spread direction
    scored = _add_spread(scored)

    # WHY engine
    row = scored.iloc[0]
    feat_row = feature_df[FEATURE_COLS].iloc[0]

    explanation = {}
    if _why is not None:
        explanation = _why.explain_single(
            cell_features=feat_row,
            fire_prob=float(row["fire_prob"]),
            risk_tier=str(row["risk_tier"]),
            model_std=float(row.get("model_std", 0)),
        )

    # Spread direction text
    bearing = row.get("spread_bearing_deg")
    if not pd.isna(bearing):
        from src.spread.direction import bearing_to_compass

        explanation["spread_direction"] = bearing_to_compass(float(bearing))
        explanation["spread_bearing_deg"] = round(float(bearing), 1)
        explanation["spread_intensity"] = str(row.get("spread_intensity", "none"))
    else:
        explanation["spread_direction"] = None
        explanation["spread_bearing_deg"] = None
        explanation["spread_intensity"] = "none"

    def _safe(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 4)

    response = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 2),
        "latitude": lat,
        "longitude": lon,
        "forecast_day": forecast_days,
        "forecast_date": (
            str(scored["forecast_date"].iloc[0])
            if "forecast_date" in scored.columns
            else date.today().isoformat()
        ),
        "weather": {
            "temp_c": _safe(row.get("temp")),
            "humidity_pct": _safe(row.get("humidity")),
            "wind_ms": _safe(row.get("wind")),
            "vpd_kpa": _safe(row.get("vpd")),
            "elevation_m": _safe(row.get("elevation_m")),
        },
        "fri": _safe(row.get("fri")),
        "fri_tier": str(row.get("fri_tier")) if not pd.isna(row.get("fri_tier")) else None,
        **explanation,
    }

    return json_response(response)


@router.get("/map/24h", tags=["Map"])
async def map_24h(
    min_prob: float = Query(0.15, description="Minimum P(fire) to include"),
    include_arrows: bool = Query(True, description="Include spread arrow layer"),
):
    """Generate 24h GeoJSON for map rendering."""
    require_engine()
    from src.api.dependencies import engine as _eng

    t0 = time.time()
    await _ensure_cache()

    if prediction_cache.is_warming_or_refreshing():
        ready = await prediction_cache.wait_for_ready(timeout=10.0)
        if not ready:
            raise HTTPException(
                status_code=503,
                detail="Cache is currently warming up, please try again in a few moments.",
            )

    if prediction_cache.state == CacheState.READY and prediction_cache.is_current:
        scored_df = prediction_cache.scored_24h.copy()
    else:
        grid_df = _eng.get_inference_grid()
        scored_df = _eng.predict_now(grid_df=grid_df)
        if include_arrows:
            scored_df = _add_spread(scored_df)

    today_str = date.today().isoformat()
    geojson_grid = scored_df_to_geojson(
        scored_df, min_prob=min_prob, forecast_date=today_str
    )

    response = {"elapsed_s": round(time.time() - t0, 2)}
    if include_arrows:
        response["arrows"] = build_spread_arrow_geojson(scored_df)

    return json_response({**geojson_grid, **response})


@router.get("/map/7day", tags=["Map"])
async def map_7day(min_prob: float = Query(0.15)):
    """Generate full 7-day GeoJSON snapshot pack."""
    require_engine()
    from src.api.dependencies import engine as _eng

    t0 = time.time()
    await _ensure_cache()

    if prediction_cache.is_warming_or_refreshing():
        ready = await prediction_cache.wait_for_ready(timeout=10.0)
        if not ready:
            raise HTTPException(
                status_code=503,
                detail="Cache is currently warming up, please try again in a few moments.",
            )

    if prediction_cache.state == CacheState.READY and prediction_cache.is_current:
        day_results = prediction_cache.results_7day
    else:
        grid_df = _eng.get_inference_grid()
        day_results = _eng.predict_7day(grid_df=grid_df)

    snapshots = build_7day_geojson_pack(day_results, min_prob=min_prob)

    return json_response(
        {
            "elapsed_s": round(time.time() - t0, 2),
            "n_snapshots": len(snapshots),
            "geojson_snapshots": snapshots,
        }
    )


@router.get("/map/evi", tags=["Map"])
async def map_evi():
    """Generate GeoJSON for the EVI (Enhanced Vegetation Index) layer."""
    require_engine()
    from src.api.dependencies import engine as _eng
    from src.inference.map_export import _cell_polygon, _safe_float

    grid_df = _eng.get_inference_grid()
    features = []

    for _, row in grid_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        val = row.get("evi")
        if pd.isna(val):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": _cell_polygon(lat, lon, resolution=0.1),
                "properties": {
                    "cell_id": f"{lat}_{lon}",
                    "evi": _safe_float(val),
                },
            }
        )

    return json_response({"type": "FeatureCollection", "features": features})
