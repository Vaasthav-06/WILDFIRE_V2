"""
src/api/app.py

FastAPI operational deployment API for the Wildfire Risk Prediction System.

Endpoints
---------
GET  /health                    — Liveness check
GET  /model/summary             — Training-time model performance summary
POST /predict/24h               — 24-hour risk grid for India (or custom area)
POST /predict/7day              — 7-day risk grid with KBDI propagation
GET  /predict/cell/{lat}/{lon}  — Single-cell prediction with full WHY explanation
GET  /map/24h                   — GeoJSON FeatureCollection for the risk map
GET  /map/7day                  — 7-day GeoJSON snapshot pack for temporal slider

Design decisions
----------------
- The engine and WHY engine are instantiated ONCE at startup (module-level).
  FastAPI is an ASGI server — shared state is safe for read-only model access.
- All responses are JSON-serialisable. NaN → null handled by custom encoder.
- The /map/* endpoints return pre-built GeoJSON for efficient frontend rendering.
- Spread direction computation is gated on a risk threshold — only high-risk
  cells pay the spread calculation cost.
- SHAP computation is gated behind the alert threshold — only flagged cells
  pay the SHAP cost.

Usage
-----
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

    # Or with gunicorn for production:
    gunicorn src.api.app:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    # Note: Use w=1 because the model is loaded into process memory.
    # Multiple workers would each load the model — use a reverse proxy + single worker.
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from loguru import logger

from src.config import MODELS_DIR, RISK_THRESHOLD_HIGH
from src.inference.engine import WildfireInferenceEngine
from src.inference.forecast_ingest import build_india_inference_grid
from src.spread.direction import compute_spread_vectors
from src.explainability.why_engine import WhyEngine
from src.inference.map_export import (
    scored_df_to_geojson,
    build_7day_geojson_pack,
    build_spread_arrow_geojson,
)

# ── NaN-safe JSON encoder ─────────────────────────────────────────────────────


class NanSafeEncoder(json.JSONEncoder):
    """Replace NaN/Inf with null for valid JSON output."""

    def default(self, obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (date, datetime)):
            return str(obj)
        return super().default(obj)


def _json_response(data: dict, status_code: int = 200) -> JSONResponse:
    """Return a JSONResponse with NaN-safe serialisation."""
    content = json.loads(json.dumps(data, cls=NanSafeEncoder))
    return JSONResponse(content=content, status_code=status_code)


# ── App startup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Wildfire Risk Prediction API",
    description=(
        "Operational wildfire risk inference for India. "
        "Predicts fire probability using ERA5-trained ensemble model "
        "with Open-Meteo forecast data. "
        "Provides 24h and 7-day forecasts, spread direction, "
        "and SHAP-based natural language explanations."
    ),
    version="2.0.0",
    docs_url="/docs",
)

# CORS — allow any origin in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module-level singletons ───────────────────────────────────────────────────
# Loaded once at startup. FastAPI handles concurrency — the model is read-only.

_engine: Optional[WildfireInferenceEngine] = None
_why_engine: Optional[WhyEngine] = None
_startup_time: Optional[datetime] = None


@app.on_event("startup")
async def load_model():
    """Load the production model and WHY engine at startup."""
    global _engine, _why_engine, _startup_time

    logger.info("Loading wildfire risk prediction model...")
    _startup_time = datetime.utcnow()

    try:
        _engine = WildfireInferenceEngine(grid_resolution=0.1)
        _why_engine = WhyEngine(
            model=_engine.model,
            alert_threshold=RISK_THRESHOLD_HIGH,
            top_n_reasons=3,
            uncertainty_threshold=0.15,
        )
        logger.info(f"Model loaded successfully: {_engine.summary()}")
    except FileNotFoundError as e:
        logger.error(
            f"production_model.pkl not found: {e}. "
            "Run scripts/train_models.py first to generate the model file."
        )
        # Don't crash the server — /health will report degraded status
        _engine = None
        _why_engine = None


def _require_engine():
    """Raise 503 if the model is not loaded."""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Run scripts/train_models.py to generate production_model.pkl, "
                "then restart the API server."
            ),
        )


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
    baseline_kbdi: Optional[List[float]] = Field(
        None,
        description=(
            "Current KBDI per grid cell [n_cells]. "
            "If null, uses 200.0 (moderate drought) as conservative default. "
            "For accuracy, pass actual KBDI values from recent training data."
        ),
    )
    include_spread: bool = Field(True, description="Compute spread vectors")
    include_explanation: bool = Field(
        False, description="Run WHY engine (expensive for 7 days)"
    )
    min_prob_geojson: float = Field(0.15)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_grid(area: Optional[CustomAreaRequest]) -> pd.DataFrame:
    """Build India grid or custom area grid."""
    if area is None:
        return _engine.get_inference_grid()

    lats = np.arange(area.lat_min, area.lat_max, area.resolution)
    lons = np.arange(area.lon_min, area.lon_max, area.resolution)
    rows = [
        {"latitude": round(float(la), 4), "longitude": round(float(lo), 4)}
        for la in lats
        for lo in lons
    ]
    grid = pd.DataFrame(rows)
    grid["cell_id"] = grid["latitude"].astype(str) + "_" + grid["longitude"].astype(str)
    return grid


def _add_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Run spread vector computation if wind components are present."""
    if "wind_u" not in df.columns or "wind_v" not in df.columns:
        logger.warning("wind_u/wind_v not available — skipping spread computation")
        return df
    return compute_spread_vectors(df, risk_threshold=RISK_THRESHOLD_HIGH)


def _add_explanations(
    scored_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> list[dict]:
    """Run WHY engine for flagged cells."""
    try:
        return _why_engine.explain(scored_df, feature_df)
    except Exception as e:
        logger.warning(f"WHY engine failed: {e}")
        return []


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health():
    """
    Liveness and readiness check.

    Returns model status and uptime. Used by load balancers and monitoring.
    """
    model_ok = _engine is not None
    uptime_s = (
        (datetime.utcnow() - _startup_time).total_seconds() if _startup_time else 0
    )
    return {
        "status": "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "uptime_s": round(uptime_s, 1),
        "timestamp": datetime.utcnow().isoformat(),
        "model_summary": _engine.summary() if model_ok else None,
    }


@app.get("/model/summary", tags=["Model"])
async def model_summary():
    """Return training-time model performance metrics."""
    _require_engine()
    return _json_response(_engine.summary())


@app.post("/predict/24h", tags=["Predictions"])
async def predict_24h(request: Predict24hRequest):
    """
    Generate 24-hour wildfire risk predictions.

    Fetches peak-weather composite (max temp, min humidity, max wind)
    for the next 24 hours and scores each grid cell.

    Returns a JSON response with:
    - summary: aggregate statistics
    - predictions: per-cell results (sorted by risk descending)
    - geojson_grid: GeoJSON FeatureCollection for the risk map
    - geojson_arrows: GeoJSON LineString overlay for spread arrows
    - explanations: WHY engine output for high-risk cells

    Notes
    -----
    - Default India grid: ~3,200 cells at 0.1° resolution
    - Typical latency: 15–60s (dominated by Open-Meteo API calls)
    - For development, use a small custom area (e.g. single state)
    """
    _require_engine()
    t0 = time.time()

    grid_df = _build_grid(request.area)
    logger.info(f"POST /predict/24h | {len(grid_df):,} cells")

    # 1. Score
    scored_df = _engine.predict_now(grid_df=grid_df)

    # Build feature matrix for WHY engine (features already computed inside _score)
    # We re-run build_features here to get the feature DataFrame.
    # In production, _score() should return the feature matrix to avoid double computation.
    feature_df = None
    if request.include_explanation:
        from src.features.engineering import build_features
        from src.data.weather import fetch_forecast_weather

        feature_df = build_features(scored_df.copy(), use_elevation_api=False)

    # 2. Spread direction
    if request.include_spread:
        scored_df = _add_spread(scored_df)

    # 3. WHY engine
    explanations = []
    if request.include_explanation and feature_df is not None:
        from src.config import FEATURE_COLS

        explanations = _add_explanations(scored_df, feature_df[FEATURE_COLS])

    # 4. GeoJSON
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

    # 5. Summary statistics
    high_risk = scored_df[scored_df["fire_prob"] >= RISK_THRESHOLD_HIGH]
    elapsed = round(time.time() - t0, 2)

    tier_counts = scored_df["risk_tier"].value_counts().to_dict()

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
        "predictions": json.loads(
            json.dumps(
                scored_df[
                    [
                        "cell_id",
                        "latitude",
                        "longitude",
                        "fire_prob",
                        "fire_pred",
                        "risk_tier",
                        "model_std",
                        "temp",
                        "humidity",
                        "wind",
                        "vpd",
                        "kbdi_approx",
                        "spread_bearing_deg",
                        "spread_intensity",
                    ]
                ]
                .sort_values("fire_prob", ascending=False)
                .head(500)  # Top 500 highest-risk cells in predictions list
                .to_dict(orient="records"),
                cls=NanSafeEncoder,
            )
        ),
        "geojson_grid": geojson_grid,
        "geojson_arrows": geojson_arrows,
        "explanations": explanations,
        "n_explanations": len(explanations),
    }

    logger.info(
        f"POST /predict/24h complete in {elapsed}s | {high_risk.shape[0]} high-risk cells"
    )
    return _json_response(response)


@app.post("/predict/7day", tags=["Predictions"])
async def predict_7day(request: Predict7dayRequest):
    """
    Generate 7-day daily wildfire risk predictions.

    For each of the next 7 days:
    - Fetches daily max/min weather variables
    - Propagates KBDI forward (accounts for forecasted precipitation)
    - Scores all grid cells

    Returns:
    - 7 GeoJSON snapshots (geojson_snapshots[0] = tomorrow, ..., [6] = day+7)
    - Daily summaries (mean risk, high-risk cell counts per day)

    Notes
    -----
    - Provide baseline_kbdi for accurate drought tracking.
    - KBDI without a baseline defaults to 200 (moderate drought).
    - Without explanations (default), this endpoint completes in ~30–90s.
    """
    _require_engine()
    t0 = time.time()

    grid_df = _build_grid(request.area)
    n_cells = len(grid_df)
    logger.info(f"POST /predict/7day | {n_cells:,} cells")

    baseline_kbdi = None
    if request.baseline_kbdi:
        if len(request.baseline_kbdi) != n_cells:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"baseline_kbdi length ({len(request.baseline_kbdi)}) "
                    f"must match grid size ({n_cells}). "
                    "Leave null to use the default."
                ),
            )
        baseline_kbdi = np.array(request.baseline_kbdi)

    # Predict 7 days
    day_results = _engine.predict_7day(
        grid_df=grid_df,
        baseline_kbdi=baseline_kbdi,
    )

    # Spread direction per day (optional)
    if request.include_spread:
        day_results = [_add_spread(df) for df in day_results]

    # WHY engine per day (expensive — off by default for 7-day)
    explanations_by_day: list = [None] * 7
    if request.include_explanation:
        from src.features.engineering import build_features
        from src.config import FEATURE_COLS

        explanations_by_day = []
        for day_df in day_results:
            feat_df = build_features(day_df.copy(), use_elevation_api=False)
            expl = _add_explanations(day_df, feat_df[FEATURE_COLS])
            explanations_by_day.append(expl)

    # Build 7-day GeoJSON pack
    resolution = getattr(request.area, "resolution", 0.1)
    geojson_snapshots = build_7day_geojson_pack(
        day_results=day_results,
        explanations_by_day=(
            explanations_by_day if request.include_explanation else None
        ),
        resolution=resolution,
        min_prob=request.min_prob_geojson,
    )

    # Daily summary stats
    daily_summaries = []
    for day_idx, df_day in enumerate(day_results):
        daily_summaries.append(
            {
                "day": day_idx + 1,
                "forecast_date": (
                    str(df_day.get("forecast_date", pd.Series([""])).iloc[0])
                    if "forecast_date" in df_day.columns
                    else None
                ),
                "mean_fire_prob": round(float(df_day["fire_prob"].mean()), 4),
                "max_fire_prob": round(float(df_day["fire_prob"].max()), 4),
                "n_high_risk": int((df_day["fire_prob"] >= RISK_THRESHOLD_HIGH).sum()),
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
    return _json_response(response)


@app.get("/predict/cell/{lat}/{lon}", tags=["Predictions"])
async def predict_cell(
    lat: float,
    lon: float,
    forecast_days: int = Query(
        1, ge=1, le=7, description="1 = next 24h, 2–7 = specific forecast day"
    ),
):
    """
    Single-cell deep prediction with full WHY explanation.

    Returns the complete explanation for one geographic point:
    - Fire probability and risk tier
    - Top 3 reasons (natural language)
    - Mitigating factors
    - Full SHAP values
    - Spread direction
    - Model confidence

    Args
    ----
    lat:           Latitude (8.0–37.6 for India)
    lon:           Longitude (68.1–97.4 for India)
    forecast_days: 1 = next 24 hours, 2–7 = specific forecast day

    Use case: firefighter app queries a specific forest location before deployment.
    """
    _require_engine()

    from src.config import INDIA_BOUNDS, FEATURE_COLS
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
                "cell_id": f"{round(lat,4)}_{round(lon,4)}",
            }
        ]
    )

    # Fetch and score
    if forecast_days == 1:
        scored = _engine.predict_now(grid_df=cell_grid)
    else:
        day_results = _engine.predict_7day(grid_df=cell_grid)
        scored = day_results[min(forecast_days - 1, 6)]

    # Feature engineering for SHAP
    feature_df = build_features(scored.copy(), use_elevation_api=False)

    # Spread direction
    scored = _add_spread(scored)

    # WHY engine
    row = scored.iloc[0]
    feat_row = feature_df[FEATURE_COLS].iloc[0]
    explanation = _why_engine.explain_single(
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

    response = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 2),
        "latitude": lat,
        "longitude": lon,
        "forecast_day": forecast_days,
        "forecast_date": (
            str(
                scored.get("forecast_date", pd.Series([date.today().isoformat()])).iloc[
                    0
                ]
            )
            if "forecast_date" in scored.columns
            else date.today().isoformat()
        ),
        "weather": {
            "temp_c": json.loads(
                json.dumps(float(row.get("temp", np.nan)), cls=NanSafeEncoder)
            ),
            "humidity_pct": json.loads(
                json.dumps(float(row.get("humidity", np.nan)), cls=NanSafeEncoder)
            ),
            "wind_ms": json.loads(
                json.dumps(float(row.get("wind", np.nan)), cls=NanSafeEncoder)
            ),
            "vpd_kpa": json.loads(
                json.dumps(float(row.get("vpd", np.nan)), cls=NanSafeEncoder)
            ),
            "kbdi": json.loads(
                json.dumps(float(row.get("kbdi_approx", np.nan)), cls=NanSafeEncoder)
            ),
        },
        **explanation,
    }

    return _json_response(response)


@app.get("/map/24h", tags=["Map"])
async def map_24h(
    min_prob: float = Query(0.15, description="Minimum P(fire) to include"),
    include_arrows: bool = Query(True, description="Include spread arrow layer"),
):
    """
    Convenience endpoint: generate 24h GeoJSON without explanations (faster).
    Suitable for map tile server integration.
    """
    _require_engine()
    t0 = time.time()

    grid_df = _engine.get_inference_grid()
    scored_df = _engine.predict_now(grid_df=grid_df)

    if include_arrows:
        scored_df = _add_spread(scored_df)

    today_str = date.today().isoformat()
    geojson_grid = scored_df_to_geojson(
        scored_df, min_prob=min_prob, forecast_date=today_str
    )

    response = {"elapsed_s": round(time.time() - t0, 2)}
    if include_arrows:
        response["arrows"] = build_spread_arrow_geojson(scored_df)

    return _json_response({**geojson_grid, **response})


@app.get("/map/7day", tags=["Map"])
async def map_7day(
    min_prob: float = Query(0.15),
):
    """
    Convenience endpoint: generate full 7-day GeoJSON snapshot pack.
    Used by the frontend temporal slider — all 7 days in one response.
    """
    _require_engine()
    t0 = time.time()

    grid_df = _engine.get_inference_grid()
    day_results = _engine.predict_7day(grid_df=grid_df)
    snapshots = build_7day_geojson_pack(day_results, min_prob=min_prob)

    return _json_response(
        {
            "elapsed_s": round(time.time() - t0, 2),
            "n_snapshots": len(snapshots),
            "geojson_snapshots": snapshots,
        }
    )
