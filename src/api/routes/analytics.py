"""
src/api/routes/analytics.py

Analytics endpoints for model performance and environmental trends.

IMPORTANT: This module does NOT generate fake data. If historical data
is unavailable, endpoints return empty arrays with an honest status message.

Endpoints
---------
GET /analytics/feature-importance   — Model feature importances
GET /analytics/validation-history   — Real validation metrics log
GET /analytics/evi-trend            — EVI trend for a region
GET /analytics/fire-history-stats   — Historical fire counts from FIRMS
"""

import json
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Query
from loguru import logger

from src.config import DATA_RAW, MODELS_DIR
from src.api.dependencies import require_engine

router = APIRouter()


# ── Feature Importance ────────────────────────────────────────────────────────


@router.get("/analytics/feature-importance", tags=["Analytics"])
async def feature_importance():
    """
    Return model feature importances.
    Extracts from the production model's internal feature_importances_.
    """
    require_engine()
    from src.api.dependencies import engine as _eng

    model = _eng.model

    # Unwrap pipeline if necessary
    clf = model
    if hasattr(model, "named_steps"):
        clf = list(model.named_steps.values())[-1]

    # Extract importances
    importance_attr = None
    if hasattr(clf, "feature_importances_"):
        importance_attr = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance_attr = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)

    if importance_attr is None:
        return {"features": [], "status": "Model type does not expose feature importances."}

    # Get feature names
    feature_names = None
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    elif hasattr(clf, "feature_names_in_"):
        feature_names = list(clf.feature_names_in_)
    else:
        feature_names = [f"feature_{i}" for i in range(len(importance_attr))]

    features = [
        {"name": name, "importance": round(float(imp), 6)}
        for name, imp in zip(feature_names, importance_attr)
    ]
    features.sort(key=lambda x: x["importance"], reverse=True)

    return {"features": features}


# ── Validation History ────────────────────────────────────────────────────────

VALIDATION_LOG_PATH = MODELS_DIR / "validation_history.json"


@router.get("/analytics/validation-history", tags=["Analytics"])
async def validation_history():
    """
    Return REAL validation history from the validation log file.
    
    The log file is written by the training pipeline each time a model
    is validated against held-out data. If no log exists, returns an empty
    array with an honest status message.
    
    NOTE: This endpoint previously generated random fake data. That has
    been removed entirely. Only real metrics are served.
    """
    if VALIDATION_LOG_PATH.exists():
        try:
            with open(VALIDATION_LOG_PATH, "r") as f:
                runs = json.load(f)
            return {"runs": runs, "status": "ok", "source": str(VALIDATION_LOG_PATH)}
        except Exception as e:
            logger.warning(f"Could not read validation log: {e}")
            return {"runs": [], "status": f"Error reading log: {e}"}
    else:
        return {
            "runs": [],
            "status": "no_history",
            "message": (
                "No validation history available yet. "
                "Run the training pipeline with validation logging to populate this."
            ),
        }


# ── EVI Trend ─────────────────────────────────────────────────────────────────

# KMTR region sample points for meaningful EVI sampling
REGION_SAMPLE_POINTS = {
    "kmtr": [
        (8.45, 77.25), (8.50, 77.30), (8.55, 77.35),
        (8.60, 77.40), (8.65, 77.45), (8.70, 77.50),
        (8.75, 77.35), (8.80, 77.40),
    ],
    "tamil_nadu": [
        (9.0, 77.5), (10.5, 78.5), (11.0, 79.0),
        (12.0, 79.5), (8.5, 77.3), (10.0, 78.0),
    ],
}


@router.get("/analytics/evi-trend", tags=["Analytics"])
async def evi_trend(
    region: str = Query("kmtr", description="Region to sample EVI for"),
):
    """
    EVI trend over the last 12 months for representative points in a region.
    
    Samples multiple grid cells (not just one hardcoded coordinate) and
    returns the spatial mean for each month. This gives a meaningful
    vegetation health trend for the entire region.
    """
    from src.api.dependencies import evi_lookup as _evi

    if _evi is None:
        return {
            "points": [],
            "status": "evi_lookup_unavailable",
            "message": "EVI data not loaded. Place MODIS NetCDF files in data/raw/evi/",
        }

    sample_points = REGION_SAMPLE_POINTS.get(region.lower(), REGION_SAMPLE_POINTS["kmtr"])

    # Build sample grid
    sample_rows = [{"latitude": lat, "longitude": lon} for lat, lon in sample_points]
    sample_df = pd.DataFrame(sample_rows)

    months = pd.date_range(
        end=date.today(), periods=12, freq="MS"
    ).strftime("%Y-%m-%d").tolist()

    evi_points = []
    for month_str in months:
        try:
            enriched = _evi.enrich_dataframe(
                sample_df.assign(acq_date=month_str), date_col="acq_date"
            )
            mean_evi = float(enriched["evi"].mean())
            if not np.isnan(mean_evi):
                evi_points.append({"date": month_str, "mean_evi": round(mean_evi, 4)})
        except Exception as e:
            logger.debug(f"EVI lookup failed for {month_str}: {e}")
            continue

    return {
        "points": evi_points,
        "region": region,
        "n_sample_points": len(sample_points),
        "status": "ok" if evi_points else "no_data",
    }


# ── Fire History Stats ────────────────────────────────────────────────────────


@router.get("/analytics/fire-history-stats", tags=["Analytics"])
async def fire_history_stats():
    """
    Historical fire statistics from FIRMS CSV data.
    Reads from data/raw/filtered_*.csv files.
    """
    # Try to find FIRMS CSV
    csv_candidates = sorted(DATA_RAW.glob("filtered_*.csv"), reverse=True)
    if not csv_candidates:
        return {
            "total_fires": 0,
            "by_year": [],
            "by_severity": [],
            "status": "no_firms_data",
            "message": "No FIRMS CSV found in data/raw/",
        }

    csv_path = csv_candidates[0]
    logger.info(f"Loading fire history from {csv_path.name}")

    try:
        # Read only needed columns to save memory
        cols = ["acq_date", "brightness", "confidence"]
        available_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        use_cols = [c for c in cols if c in available_cols]

        df = pd.read_csv(csv_path, usecols=use_cols, low_memory=False)
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
        df["year"] = df["acq_date"].dt.year

        by_year = (
            df.groupby("year")
            .size()
            .reset_index(name="count")
            .sort_values("year")
            .to_dict(orient="records")
        )

        by_severity = []
        if "brightness" in df.columns:
            bins = [0, 320, 350, 400, float("inf")]
            labels = ["Low", "Moderate", "High", "Extreme"]
            df["severity"] = pd.cut(df["brightness"], bins=bins, labels=labels)
            by_severity = (
                df["severity"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "severity", "severity": "severity", "count": "count"})
                .to_dict(orient="records")
            )

        return {
            "total_fires": len(df),
            "by_year": by_year,
            "by_severity": by_severity,
            "data_source": csv_path.name,
            "status": "ok",
        }
    except Exception as e:
        logger.error(f"Fire history stats failed: {e}")
        return {"total_fires": 0, "by_year": [], "by_severity": [], "status": f"error: {e}"}
