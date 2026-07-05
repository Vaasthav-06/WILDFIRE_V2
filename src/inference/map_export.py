"""
src/inference/map_export.py

GeoJSON export and map rendering utilities.
Converts the scored grid DataFrame into frontend-ready polygon artifacts.

Features:
  1. GeoJSON FeatureCollection — grid cells rendered as Polygons (rectangles)
  2. Integrated SHAP explanations — attaches WHY Engine descriptions to grid cells
  3. Spread arrow overlay — directional vectors for active risk zones
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional, Union

# Risk tier → hex colour mapping
RISK_COLOURS = {
    "low": "#2ecc71",
    "moderate": "#f1c40f",
    "high": "#e67e22",
    "extreme": "#c0392b",
}

# Minimum opacity (even very uncertain cells remain slightly visible)
OPACITY_MIN = 0.25
OPACITY_MAX = 0.80


def _cell_polygon(lat: float, lon: float, resolution: float = 0.1) -> dict:
    """
    Build a GeoJSON Polygon for a grid cell.
    The cell is centred on (lat, lon) with ±resolution/2 bounds.
    """
    half = resolution / 2
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - half, lat - half],  # SW
                [lon + half, lat - half],  # SE
                [lon + half, lat + half],  # NE
                [lon - half, lat + half],  # NW
                [lon - half, lat - half],  # close
            ]
        ],
    }


def _compute_opacity(model_std: float) -> float:
    """
    Map inter-model standard deviation to cell opacity.
    Low uncertainty (std≈0) → opacity = OPACITY_MAX
    High uncertainty (std≥0.3) → opacity = OPACITY_MIN
    """
    if pd.isna(model_std) or model_std <= 0:
        return OPACITY_MAX
    opacity = OPACITY_MAX - (min(model_std, 0.3) / 0.3) * (OPACITY_MAX - OPACITY_MIN)
    return round(float(opacity), 2)


def _safe_float(val) -> Optional[float]:
    """Return float or None, never NaN (NaN breaks JSON serialisation)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def scored_df_to_geojson(
    scored_df: pd.DataFrame,
    explanations: Optional[list[dict]] = None,
    resolution: float = 0.05,
    min_prob: float = 0.05,
    forecast_date: Optional[str] = None,
    forecast_day: Optional[int] = None,
) -> dict:
    """
    Converts the scored dataframe into a GeoJSON FeatureCollection of POLYGONS (grid cells)
    instead of Points, creating a continuous risk map surface. Fully compatible with
    explanations and day parameters.
    """
    features = []
    half_res = resolution / 2.0

    # Build explanations lookup for fast matching
    expl_lookup = {}
    if explanations:
        for e in explanations:
            expl_lookup[e["cell_id"]] = e

    # Filter out low-probability cells to keep file sizes optimal
    df = scored_df[scored_df["fire_prob"] >= min_prob].copy()

    for _, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cid = str(row.get("cell_id", f"{lat}_{lon}"))

        # Calculate the 4 corners of the grid cell
        bottom_left = [lon - half_res, lat - half_res]
        bottom_right = [lon + half_res, lat - half_res]
        top_right = [lon + half_res, lat + half_res]
        top_left = [lon - half_res, lat + half_res]

        # Closed polygon loop coordinates
        polygon_coords = [[bottom_left, bottom_right, top_right, top_left, bottom_left]]

        # Package the ML properties
        properties = {
            "cell_id": cid,
            "fire_prob": float(row["fire_prob"]),
            "fire_pred": int(row.get("fire_pred", 0)),
            "risk_tier": str(row.get("risk_tier", "unknown")),
            "model_std": _safe_float(row.get("model_std", 0.0)),
            "date": forecast_date or "",
            "forecast_day": forecast_day,
            "temp": _safe_float(row.get("temp")),
            "humidity": _safe_float(row.get("humidity")),
            "wind": _safe_float(row.get("wind")),
            "vpd": _safe_float(row.get("vpd")),
        }

        # Add spread vectors if they were computed
        if "spread_bearing_deg" in row and not pd.isna(row["spread_bearing_deg"]):
            properties["spread_bearing_deg"] = _safe_float(row["spread_bearing_deg"])
            properties["spread_intensity"] = str(row.get("spread_intensity", "none"))

        # Inject explanations if available for detailed sidebars and tooltips
        if cid in expl_lookup:
            e = expl_lookup[cid]
            properties["model_confidence"] = e.get("model_confidence", "HIGH")
            properties["top_reasons"] = e.get("top_reasons", [])
            properties["mitigating_factors"] = e.get("mitigating_factors", [])
            properties["raw_shap"] = e.get("raw_shap", {})

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": polygon_coords
            },
            "properties": properties
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


def build_7day_geojson_pack(
    day_results: list[pd.DataFrame],
    explanations_by_day: Optional[list[list[dict]]] = None,
    resolution: float = 0.1,
    min_prob: float = 0.15,
) -> list[dict]:
    """
    Build a list of 7 GeoJSON FeatureCollections, one for each day of the forecast.
    """
    features_pack = []
    for day_idx, df_day in enumerate(day_results):
        date_str = str(df_day["acq_date"].iloc[0])[:10] if "acq_date" in df_day.columns else ""
        expls = explanations_by_day[day_idx] if (explanations_by_day and day_idx < len(explanations_by_day)) else None
        
        geojson = scored_df_to_geojson(
            scored_df=df_day,
            explanations=expls,
            resolution=resolution,
            min_prob=min_prob,
            forecast_date=date_str,
            forecast_day=day_idx + 1,
        )
        features_pack.append(geojson)
    return features_pack


def build_spread_arrow_geojson(
    scored_df: pd.DataFrame,
    risk_threshold: float = 0.40,
) -> dict:
    """
    Build a GeoJSON FeatureCollection of LineStrings for fire spread arrows.
    """
    ARROW_LENGTH = {
        "calm": 0.03,
        "light": 0.06,
        "moderate": 0.12,
        "rapid": 0.22,
        "extreme": 0.35,
    }

    features = []
    flagged = scored_df[
        (scored_df["fire_prob"] >= risk_threshold)
        & scored_df["spread_bearing_deg"].notna()
    ]

    for _, row in flagged.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        bearing = float(row["spread_bearing_deg"])
        intensity = str(row.get("spread_intensity", "moderate"))

        arrow_len = ARROW_LENGTH.get(intensity, 0.12)
        bearing_rad = np.deg2rad(bearing)
        dlat = arrow_len * np.cos(bearing_rad)
        dlon = arrow_len * np.sin(bearing_rad)

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [lon, lat],
                        [lon + dlon, lat + dlat],
                    ],
                },
                "properties": {
                    "cell_id": str(row.get("cell_id", f"{lat}_{lon}")),
                    "fire_prob": round(float(row["fire_prob"]), 4),
                    "spread_bearing": round(bearing, 1),
                    "spread_intensity": intensity,
                    "risk_tier": str(row.get("risk_tier", "high")),
                    "arrow_colour": RISK_COLOURS.get(
                        str(row.get("risk_tier", "high")), "#c0392b"
                    ),
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features
    }


def save_geojson(geojson_dict: dict, file_path: Union[str, Path]):
    """Saves the GeoJSON dictionary directly to a physical file."""
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_dict, f, indent=2)
    logger.info(f"Saved GeoJSON -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")