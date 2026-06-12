"""
src/inference/map_export.py

GeoJSON export and map rendering utilities.

Converts the scored grid DataFrame into frontend-ready artifacts:

  1. GeoJSON FeatureCollection — grid cells as polygons with risk properties
  2. Spread arrow overlay      — directional arrows for high-risk cells
  3. 7-day temporal snapshots  — one GeoJSON per forecast day
  4. Folium interactive map    — for local development and notebook viewing

GeoJSON schema per feature (cell):
  geometry:    Polygon (0.1° × 0.1° bounding box)
  properties:
    cell_id            — stable identifier
    fire_prob          — P(fire) ∈ [0, 1]
    risk_tier          — "low"|"moderate"|"high"|"extreme"
    model_confidence   — "HIGH"|"MEDIUM"|"LOW"
    top_reason_short   — First sentence from WHY engine (≤80 chars)
    spread_bearing_deg — 0–360° or null
    spread_direction   — "NE" etc. or null
    spread_intensity   — "moderate" etc. or null
    forecast_date      — "YYYY-MM-DD"
    forecast_day       — 1–7 (null for 24h)
    temp               — °C
    humidity           — %
    wind               — m/s
    vpd                — kPa
    kbdi_approx        — 0–800

Colour encoding (matches risk_tier):
  low:      #2ecc71  (green)
  moderate: #f1c40f  (yellow)
  high:     #e67e22  (orange)
  extreme:  #c0392b  (deep red)

  Opacity: 0.3 + 0.5 × (1 - model_std / 0.3)
  High uncertainty → semi-transparent cell (visual uncertainty signal)
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

    This makes uncertain cells semi-transparent without hiding them.
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


# ── 24h GeoJSON export ────────────────────────────────────────────────────────


def scored_df_to_geojson(
    scored_df: pd.DataFrame,
    explanations: Optional[list[dict]] = None,
    resolution: float = 0.1,
    min_prob: float = 0.15,
    forecast_date: Optional[str] = None,
    forecast_day: Optional[int] = None,
) -> dict:
    """
    Convert a scored grid DataFrame to a GeoJSON FeatureCollection.

    Args:
        scored_df:     Output from WildfireInferenceEngine._score()
                       (optionally augmented with spread columns).
        explanations:  List of WHY engine dicts. If provided, top_reason_short
                       is added to each high-risk cell's properties.
        resolution:    Grid cell size in degrees (must match inference grid).
        min_prob:      Cells with P(fire) < min_prob are EXCLUDED from output
                       to reduce payload size. Callers can set to 0 to include all.
        forecast_date: ISO date string for this snapshot.
        forecast_day:  1–7 for 7-day forecasts; None for 24h.

    Returns:
        GeoJSON FeatureCollection dict ready for json.dumps().
        Estimated size: ~400 bytes/cell at 0.1° → ~1.3 MB for India at 3200 cells.
        With min_prob=0.15, typical payload ≈ 200–600 KB.
    """
    # Build explanation lookup keyed by cell_id
    expl_lookup: dict[str, dict] = {}
    if explanations:
        for expl in explanations:
            expl_lookup[expl["cell_id"]] = expl

    features = []
    n_skipped = 0

    for _, row in scored_df.iterrows():
        prob = float(row.get("fire_prob", 0))
        if prob < min_prob:
            n_skipped += 1
            continue

        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cid = str(row.get("cell_id", f"{lat}_{lon}"))
        tier = str(row.get("risk_tier", "low"))
        std = float(row.get("model_std", 0))

        # Spread direction (may be None)
        spread_bearing = _safe_float(row.get("spread_bearing_deg"))
        spread_direction = None
        spread_intensity = str(row.get("spread_intensity", "none"))
        if spread_bearing is not None:
            from src.spread.direction import bearing_to_compass

            spread_direction = bearing_to_compass(spread_bearing)

        # Top reason (first sentence from WHY engine if available)
        top_reason = None
        confidence = "HIGH"
        if cid in expl_lookup:
            expl = expl_lookup[cid]
            reasons = expl.get("top_reasons", [])
            top_reason = reasons[0][:100] if reasons else None  # cap at 100 chars
            confidence = expl.get("model_confidence", "HIGH")

        props = {
            "cell_id": cid,
            "fire_prob": round(prob, 4),
            "risk_tier": tier,
            "risk_colour": RISK_COLOURS.get(tier, "#95a5a6"),
            "opacity": _compute_opacity(std),
            "model_confidence": confidence,
            "model_std": round(std, 4),
            "top_reason_short": top_reason,
            "spread_bearing_deg": spread_bearing,
            "spread_direction": spread_direction,
            "spread_intensity": spread_intensity,
            "forecast_date": forecast_date,
            "forecast_day": forecast_day,
            # Weather snapshot for tooltip
            "temp_c": _safe_float(row.get("temp")),
            "humidity_pct": _safe_float(row.get("humidity")),
            "wind_ms": _safe_float(row.get("wind")),
            "vpd_kpa": _safe_float(row.get("vpd")),
            "kbdi": _safe_float(row.get("kbdi_approx")),
        }

        features.append(
            {
                "type": "Feature",
                "geometry": _cell_polygon(lat, lon, resolution),
                "properties": props,
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "n_cells": len(features),
            "n_skipped": n_skipped,
            "min_prob": min_prob,
            "resolution_deg": resolution,
            "forecast_date": forecast_date,
            "forecast_day": forecast_day,
            "tier_counts": {
                tier: sum(1 for f in features if f["properties"]["risk_tier"] == tier)
                for tier in RISK_COLOURS
            },
        },
    }

    logger.info(
        f"GeoJSON built: {len(features):,} cells exported, "
        f"{n_skipped:,} skipped (P < {min_prob})"
    )
    return geojson


# ── 7-day temporal snapshot pack ─────────────────────────────────────────────


def build_7day_geojson_pack(
    day_results: list[pd.DataFrame],
    explanations_by_day: Optional[list[list[dict]]] = None,
    resolution: float = 0.1,
    min_prob: float = 0.15,
) -> list[dict]:
    """
    Build a list of 7 GeoJSON snapshots for the temporal slider.

    Each snapshot is a complete FeatureCollection for one forecast day.
    The frontend loads all 7 at once and renders them via a time slider
    without additional API calls.

    Args:
        day_results:           List of 7 DataFrames from engine.predict_7day().
        explanations_by_day:   Optional list of 7 explanation lists.
        resolution:            Grid cell size in degrees.
        min_prob:              Minimum probability to include.

    Returns:
        List of 7 GeoJSON dicts. Client accesses snapshots[0..6].
    """
    snapshots = []

    for day_idx, df_day in enumerate(day_results):
        forecast_date = (
            str(df_day["forecast_date"].iloc[0])
            if "forecast_date" in df_day.columns
            else None
        )
        explanations = (
            explanations_by_day[day_idx]
            if explanations_by_day and day_idx < len(explanations_by_day)
            else None
        )

        geojson = scored_df_to_geojson(
            scored_df=df_day,
            explanations=explanations,
            resolution=resolution,
            min_prob=min_prob,
            forecast_date=forecast_date,
            forecast_day=day_idx + 1,
        )
        snapshots.append(geojson)

    logger.info(f"7-day GeoJSON pack: {len(snapshots)} snapshots built")
    return snapshots


# ── Spread arrow overlay ──────────────────────────────────────────────────────


def build_spread_arrow_geojson(
    scored_df: pd.DataFrame,
    risk_threshold: float = 0.40,
) -> dict:
    """
    Build a GeoJSON LineString overlay for spread direction arrows.

    Each arrow is a LineString from the cell centroid in the direction of
    predicted fire spread. Arrow length scales with spread intensity.

    This is a SEPARATE layer from the risk grid — rendered on top of it.
    The frontend can toggle this layer independently.

    Arrow scaling:
        calm:     0.03° (very short)
        light:    0.06°
        moderate: 0.12°
        rapid:    0.22°
        extreme:  0.35°

    Returns:
        GeoJSON FeatureCollection of LineString features.
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

        # Convert bearing to dx, dy in degrees
        bearing_rad = np.deg2rad(bearing)
        # bearing 0° = North = +lat direction
        # bearing 90° = East = +lon direction
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

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "n_arrows": len(features),
            "risk_threshold": risk_threshold,
            "layer_type": "spread_arrows",
        },
    }

    logger.info(f"Spread arrow overlay: {len(features):,} arrows generated")
    return geojson


# ── Save to disk ──────────────────────────────────────────────────────────────


def save_geojson(geojson: dict, path: Union[str, Path]) -> None:
    """Save a GeoJSON dict to disk with compact separators to minimise file size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"), ensure_ascii=False)
    size_kb = path.stat().st_size / 1024
    logger.info(f"Saved GeoJSON → {path} ({size_kb:.1f} KB)")


# ── Folium map (development / notebook) ──────────────────────────────────────


def render_folium_map(
    scored_df: pd.DataFrame,
    explanations: Optional[list[dict]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> "folium.Map":
    """
    Render an interactive Folium map of the risk grid.

    This is a development/notebook tool — NOT the production frontend.
    The production frontend renders the GeoJSON client-side with Leaflet/Mapbox.

    Layers:
      1. Risk grid (coloured polygons)
      2. Spread arrows (for high-risk cells)
      3. Tooltip on hover (fire_prob, tier, top reason, weather snapshot)

    Args:
        scored_df:    Scored grid DataFrame.
        explanations: WHY engine output (optional).
        output_path:  If provided, saves the map as HTML.

    Returns:
        folium.Map object.
    """
    try:
        import folium
    except ImportError:
        raise ImportError(
            "folium is required for render_folium_map(). "
            "Install with: pip install folium"
        )

    expl_lookup = {}
    if explanations:
        for e in explanations:
            expl_lookup[e["cell_id"]] = e

    # Centre map on India
    india_center = [22.0, 82.0]
    fmap = folium.Map(
        location=india_center,
        zoom_start=5,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # Risk grid layer
    risk_layer = folium.FeatureGroup(name="Wildfire Risk Grid", show=True)
    arrow_layer = folium.FeatureGroup(name="Spread Direction", show=True)

    half = 0.1 / 2

    for _, row in scored_df.iterrows():
        prob = float(row.get("fire_prob", 0))
        if prob < 0.10:
            continue

        lat = float(row["latitude"])
        lon = float(row["longitude"])
        tier = str(row.get("risk_tier", "low"))
        std = float(row.get("model_std", 0))
        cid = str(row.get("cell_id", f"{lat}_{lon}"))
        color = RISK_COLOURS.get(tier, "#95a5a6")
        opacity = _compute_opacity(std)

        # Tooltip content
        expl = expl_lookup.get(cid, {})
        top_reason = (expl.get("top_reasons") or ["No explanation available"])[0]
        confidence = expl.get("model_confidence", "—")

        tooltip_html = (
            f"<b>P(Fire): {prob:.1%}</b> [{tier.upper()}]<br>"
            f"Confidence: {confidence}<br>"
            f"Temp: {row.get('temp', 'N/A')}°C | "
            f"Wind: {row.get('wind', 'N/A')} m/s<br>"
            f"VPD: {row.get('vpd', 'N/A'):.2f} kPa | "
            f"KBDI: {row.get('kbdi_approx', 'N/A'):.0f}<br>"
            f"<i>{top_reason}</i>"
        )

        folium.Rectangle(
            bounds=[[lat - half, lon - half], [lat + half, lon + half]],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            weight=0,
            tooltip=folium.Tooltip(tooltip_html),
        ).add_to(risk_layer)

        # Spread arrow
        bearing = row.get("spread_bearing_deg")
        if prob >= 0.40 and bearing is not None and not pd.isna(bearing):
            intensity = str(row.get("spread_intensity", "moderate"))
            ARROW_LENGTH = {
                "calm": 0.03,
                "light": 0.06,
                "moderate": 0.12,
                "rapid": 0.22,
                "extreme": 0.35,
            }
            arrow_len = ARROW_LENGTH.get(intensity, 0.12)
            bearing_rad = np.deg2rad(float(bearing))
            dlat = arrow_len * np.cos(bearing_rad)
            dlon = arrow_len * np.sin(bearing_rad)
            folium.PolyLine(
                locations=[[lat, lon], [lat + dlat, lon + dlon]],
                color=color,
                weight=2,
                opacity=0.9,
                tooltip=f"Spread: {bearing:.0f}° ({intensity})",
            ).add_to(arrow_layer)

    risk_layer.add_to(fmap)
    arrow_layer.add_to(fmap)
    folium.LayerControl().add_to(fmap)

    if output_path:
        fmap.save(str(output_path))
        logger.info(f"Folium map saved → {output_path}")

    return fmap
