"""
scripts/run_case_study.py

Comprehensive case study analysis for Tamil Nadu wildfire events.
Now includes:
  - Wildfire risk prediction (existing)
  - Forest cover mapping (new)
  - Habitat change tracking (new)
  - Wildlife habitat suitability (new)
  - Fire spread vectors (existing)
  - GeoJSON export for dashboard (existing)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from src.config import DATA_RAW
from src.inference.engine import WildfireInferenceEngine, _prob_to_tier
from src.data.era5 import ERA5Lookup
from src.data.evi import EVILookup
from src.features.engineering import add_cyclical_month, add_interactions
from scripts.extract_real_features import sample_raster, ESA_CLASSES
from src.spread.direction import compute_spread_vectors
from src.inference.map_export import scored_df_to_geojson, save_geojson
from src.ecology.habitat_monitoring import (
    WildlifeHabitatScore,
    monitor_region_habitat,
    classify_habitat_health,
)

# ── Target events ─────────────────────────────────────────────────────────────

TARGET_ZONES = {
    "Kodaikanal_2024": {
        "lat_min": 10.0, "lat_max": 10.5,
        "lon_min": 77.2, "lon_max": 77.8,
        "date": "2024-05-01",
    },
    "Boluvampatti_2025": {
        "lat_min": 10.8, "lat_max": 11.2,
        "lon_min": 76.5, "lon_max": 77.0,
        "date": "2025-02-20",
    },
    "Mudumalai_Sathyamangalam_2024": {
        "lat_min": 11.4, "lat_max": 11.8,
        "lon_min": 76.4, "lon_max": 77.4,
        "date": "2024-03-15",
    },
}

# ESA land cover labels for the report
ESA_LABELS = {
    "lc_tree_cover":          "Dense Tree Cover",
    "lc_shrubland":           "Shrubland",
    "lc_grassland":           "Grassland",
    "lc_cropland":            "Cropland",
    "lc_built_up":            "Built-up / Urban",
    "lc_sparse_veg":          "Sparse Vegetation",
    "lc_snow_ice":            "Snow / Ice",
    "lc_water":               "Water Bodies",
    "lc_herbaceous_wetland":  "Herbaceous Wetland",
    "lc_mangroves":           "Mangroves",
    "lc_moss_lichen":         "Moss / Lichen",
}

RESULTS_DIR  = Path("results/metrics")
FIGURES_DIR  = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_local_grid(bounds: dict, resolution: float = 0.05) -> pd.DataFrame:
    lats = np.arange(bounds["lat_min"], bounds["lat_max"], resolution)
    lons = np.arange(bounds["lon_min"], bounds["lon_max"], resolution)
    rows = [
        {"latitude": round(float(la), 4), "longitude": round(float(lo), 4)}
        for la in lats for lo in lons
    ]
    df = pd.DataFrame(rows)
    df["cell_id"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
    return df


def forest_cover_report(df: pd.DataFrame, event_name: str) -> dict:
    """
    Calculate what percentage of the study area is each land cover type.
    Uses the ESA columns already in df (lc_tree_cover, lc_shrubland, etc.)
    """
    total = len(df)
    report = {"event": event_name, "total_cells": total}

    for col, label in ESA_LABELS.items():
        if col in df.columns:
            count = int(df[col].sum())
            report[label] = {
                "cells": count,
                "pct":   round(count / total * 100, 2),
            }

    # Dominant land cover
    lc_cols  = [c for c in ESA_LABELS if c in df.columns]
    dom_col  = df[lc_cols].sum().idxmax() if lc_cols else "unknown"
    report["dominant_cover"] = ESA_LABELS.get(dom_col, dom_col)

    return report


def habitat_months_for_event(event_date: str) -> list:
    """Return 6 monthly date strings ending on event_date."""
    event = pd.Timestamp(event_date)
    months = pd.date_range(end=event, periods=6, freq="MS")
    return [str(m.date()) for m in months]


def print_section(title: str):
    logger.info("")
    logger.info("─" * 60)
    logger.info(f"  {title}")
    logger.info("─" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  WILDFIRE_V2  —  COMPREHENSIVE CASE STUDY ANALYSIS")
    logger.info("=" * 60)

    engine = WildfireInferenceEngine()
    era5   = ERA5Lookup(years=[2024, 2025])
    evi    = EVILookup()

    for event_name, config in TARGET_ZONES.items():
        logger.info("")
        logger.info("█" * 60)
        logger.info(f"  CASE STUDY: {event_name.upper()}")
        logger.info(f"  Date: {config['date']}")
        logger.info("█" * 60)

        # ── 1. Build grid ──────────────────────────────────────────────
        print_section("Step 1 — Building spatial grid")
        df = build_local_grid(config, resolution=0.05)
        df["acq_date"] = config["date"]
        logger.info(f"Grid: {len(df)} cells at 0.05° resolution")

        # ── 2. Spatial features ────────────────────────────────────────
        print_section("Step 2 — Extracting SRTM elevation and ESA land cover")
        df = sample_raster(df, DATA_RAW / "srtm_india.tif", "elevation_m")
        df = sample_raster(df, DATA_RAW / "esa_landcover_india.vrt", "raw_landcover_code")
        for code, col_name in ESA_CLASSES.items():
            df[col_name] = (df["raw_landcover_code"] == code).astype(int)
        logger.info("Raster sampling complete")

        # ── 3. Forest cover report ─────────────────────────────────────
        print_section("Step 3 — Forest Cover Mapping")
        fc_report = forest_cover_report(df, event_name)
        logger.info(f"Dominant cover type: {fc_report['dominant_cover']}")
        for label, stats in fc_report.items():
            if isinstance(stats, dict):
                logger.info(f"  {label:30s}: {stats['pct']:6.2f}%  ({stats['cells']} cells)")

        fc_path = RESULTS_DIR / f"{event_name}_forest_cover.json"
        fc_path.write_text(json.dumps(fc_report, indent=2))
        logger.info(f"Saved → {fc_path}")

        # ── 4. Climate + vegetation enrichment ────────────────────────
        print_section("Step 4 — ERA5 weather and MODIS EVI enrichment")
        df = era5.enrich_dataframe(df, date_col="acq_date", hour=12)
        df = evi.enrich_dataframe(df, date_col="acq_date")
        df = add_cyclical_month(df, "acq_date")
        df = add_interactions(df)
        df = df.dropna(subset=["temp", "evi"])
        logger.info(f"Valid cells after enrichment: {len(df)}")

        # ── 5. Habitat change tracking ─────────────────────────────────
        print_section("Step 5 — Habitat Change Tracking (EVI time series)")
        months = habitat_months_for_event(config["date"])
        logger.info(f"Analysing months: {months}")
        habitat_df = monitor_region_habitat(
            grid_df=df,
            evi_lookup=evi,
            months=months,
            region_name=event_name,
        )

        n_excellent = (habitat_df["health_status"] == "Excellent").sum()
        n_good      = (habitat_df["health_status"] == "Good").sum()
        n_stressed  = (habitat_df["health_status"] == "Stressed").sum()
        n_degraded  = (habitat_df["health_status"] == "Degraded").sum()
        n_concern   = habitat_df["has_concern"].sum()

        logger.info(f"  Excellent: {n_excellent} | Good: {n_good} | Stressed: {n_stressed} | Degraded: {n_degraded}")
        logger.info(f"  Cells with significant EVI drop: {n_concern}")

        habitat_csv = RESULTS_DIR / f"{event_name}_habitat_changes.csv"
        habitat_df.to_csv(habitat_csv, index=False)
        logger.info(f"Saved → {habitat_csv}")

        # ── 6. Wildlife habitat suitability ───────────────────────────
        print_section("Step 6 — Wildlife Habitat Suitability")
        df["tiger_score"]    = WildlifeHabitatScore.tiger_score(df)
        df["elephant_score"] = WildlifeHabitatScore.elephant_score(df)
        df["leopard_score"]  = WildlifeHabitatScore.leopard_score(df)

        THRESHOLD = 0.5
        n_tiger    = (df["tiger_score"]    > THRESHOLD).sum()
        n_elephant = (df["elephant_score"] > THRESHOLD).sum()
        n_leopard  = (df["leopard_score"]  > THRESHOLD).sum()
        logger.info(f"  Suitable for tigers:    {n_tiger} cells")
        logger.info(f"  Suitable for elephants: {n_elephant} cells")
        logger.info(f"  Suitable for leopards:  {n_leopard} cells")

        # ── 7. Wildfire risk prediction ────────────────────────────────
        print_section("Step 7 — Wildfire Risk Prediction")
        X = engine._validate_feature_schema(df)
        fire_prob = engine.model.predict_proba(X)[:, 1]

        df["fire_prob"]  = np.round(fire_prob, 4)
        df["risk_tier"]  = [_prob_to_tier(p) for p in fire_prob]
        df["model_std"]  = engine._compute_model_disagreement(X)
        df["wind_u"]     = -df["wind"] * 0.8
        df["wind_v"]     =  df["wind"] * 0.2

        n_high    = (fire_prob >= 0.50).sum()
        n_extreme = (fire_prob >= 0.65).sum()
        logger.info(f"  High risk (≥0.50):    {n_high} cells")
        logger.info(f"  Extreme risk (≥0.65): {n_extreme} cells")
        logger.info(f"  Mean probability:     {fire_prob.mean():.3f}")

        # ── 8. Fire spread vectors ─────────────────────────────────────
        print_section("Step 8 — Fire Spread Direction")
        df = compute_spread_vectors(df, risk_threshold=0.40)
        logger.info("Spread vectors computed")

        # ── 9. Full event summary ──────────────────────────────────────
        print_section("Summary")
        logger.info(f"  Total cells analysed : {len(df)}")
        logger.info(f"  Dominant land cover  : {fc_report['dominant_cover']}")
        logger.info(f"  Mean EVI             : {df['evi'].mean():.3f}")
        logger.info(f"  Mean temperature     : {df['temp'].mean():.1f} °C")
        logger.info(f"  Mean fire probability: {df['fire_prob'].mean():.3f}")
        logger.info(f"  High-risk cells      : {n_high}")
        logger.info(f"  Habitat concerns     : {n_concern} cells with EVI drop")
        logger.info(f"  Tiger habitat cells  : {n_tiger}")
        logger.info(f"  Elephant habitat     : {n_elephant}")

        # ── 10. Export GeoJSON to dashboard ──────────────────────────
        print_section("Step 9 — Exporting to Dashboard")
        out_file = Path(f"wildfire-react-dashboard/public/{event_name}.geojson")
        geojson  = scored_df_to_geojson(
            scored_df=df,
            resolution=0.05,
            min_prob=0.05,
            forecast_date=config["date"],
        )
        save_geojson(geojson, out_file)
        logger.info(f"GeoJSON saved → {out_file}")

        logger.info(f"\n✅  {event_name} complete!")

    logger.info("")
    logger.info("=" * 60)
    logger.info("  ALL CASE STUDIES COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()