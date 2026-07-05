"""
scripts/predict_poaching_risk.py

Generates poaching risk maps for the three Tamil Nadu case study regions.
Run this after run_case_study.py has already extracted spatial features.

Usage:
    python scripts/predict_poaching_risk.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from src.config import DATA_RAW, DATA_PROCESSED
from src.wildlife.poaching_risk import PoachingRiskModel, build_poaching_features
from src.ecology.habitat_monitoring import WildlifeHabitatScore
from scripts.extract_real_features import sample_raster, ESA_CLASSES
from src.data.era5 import ERA5Lookup
from src.data.evi import EVILookup
from src.features.engineering import add_cyclical_month, add_interactions

CASE_STUDIES = {
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

OUTPUT_DIR   = Path("results/metrics")
GEOJSON_DIR  = Path("wildfire-react-dashboard/public")
MODELS_DIR   = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_grid(bounds: dict, resolution: float = 0.05) -> pd.DataFrame:
    lats = np.arange(bounds["lat_min"], bounds["lat_max"], resolution)
    lons = np.arange(bounds["lon_min"], bounds["lon_max"], resolution)
    rows = [
        {"latitude": round(float(la), 4), "longitude": round(float(lo), 4)}
        for la in lats for lo in lons
    ]
    return pd.DataFrame(rows)


def export_geojson(df: pd.DataFrame, event_name: str):
    """Save poaching risk as GeoJSON for the React dashboard."""
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            },
            "properties": {
                "poaching_risk":  round(float(row["poaching_risk"]), 4),
                "poaching_tier":  str(row["poaching_tier"]),
                "wildlife_index": round(float(row["wildlife_index"]), 4),
                "elevation_m":    round(float(row["elevation_m"]), 1),
                "evi":            round(float(row["evi"]), 4),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "name": f"{event_name} — Poaching Risk",
        "features": features,
    }

    out_path = GEOJSON_DIR / f"{event_name}_poaching.geojson"
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    logger.info(f"[Poaching] GeoJSON saved → {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("POACHING RISK PREDICTION — Tamil Nadu Case Studies")
    logger.info("=" * 60)

    era5 = ERA5Lookup(years=[2024, 2025])
    evi  = EVILookup()

    for event_name, config in CASE_STUDIES.items():
        logger.info(f"\n{'─'*50}")
        logger.info(f"Region: {event_name}")
        logger.info(f"{'─'*50}")

        # 1. Build grid
        df = build_grid(config, resolution=0.05)
        df["acq_date"] = config["date"]
        logger.info(f"Grid: {len(df)} cells")

        # 2. Spatial features (elevation + land cover)
        df = sample_raster(df, DATA_RAW / "srtm_india.tif", "elevation_m")
        df = sample_raster(df, DATA_RAW / "esa_landcover_india.vrt", "raw_landcover_code")
        for code, col_name in ESA_CLASSES.items():
            df[col_name] = (df["raw_landcover_code"] == code).astype(int)

        # 3. Weather + EVI
        df = era5.enrich_dataframe(df, date_col="acq_date", hour=12)
        df = evi.enrich_dataframe(df, date_col="acq_date")
        df = add_cyclical_month(df, "acq_date")
        df = add_interactions(df)
        df = df.dropna(subset=["temp", "evi"])
        logger.info(f"After enrichment: {len(df)} valid cells")

        # 4. Wildlife habitat scores
        df["tiger_score"]    = WildlifeHabitatScore.tiger_score(df)
        df["elephant_score"] = WildlifeHabitatScore.elephant_score(df)
        df["leopard_score"]  = WildlifeHabitatScore.leopard_score(df)
        df["wildlife_index"] = (
            df["tiger_score"] * 0.5 +
            df["elephant_score"] * 0.3 +
            df["leopard_score"] * 0.2
        )

        # 5. Build poaching feature matrix
        # Pass road_points / village_points / incident_df / patrol_df
        # if you have them. Otherwise the function uses safe defaults.
        feat_df = build_poaching_features(
            grid_df       = df,
            road_points   = None,   # Replace with pd.read_csv("data/raw/roads.csv") when available
            village_points = None,  # Replace with pd.read_csv("data/raw/villages.csv") when available
            incident_df   = None,   # Replace with pd.read_csv("data/raw/poaching_incidents.csv") when available
            patrol_df     = None,   # Replace with pd.read_csv("data/raw/patrol_logs.csv") when available
        )

        # 6. Train with synthetic labels (replace with real incidents when available)
        #    Synthetic rule: high wildlife index + close to road + long since patrol = likely poaching
        synthetic_prob = (
            df["wildlife_index"].values * 0.5 +
            (1 / (feat_df["distance_to_road_km"].values + 1)) * 0.3 +
            (feat_df["days_since_last_patrol"].values / 30).clip(0, 1) * 0.2
        )
        synthetic_labels = pd.Series(
            (synthetic_prob > 0.40).astype(int), index=feat_df.index
        )
        logger.info(f"Synthetic positives: {synthetic_labels.sum()} / {len(synthetic_labels)}")

        from src.wildlife.poaching_risk import PoachingRiskScorer

        scorer = PoachingRiskScorer()
        df = scorer.compute_score(feat_df)
        logger.info(f"Rule-based poaching risk computed for {len(df)} cells")

        # Save the scoring logic for audit/transparency
        PoachingRiskScorer.save_metadata(OUTPUT_DIR / "poaching_scoring_logic.json")

        # 8. Summary
        for tier in ["Critical", "High", "Moderate", "Low"]:
            n = (df["poaching_tier"] == tier).sum()
            logger.info(f"  {tier}: {n} cells")

        # 9. Save
        csv_out = OUTPUT_DIR / f"{event_name}_poaching_risk.csv"
        df[["latitude", "longitude", "elevation_m", "evi",
            "wildlife_index", "poaching_risk", "poaching_tier"]].to_csv(csv_out, index=False)
        logger.info(f"CSV → {csv_out}")

        scorer.save(str(MODELS_DIR / f"poaching_model_{event_name}.pkl"))
        export_geojson(df, event_name)

    logger.info("\n✅ Poaching risk analysis complete!")


if __name__ == "__main__":
    main()