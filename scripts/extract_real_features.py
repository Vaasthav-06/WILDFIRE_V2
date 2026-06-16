"""
scripts/extract_real_features.py

Extracts real physical measurements (Elevation + Land Cover) from local GeoTIFFs
by sampling spatial coordinates, then applies standard feature engineering.
Outputs the final 'training_features.csv' used by the model.
"""

import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.rule import Rule

from src.config import DATA_PROCESSED, DATA_RAW, FEATURE_COLS
from src.features.engineering import build_features
from src.data.evi import EVILookup  # Adjust import based on your exact module structure

logger.add("logs/extract_real_features.log", rotation="10 MB", level="INFO")
console = Console()

def phase(title: str):
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))

# ESA WorldCover Map Values to Column mapping
ESA_CLASSES = {
    10: "lc_tree_cover",
    20: "lc_shrubland",
    30: "lc_grassland",
    40: "lc_cropland",
    50: "lc_built_up",
    60: "lc_sparse_veg",
    70: "lc_snow_ice",
    80: "lc_water",
    90: "lc_herbaceous_wetland",
    95: "lc_mangroves",
    100: "lc_moss_lichen"
}

def sample_raster(df: pd.DataFrame, raster_path: Path, col_name: str) -> pd.DataFrame:
    """Samples a raster file at exactly the lat/lon points in the dataframe."""
    logger.info(f"Sampling raster: {raster_path.name}")
    
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file missing: {raster_path}")

    # rasterio expects (longitude, latitude) coordinate pairs
    coords = [(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    
    with rasterio.open(raster_path) as src:
        # Sample the raster; returns a generator of arrays
        sampled_values = [val[0] for val in src.sample(coords)]
        
    df[col_name] = sampled_values
    return df

def main():
    phase("Step 1: Load Base Training Data")
    input_path = DATA_PROCESSED / "training_data.csv"
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path.name}")

    phase("Step 2: Geospatial Raster Extraction")
    # 1. Elevation (SRTM)
    srtm_path = DATA_RAW / "srtm_india.tif"
    df = sample_raster(df, srtm_path, "elevation_m")
    
    # 2. Land Cover (ESA WorldCover Virtual Spatial Index Header)
    # Changed from 'esa_landcover_india.tif' to target our optimized multi-tile VRT index
    esa_vrt_path = DATA_RAW / "esa_landcover_india.vrt"
    df = sample_raster(df, esa_vrt_path, "raw_landcover_code")

    # One-hot encode the land cover codes exactly matching config.py
    logger.info("One-hot encoding ESA WorldCover categorical classes...")
    for code, col_name in ESA_CLASSES.items():
        df[col_name] = (df["raw_landcover_code"] == code).astype(int)
    
    # Drop the raw code column to keep it clean
    df.drop(columns=["raw_landcover_code"], inplace=True)

    # ... [Keep the remaining Steps 3 & 4 unchanged] ...

    phase("Step 3: Base Feature Engineering")
    # Apply time encodings, EVI lookups, and interactions
    evi_lookup = EVILookup()
    df = build_features(df, date_col="acq_date", evi_lookup=evi_lookup)

    phase("Step 4: Final Validation & Save")
    # Ensure all target features are present and clean
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CRITICAL: Missing columns before save: {missing_cols}")

    # Keep only the features + target + essential metadata
    output_path = DATA_PROCESSED / "training_features.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Final feature matrix saved → {output_path}")
    logger.info(f"Ready for Model Training. Matrix shape: {df.shape}")

if __name__ == "__main__":
    main()