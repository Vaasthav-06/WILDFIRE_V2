"""
scripts/build_inference_cache.py

Generates the static spatial grid cache for daily live inference.
Extracts Elevation and Land Cover for the 0.1° India grid and saves it to a fast Parquet file.
"""

import pandas as pd
from pathlib import Path
from loguru import logger
from rich.console import Console

from src.config import DATA_CACHE, DATA_RAW
from src.inference.forecast_ingest import build_india_inference_grid
from scripts.extract_real_features import sample_raster, ESA_CLASSES
from src.data.evi import EVILookup

logger.add("logs/build_cache.log", rotation="10 MB")
console = Console()

def main():
    logger.info("Step 1: Building base India coordinate grid (0.1° resolution)...")
    grid_df = build_india_inference_grid(resolution=0.1)
    
    logger.info(f"Step 2: Sampling SRTM Elevation for {len(grid_df):,} cells...")
    srtm_path = DATA_RAW / "srtm_india.tif"
    grid_df = sample_raster(grid_df, srtm_path, "elevation_m")
    
    logger.info("Step 3: Sampling ESA WorldCover Land Cover...")
    esa_path = DATA_RAW / "esa_landcover_india.tif"
    grid_df = sample_raster(grid_df, esa_path, "raw_landcover_code")
    
    for code, col_name in ESA_CLASSES.items():
        grid_df[col_name] = (grid_df["raw_landcover_code"] == code).astype(int)
    grid_df.drop(columns=["raw_landcover_code"], inplace=True)
    
    logger.info("Step 4: Fetching historical EVI median proxy...")
    # Add a dummy date to fetch a representative EVI baseline for the grid
    grid_df["acq_date"] = "2024-01-01" 
    evi = EVILookup()
    grid_df = evi.enrich_dataframe(grid_df, date_col="acq_date")
    grid_df.drop(columns=["acq_date"], inplace=True)
    
    # Save the cache
    out_path = DATA_CACHE / "india_spatial_grid.parquet"
    grid_df.to_parquet(out_path, index=False)
    logger.info(f"Success! Spatial grid cached → {out_path}")

if __name__ == "__main__":
    main()