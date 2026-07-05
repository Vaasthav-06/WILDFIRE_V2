import pandas as pd
from loguru import logger
from src.inference.engine import WildfireInferenceEngine
from src.data.era5 import ERA5Lookup, download_era5
from src.config import DATA_RAW

def main():
    logger.info("Step 1: Downloading 2026 ERA5 NetCDF Archive via CDS...")
    # This downloads the raw climate data for India for 2026
    # Note: Ensure your ~/.cdsapirc file is configured with your API key
    download_era5([2026])
    
    logger.info("Step 2: Loading spatial grid and ERA5 engine...")
    engine = WildfireInferenceEngine()
    grid = engine.get_inference_grid()
    
    # Initialize the fast xarray lookup for the downloaded 2026 data
    era5 = ERA5Lookup([2026])
    
    logger.info("Step 3: Flattening 6 months of spatial data into Parquet format...")
    # Generate dates from Jan 1, 2026 to Today
    dates = pd.date_range(start="2026-01-01", end="2026-06-23", freq="D")
    
    all_days = []
    
    # Process day-by-day to manage RAM efficiently
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        day_df = grid[['latitude', 'longitude']].copy()
        day_df['date'] = date_str
        
        # Use your project's built-in vector enrichment
        day_enriched = era5.enrich_dataframe(day_df, date_col='date', hour=12)
        all_days.append(day_enriched)
        
    logger.info("Step 4: Compiling and saving the massive offline database...")
    final_df = pd.concat(all_days, ignore_index=True)
    
    out_path = DATA_RAW / "india_weather_2026.parquet"
    final_df.to_parquet(out_path, index=False)
    
    logger.info(f"Success! Saved offline weather database to {out_path}")

if __name__ == "__main__":
    main()