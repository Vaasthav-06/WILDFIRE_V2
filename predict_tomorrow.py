import sys
import time
from loguru import logger
from datetime import date, timedelta
from src.inference.engine import WildfireInferenceEngine
from src.inference.map_export import scored_df_to_geojson, save_geojson

def main():
    logger.info("Initializing Wildfire Inference Engine for Tomorrow's Prediction...")
    
    try:
        engine = WildfireInferenceEngine()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
        
    logger.info("Model loaded successfully.")
    
    try:
        # Get full inference grid
        full_grid = engine.get_inference_grid()
        
        # Sample a small grid to ensure we stay under the API limits
        # We will use batching implicitly inside the engine's fetch logic
        small_grid = full_grid.sample(400, random_state=42).copy()
        
        logger.info(f"Fetching 7-day forecast for a sample of {len(small_grid)} cells in slow batches...")
        
        # predict_7day returns a list of 7 DataFrames (tomorrow is index 0)
        day_results = engine.predict_7day(grid_df=small_grid)
        
        # Extract Tomorrow's prediction (Day 1 of the 7-day forecast)
        tomorrow_df = day_results[0]
        tomorrow_date = (date.today() + timedelta(days=1)).isoformat()
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)
        
    logger.info("Prediction complete!")
    
    # Save GeoJSON for the React Frontend
    geojson_output = "wildfire-react-dashboard/public/predictions.geojson"
    logger.info(f"Generating GeoJSON for Tomorrow ({tomorrow_date})...")
    geojson_dict = scored_df_to_geojson(
        scored_df=tomorrow_df,
        explanations=None,
        resolution=0.1,
        min_prob=0.01,
        forecast_date=tomorrow_date
    )
    save_geojson(geojson_dict, geojson_output)
    logger.info(f"Frontend predictions updated at {geojson_output} for {tomorrow_date}")

if __name__ == "__main__":
    main()
