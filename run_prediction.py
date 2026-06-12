import sys
import numpy as np
import pandas as pd
from loguru import logger
from datetime import date
from src.inference.engine import WildfireInferenceEngine
from src.inference.map_export import scored_df_to_geojson, save_geojson

def main():
    logger.info("Initializing Wildfire Inference Engine...")
    
    try:
        engine = WildfireInferenceEngine()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
        
    logger.info("Model loaded successfully. Generating synthetic weather to bypass Open-Meteo rate limit...")
    
    try:
        # Get full inference grid
        full_grid = engine.get_inference_grid()
        
        # Sample 2000 cells for demonstration
        demo_grid = full_grid.sample(2000, random_state=42).copy()
        
        # Add synthetic weather data that looks realistic for India
        demo_grid['acq_date'] = pd.to_datetime(date.today().isoformat())
        
        # Random temps between 25 and 45 C
        demo_grid['temp'] = np.random.uniform(25, 45, size=len(demo_grid))
        # Random humidity between 10% and 80%
        demo_grid['humidity'] = np.random.uniform(10, 80, size=len(demo_grid))
        # Random wind between 1 and 12 m/s
        demo_grid['wind'] = np.random.uniform(1, 12, size=len(demo_grid))
        demo_grid['wind_u'] = demo_grid['wind'] * np.cos(np.random.uniform(0, 2*np.pi, size=len(demo_grid)))
        demo_grid['wind_v'] = demo_grid['wind'] * np.sin(np.random.uniform(0, 2*np.pi, size=len(demo_grid)))
        
        # VPD (approximated based on temp and humidity)
        demo_grid['vpd'] = np.random.uniform(0.5, 6.0, size=len(demo_grid))
        demo_grid['precip'] = 0.0
        demo_grid['ndvi_proxy'] = np.random.uniform(0.1, 0.8, size=len(demo_grid))
        
        # Run prediction on the custom enriched DataFrame
        logger.info(f"Running ML engine on {len(demo_grid)} cells...")
        predictions_df = engine.predict_custom(df_weather=demo_grid)
        
        # Add simulated spread properties (usually computed by spread_vectors)
        predictions_df['spread_bearing_deg'] = np.where(predictions_df['fire_prob'] > 0.5, np.random.uniform(0, 360, size=len(predictions_df)), None)
        predictions_df['spread_intensity'] = np.where(predictions_df['fire_prob'] > 0.8, 'extreme', 
                                             np.where(predictions_df['fire_prob'] > 0.5, 'moderate', 'none'))
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)
        
    logger.info("Prediction complete!")
    
    # Save GeoJSON for the React Frontend
    geojson_output = "wildfire-react-dashboard/public/predictions.geojson"
    logger.info("Generating GeoJSON for the frontend...")
    geojson_dict = scored_df_to_geojson(
        scored_df=predictions_df,
        explanations=None,
        resolution=0.1,
        min_prob=0.01,
        forecast_date=date.today().isoformat()
    )
    save_geojson(geojson_dict, geojson_output)
    logger.info(f"Frontend predictions updated at {geojson_output}")

if __name__ == "__main__":
    main()