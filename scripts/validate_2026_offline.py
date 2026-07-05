import pandas as pd
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from src.inference.engine import WildfireInferenceEngine
from src.config import DATA_RAW

console = Console()

def calculate_vpd(temp_series: pd.Series, humidity_series: pd.Series) -> pd.Series:
    """Vectorized calculation of Vapor Pressure Deficit (VPD) in kPa."""
    # Saturation Vapor Pressure (svp)
    svp = 0.6108 * np.exp((17.27 * temp_series) / (temp_series + 237.3))
    # Actual Vapor Pressure (act_vp)
    act_vp = svp * (humidity_series / 100.0)
    return svp - act_vp

def main():
    logger.info("Initializing Offline Vectorized Early-Warning Audit...")
    
    # 1. Load the NASA FIRMS Data (The 5,000 real fires)
    firms_path = DATA_RAW / "firms_2026_ytd.csv"
    if not firms_path.exists():
        logger.error(f"Missing fire data at {firms_path}")
        return
        
    fires_df = pd.read_csv(firms_path)
    
    # Filter for nominal/high confidence
    if 'confidence' in fires_df.columns:
        fires_df = fires_df[fires_df['confidence'].isin(['n', 'h', 'nominal', 'high']) | (fires_df['confidence'] > 50)].copy()

    # Standardize coordinate resolution
    fires_df['latitude'] = fires_df['latitude'].round(1)
    fires_df['longitude'] = fires_df['longitude'].round(1)
    
    # Calculate the T-1 Early Warning Date (Day Before Ignition)
    fires_df['acq_date'] = pd.to_datetime(fires_df['acq_date'].astype(str).str[:10])
    fires_df['target_weather_date'] = fires_df['acq_date'] - pd.Timedelta(days=1)
    
    # Convert back to string for clean merging
    fires_df['target_weather_date'] = fires_df['target_weather_date'].dt.strftime("%Y-%m-%d")
    fires_df['actual_fire_date'] = fires_df['acq_date'].dt.strftime("%Y-%m-%d")

    # 2. Load the Offline Weather Dataset
    weather_path = DATA_RAW / "india_weather_2026.parquet" # Or .csv if you prefer
    if not weather_path.exists():
        logger.error(f"Missing offline weather database at {weather_path}. Please download bulk data first.")
        return
        
    logger.info("Loading massive offline weather dataset into RAM...")
    if weather_path.suffix == '.parquet':
        weather_df = pd.read_parquet(weather_path)
    else:
        weather_df = pd.read_csv(weather_path)
        
    weather_df['date'] = weather_df['date'].astype(str)
    weather_df['latitude'] = weather_df['latitude'].round(1)
    weather_df['longitude'] = weather_df['longitude'].round(1)

    # 3. The Vectorized Merge (Instantaneous Cross-Reference)
    logger.info("Performing spatial-temporal cross-referencing...")
    merged_df = pd.merge(
        fires_df, 
        weather_df, 
        left_on=['latitude', 'longitude', 'target_weather_date'],
        right_on=['latitude', 'longitude', 'date'],
        how='inner'
    )
    
    # 4. Snap to Inference Grid (Topography + Landcover)
    engine = WildfireInferenceEngine()
    grid_df = engine.get_inference_grid()
    grid_df['latitude'] = grid_df['latitude'].round(1)
    grid_df['longitude'] = grid_df['longitude'].round(1)
    
    final_df = pd.merge(merged_df, grid_df, on=['latitude', 'longitude'], how='inner')
    
    # Calculate VPD offline using the vectorized function
    final_df['vpd'] = calculate_vpd(final_df['temp'], final_df['humidity'])
    final_df['precip'] = 0.0 # Default assumption unless included in offline weather
    
    # Clean up duplicate fires in the same cell on the same day
    final_df = final_df.drop_duplicates(subset=['latitude', 'longitude', 'actual_fire_date'])

    if final_df.empty:
        logger.warning("No matches found. Ensure date formats and coordinate rounding match exactly.")
        return

    logger.info(f"Successfully cross-referenced {len(final_df)} historical fires. Running XGBoost...")

    # 5. Run the Engine & Score
    results = engine.predict_custom(final_df)
    
    results['early_warning_success'] = results['risk_tier'].isin(['high', 'extreme'])
    caught_count = results['early_warning_success'].sum()
    total_evaluated = len(results)
    recall = (caught_count / total_evaluated) * 100

    # 6. Output Final Metrics
    table = Table(title="Offline Vectorized Early-Warning Audit (2026 YTD)", style="cyan")
    table.add_column("Metric", justify="left", style="white", no_wrap=True)
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Total Validated 2026 Fires", str(total_evaluated))
    table.add_row("Successfully Warned 24h Prior", str(caught_count))
    table.add_row("Early Warning Reliability (Recall)", f"{recall:.1f}%")
    table.add_row("Execution Strategy", "100% Offline Vectorized Join")
    
    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    main()