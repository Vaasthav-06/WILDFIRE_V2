"""
src/risk/fire_risk_index.py

Unified Fire Risk Index (FRI).
Provides a transparent, formula-based composite risk score that complements the ML probability.
"""

import pandas as pd
import numpy as np

# Component weights (must sum to 1.0)
WEIGHTS = {
    "vegetation_stress": 0.25,
    "historical_density": 0.20,
    "terrain_risk": 0.15,
    "fuel_load": 0.15,
    "weather_severity": 0.25,
}

def normalize(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """Min-max normalization, clamped to [0, 1]."""
    if max_val == min_val:
        return pd.Series(0, index=series.index)
    normalized = (series - min_val) / (max_val - min_val)
    return normalized.clip(0.0, 1.0)

def compute_fri(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Fire Risk Index (0-100) for a dataframe of cells.
    Expects standard feature columns. Missing columns will assume a neutral (0) contribution.
    """
    result = df.copy()
    
    # 1. Vegetation Stress (Lower EVI = Higher Stress)
    if "evi" in df.columns:
        # High EVI (~0.6) = healthy, Low EVI (~0.0) = dry/stress
        veg_stress = 1.0 - normalize(df["evi"], 0.0, 0.6)
    else:
        veg_stress = pd.Series(0.0, index=df.index)
        
    # 2. Historical Density
    if "fire_count_5yr" in df.columns:
        # Cap at 50 historical fires for normalization
        hist_density = normalize(df["fire_count_5yr"], 0, 50)
    else:
        hist_density = pd.Series(0.0, index=df.index)
        
    # 3. Terrain Risk
    terrain_risk = pd.Series(0.0, index=df.index)
    if "elevation_m" in df.columns:
        # Lower elevation generally easier to burn (drier, more human access)
        # Higher elevation = cooler, wetter
        terrain_risk += (1.0 - normalize(df["elevation_m"], 0, 2500)) * 0.5
    if "slope_deg" in df.columns:
        # Steeper slope = faster fire spread
        terrain_risk += normalize(df["slope_deg"], 0, 45) * 0.5
        
    # 4. Fuel Load
    fuel_load = pd.Series(0.0, index=df.index)
    if "lc_tree_cover" in df.columns and "lc_shrubland" in df.columns and "lc_grassland" in df.columns:
        # Trees + Shrubs + Grass = Fuel
        total_fuel = df["lc_tree_cover"] + df["lc_shrubland"] + df["lc_grassland"]
        fuel_load = normalize(total_fuel, 0, 100)
        
    # 5. Weather Severity
    weather_severity = pd.Series(0.0, index=df.index)
    if "temp" in df.columns and "humidity" in df.columns and "wind" in df.columns:
        temp_norm = normalize(df["temp"], 25.0, 45.0)
        hum_norm = 1.0 - normalize(df["humidity"], 10.0, 60.0) # lower hum = higher severity
        wind_norm = normalize(df["wind"], 0.0, 15.0)
        
        weather_severity = (temp_norm * 0.4) + (hum_norm * 0.4) + (wind_norm * 0.2)

    # Composite Index (0 to 1.0)
    raw_fri = (
        (veg_stress * WEIGHTS["vegetation_stress"]) +
        (hist_density * WEIGHTS["historical_density"]) +
        (terrain_risk * WEIGHTS["terrain_risk"]) +
        (fuel_load * WEIGHTS["fuel_load"]) +
        (weather_severity * WEIGHTS["weather_severity"])
    )
    
    # Scale to 0-100
    result["fri"] = (raw_fri * 100).round(1)
    
    # Categorization
    result["fri_tier"] = pd.cut(
        result["fri"],
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=["Low", "Moderate", "High", "Extreme"]
    )
    
    return result
