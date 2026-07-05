"""
src/data/firms_sync.py

NASA FIRMS Data Sync
Polls the FIRMS API for new fire alerts and caches them locally for the map overlay.
"""

import os
import time
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from loguru import logger
from pathlib import Path

from src.config import DATA_CACHE, DATA_RAW, INDIA_BOUNDS

FIRMS_API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_CACHE_FILE = DATA_CACHE / "firms_active_fires.csv"

def get_firms_key():
    # Attempt to read from env or use a default test key (note: in prod, use real key)
    # The default key is just a placeholder for demo purposes if not set
    return os.getenv("FIRMS_API_KEY", "b33633ab0a7fc936e760ef162f2cc3f7")

def fetch_recent_firms_data(days: int = 2) -> pd.DataFrame:
    """
    Fetch FIRMS active fire alerts for India for the last `days` days.
    """
    api_key = get_firms_key()
    if not api_key:
        logger.warning("No FIRMS_API_KEY set. Cannot fetch live fire data.")
        return pd.DataFrame()
        
    # NASA FIRMS API expects bounds as: W, S, E, N
    w = INDIA_BOUNDS["lon_min"]
    s = INDIA_BOUNDS["lat_min"]
    e = INDIA_BOUNDS["lon_max"]
    n = INDIA_BOUNDS["lat_max"]
    
    # We use VIIRS NOAA20 (N20) or VIIRS Suomi NPP (VIIRS_SNPP) for best resolution
    source = "VIIRS_SNPP_N20" 
    
    url = f"{FIRMS_API_URL}/{api_key}/{source}/{w},{s},{e},{n}/{days}"
    logger.info(f"Fetching FIRMS data from: {url.replace(api_key, 'REDACTED')}")
    
    try:
        df = pd.read_csv(url)
        if df.empty:
            logger.info("No active fires reported in this period.")
            return df
            
        logger.info(f"Fetched {len(df)} active fire alerts.")
        
        # Ensure we have standard columns
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Add fetched timestamp for cache invalidation logic
            df['fetched_at'] = datetime.now().isoformat()
            return df
        else:
            logger.error("FIRMS response missing lat/lon columns.")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Failed to fetch FIRMS data: {e}")
        return pd.DataFrame()

def sync_firms_data():
    """
    Update the local cache of active fires.
    Designed to be run as a scheduled job (e.g., every 6 hours).
    """
    df_new = fetch_recent_firms_data(days=3)
    
    if df_new.empty:
        return
        
    # Ensure cache directory exists
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    
    if FIRMS_CACHE_FILE.exists():
        try:
            df_existing = pd.read_csv(FIRMS_CACHE_FILE)
            # Merge and deduplicate
            # We identify unique fires by lat, lon, and acq_date, acq_time
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            subset_cols = ['latitude', 'longitude', 'acq_date', 'acq_time']
            # Make sure subset_cols exist
            available_cols = [c for c in subset_cols if c in df_combined.columns]
            
            df_combined = df_combined.drop_duplicates(subset=available_cols, keep='last')
            
            # Keep only fires from the last 7 days to prevent unbounded growth
            cutoff_date = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
            if 'acq_date' in df_combined.columns:
                df_combined = df_combined[df_combined['acq_date'] >= cutoff_date]
                
            df_combined.to_csv(FIRMS_CACHE_FILE, index=False)
            logger.info(f"Updated FIRMS cache. Total active records: {len(df_combined)}")
            
        except Exception as e:
            logger.error(f"Error merging FIRMS cache: {e}. Overwriting with new data.")
            df_new.to_csv(FIRMS_CACHE_FILE, index=False)
    else:
        df_new.to_csv(FIRMS_CACHE_FILE, index=False)
        logger.info("Created new FIRMS cache.")

def get_firms_geojson(region: dict = None, hours: int = 24) -> dict:
    """
    Return cached FIRMS data as a GeoJSON FeatureCollection.
    Optionally filter by region and time recency.
    """
    if not FIRMS_CACHE_FILE.exists():
        return {"type": "FeatureCollection", "features": []}
        
    try:
        df = pd.read_csv(FIRMS_CACHE_FILE)
    except Exception:
        return {"type": "FeatureCollection", "features": []}
        
    if df.empty:
        return {"type": "FeatureCollection", "features": []}
        
    # Time filter
    if 'acq_date' in df.columns and 'acq_time' in df.columns:
        # Build naive datetime strings
        # acq_time is often a padded integer like '0630' or '1450'
        # We need to handle various formats gracefully.
        def _parse_time(t):
            t_str = str(t).zfill(4)
            return f"{t_str[:2]}:{t_str[2:]}:00"
            
        try:
            dt_strs = df['acq_date'] + " " + df['acq_time'].apply(_parse_time)
            df['datetime'] = pd.to_datetime(dt_strs, errors='coerce')
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            df = df[df['datetime'] >= cutoff]
        except Exception as e:
            logger.warning(f"Failed to parse FIRMS datetime for filtering: {e}")
            
    # Region filter
    if region:
        df = df[
            (df['latitude'] >= region['lat_min']) &
            (df['latitude'] <= region['lat_max']) &
            (df['longitude'] >= region['lon_min']) &
            (df['longitude'] <= region['lon_max'])
        ]
        
    features = []
    for _, row in df.iterrows():
        props = row.dropna().to_dict()
        # Drop the datetime object as it's not JSON serializable easily
        if 'datetime' in props:
            props['datetime'] = str(props['datetime'])
            
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['longitude']), float(row['latitude'])]
            },
            "properties": props
        })
        
    return {"type": "FeatureCollection", "features": features}
