"""
NASA FIRMS fire data ingestion.
Pulls VIIRS SNPP active fire detections for India.
Filters to landmass, validates columns, saves raw CSV.
"""

import time
import requests
import pandas as pd
from pathlib import Path
# pyrefly: ignore [missing-import]
from loguru import logger

from src.config import (
    NASA_FIRMS_API_KEY,
    INDIA_BOUNDS,
    DATA_RAW,
    DATA_CACHE,
)

# ── Constants ────────────────────────────────────────────────
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Approximate India land polygon exclusions (ocean / sea zones)
OCEAN_EXCLUSIONS = [
    {"lat_max": 20, "lon_max": 72.5},  # Arabian Sea
    {"lat_max": 15, "lon_min": 85.0},  # Bay of Bengal
    {"lat_max": 10, "lon_min": 80.0},  # Southern tip
]

REQUIRED_COLS = {"latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"}


def _vectorised_landmass_filter(df: pd.DataFrame) -> pd.Series:
    """
    Vectorised bounding box and ocean exclusion filter.
    Orders of magnitude faster than row-wise apply.
    """
    b = INDIA_BOUNDS
    in_bounds = df["latitude"].between(b["lat_min"], b["lat_max"]) & df[
        "longitude"
    ].between(b["lon_min"], b["lon_max"])

    # Ocean exclusions mapping to OCEAN_EXCLUSIONS
    arabian_sea = (df["latitude"] <= 20) & (df["longitude"] <= 72.5)
    bay_of_bengal = (df["latitude"] <= 15) & (df["longitude"] >= 85.0)
    southern_tip = (df["latitude"] <= 10) & (df["longitude"] >= 80.0)

    is_ocean = arabian_sea | bay_of_bengal | southern_tip

    return in_bounds & ~is_ocean


def _confidence_frp_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter FIRMS data by detection confidence and Fire Radiative Power (FRP).
    - Confidence: retain 'n' (nominal) and 'h' (high), drop 'l' (low)
    - FRP: retain frp >= 1.0 MW (standard operational floor)
    
    Agricultural Burning Note (Task 3):
    This system retains agricultural fires (e.g. Punjab/Haryana stubble burning in
    Oct-Nov and Apr-May) as legitimate positives, rather than masking the IGP zone.
    The rationale is that the model functions as an all-cause fire risk system,
    and agricultural fires still pose particulate and spread risks.
    """
    if df.empty:
        return df

    # Confidence filter
    if "confidence" in df.columns:
        valid_conf = {"n", "h", "nominal", "high"}
        mask_conf = df["confidence"].astype(str).str.lower().isin(valid_conf)
        df = df[mask_conf]
        logger.info(f"Rows after confidence filter (dropped low): {len(df)}")
        
    # FRP filter
    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
        df = df[df["frp"] >= 1.0]
        logger.info(f"Rows after FRP filter (>= 1.0): {len(df)}")
        
    return df.reset_index(drop=True)


def fetch_live_firms(
    source: str = "VIIRS_SNPP_NRT",
    days: int = 10,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch recent active fire detections from NASA FIRMS.

    Args:
        source: FIRMS data source identifier.
        days:   Number of recent days to pull (max 10 for NRT).
        save:   Whether to write the raw file to DATA_RAW.

    Returns:
        Filtered DataFrame of India fire detections.
    """
    if not NASA_FIRMS_API_KEY or NASA_FIRMS_API_KEY == "your_key_here":
        logger.warning("No NASA FIRMS API key set — falling back to public URL")
        url = (
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
            "suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_24h.csv"
        )
    else:
        # Bounding box: India
        west = INDIA_BOUNDS["lon_min"]
        south = INDIA_BOUNDS["lat_min"]
        east = INDIA_BOUNDS["lon_max"]
        north = INDIA_BOUNDS["lat_max"]
        url = (
            f"{FIRMS_BASE}/{NASA_FIRMS_API_KEY}/{source}/"
            f"{west},{south},{east},{north}/{days}"
        )

    logger.info(f"Fetching FIRMS data from: {url}")

    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            wait = 2**attempt * 3
            logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError("NASA FIRMS fetch failed after 4 attempts")

    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    logger.info(f"Raw rows fetched: {len(df)}")

    # ── Validate columns ────────────────────────────────────
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"FIRMS response missing columns: {missing}")

    # ── Normalise ────────────────────────────────────────────
    df.columns = [c.lower() for c in df.columns]
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    # ── Landmass filter ──────────────────────────────────────
    mask = _vectorised_landmass_filter(df)
    df = df[mask].reset_index(drop=True)
    logger.info(f"Rows after India landmass filter: {len(df)}")

    # ── Confidence and FRP filter ─────────────────────────────
    df = _confidence_frp_filter(df)

    if save:
        out = DATA_RAW / "firms_live.csv"
        df.to_csv(out, index=False)
        logger.info(f"Saved raw FIRMS data → {out}")

    return df


def load_historical_firms(path: Path) -> pd.DataFrame:
    """
    Load a pre-downloaded historical FIRMS CSV (e.g. filtered_2024_2026.csv).
    Applies the same normalisation and landmass filter as the live fetch.
    """
    logger.info(f"Loading historical FIRMS data from {path}")
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [c.lower() for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Historical file missing columns: {missing}")

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    mask = _vectorised_landmass_filter(df)
    df = df[mask].reset_index(drop=True)
    logger.info(f"Historical rows after filter: {len(df):,}")

    # ── Confidence and FRP filter ─────────────────────────────
    df = _confidence_frp_filter(df)

    return df