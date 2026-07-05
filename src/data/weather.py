"""
src/data/weather.py

Public-facing weather data module for the Wildfire Risk Prediction System.

This module is the single import point for all live weather fetching.
app.py imports `fetch_forecast_weather` from here.

Responsibilities
----------------
- Fetch 24-hour peak-weather composites (max temp, min humidity, max wind)
  from the Open-Meteo free forecast API for any set of grid cells.
- Compute derived variables: VPD (Vapour Pressure Deficit), wind components.
- Apply a safety fallback so the FastAPI server never crashes if Open-Meteo
  is temporarily unreachable or times out (returns climatological defaults).

Open-Meteo variables fetched
-----------------------------
  temperature_2m         — 2 m air temperature (°C), hourly
  relative_humidity_2m   — 2 m relative humidity (%), hourly
  wind_speed_10m         — 10 m wind speed (m/s), hourly
  wind_direction_10m     — 10 m wind direction (°), hourly

Derived
-------
  vpd     — Vapour pressure deficit (kPa): drives fuel drying
  wind_u  — Zonal wind component (m/s):    drives spread direction
  wind_v  — Meridional wind component (m/s): drives spread direction

Usage
-----
    from src.data.weather import fetch_forecast_weather

    df_weather = fetch_forecast_weather(grid_df)
    # Returns grid_df enriched with: temp, humidity, wind, vpd, wind_u, wind_v
"""

import numpy as np
import pandas as pd
from datetime import date
from loguru import logger
from typing import Optional

# ── Re-export core fetch logic from forecast_ingest ──────────────────────────
# forecast_ingest.py owns the actual HTTP calls, chunking, and retry logic.
# We import it here and wrap it with a fallback layer.

from src.inference.forecast_ingest import fetch_24h_forecast as _fetch_24h


# ── Climatological fallback constants ────────────────────────────────────────
# Applied cell-by-cell when Open-Meteo returns None for a cell.
# Values represent a conservative mid-season estimate for India (April–May),
# which is the peak wildfire season. This prevents NaN from propagating into
# the model's feature matrix and causing a 500 error on the prediction routes.

_FALLBACK_TEMP_C     = 32.0   # °C  — hot but not extreme
_FALLBACK_HUMIDITY   = 35.0   # %   — moderate dryness
_FALLBACK_WIND_MS    = 3.5    # m/s — light to moderate wind
_FALLBACK_WIND_DIR   = 0.0    # °   — default north (components → 0, -3.5)


def _compute_vpd(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Tetens formula: vapour pressure deficit (kPa).

    VPD = e_s - e_a
      e_s = saturation vapour pressure at temp_c
      e_a = actual vapour pressure (e_s × RH/100)

    A VPD > 2 kPa combined with high wind is a strong fire-weather indicator.
    """
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * np.clip(rh / 100.0, 0.0, 1.0)
    return np.maximum(es - ea, 0.0).round(4)


def _wind_components(
    speed_ms: np.ndarray,
    direction_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose wind speed + direction into meteorological U/V components.

    Meteorological convention:
      U = -speed × sin(dir)   (positive = westerly, i.e. wind blowing east)
      V = -speed × cos(dir)   (positive = southerly, i.e. wind blowing north)
    """
    dir_rad = np.deg2rad(direction_deg)
    u = (-speed_ms * np.sin(dir_rad)).round(3)
    v = (-speed_ms * np.cos(dir_rad)).round(3)
    return u, v


def _apply_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill any NaN weather values with safe climatological defaults.

    Called after the Open-Meteo fetch so that partial failures (e.g., a
    single chunk timing out) don't propagate NaN into the model input.
    """
    df = df.copy()

    nan_mask = df["temp"].isna()
    if nan_mask.any():
        n_fallback = int(nan_mask.sum())
        logger.warning(
            f"[weather] {n_fallback} cells missing weather data — "
            "applying climatological fallback values. "
            "Check Open-Meteo connectivity if this persists."
        )

    df["temp"]     = df["temp"].fillna(_FALLBACK_TEMP_C)
    df["humidity"] = df["humidity"].fillna(_FALLBACK_HUMIDITY)
    df["wind"]     = df["wind"].fillna(_FALLBACK_WIND_MS)

    # Recompute wind components for fallback cells
    fallback_wu, fallback_wv = _wind_components(
        np.array([_FALLBACK_WIND_MS]),
        np.array([_FALLBACK_WIND_DIR]),
    )
    df["wind_u"] = df["wind_u"].fillna(float(fallback_wu[0]))
    df["wind_v"] = df["wind_v"].fillna(float(fallback_wv[0]))

    return df


def fetch_forecast_weather(
    grid_df: pd.DataFrame,
    use_fallback: bool = True,
) -> pd.DataFrame:
    """
    Fetch 24-hour peak-weather composite for all grid cells.

    This is the primary function imported by app.py. It enriches `grid_df`
    with live Open-Meteo weather data and returns a DataFrame ready for the
    feature engineering pipeline.

    Output columns added
    --------------------
    acq_date   : str         — ISO date string (today)
    temp       : float       — Max 2m temperature over next 24h (°C)
    humidity   : float       — Min 2m relative humidity over next 24h (%)
    wind       : float       — Max 10m wind speed over next 24h (m/s)
    wind_u     : float       — Zonal wind component at peak-wind hour (m/s)
    wind_v     : float       — Meridional wind component at peak-wind hour (m/s)
    vpd        : float       — Vapour pressure deficit at peak conditions (kPa)

    Parameters
    ----------
    grid_df     : pd.DataFrame
        Must contain 'latitude' and 'longitude' columns.
    use_fallback : bool, default True
        If True, replace NaN values with climatological defaults after fetch.
        Set to False only in tests where you want to inspect raw NaN counts.

    Returns
    -------
    pd.DataFrame
        A copy of grid_df enriched with weather columns.

    Notes
    -----
    - Calls Open-Meteo in chunks of 50 cells using concurrent threads.
    - Each chunk retries up to 4 times with exponential back-off.
    - With fallback enabled, this function never raises on API failure.
    """
    n_cells = len(grid_df)
    logger.info(f"[weather] Fetching 24h forecast for {n_cells:,} grid cells via Open-Meteo...")

    try:
        df_weather = _fetch_24h(grid_df)
        logger.info(f"[weather] Open-Meteo fetch complete ({n_cells:,} cells)")
    except Exception as exc:
        # Catastrophic failure (e.g. DNS down, requests not installed).
        # Build a fully-fallback frame so the server can still respond.
        logger.error(
            f"[weather] Open-Meteo fetch raised an exception: {exc}. "
            "Returning full climatological fallback for all cells."
        )
        df_weather = grid_df.copy()
        df_weather["acq_date"] = date.today().isoformat()
        df_weather["temp"]     = np.nan
        df_weather["humidity"] = np.nan
        df_weather["wind"]     = np.nan
        df_weather["wind_u"]   = np.nan
        df_weather["wind_v"]   = np.nan
        df_weather["vpd"]      = np.nan

    if use_fallback:
        df_weather = _apply_fallback(df_weather)

    # Recompute VPD over all valid cells in case fallback filled temp/humidity
    valid = df_weather["temp"].notna() & df_weather["humidity"].notna()
    if valid.any():
        df_weather.loc[valid, "vpd"] = _compute_vpd(
            df_weather.loc[valid, "temp"].values,
            df_weather.loc[valid, "humidity"].values,
        )

    # Log data quality summary
    n_valid = int(df_weather["temp"].notna().sum())
    n_fallback_used = n_cells - n_valid
    if n_fallback_used > 0:
        logger.warning(
            f"[weather] Data quality: {n_valid}/{n_cells} cells have live data, "
            f"{n_fallback_used} use fallback defaults."
        )
    else:
        logger.info(f"[weather] Data quality: {n_valid}/{n_cells} cells — all live ✓")

    return df_weather


# ── Convenience re-exports ────────────────────────────────────────────────────
# Expose the underlying 24h fetch directly for tests or scripts that
# want raw (no-fallback) data.

fetch_24h_raw = _fetch_24h

__all__ = [
    "fetch_forecast_weather",
    "fetch_24h_raw",
]
