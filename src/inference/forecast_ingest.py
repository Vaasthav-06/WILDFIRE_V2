"""
src/inference/forecast_ingest.py

Operational forecast data ingestion for wildfire risk inference.

Replaces ERA5 (retrospective reanalysis) with Open-Meteo Forecast API
for live and future predictions. The feature schema produced here is
IDENTICAL to the training schema — same column names, same units, same
physical meaning. The static production_model.pkl is unaware of whether
its inputs came from ERA5 or Open-Meteo; it only sees the feature vector.

Outputs
-------
Two public functions called by the inference engine:

  fetch_24h_forecast(grid_df)
      Returns the peak-weather composite for the next 24 hours.
      One row per grid cell — worst-case (highest VPD, highest wind,
      lowest humidity) across the 24-hour window.

  fetch_7day_forecast(grid_df)
      Returns 7 DataFrames (one per day), each with one row per grid cell.
      KBDI is propagated forward sequentially (stateful — depends on
      yesterday's KBDI and today's forecasted precipitation).

Weather variables fetched (matching ERA5 training columns exactly):
  temp      — 2m temperature max (°C)
  humidity  — 2m relative humidity min (%)
  wind      — 10m wind speed max (m/s)
  precip    — total precipitation sum (mm)  [for KBDI propagation]
  wind_u    — 10m u-component of wind (m/s) [for spread direction]
  wind_v    — 10m v-component of wind (m/s) [for spread direction]
  vpd       — Vapor Pressure Deficit (kPa)  [derived]
"""

import time
import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta
from loguru import logger
from typing import Optional

# Open-Meteo forecast endpoint (no API key needed for free tier)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Batch size: Open-Meteo supports up to 1000 locations per request
# We use 50 to stay well within rate limits and avoid payload size issues
CHUNK_SIZE = 50

# Forecast variables we need from Open-Meteo hourly API
# wind_u/v let us compute both scalar wind speed and direction vector
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

# Wind component variables for spread direction calculation
# We re-derive u/v from speed + direction (Open-Meteo doesn't return u/v directly)
DAILY_VARS = [
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "precipitation_sum",
    "et0_fao_evapotranspiration",  # Used in KBDI propagation
]


# ── HTTP helper ───────────────────────────────────────────────────────────────


def _safe_get(
    url: str,
    params: dict,
    timeout: int = 30,
    max_retries: int = 4,
) -> Optional[dict]:
    """Retry-safe GET with exponential backoff. Returns parsed JSON or None."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = (2**attempt) * 5
                logger.warning(f"Rate limit hit. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                logger.warning(
                    f"HTTP {r.status_code} on attempt {attempt+1}: {r.text[:200]}"
                )
                time.sleep(2)
        except requests.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1}. Retrying...")
            time.sleep(3)
        except requests.ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt+1}: {e}")
            time.sleep(5)
    logger.error(f"All {max_retries} attempts failed for {url}")
    return None


# ── VPD derivation (matches ERA5 enrich_dataframe exactly) ───────────────────


def _compute_vpd(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Tetens formula — identical to ERA5Lookup implementation.
    Ensures VPD values are on the same scale as training data.
    """
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * np.clip(rh / 100.0, 0, 1)
    return np.maximum(es - ea, 0.0).round(4)


# ── Wind u/v decomposition ────────────────────────────────────────────────────


def _wind_components(speed_ms: np.ndarray, direction_deg: np.ndarray):
    """
    Decompose scalar wind speed + meteorological direction into u/v vectors.

    Meteorological convention: direction FROM which wind is blowing,
    measured clockwise from North (0° = wind from North, 90° = from East).

    u = eastward component  (positive = wind blowing toward East)
    v = northward component (positive = wind blowing toward North)

    Math:
        dir_rad = direction_deg * π/180
        u = -speed × sin(dir_rad)   ← negative because direction is FROM
        v = -speed × cos(dir_rad)
    """
    dir_rad = np.deg2rad(direction_deg)
    u = -speed_ms * np.sin(dir_rad)
    v = -speed_ms * np.cos(dir_rad)
    return u.round(3), v.round(3)


# ── KBDI forward propagation ──────────────────────────────────────────────────


def propagate_kbdi_forward(
    baseline_kbdi: np.ndarray,
    forecasted_precip_mm: list[np.ndarray],
    forecasted_temp_c: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Propagate KBDI forward over a multi-day forecast window.

    KBDI is the only stateful feature in the pipeline. Its value on day N+1
    depends on day N's KBDI plus the net moisture balance of day N.

    This function mirrors the per-cell KBDI logic in engineering.py but
    operates forward in time rather than over historical rolling windows.

    KBDI daily update equation (Keetch-Byram 1968):
        ET_demand  = (0.968 × exp(0.0875 × T + 1.5552) − 8.30) / 1000
        deficit_d  = max(ET_demand − precip_mm, 0)
        KBDI_new   = clip(KBDI_old + deficit_d × 800, 0, 800)

    Args:
        baseline_kbdi:        KBDI values for current day [n_cells]
        forecasted_precip_mm: List of length n_days, each [n_cells]
        forecasted_temp_c:    List of length n_days, each [n_cells]

    Returns:
        List of n_days KBDI arrays, each [n_cells].
        kbdi_sequence[0] = KBDI after day 1 of forecast, etc.
    """
    assert len(forecasted_precip_mm) == len(
        forecasted_temp_c
    ), "precip and temp lists must have equal length (one entry per day)"

    n_days = len(forecasted_precip_mm)
    kbdi = baseline_kbdi.copy().astype(float)
    kbdi_seq = []

    for day_idx in range(n_days):
        temp = np.clip(forecasted_temp_c[day_idx], 0, 55)
        precip = np.clip(forecasted_precip_mm[day_idx], 0, None)

        # Crane 1982 ET approximation (same formula as engineering.py)
        et_demand = (
            np.clip(0.968 * np.exp(0.0875 * temp + 1.5552) - 8.30, 0, None) / 1000.0
        )

        daily_deficit = np.clip(et_demand - precip, 0, None)
        kbdi = np.clip(kbdi + daily_deficit * 800, 0, 800)
        kbdi_seq.append(kbdi.copy().round(1))

    logger.debug(
        f"KBDI propagation: baseline mean={baseline_kbdi.mean():.1f}, "
        f"day-7 mean={kbdi_seq[-1].mean():.1f}"
    )
    return kbdi_seq


# ── Core forecast fetch ───────────────────────────────────────────────────────


def _fetch_daily_forecast_chunk(
    lats: list[float],
    lons: list[float],
    forecast_days: int = 7,
) -> Optional[list[dict]]:
    """
    Fetch daily forecast for a batch of locations from Open-Meteo.
    Returns list of per-location response dicts, or None on failure.
    """
    params = {
        "latitude": ",".join(str(round(la, 4)) for la in lats),
        "longitude": ",".join(str(round(lo, 4)) for lo in lons),
        "daily": ",".join(DAILY_VARS),
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",  # Always m/s — matches ERA5 training units
        "timezone": "Asia/Kolkata",
    }

    data = _safe_get(FORECAST_URL, params)
    if data is None:
        return None

    # Open-Meteo returns a list when multiple locations are requested
    if isinstance(data, dict):
        data = [data]

    return data


def _fetch_hourly_forecast_chunk(
    lats: list[float],
    lons: list[float],
    forecast_hours: int = 24,
) -> Optional[list[dict]]:
    """
    Fetch hourly forecast for the next 24 hours for a batch of locations.
    """
    params = {
        "latitude": ",".join(str(round(la, 4)) for la in lats),
        "longitude": ",".join(str(round(lo, 4)) for lo in lons),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": 1,
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
    }

    data = _safe_get(FORECAST_URL, params)
    if data is None:
        return None

    if isinstance(data, dict):
        data = [data]

    return data


# ── 24-hour forecast ──────────────────────────────────────────────────────────

# ── 24-hour forecast ──────────────────────────────────────────────────────────


def fetch_24h_forecast(grid_df: pd.DataFrame) -> pd.DataFrame:
    result = grid_df.copy()

    # Stamp today's date so build_features() can compute cyclical month
    # and KBDI. This is the reference date for the 24-hour forecast window.
    result["acq_date"] = date.today().isoformat()

    result["temp"] = np.nan
    result["humidity"] = np.nan
    result["wind"] = np.nan
    result["precip"] = np.nan
    result["wind_u"] = np.nan
    result["wind_v"] = np.nan
    # ... rest of function unchanged

    lats = grid_df["latitude"].tolist()
    lons = grid_df["longitude"].tolist()
    n = len(lats)

    logger.info(f"Fetching 24h hourly forecast for {n:,} grid cells...")

    filled = 0
    for i in range(0, n, CHUNK_SIZE):
        chunk_lats = lats[i : i + CHUNK_SIZE]
        chunk_lons = lons[i : i + CHUNK_SIZE]
        chunk_idx = grid_df.index[i : i + CHUNK_SIZE]

        responses = _fetch_hourly_forecast_chunk(
            chunk_lats, chunk_lons, forecast_hours=24
        )
        if responses is None:
            logger.warning(
                f"Chunk {i//CHUNK_SIZE + 1}: fetch failed, skipping {len(chunk_lats)} cells"
            )
            continue

        for j, resp in enumerate(responses):
            if j >= len(chunk_idx):
                break
            idx = chunk_idx[j]
            try:
                hourly = resp.get("hourly", {})
                t_arr = np.array(
                    hourly.get("temperature_2m", [np.nan] * 24), dtype=float
                )
                rh_arr = np.array(
                    hourly.get("relative_humidity_2m", [np.nan] * 24), dtype=float
                )
                ws_arr = np.array(
                    hourly.get("wind_speed_10m", [np.nan] * 24), dtype=float
                )
                wd_arr = np.array(
                    hourly.get("wind_direction_10m", [np.nan] * 24), dtype=float
                )
                pr_arr = np.array(
                    hourly.get("precipitation", [np.nan] * 24), dtype=float
                )

                # Peak-weather composite
                peak_temp = float(np.nanmax(t_arr))
                peak_wind = float(np.nanmax(ws_arr))
                min_hum = float(np.nanmin(rh_arr))
                total_prec = float(np.nansum(pr_arr))

                # Wind u/v from the hour of peak wind speed
                peak_wind_hour = int(np.nanargmax(ws_arr))
                peak_dir = (
                    float(wd_arr[peak_wind_hour])
                    if not np.isnan(wd_arr[peak_wind_hour])
                    else 0.0
                )
                wu, wv = _wind_components(
                    np.array([peak_wind]),
                    np.array([peak_dir]),
                )

                result.at[idx, "temp"] = round(peak_temp, 2)
                result.at[idx, "humidity"] = round(min_hum, 2)
                result.at[idx, "wind"] = round(peak_wind, 3)
                result.at[idx, "precip"] = round(total_prec, 3)
                result.at[idx, "wind_u"] = float(wu[0])
                result.at[idx, "wind_v"] = float(wv[0])
                filled += 1

            except (KeyError, IndexError, TypeError, ValueError) as e:
                logger.debug(f"Cell {idx} parse error: {e}")
                continue

        logger.info(
            f"  24h forecast: chunk {i//CHUNK_SIZE + 1}/{(n-1)//CHUNK_SIZE + 1} "
            f"| filled so far: {filled}/{n}"
        )
        time.sleep(0.15)  # polite rate limiting

    # Derive VPD after all cells are filled
    valid = result["temp"].notna() & result["humidity"].notna()
    result.loc[valid, "vpd"] = _compute_vpd(
        result.loc[valid, "temp"].values,
        result.loc[valid, "humidity"].values,
    )

    fill_rate = valid.sum() / n * 100
    logger.info(
        f"24h forecast complete: {valid.sum():,}/{n:,} cells filled ({fill_rate:.1f}%)"
    )
    return result


# ── 7-day forecast ────────────────────────────────────────────────────────────


def fetch_7day_forecast(
    grid_df: pd.DataFrame,
    baseline_kbdi: Optional[np.ndarray] = None,
) -> list[pd.DataFrame]:
    """
    Fetch 7-day daily forecast and return one enriched DataFrame per day.

    KBDI is the ONLY stateful feature — it depends on yesterday's value.
    All other features (vpd, wind, ndvi_proxy, elevation, ecoregion) are
    either stateless or change slowly enough to treat as fixed over 7 days.

    Strategy per day:
      - temp    → daily max temperature (°C)
      - humidity → daily min relative humidity (%)
      - wind     → daily max wind speed (m/s)
      - precip   → daily precipitation sum (mm)
      - wind_u/v → from dominant wind direction at max wind speed
      - kbdi     → propagated forward from baseline using forecasted precip/temp

    The 'acq_date' column in each returned DataFrame is set to the
    corresponding forecast date — this is required by build_features()
    for cyclical month encoding and NDVI proxy calculation.

    Args:
        grid_df:        DataFrame with 'latitude' and 'longitude'.
        baseline_kbdi:  Current KBDI values [n_cells]. If None, uses 200.0
                        (moderate drought) as a conservative default.

    Returns:
        List of 7 DataFrames, indexed [0..6] = [day+1..day+7].
        Each has full weather enrichment ready for feature engineering.
    """
    n = len(grid_df)
    today = date.today()

    if baseline_kbdi is None:
        logger.warning(
            "No baseline KBDI provided. Using 200.0 (moderate drought) as default. "
            "For production accuracy, pass actual KBDI from the training dataset "
            "for the current date."
        )
        baseline_kbdi = np.full(n, 200.0)

    logger.info(f"Fetching 7-day daily forecast for {n:,} grid cells...")

    lats = grid_df["latitude"].tolist()
    lons = grid_df["longitude"].tolist()

    # Collect raw daily arrays per cell: shape (n_cells, 7) after assembly
    all_temp = np.full((n, 7), np.nan)
    all_hum = np.full((n, 7), np.nan)
    all_wind = np.full((n, 7), np.nan)
    all_wdir = np.full((n, 7), np.nan)
    all_precip = np.full((n, 7), np.nan)

    filled = 0
    for i in range(0, n, CHUNK_SIZE):
        chunk_lats = lats[i : i + CHUNK_SIZE]
        chunk_lons = lons[i : i + CHUNK_SIZE]

        responses = _fetch_daily_forecast_chunk(chunk_lats, chunk_lons, forecast_days=7)
        if responses is None:
            logger.warning(f"Chunk {i//CHUNK_SIZE + 1}: fetch failed")
            continue

        for j, resp in enumerate(responses):
            global_idx = i + j
            if global_idx >= n:
                break
            try:
                daily = resp.get("daily", {})
                # Open-Meteo returns lists of length = forecast_days (7)
                t_arr = np.array(
                    daily.get("temperature_2m_max", [np.nan] * 7), dtype=float
                )
                rh_arr = np.array(
                    daily.get("relative_humidity_2m_min", [np.nan] * 7), dtype=float
                )
                ws_arr = np.array(
                    daily.get("wind_speed_10m_max", [np.nan] * 7), dtype=float
                )
                wd_arr = np.array(
                    daily.get("wind_direction_10m_dominant", [np.nan] * 7), dtype=float
                )
                pr_arr = np.array(
                    daily.get("precipitation_sum", [np.nan] * 7), dtype=float
                )

                all_temp[global_idx] = t_arr
                all_hum[global_idx] = rh_arr
                all_wind[global_idx] = ws_arr
                all_wdir[global_idx] = wd_arr
                all_precip[global_idx] = pr_arr
                filled += 1

            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Cell {global_idx} parse error: {e}")
                continue

        logger.info(
            f"  7-day fetch: chunk {i//CHUNK_SIZE + 1}/{(n-1)//CHUNK_SIZE + 1} "
            f"| cells filled: {filled}"
        )
        time.sleep(1.5)

    logger.info(f"7-day raw fetch complete: {filled}/{n} cells have data")

    # KBDI forward propagation over 7 days
    kbdi_sequence = propagate_kbdi_forward(
        baseline_kbdi=baseline_kbdi,
        forecasted_precip_mm=[all_precip[:, d] for d in range(7)],
        forecasted_temp_c=[all_temp[:, d] for d in range(7)],
    )

    # Build one DataFrame per forecast day
    daily_frames = []
    for day_offset in range(7):
        forecast_date = today + timedelta(days=day_offset + 1)
        df_day = grid_df.copy()

        # Set the date — required by build_features for month encoding + NDVI
        df_day["acq_date"] = forecast_date.strftime("%Y-%m-%d")

        # Weather values for this day
        t_day = all_temp[:, day_offset]
        rh_day = all_hum[:, day_offset]
        ws_day = all_wind[:, day_offset]
        wd_day = all_wdir[:, day_offset]
        pr_day = all_precip[:, day_offset]

        df_day["temp"] = np.round(t_day, 2)
        df_day["humidity"] = np.round(rh_day, 2)
        df_day["wind"] = np.round(ws_day, 3)
        df_day["precip"] = np.round(pr_day, 3)

        # Wind u/v components from speed + dominant direction
        wu, wv = _wind_components(ws_day, np.where(np.isnan(wd_day), 0.0, wd_day))
        df_day["wind_u"] = wu
        df_day["wind_v"] = wv

        # VPD
        valid = ~np.isnan(t_day) & ~np.isnan(rh_day)
        df_day["vpd"] = np.nan
        if valid.any():
            df_day.loc[valid, "vpd"] = _compute_vpd(t_day[valid], rh_day[valid])

        # KBDI — override the rolling-window KBDI from build_features
        # The forward-propagated value is authoritative for forecast mode
        df_day["kbdi_forecast"] = kbdi_sequence[day_offset]

        daily_frames.append(df_day)

    logger.info(
        f"7-day forecast ready: {len(daily_frames)} daily DataFrames, "
        f"{n:,} cells each"
    )
    return daily_frames


# ── India inference grid ──────────────────────────────────────────────────────


def build_india_inference_grid(resolution: float = 0.1) -> pd.DataFrame:
    """
    Build the India land-cell inference grid.

    Resolution 0.1° ≈ 11km per cell. At this resolution, India
    (68.1°–97.4°E, 8.0°–37.6°N) yields ~3,200 land cells.

    For batch inference this is fine as a flat list. For map rendering
    the caller converts to GeoJSON polygons.

    Args:
        resolution: Grid spacing in degrees. Use 0.1 for operational
                    inference, 0.5 for development/testing.

    Returns:
        DataFrame with columns: latitude, longitude, cell_id
    """
    from src.config import INDIA_BOUNDS

    lat_min = INDIA_BOUNDS["lat_min"]
    lat_max = INDIA_BOUNDS["lat_max"]
    lon_min = INDIA_BOUNDS["lon_min"]
    lon_max = INDIA_BOUNDS["lon_max"]

    lats = np.arange(lat_min, lat_max, resolution)
    lons = np.arange(lon_min, lon_max, resolution)

    grid_rows = []
    for lat in lats:
        for lon in lons:
            grid_rows.append(
                {
                    "latitude": round(float(lat), 4),
                    "longitude": round(float(lon), 4),
                }
            )

    grid = pd.DataFrame(grid_rows)
    grid["cell_id"] = grid["latitude"].astype(str) + "_" + grid["longitude"].astype(str)

    logger.info(
        f"India inference grid built: {len(grid):,} cells "
        f"at {resolution}° resolution"
    )
    return grid.reset_index(drop=True)
