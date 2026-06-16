import time
import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta
from loguru import logger
from typing import Optional

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CHUNK_SIZE = 50

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
]

DAILY_VARS = [
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]

def _safe_get(url: str, params: dict, timeout: int = 30, max_retries: int = 4) -> Optional[dict]:
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = (2**attempt) * 5
                time.sleep(wait)
            else:
                time.sleep(2)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(3)
    return None

def _compute_vpd(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * np.clip(rh / 100.0, 0, 1)
    return np.maximum(es - ea, 0.0).round(4)

def _wind_components(speed_ms: np.ndarray, direction_deg: np.ndarray):
    dir_rad = np.deg2rad(direction_deg)
    u = -speed_ms * np.sin(dir_rad)
    v = -speed_ms * np.cos(dir_rad)
    return u.round(3), v.round(3)

def _fetch_daily_forecast_chunk(lats: list[float], lons: list[float], forecast_days: int = 7) -> Optional[list[dict]]:
    params = {
        "latitude": ",".join(str(round(la, 4)) for la in lats),
        "longitude": ",".join(str(round(lo, 4)) for lo in lons),
        "daily": ",".join(DAILY_VARS),
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
    }
    data = _safe_get(FORECAST_URL, params)
    return [data] if isinstance(data, dict) else data

def _fetch_hourly_forecast_chunk(lats: list[float], lons: list[float], forecast_hours: int = 24) -> Optional[list[dict]]:
    params = {
        "latitude": ",".join(str(round(la, 4)) for la in lats),
        "longitude": ",".join(str(round(lo, 4)) for lo in lons),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": 1,
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
    }
    data = _safe_get(FORECAST_URL, params)
    return [data] if isinstance(data, dict) else data

def fetch_24h_forecast(grid_df: pd.DataFrame) -> pd.DataFrame:
    result = grid_df.copy()
    result["acq_date"] = date.today().isoformat()
    result["temp"] = np.nan
    result["humidity"] = np.nan
    result["wind"] = np.nan
    result["wind_u"] = np.nan
    result["wind_v"] = np.nan

    lats = grid_df["latitude"].tolist()
    lons = grid_df["longitude"].tolist()
    n = len(lats)

    filled = 0
    for i in range(0, n, CHUNK_SIZE):
        chunk_lats = lats[i : i + CHUNK_SIZE]
        chunk_lons = lons[i : i + CHUNK_SIZE]
        chunk_idx = grid_df.index[i : i + CHUNK_SIZE]

        responses = _fetch_hourly_forecast_chunk(chunk_lats, chunk_lons, forecast_hours=24)
        if not responses: continue

        for j, resp in enumerate(responses):
            if j >= len(chunk_idx): break
            idx = chunk_idx[j]
            try:
                hourly = resp.get("hourly", {})
                t_arr = np.array(hourly.get("temperature_2m", [np.nan]*24), dtype=float)
                rh_arr = np.array(hourly.get("relative_humidity_2m", [np.nan]*24), dtype=float)
                ws_arr = np.array(hourly.get("wind_speed_10m", [np.nan]*24), dtype=float)
                wd_arr = np.array(hourly.get("wind_direction_10m", [np.nan]*24), dtype=float)

                peak_temp = float(np.nanmax(t_arr))
                peak_wind = float(np.nanmax(ws_arr))
                min_hum = float(np.nanmin(rh_arr))

                peak_wind_hour = int(np.nanargmax(ws_arr))
                peak_dir = float(wd_arr[peak_wind_hour]) if not np.isnan(wd_arr[peak_wind_hour]) else 0.0
                wu, wv = _wind_components(np.array([peak_wind]), np.array([peak_dir]))

                result.at[idx, "temp"] = round(peak_temp, 2)
                result.at[idx, "humidity"] = round(min_hum, 2)
                result.at[idx, "wind"] = round(peak_wind, 3)
                result.at[idx, "wind_u"] = float(wu[0])
                result.at[idx, "wind_v"] = float(wv[0])
                filled += 1
            except Exception:
                continue
        time.sleep(0.15)

    valid = result["temp"].notna() & result["humidity"].notna()
    result.loc[valid, "vpd"] = _compute_vpd(result.loc[valid, "temp"].values, result.loc[valid, "humidity"].values)
    return result

def fetch_7day_forecast(grid_df: pd.DataFrame) -> list[pd.DataFrame]:
    n = len(grid_df)
    today = date.today()
    lats = grid_df["latitude"].tolist()
    lons = grid_df["longitude"].tolist()

    all_temp = np.full((n, 7), np.nan)
    all_hum = np.full((n, 7), np.nan)
    all_wind = np.full((n, 7), np.nan)
    all_wdir = np.full((n, 7), np.nan)

    for i in range(0, n, CHUNK_SIZE):
        chunk_lats = lats[i : i + CHUNK_SIZE]
        chunk_lons = lons[i : i + CHUNK_SIZE]
        responses = _fetch_daily_forecast_chunk(chunk_lats, chunk_lons, forecast_days=7)
        if not responses: continue

        for j, resp in enumerate(responses):
            global_idx = i + j
            if global_idx >= n: break
            try:
                daily = resp.get("daily", {})
                all_temp[global_idx] = np.array(daily.get("temperature_2m_max", [np.nan]*7), dtype=float)
                all_hum[global_idx] = np.array(daily.get("relative_humidity_2m_min", [np.nan]*7), dtype=float)
                all_wind[global_idx] = np.array(daily.get("wind_speed_10m_max", [np.nan]*7), dtype=float)
                all_wdir[global_idx] = np.array(daily.get("wind_direction_10m_dominant", [np.nan]*7), dtype=float)
            except Exception:
                continue
        time.sleep(1.5)

    daily_frames = []
    for day_offset in range(7):
        forecast_date = today + timedelta(days=day_offset + 1)
        df_day = grid_df.copy()
        df_day["acq_date"] = forecast_date.strftime("%Y-%m-%d")

        df_day["temp"] = np.round(all_temp[:, day_offset], 2)
        df_day["humidity"] = np.round(all_hum[:, day_offset], 2)
        df_day["wind"] = np.round(all_wind[:, day_offset], 3)

        wu, wv = _wind_components(all_wind[:, day_offset], np.where(np.isnan(all_wdir[:, day_offset]), 0.0, all_wdir[:, day_offset]))
        df_day["wind_u"] = wu
        df_day["wind_v"] = wv

        valid = ~np.isnan(df_day["temp"]) & ~np.isnan(df_day["humidity"])
        df_day["vpd"] = np.nan
        if valid.any():
            df_day.loc[valid, "vpd"] = _compute_vpd(df_day.loc[valid, "temp"], df_day.loc[valid, "humidity"])
        
        daily_frames.append(df_day)

    return daily_frames

def build_india_inference_grid(resolution: float = 0.1) -> pd.DataFrame:
    from src.config import INDIA_BOUNDS
    lats = np.arange(INDIA_BOUNDS["lat_min"], INDIA_BOUNDS["lat_max"], resolution)
    lons = np.arange(INDIA_BOUNDS["lon_min"], INDIA_BOUNDS["lon_max"], resolution)
    grid_rows = [{"latitude": round(float(la), 4), "longitude": round(float(lo), 4)} for la in lats for lo in lons]
    grid = pd.DataFrame(grid_rows)
    grid["cell_id"] = grid["latitude"].astype(str) + "_" + grid["longitude"].astype(str)
    return grid.reset_index(drop=True)