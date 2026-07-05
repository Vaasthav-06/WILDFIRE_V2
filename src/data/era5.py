"""
ERA5 climate data ingestion and local lookup.

Workflow:
  1. download_era5()     — downloads .nc files from CDS (run once)
  2. ERA5Lookup          — fast local (lat, lon, date) → weather lookup
                           zero network calls after download

Variables downloaded:
  - 2m temperature          (for temp, VPD)
  - 2m dewpoint temperature  (for VPD calculation)
  - 10m wind speed (u + v)  (combined → scalar wind)
  - Total precipitation      (antecedent dryness)
  - Surface pressure         (for VPD)
"""

import cdsapi
import numpy as np
import xarray as xr
from pathlib import Path
from loguru import logger
from datetime import datetime
import sys

# Force Python to see the main root directory so 'from src...' works
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_RAW, INDIA_BOUNDS

ERA5_DIR = DATA_RAW / "era5"
ERA5_DIR.mkdir(parents=True, exist_ok=True)

CDS_VARIABLES_INSTANT = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]

CDS_VARIABLES_ACCUM = [
    "total_precipitation",
]

CDS_AREA = [
    INDIA_BOUNDS["lat_max"],
    INDIA_BOUNDS["lon_min"],
    INDIA_BOUNDS["lat_min"],
    INDIA_BOUNDS["lon_max"],
]


def download_era5(years: list[int], overwrite: bool = False) -> list[Path]:
    """
    Download ERA5 data for India. Saves instant and accum as separate files.
    Makes two distinct requests to guarantee split file structure without zip parsing.
    """
    import zipfile, shutil, tempfile
    c = cdsapi.Client()
    paths = []

    # FIX: Removed the "year" key from here
    common_request = {
        "product_type": "reanalysis",
        "month":  [f"{m:02d}" for m in range(1, 13)],
        "day":    [f"{d:02d}" for d in range(1, 32)],
        "time":   ["06:00", "12:00", "15:00"],
        "area":   CDS_AREA,
        "format": "netcdf",
    }

    for year in years:
        instant_path = ERA5_DIR / f"era5_{year}_instant.nc"
        accum_path   = ERA5_DIR / f"era5_{year}_accum.nc"

        if instant_path.exists() and accum_path.exists() and not overwrite:
            logger.info(f"ERA5 {year} already exists — skipping")
            paths += [instant_path, accum_path]
            continue

        for var_list, out_path, label in [
            (CDS_VARIABLES_INSTANT, instant_path, "instant"),
            (CDS_VARIABLES_ACCUM,   accum_path,   "accum"),
        ]:
            if out_path.exists() and not overwrite:
                continue
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            # FIX: Injected the "year" key dynamically inside the loop
            req = {**common_request, "variable": var_list, "year": [str(year)]}
            
            logger.info(f"Requesting ERA5 {year} {label} from CDS...")
            c.retrieve("reanalysis-era5-single-levels", req, str(tmp_path))
            shutil.copy(tmp_path, out_path)
            tmp_path.unlink(missing_ok=True)
            logger.info(f"Saved → {out_path}")

        paths += [instant_path, accum_path]
    return paths


class ERA5Lookup:
    """
    Fast local lookup using the split instant/accum file structure.
    Loads both datasets into memory once, serves queries in microseconds.
    """

    def __init__(self, years: list[int]):
        self.ds_instant, self.ds_accum = self._load(years)
        logger.info(
            f"ERA5Lookup ready | "
            f"instant vars: {list(self.ds_instant.data_vars)} | "
            f"accum vars: {list(self.ds_accum.data_vars)}"
        )

    def _load(self, years: list[int]):
        instant_paths, accum_paths = [], []

        for year in years:
            ip = ERA5_DIR / f"era5_{year}_instant.nc"
            ap = ERA5_DIR / f"era5_{year}_accum.nc"
            if not ip.exists():
                raise FileNotFoundError(f"Missing: {ip}")
            if not ap.exists():
                raise FileNotFoundError(f"Missing: {ap}")
            instant_paths.append(ip)
            accum_paths.append(ap)

        ds_i = xr.open_mfdataset(instant_paths, combine="by_coords", engine="netcdf4")
        ds_a = xr.open_mfdataset(accum_paths, combine="by_coords", engine="netcdf4")
        return ds_i, ds_a

    def query(self, lat: float, lon: float, date: str, hour: int = 12) -> dict:
        """Single point lookup. Returns dict with temp, humidity, wind, precip, vpd."""
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")

        target = np.datetime64(date.strftime(f"%Y-%m-%dT{hour:02d}:00:00"))

        pt_i = self.ds_instant.sel(latitude=lat, longitude=lon, method="nearest").sel(
            valid_time=target, method="nearest"
        )

        pt_a = self.ds_accum.sel(latitude=lat, longitude=lon, method="nearest").sel(
            valid_time=target, method="nearest"
        )

        t2m = float(pt_i["t2m"].values)
        td2m = float(pt_i["d2m"].values)
        u10 = float(pt_i["u10"].values)
        v10 = float(pt_i["v10"].values)
        sp = float(pt_i["sp"].values)
        tp = float(pt_a["tp"].values)

        temp_c = t2m - 273.15
        dewpoint_c = td2m - 273.15
        wind_ms = float(np.sqrt(u10**2 + v10**2))
        precip_mm = tp * 1000
        humidity = _relative_humidity(temp_c, dewpoint_c)
        vpd = _vpd(temp_c, humidity)

        return {
            "temp": round(temp_c, 2),
            "dewpoint": round(dewpoint_c, 2),
            "wind": round(wind_ms, 3),
            "precip": round(precip_mm, 3),
            "pressure": round(sp, 1),
            "humidity": round(humidity, 2),
            "vpd": round(vpd, 4),
        }

    def enrich_dataframe(
        self,
        df,
        date_col: str = "acq_date",
        hour: int = 12,
    ):
        """
        Enrich entire DataFrame with ERA5 weather. Groups by date for speed.
        Processes ~1M rows in seconds using fully vectorized xarray indexing.
        """
        import pandas as pd
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TaskProgressColumn,
            TimeElapsedColumn,
        )

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        out_cols = ["temp", "dewpoint", "humidity", "wind", "precip", "vpd"]
        for col in out_cols:
            df[col] = np.nan

        grouped = df.groupby(df[date_col].dt.date)
        logger.info(f"Enriching {len(df):,} rows across {len(grouped)} unique dates...")

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("ERA5 enrichment", total=len(grouped))

            for date_val, group in grouped:
                date_str = str(date_val)
                target = np.datetime64(f"{date_str}T{hour:02d}:00:00")

                try:
                    day_i = self.ds_instant.sel(valid_time=target, method="nearest")
                    day_a = self.ds_accum.sel(valid_time=target, method="nearest")
                except Exception:
                    progress.advance(task)
                    continue

                lats = xr.DataArray(group["latitude"].values, dims="points")
                lons = xr.DataArray(group["longitude"].values, dims="points")

                pt_i = day_i.sel(latitude=lats, longitude=lons, method="nearest")
                pt_a = day_a.sel(latitude=lats, longitude=lons, method="nearest")

                t2m = pt_i["t2m"].values - 273.15
                td2m = pt_i["d2m"].values - 273.15
                u10 = pt_i["u10"].values
                v10 = pt_i["v10"].values
                tp = pt_a["tp"].values * 1000

                wind_ms = np.sqrt(u10**2 + v10**2)

                a, b = 17.625, 243.04
                hum = np.clip(
                    100
                    * np.exp((a * td2m) / (b + td2m))
                    / np.exp((a * t2m) / (b + t2m)),
                    0,
                    100,
                )
                es = 0.6108 * np.exp((17.27 * t2m) / (t2m + 237.3))
                ea = es * (hum / 100.0)
                vpd = np.maximum(es - ea, 0.0)

                df.loc[group.index, "temp"] = np.round(t2m, 2)
                df.loc[group.index, "dewpoint"] = np.round(td2m, 2)
                df.loc[group.index, "wind"] = np.round(wind_ms, 3)
                df.loc[group.index, "precip"] = np.round(tp, 3)
                df.loc[group.index, "humidity"] = np.round(hum, 2)
                df.loc[group.index, "vpd"] = np.round(vpd, 4)

                progress.advance(task)

        null_pct = df[out_cols].isnull().mean() * 100
        logger.info(f"Null rates after enrichment:\n{null_pct.round(1)}")
        return df


# ── Physics functions ────────────────────────────────────────────────────────


def _relative_humidity(temp_c: float, dewpoint_c: float) -> float:
    """
    Magnus formula: RH from temperature and dewpoint.
    More accurate than lookup tables for the India temperature range.
    """
    if np.isnan(temp_c) or np.isnan(dewpoint_c):
        return np.nan
    a, b = 17.625, 243.04
    rh = 100 * (
        np.exp((a * dewpoint_c) / (b + dewpoint_c))
        / np.exp((a * temp_c) / (b + temp_c))
    )
    return float(np.clip(rh, 0, 100))


def _vpd(temp_c: float, rh: float) -> float:
    """
    Vapor Pressure Deficit from temperature and relative humidity.
    Tetens formula — matches your existing VPD implementation.
    """
    if np.isnan(temp_c) or np.isnan(rh):
        return np.nan
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh / 100.0)
    return float(max(es - ea, 0.0))