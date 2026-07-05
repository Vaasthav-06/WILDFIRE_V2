"""
src/data/evi.py

MODIS EVI (Enhanced Vegetation Index) lookup.

Data source : NASA AppEEARS — MOD13A3.061
Variable    : _1_km_monthly_EVI
Coverage    : 2018-01-01 → 2025-12-01 (full pipeline window)
Resolution  : 1 km monthly composites, matched to ERA5 0.25° grid via
              nearest-neighbour lookup on (lat, lon, year-month).

Scale factor: raw MODIS EVI values are stored as integers × 10000.
              Valid range after scaling: -0.2 to 1.0.
              Values outside this range are MODIS fill values and are
              set to NaN. With full 2018-2025 satellite coverage there
              is no proxy fallback — NaN rows are logged as warnings.

Design:
  - One xarray Dataset loaded once at init from all NetCDF files in EVI_DIR.
  - enrich_dataframe() groups by year-month and does a vectorised
    nearest-neighbour xarray .sel() for all points in that month at once.
  - No row-wise loops.
"""

import numpy as np
import pandas as pd
import xarray as xr
import cftime
from pathlib import Path
from loguru import logger
from typing import Optional

from src.config import DATA_RAW

EVI_DIR = DATA_RAW / "evi"
EVI_VAR = "_1_km_monthly_EVI" # exact MODIS variable name in the NetCDF files

# MODIS EVI valid range after dividing by 10000
EVI_MIN = -0.2
EVI_MAX =  1.0


class EVILookup:
    """
    Monthly MODIS EVI lookup for the full 2018-2025 window.

    Usage:
        evi = EVILookup()
        df  = evi.enrich_dataframe(df, date_col="acq_date")
        # Adds column 'evi' to df.
    """

    def __init__(self):
        nc_files = sorted(EVI_DIR.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No EVI NetCDF files found in {EVI_DIR}. "
                "Download from NASA AppEEARS (MOD13A3.061, _1_km_monthly_EVI)."
            )

        # Suppress xarray FutureWarning about data_vars default
        
        self.ds = xr.open_mfdataset(
            nc_files,
            combine="by_coords",
            data_vars="minimal",
        )

        # --- THE DIMENSION RENAME FIX ---
        # NASA AppEEARS uses 'lat' and 'lon'. We must rename them to 
        # 'latitude' and 'longitude' to match our pipeline's feature columns.
        rename_map = {}
        if 'lat' in self.ds.dims: rename_map['lat'] = 'latitude'
        if 'lon' in self.ds.dims: rename_map['lon'] = 'longitude'
        if rename_map:
            self.ds = self.ds.rename(rename_map)
        # --------------------------------

        # --- THE CALENDAR FIX ---
        # NASA NetCDF files use a Julian calendar which crashes Pandas datetime matching.
        try:
            self.ds['time'] = pd.to_datetime([str(t)[:10] for t in self.ds.time.values])
        except Exception as e:
            logger.warning(f"Could not convert time calendar: {e}")
        # ------------------------

        # Verify the expected variable exists
        if EVI_VAR not in self.ds.data_vars:
            available = list(self.ds.data_vars)
            raise ValueError(
                f"Expected variable '{EVI_VAR}' not found in NetCDF files. "
                f"Available variables: {available}. "
                f"Re-download from AppEEARS selecting the EVI layer specifically."
            )

        time_min = str(self.ds.time.values[0])[:10]
        time_max = str(self.ds.time.values[-1])[:10]
        logger.info(
            f"EVILookup ready | "
            f"var={EVI_VAR} | "
            f"time={time_min} → {time_max} | "
            f"files={len(nc_files)}"
        )

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str = "acq_date",
        date_range: Optional[tuple[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Add real MODIS EVI values to df, matched by (lat, lon, year-month).

        Adds column: 'evi'  (float, range -0.2 to 1.0, scaled from raw × 10000)

        If date_range=(from, to) is provided, df is expanded to a time-series
        for every row across the months in the range.
        """
        import pandas as pd
        df = df.copy()
        if date_range:
            from_date, to_date = date_range
            dates = pd.date_range(start=from_date, end=to_date, freq="MS")
            if len(dates) == 0:
                dates = pd.date_range(start=from_date, end=to_date, periods=1)
            expanded_rows = []
            for _, row in df.iterrows():
                for d in dates:
                    new_row = row.to_dict()
                    new_row[date_col] = d
                    expanded_rows.append(new_row)
            df = pd.DataFrame(expanded_rows)

        df = df.copy()
        df["_dt"]    = pd.to_datetime(df[date_col])
        df["_month"] = df["_dt"].dt.to_period("M")
        df["evi"]    = np.nan

        unique_months = df["_month"].unique()
        logger.info(
            f"Enriching EVI for {len(df):,} rows "
            f"across {len(unique_months)} unique months..."
        )

        n_filled = 0
        n_missing_month = 0

        for period in unique_months:
            group = df[df["_month"] == period]
            ts = period.to_timestamp()

            # Try numpy datetime64 first (standard calendar NetCDFs)
            # Fall back to cftime.DatetimeJulian (MODIS AppEEARS uses Julian calendar)
            try:
                target = np.datetime64(str(ts))
                month_ds = self.ds.sel(time=target, method="nearest")
            except Exception:
                try:
                    # MODIS AppEEARS stores time as cftime.DatetimeJulian
                    target_cf = cftime.DatetimeJulian(ts.year, ts.month, ts.day)
                    month_ds = self.ds.sel(time=target_cf, method="nearest")
                except Exception as e:
                    logger.warning(
                        f"EVI: could not select month {period} from dataset: {e}"
                    )
                    n_missing_month += 1
                    continue

            lats = xr.DataArray(group["latitude"].values,  dims="points")
            lons = xr.DataArray(group["longitude"].values, dims="points")

            raw_vals = month_ds[EVI_VAR].sel(
                latitude=lats, longitude=lons, method="nearest"
            ).values

            # xarray usually auto-applies NetCDF scale factors. 
            # We only divide by 10000 if it looks like raw integers.
            scaled = raw_vals.astype(float)
            if np.nanmax(scaled) > 2.0:
                scaled = scaled / 10000.0

            # Mask fill values — anything outside valid EVI range is a fill
            scaled[(scaled < EVI_MIN) | (scaled > EVI_MAX)] = np.nan

            df.loc[group.index, "evi"] = np.round(scaled, 4)
            n_filled += (~np.isnan(scaled)).sum()

        # Report coverage
        n_null = df["evi"].isna().sum()
        pct_null = n_null / len(df) * 100

        if n_missing_month > 0:
            logger.warning(
                f"EVI: {n_missing_month} months had no matching satellite data"
            )

        if n_null > 0:
            logger.warning(
                f"EVI: {n_null:,} rows ({pct_null:.1f}%) have NaN EVI after lookup. "
                f"These may be ocean cells, cloud-masked pixels, or dates outside "
                f"the 2018-2025 window."
            )
        else:
            logger.info("EVI: 100% coverage — no NaN values")

        df = df.drop(columns=["_dt", "_month"])
        logger.info(
            f"EVI enrichment complete | "
            f"mean={df['evi'].mean():.3f} | "
            f"range=[{df['evi'].min():.3f}, {df['evi'].max():.3f}]"
        )
        return df