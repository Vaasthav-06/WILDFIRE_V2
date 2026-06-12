"""
src/features/engineering.py

Feature engineering for wildfire risk prediction.

Feature set (15 features, post-pruning):
  Ecoregion (5)  : eco_tropical_moist, eco_tropical_dry, eco_semi_arid,
                   eco_montane, eco_subtropical
  Temporal  (2)  : month_sin, month_cos
  Weather   (4)  : temp, humidity, wind, vpd  [ERA5 reanalysis]
  Drought   (1)  : kbdi_approx  [derived from ERA5 precip + temp]
  Vegetation(1)  : evi  [MODIS EVI 2018-2025, real satellite data]
  Interactions(2): vpd_wind, temp_dryness

Removed features:
  elevation    — proxy only (bounding-box lookup, no real DEM), near-zero
                 ablation impact (-0.0015 AUC). Drop until SRTM integrated.
  dist_road_km — proxy only (std=1.5 km across continent), zero discrimination
                 power. Drop until real OSM data integrated.
"""

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

from src.config import DATA_CACHE


# ── 1. Cyclical month encoding ───────────────────────────────────────────────


def add_cyclical_month(
    df: pd.DataFrame,
    date_col: str = "acq_date",
) -> pd.DataFrame:
    """
    Encode month as sine/cosine pair.
    Preserves Dec→Jan continuity unlike raw month integer.
    """
    df = df.copy()
    month = pd.to_datetime(df[date_col]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


# ── 2. KBDI per location ─────────────────────────────────────────────────────


def add_kbdi(
    df: pd.DataFrame,
    date_col: str = "acq_date",
    precip_col: str = "precip",
    temp_col: str = "temp",
    grid_resolution: float = 0.25,
) -> pd.DataFrame:
    """
    Keetch-Byram Drought Index approximation computed per grid cell.

    Physically grounded: derived from ERA5 temperature and precipitation
    using the published Keetch-Byram formula. Not a measured quantity —
    describe in paper as "KBDI approximated from ERA5 reanalysis at 0.25°".

    Uses time-aware rolling (30 calendar days) to correctly handle sparse
    datasets — rolling on row count would be wrong here.

    Grid snapping: uses (coord / 0.25).round() * 0.25 to match ERA5 exactly.
    raw round(2) does NOT guarantee grid alignment.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_col])

    # Correct grid snapping — matches ERA5 0.25° resolution exactly
    df["_lat_r"] = ((df["latitude"] / grid_resolution).round() * grid_resolution).round(2)
    df["_lon_r"] = ((df["longitude"] / grid_resolution).round() * grid_resolution).round(2)
    df["_cell"]  = df["_lat_r"].astype(str) + "_" + df["_lon_r"].astype(str)

    temp = df[temp_col].clip(0, 55)
    df["_et"] = (0.968 * np.exp(0.0875 * temp + 1.5552) - 8.30).clip(lower=0) / 1000.0
    precip = df[precip_col].clip(lower=0)
    df["_deficit"] = (df["_et"] - precip).clip(lower=0)

    original_index = df.index.copy()
    df = df.sort_values(["_cell", "_date"]).copy()

    # Vectorised per-cell time-aware rolling (30 calendar days)
    df["kbdi_approx"] = (
        df.groupby("_cell", group_keys=False).apply(
            lambda g: pd.Series(
                g.set_index("_date")["_deficit"]
                .rolling("30D", min_periods=1)
                .sum()
                .values,
                index=g.index,
            )
        )
        * 800
    ).clip(0, 800)

    df = df.loc[original_index]
    df = df.drop(
        columns=["_date", "_lat_r", "_lon_r", "_cell", "_et", "_deficit"],
        errors="ignore",
    )

    logger.info(
        f"KBDI (per-location, time-aware): "
        f"mean={df['kbdi_approx'].mean():.1f}, "
        f"max={df['kbdi_approx'].max():.1f}"
    )
    return df


# ── 3. Ecoregion encoding ────────────────────────────────────────────────────


def add_ecoregion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace raw lat/lon with ecoregion binary encoding.

    Raw coordinates let the model memorise "block X = high fire risk"
    without learning any weather relationships. Ecoregion encoding
    preserves the geographic signal at the right level of abstraction.

    Five zones based on WWF India ecoregion classification:
      tropical_moist : Kerala, NE India — humid evergreen forest
      tropical_dry   : Deccan, Chhattisgarh — dry deciduous forest
      semi_arid      : Rajasthan, Gujarat — scrub and grassland
      montane        : Himalayas, upper Western Ghats
      subtropical    : Indo-Gangetic Plain — cropland and dry forest

    Note: A point can belong to multiple zones (overlapping definitions).
    This is intentional — ecoregion boundaries are fuzzy in reality.

    Note on elevation dependency: previously ecoregion used an elevation
    proxy. Since elevation has been dropped, montane classification uses
    latitude (>28°N) only — this is conservative but avoids a phantom
    dependency on a removed feature.
    """
    df = df.copy()
    lat = df["latitude"]
    lon = df["longitude"]

    df["eco_tropical_moist"] = (
        ((lat < 20) & (lon > 75) & (lon < 82))
        | ((lat < 28) & (lon > 88))   # Northeast India
    ).astype(int)

    df["eco_tropical_dry"] = (
        (lat.between(15, 25)) & (lon.between(75, 87)) & (lat >= 18)
    ).astype(int)

    df["eco_semi_arid"] = (
        (lon < 75) | ((lat.between(22, 28)) & (lon < 77))
    ).astype(int)

    # Montane: latitude-only (elevation proxy removed)
    df["eco_montane"] = (lat > 28).astype(int)

    df["eco_subtropical"] = (
        lat.between(25, 30) & lon.between(74, 88)
    ).astype(int)

    eco_cols = [
        "eco_tropical_moist",
        "eco_tropical_dry",
        "eco_semi_arid",
        "eco_montane",
        "eco_subtropical",
    ]
    logger.info(f"Ecoregion distribution:\n{df[eco_cols].sum().to_string()}")
    return df


# ── 4. Interaction features ──────────────────────────────────────────────────


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-informed interaction features.

    vpd_wind     : VPD × wind — combined fire spread potential.
                   Empirically validated: high VPD + high wind = extreme
                   fire weather (used in McArthur Fire Danger Rating).
    temp_dryness : temp × normalised KBDI — heat under drought stress.
                   Captures compound drought-heat events that precede
                   India's worst fire seasons.
    """
    df = df.copy()

    if "vpd" in df.columns and "wind" in df.columns:
        df["vpd_wind"] = (df["vpd"] * df["wind"]).round(4)

    if "temp" in df.columns and "kbdi_approx" in df.columns:
        kbdi_norm = df["kbdi_approx"] / 800.0
        df["temp_dryness"] = (df["temp"] * kbdi_norm).round(4)

    return df


# ── 5. Master pipeline ───────────────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    date_col: str = "acq_date",
    evi_lookup=None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Call on both training data and inference grids.

    Order matters:
      1. Cyclical month  (needs date)
      2. KBDI            (needs precip + temp, sorts by date per cell)
      3. EVI             (needs date + lat + lon; real MODIS satellite data)
      4. Ecoregion       (needs lat + lon)
      5. Interactions    (needs vpd + wind + temp + kbdi)

    Args:
        df:          Input DataFrame with ERA5 columns + latitude/longitude
        date_col:    Date column name
        evi_lookup:  Instance of EVILookup (required for training; if None,
                     raises an error rather than silently falling back to proxy)

    Returns:
        DataFrame with all FEATURE_COLS added.
    """
    from src.config import FEATURE_COLS

    logger.info(f"Building features for {len(df):,} rows...")

    df = add_cyclical_month(df, date_col)
    logger.info("  ✓ Cyclical month encoding")

    df = add_kbdi(df, date_col)
    logger.info("  ✓ KBDI per-location drought index")

    # ── EVI — real MODIS satellite data only ──
    if evi_lookup is not None:
        df = evi_lookup.enrich_dataframe(df, date_col)
        logger.info("  ✓ EVI (MODIS real values, 2018-2025)")
    else:
        raise RuntimeError(
            "EVILookup is required — no proxy fallback. "
            "Ensure data/raw/ndvi/ contains MODIS NetCDF files "
            "and pass an EVILookup instance to build_features()."
        )

    df = add_ecoregion_features(df)
    logger.info("  ✓ Ecoregion encoding")

    df = add_interactions(df)
    logger.info("  ✓ Interaction features")

    # ── Validate ──
    present = [f for f in FEATURE_COLS if f in df.columns]
    missing = [f for f in FEATURE_COLS if f not in df.columns]

    if missing:
        logger.error(f"Missing features after engineering: {missing}")
        raise ValueError(f"Feature engineering incomplete. Missing: {missing}")

    null_rates = df[present].isnull().mean() * 100
    high_null = null_rates[null_rates > 1]
    if not high_null.empty:
        logger.warning(f"High null rates (>1%):\n{high_null}")

    logger.info(
        f"Feature engineering complete. "
        f"{len(present)}/{len(FEATURE_COLS)} features ready."
    )
    return df