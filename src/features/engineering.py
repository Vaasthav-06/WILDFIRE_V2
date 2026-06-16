"""
src/features/engineering.py

Feature engineering for wildfire risk prediction (Real Data Only).

Feature set:
  Temporal      : month_sin, month_cos
  Weather       : temp, humidity, wind, vpd (ERA5)
  Vegetation    : evi (MODIS)
  Interactions  : vpd_wind

Note: Geospatial features (elevation, land cover) are extracted separately 
via raster sampling in scripts/extract_real_features.py.
"""

import numpy as np
import pandas as pd
from loguru import logger


# ── 1. Cyclical month encoding ───────────────────────────────────────────────

def add_cyclical_month(
    df: pd.DataFrame,
    date_col: str = "acq_date",
) -> pd.DataFrame:
    """Encode month as sine/cosine pair to preserve Dec→Jan continuity."""
    df = df.copy()
    month = pd.to_datetime(df[date_col]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


# ── 2. Interaction features ──────────────────────────────────────────────────

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-informed interaction features.
    vpd_wind: combined fire spread potential (extreme fire weather indicator).
    """
    df = df.copy()

    if "vpd" in df.columns and "wind" in df.columns:
        df["vpd_wind"] = (df["vpd"] * df["wind"]).round(4)

    return df


# ── 3. Master feature pipeline ───────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    date_col: str = "acq_date",
    evi_lookup=None,
) -> pd.DataFrame:
    """
    Core feature engineering pipeline (excluding spatial raster extraction).
    """
    logger.info(f"Building engineered features for {len(df):,} rows...")

    df = add_cyclical_month(df, date_col)
    logger.info("  ✓ Cyclical month encoding")

    if evi_lookup is not None:
        df = evi_lookup.enrich_dataframe(df, date_col)
        logger.info("  ✓ EVI (MODIS real values, 2018-2025)")
    else:
        raise RuntimeError("EVILookup is required. Cannot proceed without real MODIS data.")

    df = add_interactions(df)
    logger.info("  ✓ Interaction features (VPD x Wind)")

    return df