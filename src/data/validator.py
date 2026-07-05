"""
Data validation.
Run after every ingestion / enrichment step to catch issues early.
"""

import pandas as pd
from loguru import logger
from src.config import INDIA_BOUNDS


def validate_firms(df: pd.DataFrame) -> bool:
    """Basic integrity checks on a FIRMS DataFrame."""
    ok = True

    if df.empty:
        logger.error("FIRMS DataFrame is empty")
        return False

    for col in ["latitude", "longitude", "acq_date"]:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            ok = False

    lat_ok = (
        df["latitude"].between(INDIA_BOUNDS["lat_min"], INDIA_BOUNDS["lat_max"]).all()
    )
    lon_ok = (
        df["longitude"].between(INDIA_BOUNDS["lon_min"], INDIA_BOUNDS["lon_max"]).all()
    )

    if not lat_ok:
        logger.warning("Some latitudes outside India bounds")
    if not lon_ok:
        logger.warning("Some longitudes outside India bounds")

    null_pct = df[["latitude", "longitude", "acq_date"]].isnull().mean() * 100
    for col, pct in null_pct.items():
        if pct > 1:
            logger.warning(f"{col} has {pct:.1f}% nulls")

    logger.info(f"FIRMS validation: {len(df):,} rows, ok={ok}")
    return ok


def validate_enriched(df: pd.DataFrame) -> bool:
    """Check weather-enriched DataFrame before feature engineering."""
    ok = True

    for col in ["temp", "humidity", "wind"]:
        if col not in df.columns:
            logger.error(f"Missing weather column: {col}")
            ok = False
            continue

        null_pct = df[col].isnull().mean() * 100
        if null_pct > 10:
            logger.warning(f"{col}: {null_pct:.1f}% nulls (>10% threshold)")

    if "temp" in df.columns:
        out_of_range = (~df["temp"].between(-20, 55)).sum()
        if out_of_range > 0:
            logger.warning(f"{out_of_range} rows with temp outside [-20, 55]°C")

    if "humidity" in df.columns:
        out_of_range = (~df["humidity"].between(0, 100)).sum()
        if out_of_range > 0:
            logger.warning(f"{out_of_range} rows with humidity outside [0, 100]%")

    logger.info(f"Enriched data validation: {len(df):,} rows, ok={ok}")
    return ok