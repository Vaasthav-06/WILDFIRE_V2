"""
scripts/build_features.py

Run feature engineering on the enriched training data.
"""

from loguru import logger
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import DATA_PROCESSED, FEATURE_COLS, TARGET_COL
from src.features.engineering import build_features
from src.data.evi import EVILookup   # ← was NDVILookup

logger.add("logs/build_features.log", rotation="10 MB")
console = Console()


def main():
    input_path = DATA_PROCESSED / "training_data.csv"
    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} rows")

    # EVILookup is required — no proxy fallback
    try:
        evi_lookup = EVILookup()
        logger.info("EVILookup ready (MODIS EVI 2018-2025)")
    except FileNotFoundError as e:
        logger.error(
            f"MODIS EVI NetCDF files not found: {e}. "
            "Download from NASA AppEEARS (MOD13A3.061, _1_km_monthly_EVI) "
            "and place in data/raw/ndvi/."
        )
        raise

    df = build_features(df, evi_lookup=evi_lookup)   # ← removed use_elevation_api kwarg

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after engineering: {missing}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    out_path = DATA_PROCESSED / "training_features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}")

    # ── Feature summary (Rich Table) ─────────────────────────
    table = Table(title="Feature Engineering Summary", show_header=True)
    table.add_column("Feature", style="cyan")
    table.add_column("Non-null %", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")

    for col in FEATURE_COLS:
        if col in df.columns:
            s = df[col]
            table.add_row(
                col,
                f"{s.notna().mean()*100:.1f}%",
                f"{s.mean():.4f}",
                f"{s.std():.4f}",
            )

    console.print(table)


if __name__ == "__main__":
    main()