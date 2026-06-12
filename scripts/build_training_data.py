"""
scripts/build_training_data.py

Full training data build pipeline:
  1. Load historical FIRMS CSV
  2. Enrich with local ERA5 NetCDF weather grids
  3. Generate negative samples via True Absence Sampling
  4. Save final training_data.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.rule import Rule

from src.config import DATA_RAW, DATA_PROCESSED, RANDOM_SEED
from src.data.nasa_firms import load_historical_firms
from src.data.era5 import ERA5Lookup
from src.data.negative_sampling import build_balanced_dataset
from src.data.validator import validate_firms, validate_enriched

# ── Logger & UI setup ────────────────────────────────────────
logger.add("logs/build_training_data.log", rotation="10 MB", level="INFO")
console = Console()


def phase(title: str):
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))


def main(input_path: Path) -> None:
    # ── Step 1: Load FIRMS ───────────────────────────────────
    phase("Step 1: Loading FIRMS data")
    df_raw = load_historical_firms(input_path)
    validate_firms(df_raw)
    logger.info(f"Loaded: {len(df_raw):,} fire records")

    # ── Step 2: Weather enrichment (ERA5 Offline) ────────────
    phase("Step 2: Weather enrichment (ERA5 Offline)")

    era5 = ERA5Lookup(years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    df_enriched = era5.enrich_dataframe(df_raw, date_col="acq_date", hour=12)
    df_enriched = df_enriched.dropna(subset=["temp", "humidity", "wind"])
    validate_enriched(df_enriched)
    logger.info(f"After enrichment: {len(df_enriched):,} rows")

    enriched_path = DATA_PROCESSED / "firms_enriched.csv"
    df_enriched.to_csv(enriched_path, index=False)
    logger.info(f"Enriched fire records saved → {enriched_path}")

    # ── Step 3: Build balanced dataset ──────────────────────
    phase("Step 3: Negative sample generation")

    df_training = build_balanced_dataset(
        df_fire=df_enriched,
        era5_lookup=era5,
        n_negatives=len(df_enriched),
        random_seed=RANDOM_SEED,
    )

    # ── Step 4: Save ─────────────────────────────────────────
    out_path = DATA_PROCESSED / "training_data.csv"
    df_training.to_csv(out_path, index=False)

    n_fire = (df_training["fire_detected"] == 1).sum()
    n_nofire = (df_training["fire_detected"] == 0).sum()

    phase("Pipeline complete")
    logger.info(f"Output : {out_path}")
    logger.info(f"Total  : {len(df_training):,} rows")
    logger.info(f"Fire   : {n_fire:,} | No-fire: {n_nofire:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_RAW / "filtered_2018_2025.csv",
        help="Path to historical FIRMS CSV",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    main(args.input)
