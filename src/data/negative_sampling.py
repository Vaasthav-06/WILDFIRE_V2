"""
src/data/negative_sampling.py

Negative (no-fire) sample generation — date-conditional true-absence strategy.

Methodology
-----------
A location is labelled as "no-fire" for a specific date ONLY IF no fire was
recorded at that (snapped grid cell, date) combination in the FIRMS dataset.

This is called "date-conditional true absence" and is the correct strategy
because it:
  1. Preserves fire-season weather in the negatives — the model must learn
     actual weather differences between fire and no-fire, not season identity.
  2. Avoids the month-offset leakage pattern where negatives drawn from
     non-fire seasons let the model trivially separate classes by month.
  3. Uses the same ERA5 grid resolution (0.25°) for snapping so negative
     coordinates always resolve to a valid ERA5 cell.

Key correctness guarantees
--------------------------
- Grid snapping: (coord / 0.25).round() * 0.25  — NOT round(2), which does
  not guarantee alignment to the 0.25° ERA5 grid.
- Double-append bug: fixed. Each accepted candidate appends ONCE. Previous
  code appended twice (snapped + unsnapped) causing ~50% inflated counts and
  duplicate rows with slightly different weather values.
- Leakage check: vectorised set-intersection O(min(|A|,|B|)) instead of
  row-wise apply O(n). Runs in milliseconds on 5M rows.
- Accept-rate sanity: with 1.14M fire (cell, date) pairs out of 39.6M
  possible combinations (~2.9% occupied), expected accept rate from random
  sampling is ~97%. Any run reporting <80% accept rate signals a bug.
"""

import numpy as np
import pandas as pd
from loguru import logger
from src.config import INDIA_BOUNDS, RANDOM_SEED

GRID_RESOLUTION = 0.25


def _snap_to_grid(series: pd.Series, resolution: float = GRID_RESOLUTION) -> pd.Series:
    """
    Snap coordinate series to ERA5 grid resolution.
    Correct formula: (coord / res).round() * res
    NOT: round(2) — that does not guarantee grid alignment.
    """
    return ((series / resolution).round() * resolution).round(2)


def build_fire_index(df_fire: pd.DataFrame) -> set:
    """
    Build a set of (snapped_lat, snapped_lon, date_str) tuples from fire records.

    This is the ground-truth lookup used to reject candidate negatives that
    overlap with actual fire observations.
    """
    df = df_fire.copy()
    df["_lat_g"] = _snap_to_grid(df["latitude"])
    df["_lon_g"] = _snap_to_grid(df["longitude"])
    df["_date"]  = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")

    fire_index = set(
        zip(df["_lat_g"].round(2), df["_lon_g"].round(2), df["_date"])
    )
    logger.info(
        f"Fire index: {len(fire_index):,} unique (snapped_cell, date) fire observations"
    )
    return fire_index


def build_india_grid(resolution: float = GRID_RESOLUTION) -> pd.DataFrame:
    """
    Build complete India land grid at ERA5 resolution.
    All cells are pre-snapped so sampling from this grid
    always produces valid ERA5 coordinate lookups.
    """
    lats = np.arange(
        INDIA_BOUNDS["lat_min"], INDIA_BOUNDS["lat_max"], resolution
    ).round(2)
    lons = np.arange(
        INDIA_BOUNDS["lon_min"], INDIA_BOUNDS["lon_max"], resolution
    ).round(2)
    grid_lats, grid_lons = np.meshgrid(lats, lons)
    grid = pd.DataFrame(
        {"latitude": grid_lats.ravel(), "longitude": grid_lons.ravel()}
    )
    return grid


def generate_date_conditional_negatives(
    df_fire: pd.DataFrame,
    n_negatives: int,
    random_seed: int = RANDOM_SEED,
    max_attempts_multiplier: int = 5,
) -> pd.DataFrame:
    """
    Generate n_negatives date-conditional absence records.

    For each candidate (grid_cell, fire_date) pair:
      - Accept if NOT in fire_index (true absence)
      - Reject if in fire_index (would contaminate negatives)

    The candidate pool is (all India grid cells) × (all dates that have
    any fire recorded anywhere in India). This ensures:
      - Negatives are drawn from the same weather regime as fires
      - The model cannot trivially separate classes by season or geography

    Args:
        df_fire:                  Enriched fire DataFrame (positives).
        n_negatives:              Number of negative samples to generate.
                                  Typically = len(df_fire) for 1:1 balance.
        random_seed:              For reproducibility.
        max_attempts_multiplier:  Max candidates per batch = n_negatives × this.

    Returns:
        DataFrame with columns: latitude, longitude, acq_date, fire_detected=0
    """
    rng = np.random.default_rng(random_seed)
    fire_index = build_fire_index(df_fire)

    # Grid cells — all pre-snapped to ERA5 resolution
    grid = build_india_grid(GRID_RESOLUTION)
    grid_lats = grid["latitude"].values
    grid_lons = grid["longitude"].values
    n_cells = len(grid)

    # Candidate dates = all dates that have fire somewhere in India
    # Using fire dates (not hardcoded months) preserves temporal realism
    fire_dates_str = (
        pd.to_datetime(df_fire["acq_date"])
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .tolist()
    )
    n_dates = len(fire_dates_str)

    logger.info(
        f"Candidate pool: {n_cells:,} cells × {n_dates} dates = "
        f"{n_cells * n_dates:,} possible negatives"
    )

    accepted_lats  = []
    accepted_lons  = []
    accepted_dates = []

    batch_size = min(n_negatives * max_attempts_multiplier, 5_000_000)
    attempts   = 0
    max_total  = n_negatives * max_attempts_multiplier * 10

    while len(accepted_lats) < n_negatives and attempts < max_total:
        n_needed = n_negatives - len(accepted_lats)
        n_batch  = min(batch_size, n_needed * max_attempts_multiplier)

        cell_idx = rng.integers(0, n_cells, size=n_batch)
        date_idx = rng.integers(0, n_dates, size=n_batch)

        batch_lats  = grid_lats[cell_idx]
        batch_lons  = grid_lons[cell_idx]
        batch_dates = [fire_dates_str[i] for i in date_idx]

        for lat, lon, date in zip(batch_lats, batch_lons, batch_dates):
    # Use _snap_to_grid formula — must match build_fire_index() exactly
            lat_s = round(round(float(lat) / GRID_RESOLUTION) * GRID_RESOLUTION, 2)
            lon_s = round(round(float(lon) / GRID_RESOLUTION) * GRID_RESOLUTION, 2)

            if (lat_s, lon_s, date) not in fire_index:
                accepted_lats.append(lat_s)
                accepted_lons.append(lon_s)
                accepted_dates.append(date)
                if len(accepted_lats) >= n_negatives:
                    break

        attempts += n_batch
        logger.info(
            f"Progress: {len(accepted_lats):,}/{n_negatives:,} "
            f"accepted from {attempts:,} candidates"
        )

    if len(accepted_lats) < n_negatives:
        logger.warning(
            f"Only {len(accepted_lats):,} negatives generated "
            f"(requested {n_negatives:,}). "
            f"Consider increasing max_attempts_multiplier."
        )

    negatives = pd.DataFrame(
        {
            "latitude":      accepted_lats[:n_negatives],
            "longitude":     accepted_lons[:n_negatives],
            "acq_date":      accepted_dates[:n_negatives],
            "fire_detected": 0,
        }
    )
    return negatives.reset_index(drop=True)


def build_balanced_dataset(
    df_fire: pd.DataFrame,
    era5_lookup,
    n_negatives: int = None,
    random_seed: int = RANDOM_SEED,
    weather_lag_days: int = 0,
) -> pd.DataFrame:
    """
    Build a balanced fire/no-fire training dataset.

    Steps:
      1. Generate n_negatives date-conditional absence records
      2. Enrich negatives with ERA5 weather
      3. Vectorised leakage check (set intersection — O(min(|A|,|B|)))
      4. Concatenate and shuffle positives + negatives

    Args:
        df_fire:          Enriched fire DataFrame (positives).
        era5_lookup:      ERA5Lookup instance for weather enrichment.
        n_negatives:      Number of negatives. Defaults to len(df_fire) (1:1).
        random_seed:      For reproducibility.
        weather_lag_days: Days to subtract from acq_date before ERA5 lookup.
                          Must match the lag used to enrich the positive samples.
                          Default 0 = same-day (backward-compatible).

    Returns:
        Balanced DataFrame with fire_detected in {0, 1}.
    """
    if n_negatives is None:
        n_negatives = len(df_fire)

    negatives_raw = generate_date_conditional_negatives(
        df_fire, n_negatives, random_seed
    )

    logger.info("Enriching negatives with ERA5 weather...")
    negatives_enriched = era5_lookup.enrich_dataframe(
        negatives_raw, date_col="acq_date", lag_days=weather_lag_days
    )
    negatives_enriched = negatives_enriched.dropna(
        subset=["temp", "humidity", "wind", "vpd"]
    )
    negatives_enriched["fire_detected"] = 0

    # Ensure 1:1 balance after dropna
    n_final = min(len(df_fire), len(negatives_enriched))
    df_fire_out = df_fire.sample(n=n_final, random_state=random_seed).copy()
    negatives_out = negatives_enriched.sample(n=n_final, random_state=random_seed)

    df_fire_out["fire_detected"] = 1

    # ── Vectorised leakage check ─────────────────────────────────────────────
    # Check that no negative (cell, date) tuple appears in the fire index.
    # Uses set intersection — O(min(|A|,|B|)) vs O(n) for row-wise apply.
    fire_idx = build_fire_index(df_fire)

    neg_check = negatives_enriched.copy()
    neg_check["_lat_g"] = _snap_to_grid(neg_check["latitude"])
    neg_check["_lon_g"] = _snap_to_grid(neg_check["longitude"])
    neg_check["_date"]  = pd.to_datetime(neg_check["acq_date"]).dt.strftime("%Y-%m-%d")

    check_tuples = set(
        zip(
            neg_check["_lat_g"].round(2),
            neg_check["_lon_g"].round(2),
            neg_check["_date"],
        )
    )
    leaks = len(check_tuples & fire_idx)

    if leaks > 0:
        logger.error(
            f"LEAKAGE DETECTED: {leaks} negative (cell, date) pairs "
            f"match fire index. This should never happen — check sampling logic."
        )
    else:
        logger.info("Leakage check passed: 0 negatives overlap with fire records")

    # ── Combine and shuffle ──────────────────────────────────────────────────
    combined = pd.concat([df_fire_out, negatives_out], ignore_index=True)
    combined = combined.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return combined