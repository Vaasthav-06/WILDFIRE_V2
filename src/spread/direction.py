"""
src/spread/direction.py

Fire spread direction estimation for high-risk grid cells.

This module answers a physically distinct question from the inference engine:
GIVEN that fire ignites at a high-risk cell, WHERE does it spread?

Two independent vector fields are composed:
  1. Wind-driven spread   — dominant factor at wind > 15 km/h
  2. Topographic spread   — slope-driven uphill acceleration

The composite spread vector is a weighted sum:
    S = α(wind) × V_wind + β(wind) × V_slope

where α and β are dynamically computed based on wind speed.
At high wind speeds, wind dominates; in calm conditions, slope matters more.

IMPORTANT: This is directional guidance, NOT a fire perimeter simulation.
Output should be labelled "Indicative spread direction" in all UIs.
A proper perimeter simulation requires FARSITE or Prometheus.

Mathematical reference:
  Rothermel (1972), Finney (1998) — wind/slope weighting for fire spread
  Beer (1990) — combined wind-terrain factor
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

# ── Wind speed thresholds for dynamic α/β weighting ─────────────────────────
# Below CALM_THRESHOLD: slope dominates (α=0.35, β=0.65)
# Above WIND_DOMINANT:  wind dominates  (α=0.75, β=0.25)
# Between: linear interpolation

WIND_CALM_MS = 4.0  # < 4 m/s (14.4 km/h) — slope-dominant regime
WIND_DOMINANT_MS = 8.0  # > 8 m/s (28.8 km/h) — wind-dominant regime


# ── Spread intensity thresholds (composite vector magnitude) ─────────────────
# Used to map magnitude → human-readable spread intensity string

SPREAD_INTENSITY_THRESHOLDS = {
    "calm": (0.0, 0.5),
    "light": (0.5, 1.5),
    "moderate": (1.5, 3.5),
    "rapid": (3.5, 7.0),
    "extreme": (7.0, float("inf")),
}


def _magnitude_to_intensity(mag: float) -> str:
    for label, (lo, hi) in SPREAD_INTENSITY_THRESHOLDS.items():
        if lo <= mag < hi:
            return label
    return "extreme"


from scipy.ndimage import generic_filter


def _fill_nan_grid(grid: np.ndarray) -> np.ndarray:
    """Fill NaN with mean of available 3×3 neighbourhood, then global mean."""
    filled = grid.copy()
    if not np.isnan(filled).any():
        return filled

    def _mean_ignore_nan(vals):
        v = vals[~np.isnan(vals)]
        return v.mean() if len(v) > 0 else np.nan

    filled = generic_filter(filled, _mean_ignore_nan, size=3, mode="nearest")
    still_nan = np.isnan(filled)
    if still_nan.any():
        filled[still_nan] = np.nanmean(grid)
    return filled


# ── Elevation gradient (slope vector) ────────────────────────────────────────


def compute_elevation_gradient(
    grid_df: pd.DataFrame,
    resolution_deg: float = 0.1,
) -> pd.DataFrame:
    df = grid_df.copy()

    if "elevation" not in df.columns:
        from src.features.engineering import _elevation_proxy

        df["elevation"] = _elevation_proxy(df["latitude"], df["longitude"])

    lats = np.sort(df["latitude"].unique())
    lons = np.sort(df["longitude"].unique())
    n_lat, n_lon = len(lats), len(lons)

    if n_lat < 2 or n_lon < 2:
        df["slope_u"] = 0.0
        df["slope_v"] = 0.0
        return df

    dy_m = resolution_deg * 111_000.0
    dx_m = resolution_deg * 102_000.0

    elev_grid = np.full((n_lat, n_lon), np.nan)

    # FIX: Vectorized numpy indexing instead of .iterrows() loops
    lat_idx = np.clip(
        np.searchsorted(lats, df["latitude"].round(4).values), 0, n_lat - 1
    )
    lon_idx = np.clip(
        np.searchsorted(lons, df["longitude"].round(4).values), 0, n_lon - 1
    )

    elev_grid[lat_idx, lon_idx] = df["elevation"].values

    # FIX: 2D Interpolation for missing data
    elev_grid = _fill_nan_grid(elev_grid)

    grad_y, grad_x = np.gradient(elev_grid, dy_m, dx_m)

    # Map directly back via indices
    df["slope_u"] = np.round(grad_x[lat_idx, lon_idx], 6)
    df["slope_v"] = np.round(grad_y[lat_idx, lon_idx], 6)

    return df


# ── Dynamic α/β weighting ─────────────────────────────────────────────────────


def _compute_weights(wind_speed_ms: float) -> tuple[float, float]:
    """
    Compute dynamic wind/slope weighting based on wind speed.

    Returns (alpha, beta) where alpha + beta = 1.
    alpha → weight for wind vector
    beta  → weight for slope vector

    At low wind:  slope dominates (alpha=0.35, beta=0.65)
    At high wind: wind dominates  (alpha=0.75, beta=0.25)
    Between:      linear interpolation
    """
    if wind_speed_ms <= WIND_CALM_MS:
        alpha, beta = 0.35, 0.65
    elif wind_speed_ms >= WIND_DOMINANT_MS:
        alpha, beta = 0.75, 0.25
    else:
        # Linear interpolation
        t = (wind_speed_ms - WIND_CALM_MS) / (WIND_DOMINANT_MS - WIND_CALM_MS)
        alpha = 0.35 + t * (0.75 - 0.35)
        beta = 1.0 - alpha

    return round(alpha, 3), round(beta, 3)


# ── Spread vector computation ─────────────────────────────────────────────────


def compute_spread_vectors(
    scored_df: pd.DataFrame,
    risk_threshold: float = 0.40,
) -> pd.DataFrame:
    """
    Compute fire spread direction and intensity for high-risk cells.

    Only runs on cells above risk_threshold (P(fire) ≥ threshold).
    For cells below threshold, spread columns are set to NaN — there is
    no fire to spread.

    Args:
        scored_df:       Output from WildfireInferenceEngine._score().
                         Must contain: fire_prob, wind_u, wind_v,
                         latitude, longitude, elevation (optional).
        risk_threshold:  Minimum P(fire) to compute spread for.

    Returns:
        scored_df with additional columns:
          spread_bearing_deg  — direction fire spreads TOWARD (0–360°, N=0)
          spread_intensity    — "calm"|"light"|"moderate"|"rapid"|"extreme"
          spread_u            — composite spread vector east component
          spread_v            — composite spread vector north component
          wind_alpha          — wind weight used
          slope_beta          — slope weight used
    """
    df = scored_df.copy()

    # Initialise spread columns
    for col in [
        "spread_bearing_deg",
        "spread_u",
        "spread_v",
        "wind_alpha",
        "slope_beta",
    ]:
        df[col] = np.nan
    df["spread_intensity"] = "none"

    # Compute elevation gradients for the full grid (needed for slope component)
    df = compute_elevation_gradient(df)

    # Select only high-risk cells for spread computation
    high_risk_mask = df["fire_prob"] >= risk_threshold
    n_high_risk = high_risk_mask.sum()

    if n_high_risk == 0:
        logger.info("No high-risk cells above threshold — no spread vectors computed.")
        return df

    logger.info(
        f"Computing spread vectors for {n_high_risk:,} high-risk cells "
        f"(P(fire) ≥ {risk_threshold})"
    )

    # Process each high-risk cell
    idx_list = df.index[high_risk_mask].tolist()

    for idx in idx_list:
        wind_u = df.at[idx, "wind_u"]
        wind_v = df.at[idx, "wind_v"]
        slope_u = df.at[idx, "slope_u"]
        slope_v = df.at[idx, "slope_v"]

        # Skip if wind components missing
        if pd.isna(wind_u) or pd.isna(wind_v):
            continue

        # Scalar wind speed for weight computation
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        alpha, beta = _compute_weights(float(wind_speed))

        # Normalise slope vector to prevent units mismatch
        # Slope vector magnitude is in m/m (dimensionless gradient)
        # Wind vector magnitude is in m/s
        # We normalise both to unit vectors before weighting
        wind_mag = wind_speed + 1e-9
        slope_mag = np.sqrt(slope_u**2 + slope_v**2) + 1e-9

        wind_unit_u = wind_u / wind_mag
        wind_unit_v = wind_v / wind_mag
        slope_unit_u = slope_u / slope_mag
        slope_unit_v = slope_v / slope_mag

        # Composite spread vector (weighted sum of unit vectors)
        # Scaled back by wind speed so magnitude reflects actual spread potential
        comp_u = (alpha * wind_unit_u + beta * slope_unit_u) * wind_speed
        comp_v = (alpha * wind_unit_v + beta * slope_unit_v) * wind_speed

        # Bearing: direction fire spreads TOWARD (meteorological → navigation)
        # atan2(east, north) → degrees clockwise from North
        # Wind u = east component, v = north component
        bearing_rad = np.arctan2(comp_u, comp_v)  # atan2(east, north)
        bearing_deg = float(np.degrees(bearing_rad)) % 360.0

        spread_magnitude = float(np.sqrt(comp_u**2 + comp_v**2))

        df.at[idx, "spread_bearing_deg"] = round(bearing_deg, 1)
        df.at[idx, "spread_u"] = round(float(comp_u), 3)
        df.at[idx, "spread_v"] = round(float(comp_v), 3)
        df.at[idx, "wind_alpha"] = alpha
        df.at[idx, "slope_beta"] = beta
        df.at[idx, "spread_intensity"] = _magnitude_to_intensity(spread_magnitude)

    computed = high_risk_mask.sum()
    intensity_dist = df.loc[high_risk_mask, "spread_intensity"].value_counts()
    logger.info(
        f"Spread vectors computed for {computed:,} cells | "
        f"Intensity distribution:\n{intensity_dist.to_string()}"
    )

    return df


# ── Compass bearing → text ────────────────────────────────────────────────────


def bearing_to_compass(bearing_deg: float) -> str:
    """
    Convert a 0–360° bearing to a 16-point compass label.

    E.g.: 0° → "N", 45° → "NE", 157.5° → "SSE"
    """
    if pd.isna(bearing_deg):
        return "unknown"

    labels = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    idx = round(bearing_deg / 22.5) % 16
    return labels[idx]
