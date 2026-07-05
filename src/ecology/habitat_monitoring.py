"""
src/ecology/habitat_monitoring.py

Habitat change tracking and wildlife habitat suitability scoring.
Uses MODIS EVI time series to detect vegetation stress and habitat degradation.
"""

import pandas as pd
import numpy as np
from loguru import logger


# ── EVI Health Thresholds ────────────────────────────────────────────────────
# Think of EVI like a "health score" for vegetation
# 0.0 = bare rock/dead  |  0.5+ = lush dense forest

EVI_THRESHOLDS = {
    "excellent": 0.35,   # Dense healthy vegetation
    "good":      0.25,   # Moderate vegetation
    "stressed":  0.15,   # Dry / stressed vegetation
    # Below 0.15 = Degraded
}

CHANGE_ALERT_PCT = 15.0   # Flag if EVI drops more than 15% in a month


def classify_habitat_health(evi_value: float) -> str:
    """Convert a single EVI number into a health label."""
    if evi_value >= EVI_THRESHOLDS["excellent"]:
        return "Excellent"
    elif evi_value >= EVI_THRESHOLDS["good"]:
        return "Good"
    elif evi_value >= EVI_THRESHOLDS["stressed"]:
        return "Stressed"
    else:
        return "Degraded"


def detect_habitat_change(evi_series: pd.Series) -> dict:
    """
    Analyse an EVI time series for a single cell and return health summary.

    Parameters
    ----------
    evi_series : pd.Series
        Monthly EVI values ordered by date (oldest first).

    Returns
    -------
    dict with current health, trend, and alert flag.
    """
    if len(evi_series) < 2:
        return {"status": "insufficient_data"}

    current_evi   = float(evi_series.iloc[-1])
    monthly_delta = evi_series.pct_change() * 100          # % change month-to-month

    # 3-month trend
    if len(evi_series) >= 3:
        three_month_trend = float(
            (evi_series.iloc[-1] - evi_series.iloc[-3]) / evi_series.iloc[-3] * 100
        )
    else:
        three_month_trend = None

    has_significant_drop = bool((monthly_delta < -CHANGE_ALERT_PCT).any())

    return {
        "status":              "ok",
        "current_evi":         round(current_evi, 4),
        "health_status":       classify_habitat_health(current_evi),
        "trend_3month_pct":    round(three_month_trend, 2) if three_month_trend is not None else None,
        "has_significant_drop": has_significant_drop,
        "avg_evi":             round(float(evi_series.mean()), 4),
        "min_evi":             round(float(evi_series.min()), 4),
        "max_evi":             round(float(evi_series.max()), 4),
        "months_analysed":     len(evi_series),
    }


def monitor_region_habitat(
    grid_df: pd.DataFrame,
    evi_lookup,
    months: list,
    region_name: str = "Study Area",
) -> pd.DataFrame:
    """
    For every cell in grid_df, pull EVI values for each month and
    compute habitat health.

    Parameters
    ----------
    grid_df    : DataFrame with 'latitude', 'longitude' columns
    evi_lookup : EVILookup instance (already initialised)
    months     : list of date strings, e.g. ["2023-11-01", "2023-12-01", "2024-01-01"]
    region_name: Label used in log messages

    Returns
    -------
    DataFrame with one row per grid cell and habitat health columns.
    """
    logger.info(f"[Habitat] Monitoring {region_name} across {len(months)} months ...")

    results = []

    for _, row in grid_df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        evi_values = []
        for month_str in months:
            try:
                val = evi_lookup.enrich_dataframe(
                    pd.DataFrame([{"latitude": lat, "longitude": lon, "acq_date": month_str}]),
                    date_col="acq_date"
                )["evi"].iloc[0]
                evi_values.append(val)
            except Exception:
                evi_values.append(np.nan)

        evi_series = pd.Series(evi_values).dropna()
        change = detect_habitat_change(evi_series)

        results.append({
            "latitude":           lat,
            "longitude":          lon,
            "current_evi":        change.get("current_evi", np.nan),
            "health_status":      change.get("health_status", "Unknown"),
            "trend_3month_pct":   change.get("trend_3month_pct", np.nan),
            "has_concern":        change.get("has_significant_drop", False),
            "avg_evi":            change.get("avg_evi", np.nan),
        })

    df_out = pd.DataFrame(results)
    n_degraded = (df_out["health_status"] == "Degraded").sum()
    n_stressed  = (df_out["health_status"] == "Stressed").sum()

    logger.info(f"[Habitat] {region_name}: Degraded={n_degraded}, Stressed={n_stressed} / {len(df_out)} total cells")
    return df_out


# ── Wildlife Habitat Suitability ─────────────────────────────────────────────
# Scores each grid cell 0–1 for how suitable it is for each animal.
# Pure math — no images needed. Uses EVI + elevation + land cover.

class WildlifeHabitatScore:
    """
    Simple habitat suitability scoring for key Tamil Nadu wildlife.
    All scores are 0 (unsuitable) to 1 (ideal).
    """

    @staticmethod
    def tiger_score(df: pd.DataFrame) -> pd.Series:
        """
        Tigers prefer:
          - Dense tree cover
          - Mid-elevation (500–2500 m) — Nilgiri hills
          - Good vegetation (EVI > 0.25)
          - Cool-ish temperature (< 34°C)
        """
        s = pd.Series(0.0, index=df.index)
        s += (df["lc_tree_cover"] == 1).astype(float)                       * 0.35
        s += ((df["elevation_m"] >= 500) & (df["elevation_m"] <= 2500)).astype(float) * 0.25
        s += (df["evi"] > 0.25).astype(float)                               * 0.25
        s += (df["temp"] < 34).astype(float)                                * 0.15
        return s.clip(0, 1)

    @staticmethod
    def elephant_score(df: pd.DataFrame) -> pd.Series:
        """
        Elephants prefer:
          - Low-to-mid elevation (< 2000 m)
          - Tree cover OR grassland (they need variety)
          - Dense vegetation (EVI > 0.30)
        """
        s = pd.Series(0.0, index=df.index)
        s += (df["elevation_m"] < 2000).astype(float)                        * 0.30
        veg = ((df["lc_tree_cover"] == 1) | (df["lc_grassland"] == 1)).astype(float)
        s += veg                                                              * 0.35
        s += (df["evi"] > 0.30).astype(float)                               * 0.35
        return s.clip(0, 1)

    @staticmethod
    def leopard_score(df: pd.DataFrame) -> pd.Series:
        """
        Leopards are flexible:
          - Any elevation (100–2500 m)
          - Forest OR shrubland
          - Moderate vegetation (EVI > 0.18)
        """
        s = pd.Series(0.0, index=df.index)
        s += ((df["elevation_m"] >= 100) & (df["elevation_m"] <= 2500)).astype(float) * 0.30
        cover = ((df["lc_tree_cover"] == 1) | (df["lc_shrubland"] == 1)).astype(float)
        s += cover                                                            * 0.40
        s += (df["evi"] > 0.18).astype(float)                               * 0.30
        return s.clip(0, 1)