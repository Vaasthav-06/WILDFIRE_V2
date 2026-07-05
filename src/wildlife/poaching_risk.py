"""
src/wildlife/poaching_risk.py

Poaching risk prediction using tabular environmental and spatial features.
No images or CNN needed — pure ML on numbers.

Features used:
  - Distance to nearest road      (closer road = easier access for poachers)
  - Distance to nearest village   (closer village = higher human pressure)
  - Recent incident density       (past hotspots stay hotspots)
  - Days since last patrol        (un-patrolled = higher risk)
  - Wildlife suitability index    (more animals = more attractive to poachers)
  - Terrain slope                 (easier terrain = easier poaching access)
  - Vegetation density (EVI)      (dense forest = cover for poachers)
  - Elevation                     (low elevation = easier access)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from src.ecology.habitat_monitoring import WildlifeHabitatScore


FEATURE_NAMES = [
    "distance_to_road_km",
    "distance_to_village_km",
    "incident_density_6months",
    "days_since_last_patrol",
    "wildlife_index",
    "terrain_slope_deg",
    "evi",
    "elevation_m",
]


# ── Spatial helper: Haversine distance ──────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def nearest_distance_km(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    """
    For each point in grid, find the distance to the nearest target point.
    Returns array of distances in km, one per grid point.
    """
    distances = []
    for la, lo in zip(grid_lats, grid_lons):
        d = haversine_km(la, lo, target_lats, target_lons)
        distances.append(float(np.min(d)))
    return np.array(distances)


# ── Feature builder ──────────────────────────────────────────────────────────

def build_poaching_features(
    grid_df: pd.DataFrame,
    road_points: pd.DataFrame   = None,   # columns: latitude, longitude
    village_points: pd.DataFrame = None,  # columns: latitude, longitude
    incident_df: pd.DataFrame   = None,   # columns: latitude, longitude, date
    patrol_df: pd.DataFrame     = None,   # columns: latitude, longitude, patrol_date
) -> pd.DataFrame:
    """
    Build the full poaching risk feature matrix from a grid DataFrame.

    Any missing data source gets a sensible default so the pipeline
    doesn't crash if you don't have e.g. village coordinates yet.

    Parameters
    ----------
    grid_df        : Must have latitude, longitude, elevation_m, evi,
                     lc_tree_cover, lc_shrubland, lc_grassland, temp columns.
    road_points    : DataFrame of road/track points (lat/lon).
    village_points : DataFrame of village centroids (lat/lon).
    incident_df    : Historical poaching incidents with a 'date' column.
    patrol_df      : Patrol records with a 'patrol_date' column.

    Returns
    -------
    DataFrame with FEATURE_NAMES columns ready for model input.
    """
    logger.info(f"[Poaching] Building features for {len(grid_df)} cells ...")
    feat = grid_df[["latitude", "longitude"]].copy()

    # 1. Distance to road
    if road_points is not None and len(road_points) > 0:
        feat["distance_to_road_km"] = nearest_distance_km(
            grid_df["latitude"].values, grid_df["longitude"].values,
            road_points["latitude"].values, road_points["longitude"].values,
        )
    else:
        # Heuristic: cells closer to villages are also closer to roads
        logger.warning("[Poaching] No road data — using heuristic (0.7 × village dist)")
        feat["distance_to_road_km"] = grid_df.get(
            "distance_to_village_km",
            pd.Series(np.random.uniform(1, 40, len(grid_df)), index=grid_df.index)
        ) * 0.7

    # 2. Distance to village
    if village_points is not None and len(village_points) > 0:
        feat["distance_to_village_km"] = nearest_distance_km(
            grid_df["latitude"].values, grid_df["longitude"].values,
            village_points["latitude"].values, village_points["longitude"].values,
        )
    else:
        logger.warning("[Poaching] No village data — using placeholder distances")
        feat["distance_to_village_km"] = np.random.uniform(2, 60, len(grid_df))

    # 3. Recent incident density (count of incidents within 10 km, last 180 days)
    if incident_df is not None and len(incident_df) > 0:
        incident_df = incident_df.copy()
        incident_df["date"] = pd.to_datetime(incident_df["date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
        recent = incident_df[incident_df["date"] >= cutoff]

        density = []
        for _, row in grid_df.iterrows():
            if len(recent) == 0:
                density.append(0)
            else:
                dists = haversine_km(
                    row["latitude"], row["longitude"],
                    recent["latitude"].values, recent["longitude"].values,
                )
                density.append(int((dists <= 10).sum()))
        feat["incident_density_6months"] = density
    else:
        logger.warning("[Poaching] No incident data — setting density to 0")
        feat["incident_density_6months"] = 0

    # 4. Days since last patrol
    if patrol_df is not None and len(patrol_df) > 0:
        patrol_df = patrol_df.copy()
        patrol_df["patrol_date"] = pd.to_datetime(patrol_df["patrol_date"])
        today = pd.Timestamp.now()

        days_list = []
        for _, row in grid_df.iterrows():
            dists = haversine_km(
                row["latitude"], row["longitude"],
                patrol_df["latitude"].values, patrol_df["longitude"].values,
            )
            nearest_idx = int(np.argmin(dists))
            days = (today - patrol_df.iloc[nearest_idx]["patrol_date"]).days
            days_list.append(float(days))
        feat["days_since_last_patrol"] = days_list
    else:
        logger.warning("[Poaching] No patrol data — assuming 14-day default")
        feat["days_since_last_patrol"] = 14.0

    # 5. Wildlife index  (average of tiger + elephant + leopard suitability)
    tiger_s   = WildlifeHabitatScore.tiger_score(grid_df)
    elephant_s = WildlifeHabitatScore.elephant_score(grid_df)
    leopard_s  = WildlifeHabitatScore.leopard_score(grid_df)
    feat["wildlife_index"] = (tiger_s * 0.5 + elephant_s * 0.3 + leopard_s * 0.2).values

    # 6. Terrain slope  (approximated from elevation neighbours)
    #    If you have a real slope raster, sample it here instead.
    if "slope_deg" in grid_df.columns:
        feat["terrain_slope_deg"] = grid_df["slope_deg"].values
    else:
        logger.warning("[Poaching] No slope data — approximating from elevation variance")
        feat["terrain_slope_deg"] = np.clip(
            np.abs(grid_df["elevation_m"].diff().fillna(0)) / 5.5, 0, 60
        ).values   # very rough proxy

    # 7. EVI
    feat["evi"] = grid_df["evi"].values

    # 8. Elevation
    feat["elevation_m"] = grid_df["elevation_m"].values

    logger.info("[Poaching] Feature matrix ready.")
    return feat[FEATURE_NAMES]


# ── Model class ──────────────────────────────────────────────────────────────

# In src/wildlife/poaching_risk.py

class PoachingRiskScorer:
    """
    Rule-based poaching risk scoring.
    Fully transparent, no ML model, no data leakage.
    
    Risk combines three ecological/geographic factors:
      1. Wildlife suitability (50%) — more animals = more attractive target
      2. Access ease via roads (30%) — closer roads = easier poacher access
      3. Patrol neglect (20%) — longer since last patrol = riskier
    """
    
    # Risk tier thresholds (cumulative score 0–1)
    RISK_TIERS = {
        (0.0, 0.25):   "Low",
        (0.25, 0.50):  "Moderate",
        (0.50, 0.75):  "High",
        (0.75, 1.0):   "Critical",
    }
    
    WEIGHTS = {
        "wildlife": 0.50,
        "road_proximity": 0.30,
        "patrol_recency": 0.20,
    }
    
    @staticmethod
    def compute_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute poaching risk for a grid.
        
        Inputs:
          df must have: wildlife_index, distance_to_road_km, days_since_last_patrol
        """
        df_out = df.copy()
        
        # Component 1: Wildlife attractiveness (already 0–1)
        wildlife_score = df["wildlife_index"].clip(0, 1)
        
        # Component 2: Road proximity (closer = higher risk)
        # Invert distance: near road (1 km) → 1.0, far (60 km) → 0.0
        road_score = 1.0 / (df["distance_to_road_km"].clip(0.5, 60) / 10)
        road_score = road_score.clip(0, 1)
        
        # Component 3: Patrol recency (neglected areas = higher risk)
        # 0 days since patrol → 0.0, 90+ days → 1.0
        patrol_score = (df["days_since_last_patrol"] / 90).clip(0, 1)
        
        # Weighted composite
        df_out["poaching_risk"] = (
            wildlife_score * PoachingRiskScorer.WEIGHTS["wildlife"] +
            road_score * PoachingRiskScorer.WEIGHTS["road_proximity"] +
            patrol_score * PoachingRiskScorer.WEIGHTS["patrol_recency"]
        ).clip(0, 1)
        
        # Tier assignment
        df_out["poaching_tier"] = df_out["poaching_risk"].apply(
            lambda x: next(
                (tier for (lo, hi), tier in PoachingRiskScorer.RISK_TIERS.items() if lo <= x < hi),
                "Critical"
            )
        )
        
        return df_out
    
    @staticmethod
    def save_metadata(output_path: Path):
        """Save the scoring logic and weights for transparency."""
        metadata = {
            "type": "rule_based_scorer",
            "version": "1.0",
            "components": {
                "wildlife_suitability": {
                    "weight": PoachingRiskScorer.WEIGHTS["wildlife"],
                    "source": "species-centric habitat scores (tiger/elephant/leopard)",
                    "range": [0, 1]
                },
                "road_proximity": {
                    "weight": PoachingRiskScorer.WEIGHTS["road_proximity"],
                    "formula": "1 / (distance_km / 10), clipped [0,1]",
                    "interpretation": "closer roads → easier access for poachers"
                },
                "patrol_recency": {
                    "weight": PoachingRiskScorer.WEIGHTS["patrol_recency"],
                    "formula": "days_since_patrol / 90, clipped [0,1]",
                    "interpretation": "longer since patrol → higher risk"
                }
            },
            "tiers": PoachingRiskScorer.RISK_TIERS,
            "rationale": "Transparent rule-based system avoids data leakage from synthetic labels."
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Poaching risk scoring logic saved to {output_path}")