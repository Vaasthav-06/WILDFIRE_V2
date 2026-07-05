import joblib
import numpy as np
import pandas as pd
import time
from pathlib import Path
from loguru import logger
from typing import Optional, Union

from src.config import (
    FEATURE_COLS,
    MODELS_DIR,
    DATA_CACHE,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_CRITICAL,
)
from src.features.engineering import build_features
from src.data.evi import EVILookup
from src.inference.forecast_ingest import (
    fetch_24h_forecast,
    fetch_7day_forecast,
    build_inference_grid,
)

RISK_TIERS = {
    "low": (0.00, 0.20),
    "moderate": (0.20, 0.40),
    "high": (0.40, 0.65),
    "extreme": (0.65, 1.01),
}

def _prob_to_tier(prob: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "extreme"

class WildfireInferenceEngine:
    def __init__(self, model_path: Optional[Path] = None, grid_resolution: float = 0.1):
        if model_path is None:
            model_path = MODELS_DIR / "production_model.pkl"

        logger.info(f"Loading production model from: {model_path}")
        self._production = joblib.load(model_path)
        self.model = self._production["model"]
        self.threshold = self._production["threshold"]
        self.features = self._production["features"]
        self.best_name = self._production.get("best_single", "lightgbm")
        self.cv_results = self._production.get("cv_results", {})

        if hasattr(self.model, "named_estimators_"):
            for name, estimator in self.model.named_estimators_.items():
                try:
                    if 'xgb' in name.lower():
                        estimator.set_params(device='cuda')
                    elif 'lgb' in name.lower():
                        estimator.set_params(device_type='gpu')
                except Exception as e:
                    logger.debug(f"Could not set GPU params for {name}: {e}")

        self.grid_resolution = grid_resolution
        self._inference_grid = None
        self._kmtr_grid = None
        
        # Initialize EVILookup handle for feature compilation
        self.evi_lookup = EVILookup()

    def get_inference_grid(self, area: Optional[dict] = None) -> pd.DataFrame:
        is_kmtr = False
        if area:
            # Check if this is the KMTR region (approx)
            if abs(area.get("lat_min", 0) - 8.35) < 0.1 and abs(area.get("lon_min", 0) - 77.15) < 0.1:
                is_kmtr = True
        
        if is_kmtr:
            if self._kmtr_grid is None:
                self._kmtr_grid = build_inference_grid(
                    lat_min=area["lat_min"], lat_max=area["lat_max"], 
                    lon_min=area["lon_min"], lon_max=area["lon_max"], 
                    resolution=area.get("resolution", 0.05)
                )
                logger.info("Generated and cached KMTR-specific inference grid.")
            return self._kmtr_grid.copy()
            
        if self._inference_grid is None:
            cache_path = DATA_CACHE / "india_spatial_grid.parquet"
            if cache_path.exists():
                self._inference_grid = pd.read_parquet(cache_path)
                logger.info(f"Loaded spatial cache from {cache_path}")
            else:
                logger.warning(f"Spatial cache missing: {cache_path}. Building India grid...")
                self._inference_grid = build_inference_grid()
                
        grid = self._inference_grid.copy()
        if area:
            grid = grid[
                (grid["latitude"] >= area["lat_min"]) &
                (grid["latitude"] <= area["lat_max"]) &
                (grid["longitude"] >= area["lon_min"]) &
                (grid["longitude"] <= area["lon_max"])
            ]
            
        return grid

    def _validate_feature_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise ValueError(f"SCHEMA VIOLATION: Missing features: {missing}")

        X_ordered = X[self.features].copy()
        null_counts = X_ordered.isnull().sum()
        null_features = null_counts[null_counts > 0]
        if not null_features.empty:
            for col in null_features.index:
                X_ordered[col].fillna(X_ordered[col].median(), inplace=True)
        return X_ordered

    def _score(self, df_enriched: pd.DataFrame, profiler: dict = None) -> pd.DataFrame:
        t0 = time.time()
        # FIX: Pass the instantiated self.evi_lookup object to build_features
        df_featured = build_features(df_enriched, evi_lookup=self.evi_lookup)
        X = self._validate_feature_schema(df_featured)
        if profiler is not None:
            profiler["feature_eng_s"] = time.time() - t0
            t0 = time.time()

        fire_prob = self.model.predict_proba(X)[:, 1]
        model_std = self._compute_model_disagreement(X)

        df_result = df_enriched.copy()
        df_result["fire_prob"] = np.round(fire_prob, 4)
        df_result["fire_pred"] = (fire_prob >= self.threshold).astype(int)
        df_result["risk_tier"] = [_prob_to_tier(p) for p in fire_prob]
        df_result["model_std"] = np.round(model_std, 4)
        
        # Compute FRI
        try:
            from src.risk.fire_risk_index import compute_fri
            fri_df = compute_fri(df_featured)
            df_result["fri"] = fri_df["fri"]
            df_result["fri_tier"] = fri_df["fri_tier"]
        except Exception as e:
            logger.error(f"Failed to compute FRI: {e}")
            df_result["fri"] = np.nan
            df_result["fri_tier"] = None
        
        
        if profiler is not None:
            profiler["xgboost_s"] = time.time() - t0

        return df_result

    def _compute_model_disagreement(self, X: pd.DataFrame) -> np.ndarray:
        try:
            if hasattr(self.model, "named_estimators_"):
                individual_probs = []
                for name, estimator in self.model.named_estimators_.items():
                    if hasattr(estimator, "predict_proba"):
                        p = estimator.predict_proba(X)[:, 1]
                        individual_probs.append(p)
                if len(individual_probs) > 1:
                    return np.std(np.column_stack(individual_probs), axis=1)
            return np.zeros(len(X))
        except Exception:
            return np.zeros(len(X))

    def predict_now(self, grid_df: Optional[pd.DataFrame] = None, profiler: dict = None) -> pd.DataFrame:
        if grid_df is None:
            grid_df = self.get_inference_grid()
        t0 = time.time()
        df_weather = fetch_24h_forecast(grid_df)
        if profiler is not None:
            profiler["open_meteo_s"] = time.time() - t0
        return self._score(df_weather, profiler=profiler)

    def predict_7day(self, grid_df: Optional[pd.DataFrame] = None) -> list[pd.DataFrame]:
        if grid_df is None:
            grid_df = self.get_inference_grid()
        daily_frames = fetch_7day_forecast(grid_df=grid_df)
        results = []
        for day_idx, df_day in enumerate(daily_frames):
            df_scored = self._score(df_day)
            df_scored["forecast_day"] = day_idx + 1
            df_scored["forecast_date"] = df_day["acq_date"].iloc[0]
            results.append(df_scored)
        return results

    def predict_custom(self, df_weather: pd.DataFrame) -> pd.DataFrame:
        return self._score(df_weather)

    def summary(self) -> dict:
        best_cv = self.cv_results.get(self.best_name, {}).get("cv", {})
        return {
            "model_type": type(self.model).__name__,
            "best_single": self.best_name,
            "threshold": self.threshold,
            "n_features": len(self.features),
            "cv_auc_mean": best_cv.get("roc_auc_mean"),
        }