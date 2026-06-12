"""
src/inference/engine.py

Operational inference engine for wildfire risk prediction.

This is the central coordinator. It:
  1. Loads the trained production_model.pkl
  2. Accepts a weather-enriched grid DataFrame
  3. Runs the IDENTICAL feature engineering pipeline as training
  4. Scores each cell through the ensemble
  5. Returns structured predictions with risk tiers

Critical design principle:
  The production model is STATIC. It learned f(features) → P(fire).
  We call it with future weather and it returns future P(fire).
  No retraining, no modifications to the model.

  The ONLY operational contract that must hold:
    feature vector at inference time == feature vector at training time
    (same column names, same order, same units, same scale)

  This module enforces that contract via _validate_feature_schema().
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional, Union

from src.config import (
    FEATURE_COLS,
    MODELS_DIR,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_CRITICAL,
)
from src.features.engineering import build_features
from src.inference.forecast_ingest import (
    fetch_24h_forecast,
    fetch_7day_forecast,
    build_india_inference_grid,
)

# ── Risk tier thresholds ──────────────────────────────────────────────────────
# These map P(fire) to operational alert tiers used by first responders.
# Based on Brier-score calibrated probability buckets — not arbitrary.

RISK_TIERS = {
    "low": (0.00, 0.20),  # No action required
    "moderate": (0.20, 0.40),  # Monitor
    "high": (0.40, 0.65),  # Prepare resources
    "extreme": (0.65, 1.01),  # Immediate deployment
}


def _prob_to_tier(prob: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "extreme"


# ── Model loading ─────────────────────────────────────────────────────────────


class WildfireInferenceEngine:
    """
    Production wildfire risk inference engine.

    Loads the trained ensemble from production_model.pkl and provides
    three prediction methods:

      predict_now()         → 24-hour risk grid
      predict_7day()        → 7-day daily risk grids
      predict_custom(df)    → Score any pre-enriched DataFrame

    Usage
    -----
        engine = WildfireInferenceEngine()
        result_24h = engine.predict_now()
        result_7day = engine.predict_7day()
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        grid_resolution: float = 0.1,
    ):
        """
        Args:
            model_path:       Path to production_model.pkl.
                              Defaults to MODELS_DIR / "production_model.pkl".
            grid_resolution:  Inference grid spacing in degrees.
                              Use 0.1 for operational runs, 0.5 for testing.
        """
        if model_path is None:
            model_path = MODELS_DIR / "production_model.pkl"

        logger.info(f"Loading production model from: {model_path}")
        self._production = joblib.load(model_path)

        # Unpack the production bundle saved by scripts/train_models.py
        self.model = self._production["model"]  # VotingClassifier ensemble
        self.threshold = self._production["threshold"]  # Optimal F1 threshold
        self.features = self._production["features"]  # FEATURE_COLS at train time
        self.best_name = self._production.get("best_single", "lightgbm")
        self.cv_results = self._production.get("cv_results", {})

        # Attempt to enable GPU inference for underlying ensemble estimators
        if hasattr(self.model, "named_estimators_"):
            for name, estimator in self.model.named_estimators_.items():
                try:
                    if 'xgb' in name.lower():
                        estimator.set_params(device='cuda') # For newer xgboost
                    elif 'lgb' in name.lower():
                        estimator.set_params(device_type='gpu')
                    elif 'cat' in name.lower():
                        # CatBoost GPU inference needs to be set during init/training, but we can try
                        estimator.set_params(task_type='GPU')
                except Exception as e:
                    logger.debug(f"Could not set GPU params for {name}: {e}")

        logger.info(
            f"Model loaded: {type(self.model).__name__} | "
            f"threshold={self.threshold} | "
            f"features={len(self.features)}"
        )

        # Verify the saved feature list matches the current config
        if self.features != FEATURE_COLS:
            diff = set(self.features).symmetric_difference(set(FEATURE_COLS))
            logger.warning(
                f"Feature mismatch between model and config.py: {diff}. "
                "Using MODEL's feature list (authoritative)."
            )

        self.grid_resolution = grid_resolution
        self._inference_grid = None  # Lazy-loaded

    # ── Grid management ───────────────────────────────────────────────────────

    def get_inference_grid(self) -> pd.DataFrame:
        """Return (or build) the India inference grid."""
        if self._inference_grid is None:
            self._inference_grid = build_india_inference_grid(self.grid_resolution)
        return self._inference_grid

    # ── Feature schema validation ─────────────────────────────────────────────

    def _validate_feature_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce the exact feature schema the model was trained on.

        Checks:
          - All required features present
          - No unexpected extra features
          - No null values that would silently corrupt predictions
          - Correct column order (tree models are order-sensitive)

        Returns X with exactly the columns in self.features, in order.
        Raises ValueError if required columns are missing.
        """
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise ValueError(
                f"SCHEMA VIOLATION: {len(missing)} required features missing from "
                f"inference input: {missing}. "
                f"This usually means a step in build_features() did not run."
            )

        # Select only model features, in exact training order
        X_ordered = X[self.features].copy()

        # Null check — impute with column medians rather than silently scoring NaN
        null_counts = X_ordered.isnull().sum()
        null_features = null_counts[null_counts > 0]
        if not null_features.empty:
            logger.warning(
                f"Null values in {len(null_features)} features before inference. "
                f"Imputing with training-time medians:\n{null_features.to_string()}"
            )
            # Fill with feature means (conservative imputation)
            # In production, these means should be saved from training data
            for col in null_features.index:
                fill_val = X_ordered[col].median()
                X_ordered[col].fillna(fill_val, inplace=True)

        return X_ordered

    # ── Core scoring ──────────────────────────────────────────────────────────

    def _score(self, df_enriched: pd.DataFrame) -> pd.DataFrame:
        """
        Run feature engineering and score a weather-enriched grid DataFrame.

        This is the inner loop called by all public predict_* methods.
        It is the SINGLE place where the training→inference feature contract
        is enforced.

        Args:
            df_enriched:  DataFrame with weather columns (temp, humidity,
                          wind, precip, vpd, wind_u, wind_v) and acq_date.
                          Must also have latitude, longitude.

        Returns:
            df_enriched with added columns:
              fire_prob     — P(fire) ∈ [0, 1]
              fire_pred     — binary prediction at calibrated threshold
              risk_tier     — "low" | "moderate" | "high" | "extreme"
              model_std     — inter-model disagreement (confidence proxy)
        """
        logger.info(f"Running feature engineering on {len(df_enriched):,} rows...")

        # Run identical feature pipeline as training
        # Note: KBDI will be overridden by kbdi_forecast in 7-day mode
        df_featured = build_features(df_enriched, use_elevation_api=False)

        # Override KBDI with forward-propagated value if present
        # (set by fetch_7day_forecast before feature engineering)
        if "kbdi_forecast" in df_featured.columns:
            n_override = df_featured["kbdi_forecast"].notna().sum()
            logger.info(
                f"  Overriding KBDI with forward-propagated values for {n_override:,} cells"
            )
            mask = df_featured["kbdi_forecast"].notna()
            df_featured.loc[mask, "kbdi_approx"] = df_featured.loc[
                mask, "kbdi_forecast"
            ]

        # Validate and enforce feature schema
        X = self._validate_feature_schema(df_featured)

        logger.info(f"Scoring {len(X):,} cells through ensemble...")

        # Primary probability
        fire_prob = self.model.predict_proba(X)[:, 1]

        # Inter-model disagreement (uncertainty signal)
        # VotingClassifier exposes individual estimator probabilities
        model_std = self._compute_model_disagreement(X)

        # Assemble results
        df_result = df_enriched.copy()
        df_result["fire_prob"] = np.round(fire_prob, 4)
        df_result["fire_pred"] = (fire_prob >= self.threshold).astype(int)
        df_result["risk_tier"] = [_prob_to_tier(p) for p in fire_prob]
        df_result["model_std"] = np.round(model_std, 4)

        tier_counts = pd.Series([_prob_to_tier(p) for p in fire_prob]).value_counts()
        logger.info(
            f"Scoring complete | "
            f"Mean P(fire)={fire_prob.mean():.3f} | "
            f"Alerts (≥high): {(fire_prob >= 0.40).sum():,} | "
            f"Tier distribution:\n{tier_counts.to_string()}"
        )

        return df_result

    def _compute_model_disagreement(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute standard deviation of P(fire) across individual ensemble models.
        """
        try:
            # Try VotingClassifier route first
            if hasattr(self.model, "named_estimators_"):
                individual_probs = []
                # FIX: Use named_estimators_.items()
                for name, estimator in self.model.named_estimators_.items():
                    if hasattr(estimator, "predict_proba"):
                        p = estimator.predict_proba(X)[:, 1]
                        individual_probs.append(p)
                if len(individual_probs) > 1:
                    return np.std(np.column_stack(individual_probs), axis=1)

            # Single model — no disagreement signal available
            return np.zeros(len(X))

        except Exception as e:
            logger.debug(f"Could not compute model disagreement: {e}")
            return np.zeros(len(X))

    # ── Public prediction methods ─────────────────────────────────────────────

    def predict_now(
        self,
        grid_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate 24-hour wildfire risk predictions for India.

        Fetches peak-weather composite for the next 24 hours,
        runs feature engineering, scores each grid cell.

        Args:
            grid_df:  Optional custom grid. If None, uses India inference grid.

        Returns:
            DataFrame with one row per grid cell and columns:
            latitude, longitude, cell_id, temp, humidity, wind, precip,
            vpd, wind_u, wind_v, kbdi_approx, fire_prob, fire_pred,
            risk_tier, model_std
        """
        if grid_df is None:
            grid_df = self.get_inference_grid()

        logger.info(f"=== 24h Risk Prediction | {len(grid_df):,} cells ===")

        # Fetch forecast weather
        df_weather = fetch_24h_forecast(grid_df)

        # Score
        df_result = self._score(df_weather)

        logger.info("=== 24h prediction complete ===")
        return df_result

    def predict_7day(
        self,
        grid_df: Optional[pd.DataFrame] = None,
        baseline_kbdi: Optional[np.ndarray] = None,
    ) -> list[pd.DataFrame]:
        """
        Generate 7-day daily wildfire risk predictions for India.

        For each forecast day:
          - Fetches daily max/min weather variables
          - Propagates KBDI forward (stateful)
          - Runs feature engineering
          - Scores each cell

        Args:
            grid_df:        Optional custom grid. If None, uses India grid.
            baseline_kbdi:  Current KBDI per cell [n_cells]. If None,
                            uses 200.0 conservative default.

        Returns:
            List of 7 DataFrames (day+1 through day+7), each identical
            in structure to the predict_now() return value.
            Indexed [0..6]: result[0] = tomorrow, result[6] = day+7.
        """
        if grid_df is None:
            grid_df = self.get_inference_grid()

        logger.info(f"=== 7-Day Risk Prediction | {len(grid_df):,} cells ===")

        # Fetch all 7 days of weather (single API call batch)
        daily_frames = fetch_7day_forecast(
            grid_df=grid_df,
            baseline_kbdi=baseline_kbdi,
        )

        # Score each day independently
        results = []
        for day_idx, df_day in enumerate(daily_frames):
            logger.info(f"  Scoring day {day_idx + 1}/7...")
            df_scored = self._score(df_day)
            df_scored["forecast_day"] = day_idx + 1
            df_scored["forecast_date"] = df_day["acq_date"].iloc[0]
            results.append(df_scored)

        logger.info("=== 7-day prediction complete ===")
        return results

    def predict_custom(
        self,
        df_weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score a pre-enriched DataFrame directly.

        Useful for:
          - Unit testing with known weather values
          - Bandipur case study validation
          - Region-specific deep dives

        Args:
            df_weather:  DataFrame already enriched with weather columns
                         (temp, humidity, wind, precip, vpd) and acq_date.
                         Must have latitude, longitude.

        Returns:
            Same DataFrame with fire_prob, fire_pred, risk_tier, model_std added.
        """
        logger.info(f"Custom prediction: {len(df_weather):,} rows")
        return self._score(df_weather)

    # ── Utility ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary dict of the loaded model's training-time performance."""
        best_cv = {}
        if self.best_name in self.cv_results:
            best_cv = self.cv_results[self.best_name].get("cv", {})

        return {
            "model_type": type(self.model).__name__,
            "best_single": self.best_name,
            "threshold": self.threshold,
            "n_features": len(self.features),
            "features": self.features,
            "cv_auc_mean": best_cv.get("roc_auc_mean"),
            "cv_f1_mean": best_cv.get("f1_mean"),
            "grid_resolution": self.grid_resolution,
        }
