"""
src/explainability/why_engine.py

The "WHY" Engine — SHAP-based natural language explainability for firefighters.

Purpose
-------
A first responder looking at a high-risk alert does not need to know what
SHAP is. They need to know: "WHY is this cell flagged?"

This module translates numerical SHAP values into ranked, human-readable
sentences using a template system. No language model required — each feature
has a two-directional template (risk-increasing and risk-mitigating) with the
actual feature value interpolated in.

Design principles
-----------------
1. SHAP is computed ONLY for cells above an alert threshold to keep latency low.
2. The WHY engine is gated: if a cell is below threshold, no SHAP is run.
3. Uncertainty is always exposed: inter-model disagreement → confidence label.
4. Negative contributors (mitigating factors) are reported alongside positive
   ones — this is critical for trust. Responders need to know what is
   working in their favour, not just what is against them.
5. The template system is a lookup table, not a language model. This means
   it is fast, deterministic, and auditable.

Integration with the inference engine
--------------------------------------
The WHY engine is called AFTER the inference engine produces fire_prob scores.
It takes the raw feature vectors for flagged cells, computes SHAP for those
cells only, and augments the result DataFrame.

SHAP model support
------------------
  - LightGBM, XGBoost, Random Forest, CatBoost → TreeExplainer (exact, fast)
  - Logistic Regression → LinearExplainer
  - VotingClassifier → weighted average of constituent SHAP vectors
    (weights = softmax of individual model CV AUCs, or equal weights)
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

from src.config import FEATURE_COLS

# ── Feature template library ──────────────────────────────────────────────────
# Each feature has two templates:
#   "risk_up"   → when SHAP is positive (feature is increasing fire probability)
#   "risk_down" → when SHAP is negative (feature is decreasing fire probability)
#
# Template format uses {value} for the feature value and {formatted} for
# a human-readable formatted version.

FEATURE_TEMPLATES = {
    "temp": {
        "risk_up": "High temperature ({value:.1f}°C) increasing fuel ignition risk",
        "risk_down": "Mild temperature ({value:.1f}°C) reducing ignition potential",
    },
    "humidity": {
        "risk_up": "Low relative humidity ({value:.0f}%) drying out vegetation",
        "risk_down": "High humidity ({value:.0f}%) suppressing fire risk",
    },
    "wind": {
        "risk_up": "Strong winds ({value:.1f} m/s) accelerating potential spread",
        "risk_down": "Calm winds ({value:.1f} m/s) limiting fire spread",
    },
    "vpd": {
        "risk_up": "Extreme atmospheric dryness (VPD {value:.2f} kPa) stressing vegetation",
        "risk_down": "Low vapour pressure deficit (VPD {value:.2f} kPa) — conditions not critically dry",
    },
    "kbdi_approx": {
        "risk_up": "Severe drought conditions (KBDI {value:.0f}/800) — soil moisture critically depleted",
        "risk_down": "Adequate soil moisture (KBDI {value:.0f}/800) — recent rainfall limiting risk",
    },
    "ndvi_proxy": {
        "risk_up": "Dry, sparse vegetation (NDVI {value:.2f}) — high fuel load",
        "risk_down": "Dense, moist vegetation (NDVI {value:.2f}) — lower fuel flammability",
    },
    "elevation": {
        "risk_up": "Elevated terrain ({value:.0f}m) with steep slopes accelerating uphill spread",
        "risk_down": "Low-elevation flat terrain ({value:.0f}m) limiting topographic spread",
    },
    "dist_road_km": {
        "risk_up": "Close to road network ({value:.1f}km) — higher human ignition risk",
        "risk_down": "Remote location ({value:.1f}km from roads) — lower human ignition risk",
    },
    "vpd_wind": {
        "risk_up": "Combined fire weather index elevated (VPD × wind = {value:.2f}) — dangerous spread conditions",
        "risk_down": "Combined fire weather index low (VPD × wind = {value:.2f}) — spread potential limited",
    },
    "temp_dryness": {
        "risk_up": "Heat-drought compound stress ({value:.2f}) — extreme fuel curing conditions",
        "risk_down": "Moderate heat-drought index ({value:.2f}) — not critically stressed",
    },
    "month_sin": {
        "risk_up": "Seasonal timing consistent with peak fire season",
        "risk_down": "Seasonal timing not aligned with peak fire season",
    },
    "month_cos": {
        "risk_up": "Seasonal timing consistent with peak fire season",
        "risk_down": "Seasonal timing not aligned with peak fire season",
    },
    "eco_tropical_moist": {
        "risk_up": "Tropical moist forest ecoregion — fire-sensitive vegetation type",
        "risk_down": "Tropical moist forest — typically high humidity limits fire risk",
    },
    "eco_tropical_dry": {
        "risk_up": "Tropical dry forest ecoregion — high fire season susceptibility",
        "risk_down": "Tropical dry forest — outside peak burn window",
    },
    "eco_semi_arid": {
        "risk_up": "Semi-arid scrubland — chronically low fuel moisture",
        "risk_down": "Semi-arid conditions — sparse fuel load limiting fire size",
    },
    "eco_montane": {
        "risk_up": "Montane terrain — complex fire behaviour in steep slopes",
        "risk_down": "Montane terrain — higher elevation conditions currently moderate",
    },
    "eco_subtropical": {
        "risk_up": "Subtropical dry forest — fire-prone during dry season",
        "risk_down": "Subtropical zone — conditions currently within safe range",
    },
}


def _render_template(feature: str, value: float, shap_positive: bool) -> str:
    """
    Render the natural language string for a (feature, value, direction) triplet.

    Falls back to a generic template if the feature is not in the library.
    """
    tmpl = FEATURE_TEMPLATES.get(feature)

    if tmpl is None:
        direction = "increasing" if shap_positive else "decreasing"
        return (
            f"{feature.replace('_', ' ').title()} ({value:.3g}) {direction} fire risk"
        )

    key = "risk_up" if shap_positive else "risk_down"
    template_str = tmpl[key]

    # Safe format — skip {value} if feature has no numeric interpretation
    try:
        return template_str.format(value=value)
    except (KeyError, TypeError):
        return template_str


# ── SHAP explainer factory ────────────────────────────────────────────────────


def _get_shap_explainer(model):
    """
    Return the correct SHAP explainer for a given model type.

    Handles all model types present in the wildfire_v2 production bundle:
      - LightGBM, XGBoost, CatBoost, Random Forest → TreeExplainer
      - Logistic Regression (in Pipeline) → LinearExplainer
      - VotingClassifier → returns None (handled separately by _shap_for_voting)
    """
    import shap

    # Unwrap sklearn Pipeline
    clf = model
    if hasattr(model, "named_steps"):
        clf = list(model.named_steps.values())[-1]

    model_type = type(clf).__name__.lower()

    tree_types = ["lgbm", "xgb", "catboost", "forest", "tree", "gradientboost"]
    if any(t in model_type for t in tree_types):
        return shap.TreeExplainer(clf), clf, "tree"

    linear_types = ["logistic", "linear", "ridge", "lasso"]
    if any(t in model_type for t in linear_types):
        masker = shap.maskers.Independent(
            pd.DataFrame(np.zeros((1, len(FEATURE_COLS))), columns=FEATURE_COLS)
        )
        return shap.LinearExplainer(clf, masker), clf, "linear"

    # Unknown model type — try TreeExplainer as fallback
    try:
        return shap.TreeExplainer(clf), clf, "tree"
    except Exception:
        logger.warning(f"Could not create SHAP explainer for {model_type}")
        return None, clf, "unknown"


def _shap_for_voting_classifier(voting_model, X_sample):
    individual_shaps = []

    for name, estimator in voting_model.named_estimators_.items():  # ← CORRECT
        explainer, clf, etype = _get_shap_explainer(estimator)
        if explainer is None:
            logger.warning(f"Skipping {name} in VotingClassifier SHAP (no explainer)")
            continue
        try:
            sv = explainer.shap_values(X_sample)
            # Binary TreeExplainer returns [neg_class_array, pos_class_array]
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.array(sv)
            # Newer LightGBM TreeExplainer returns (n_samples, n_features, 1)
            # — squeeze out any trailing size-1 dimensions to get (n_samples, n_features)
            while sv.ndim > 2 and sv.shape[-1] == 1:
                sv = sv[..., 0]
            # Promote (n_features,) → (1, n_features) for single-row inputs
            sv = np.atleast_2d(sv)
            individual_shaps.append(sv)
            logger.debug(f"  SHAP computed for constituent: {name}")
        except Exception as e:
            logger.warning(f"SHAP failed for constituent {name}: {e}")

    if not individual_shaps:
        logger.error("No constituent SHAP values computed. Returning zeros.")
        return np.zeros((len(X_sample), len(FEATURE_COLS)))

    # Equal-weight average across all estimators
    stacked = np.stack(
        individual_shaps, axis=0
    )  # (n_estimators, n_samples, n_features)
    return np.mean(
        stacked, axis=0
    )  # (n_samples, n_features)                 # (n_samples, n_features)


# ── Main WHY engine ──────────────────────────────────────────────────────────


class WhyEngine:
    """
    SHAP-based natural language explainability engine.

    Computes SHAP values for flagged cells and translates them into
    ranked human-readable reasons for operational use.

    Usage
    -----
        why = WhyEngine(model)
        explanations = why.explain(X_flagged, feature_values_df)
        # Returns list of dicts, one per cell
    """

    def __init__(
        self,
        model,
        alert_threshold: float = 0.40,
        top_n_reasons: int = 3,
        uncertainty_threshold: float = 0.15,
    ):
        """
        Args:
            model:                The production model (VotingClassifier or single model).
            alert_threshold:      Minimum P(fire) to compute SHAP for.
            top_n_reasons:        Number of top contributing reasons to expose.
            uncertainty_threshold: Inter-model std above which confidence = "LOW".
        """
        self.model = model
        self.alert_threshold = alert_threshold
        self.top_n_reasons = top_n_reasons
        self.uncertainty_threshold = uncertainty_threshold

        logger.info(
            f"WhyEngine initialised | "
            f"alert_threshold={alert_threshold} | "
            f"top_n={top_n_reasons}"
        )

    def _compute_shap(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for X.

        Routes to VotingClassifier path or single-model path based on
        the model type.
        """
        import shap

        # VotingClassifier
        if hasattr(self.model, "estimators_"):
            logger.info(
                f"Computing SHAP for VotingClassifier with "
                f"{len(self.model.estimators_)} estimators on {len(X):,} rows..."
            )
            return _shap_for_voting_classifier(self.model, X)

        # Single model
        explainer, clf, etype = _get_shap_explainer(self.model)
        if explainer is None:
            return np.zeros((len(X), X.shape[1]))

        logger.info(f"Computing SHAP ({etype}Explainer) on {len(X):,} rows...")
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]  # binary class → positive class
        return sv

    def explain(
        self,
        scored_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> list[dict]:
        """
        Generate natural language explanations for all flagged cells.

        Args:
            scored_df:   Output from WildfireInferenceEngine._score().
                         Must have: fire_prob, model_std, cell_id (or index).
            feature_df:  The feature matrix used for scoring.
                         Must have columns matching FEATURE_COLS.
                         Shares the same index as scored_df.

        Returns:
            List of explanation dicts, one per flagged cell:
            {
              "cell_id":           str,
              "fire_prob":         float,
              "risk_tier":         str,
              "model_confidence":  "HIGH"|"MEDIUM"|"LOW",
              "top_reasons":       list[str],     # risk-increasing
              "mitigating_factors":list[str],     # risk-decreasing
              "raw_shap":          dict,           # for dashboards
              "spread_bearing":    float or None,
              "spread_direction":  str or None,
            }
        """
        # Select flagged cells only
        flagged_mask = scored_df["fire_prob"] >= self.alert_threshold
        n_flagged = flagged_mask.sum()

        if n_flagged == 0:
            logger.info("No cells above alert threshold. No explanations generated.")
            return []

        logger.info(
            f"WHY Engine: generating explanations for {n_flagged:,} flagged cells "
            f"(P(fire) ≥ {self.alert_threshold})"
        )

        flagged_scored = scored_df[flagged_mask]
        flagged_features = feature_df.loc[flagged_mask, FEATURE_COLS]

        # Compute SHAP for flagged cells only
        shap_values = self._compute_shap(flagged_features)

        explanations = []
        for i, (idx, row) in enumerate(flagged_scored.iterrows()):
            shap_row = shap_values[i]
            feat_vals = flagged_features.loc[idx]

            # Sort features by |SHAP| descending
            sorted_idx = np.argsort(np.abs(shap_row))[::-1]

            # Top N positive contributors (reasons fire is flagged)
            top_reasons = []
            for fi in sorted_idx:
                if len(top_reasons) >= self.top_n_reasons:
                    break
                if shap_row[fi] > 0.001:  # threshold to ignore negligible effects
                    feature = FEATURE_COLS[fi]
                    value = float(feat_vals.iloc[fi])
                    sentence = _render_template(feature, value, shap_positive=True)
                    top_reasons.append(sentence)

            # Mitigating factors (negative SHAP — things reducing risk)
            mitigating = []
            neg_sorted = [fi for fi in sorted_idx if shap_row[fi] < -0.001]
            for fi in neg_sorted[:2]:  # top 2 mitigating factors
                feature = FEATURE_COLS[fi]
                value = float(feat_vals.iloc[fi])
                sentence = _render_template(feature, value, shap_positive=False)
                mitigating.append(sentence)

            # Model confidence level
            model_std = float(row.get("model_std", 0.0))
            if model_std > self.uncertainty_threshold:
                confidence = "LOW"
            elif model_std > self.uncertainty_threshold * 0.5:
                confidence = "MEDIUM"
            else:
                confidence = "HIGH"

            # Spread direction (if computed)
            spread_bearing = row.get("spread_bearing_deg", np.nan)
            if pd.isna(spread_bearing):
                spread_dir = None
                spread_bearing = None
            else:
                from src.spread.direction import bearing_to_compass

                spread_dir = bearing_to_compass(spread_bearing)

            # Raw SHAP dict (for frontend dashboards)
            raw_shap = {
                FEATURE_COLS[fi]: round(float(shap_row[fi]), 5)
                for fi in range(len(FEATURE_COLS))
            }

            explanations.append(
                {
                    "cell_id": row.get("cell_id", str(idx)),
                    "latitude": float(row.get("latitude", np.nan)),
                    "longitude": float(row.get("longitude", np.nan)),
                    "fire_prob": float(row["fire_prob"]),
                    "risk_tier": row["risk_tier"],
                    "model_confidence": confidence,
                    "model_std": round(model_std, 4),
                    "top_reasons": top_reasons,
                    "mitigating_factors": mitigating,
                    "raw_shap": raw_shap,
                    "spread_bearing_deg": spread_bearing,
                    "spread_direction": spread_dir,
                    "spread_intensity": row.get("spread_intensity", "none"),
                }
            )

        logger.info(
            f"WHY Engine complete: {len(explanations):,} explanations generated | "
            f"Confidence breakdown: "
            f"HIGH={sum(1 for e in explanations if e['model_confidence']=='HIGH')}, "
            f"MEDIUM={sum(1 for e in explanations if e['model_confidence']=='MEDIUM')}, "
            f"LOW={sum(1 for e in explanations if e['model_confidence']=='LOW')}"
        )

        return explanations

    def explain_single(
        self,
        cell_features: pd.Series,
        fire_prob: float,
        risk_tier: str,
        model_std: float = 0.0,
    ) -> dict:
        """
        Generate explanation for a single cell. Useful for on-demand API queries.

        Args:
            cell_features: Feature values for one cell (pd.Series, index=FEATURE_COLS).
            fire_prob:     P(fire) from the model.
            risk_tier:     "low"|"moderate"|"high"|"extreme"
            model_std:     Inter-model disagreement.

        Returns:
            Explanation dict (same structure as explain()).
        """
        X_single = pd.DataFrame([cell_features], columns=FEATURE_COLS)
        shap_vals = self._compute_shap(X_single)[0]

        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

        top_reasons = []
        for fi in sorted_idx:
            if len(top_reasons) >= self.top_n_reasons:
                break
            if shap_vals[fi] > 0.001:
                sentence = _render_template(
                    FEATURE_COLS[fi], float(cell_features.iloc[fi]), True
                )
                top_reasons.append(sentence)

        mitigating = []
        for fi in [fi for fi in sorted_idx if shap_vals[fi] < -0.001][:2]:
            sentence = _render_template(
                FEATURE_COLS[fi], float(cell_features.iloc[fi]), False
            )
            mitigating.append(sentence)

        confidence = (
            "LOW"
            if model_std > self.uncertainty_threshold
            else "MEDIUM" if model_std > self.uncertainty_threshold * 0.5 else "HIGH"
        )

        return {
            "fire_prob": round(fire_prob, 4),
            "risk_tier": risk_tier,
            "model_confidence": confidence,
            "top_reasons": top_reasons,
            "mitigating_factors": mitigating,
            "raw_shap": {
                FEATURE_COLS[fi]: round(float(shap_vals[fi]), 5)
                for fi in range(len(FEATURE_COLS))
            },
        }
