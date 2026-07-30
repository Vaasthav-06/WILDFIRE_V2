"""
scripts/evaluate_all_models_extended.py

Extended evaluation of ALL saved models with PR-AUC and Brier Score.

Addresses reviewer point 4: the paper currently lacks PR-AUC and Brier Score
across all models. This script computes them using saved .pkl files —
NO retraining required.

Models evaluated:
  - Logistic Regression (models/logistic_regression.pkl)
  - Random Forest       (models/random_forest.pkl)
  - LightGBM            (models/lightgbm.pkl)
  - CatBoost            (models/catboost.pkl)
  - XGBoost             (models/xgboost.pkl)
  - Voting Ensemble     (models/voting_ensemble.pkl)
  - Stacking Ensemble   (models/stacking_ensemble.pkl)

Outputs:
  outputs/all_models_extended_metrics.csv

Runtime: ~15-20 min (dominated by loading 978 MB models and running predict_proba).
"""

import os
import sys
import warnings

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    recall_score, precision_score, f1_score, accuracy_score,
)
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_FEAT = ROOT / "data" / "processed" / "training_features.csv"
MODELS_DIR = ROOT / "models"
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "month_sin", "month_cos",
    "temp", "humidity", "wind", "vpd",
    "evi", "vpd_wind", "elevation_m",
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]
TARGET_COL = "fire_detected"

MODELS = [
    ("Logistic Regression",  "logistic_regression.pkl", False),
    ("Random Forest",        "random_forest.pkl",       False),
    ("LightGBM",             "lightgbm.pkl",            False),
    ("CatBoost",             "catboost.pkl",            False),
    ("XGBoost",              "xgboost.pkl",             False),
    ("Voting Ensemble",      "voting_ensemble.pkl",     True),
    ("Stacking Ensemble",    "stacking_ensemble.pkl",   True),
]

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Extended All-Model Evaluation — PR-AUC & Brier Score")
print("=" * 65)

# Step 1: Load test set
print("\n[1/2] Loading test set ...")
df = pd.read_csv(
    DATA_FEAT,
    usecols=["acq_date"] + FEATURE_COLS + [TARGET_COL],
)
df["acq_date"] = pd.to_datetime(df["acq_date"])
test_mask = df["acq_date"].dt.year >= 2024
df_test = df[test_mask].reset_index(drop=True)
X_test  = df_test[FEATURE_COLS].values
y_test  = df_test[TARGET_COL].values
print(f"   Test rows: {len(df_test):,}  |  Fire rate: {y_test.mean():.4f}")

# Step 2: Evaluate each model
print("\n[2/2] Evaluating models ...")
results = []

for model_name, pkl_name, is_ensemble in MODELS:
    pkl_path = MODELS_DIR / pkl_name
    if not pkl_path.exists():
        print(f"   SKIP: {pkl_name} not found at {pkl_path}")
        continue

    mb = pkl_path.stat().st_size / 1e6
    print(f"\n   Loading {model_name} ({mb:.1f} MB) ...")
    model = joblib.load(pkl_path)
    print(f"   Loaded. Running predict_proba ...")

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"   ERROR: predict_proba failed for {model_name}: {e}")
        continue

    y_pred = (y_prob >= 0.5).astype(int)

    row = {
        "Model":      model_name,
        "Ensemble":   is_ensemble,
        "ROC-AUC":    roc_auc_score(y_test, y_prob),
        "PR-AUC":     average_precision_score(y_test, y_prob),
        "Brier":      brier_score_loss(y_test, y_prob),
        "Recall":     recall_score(y_test, y_pred,    zero_division=0),
        "Precision":  precision_score(y_test, y_pred, zero_division=0),
        "F1":         f1_score(y_test, y_pred,        zero_division=0),
        "Accuracy":   accuracy_score(y_test, y_pred),
    }
    results.append(row)
    print(f"   {model_name}: ROC-AUC={row['ROC-AUC']:.4f}  "
          f"PR-AUC={row['PR-AUC']:.4f}  Brier={row['Brier']:.4f}  "
          f"Recall={row['Recall']:.4f}  Precision={row['Precision']:.4f}")

    # Explicitly free large models
    del model
    import gc; gc.collect()

# Save
df_results = pd.DataFrame(results)
csv_path = OUT_DIR / "all_models_extended_metrics.csv"
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 65)
print("RESULTS TABLE")
print("=" * 65)
print(df_results.to_string(index=False))
print(f"\nSaved -> {csv_path}")
print("=" * 65)
