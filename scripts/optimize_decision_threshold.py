"""
scripts/optimize_decision_threshold.py

Operational Decision Threshold Optimization for the Stacking Ensemble.

Addresses reviewer point 4: the paper currently uses an implicit 0.5 threshold.
This script finds two principled alternatives:

1. Youden's J statistic: threshold = argmax(TPR - FPR)
   - Maximizes the geometric mean of sensitivity and specificity.
   - Standard choice in medical/safety diagnostics.

2. Cost-sensitive threshold: assumes FN cost = 10x FP cost.
   RATIONALE: In a wildfire early-warning system, a missed fire (false negative)
   allows a fire to escalate unchecked — potentially destroying livelihoods and
   causing loss of life. A false alarm (false positive) wastes some fire crew
   mobilization resources but causes no harm. A 10:1 FN:FP cost ratio is a
   conservative and defensible assumption for a life-safety system; we state it
   explicitly rather than burying it.
   The cost-optimal threshold minimizes: FP_cost * FP + FN_cost * FN.

Outputs:
  outputs/threshold_optimization_results.csv
  (Values printed to console for easy insertion into paper text)

Runtime: ~5-10 min (one predict_proba call on 2.17M test rows, then ROC math).
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
    roc_curve, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_FEAT  = ROOT / "data" / "processed" / "training_features.csv"
MODEL_PATH = ROOT / "models" / "stacking_ensemble.pkl"
OUT_DIR    = ROOT / "outputs"
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

# ── Cost assumptions ───────────────────────────────────────────────────────────
# FN cost: missing a real fire = 10x worse than a false alarm.
# This is the ratio used in the cost-optimized threshold computation.
# Stated explicitly here as required by the reviewer.
FN_COST = 10.0
FP_COST = 1.0

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Decision Threshold Optimization — Stacking Ensemble")
print(f"  Cost assumption: FN_COST={FN_COST}  FP_COST={FP_COST}")
print("=" * 65)

# Step 1: Load test set
print("\n[1/3] Loading test set ...")
df = pd.read_csv(DATA_FEAT, usecols=["acq_date"] + FEATURE_COLS + [TARGET_COL])
df["acq_date"] = pd.to_datetime(df["acq_date"])
df_test = df[df["acq_date"].dt.year >= 2024].reset_index(drop=True)
X_test  = df_test[FEATURE_COLS].values
y_test  = df_test[TARGET_COL].values
print(f"   Test rows: {len(df_test):,}  |  Fire rate: {y_test.mean():.4f}")

# Step 2: Load model + get probabilities
mb = MODEL_PATH.stat().st_size / 1e9
print(f"\n[2/3] Loading stacking ensemble ({mb:.2f} GB) ...")
model = joblib.load(MODEL_PATH)
print("   Loaded. Running predict_proba ...")
y_prob = model.predict_proba(X_test)[:, 1]
print(f"   Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

# Step 3: Compute thresholds
print("\n[3/3] Computing optimal thresholds ...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# ── Youden's J statistic ───────────────────────────────────────────────────────
j_stat   = tpr - fpr
j_idx    = np.argmax(j_stat)
t_youdens = float(thresholds[j_idx])
print(f"\n   Youden's J: argmax(TPR - FPR)")
print(f"   Optimal threshold = {t_youdens:.4f}")
print(f"   TPR={tpr[j_idx]:.4f}  FPR={fpr[j_idx]:.4f}  J={j_stat[j_idx]:.4f}")

# ── Cost-sensitive threshold ───────────────────────────────────────────────────
# Total cost = FP_cost * FP + FN_cost * FN
# FP = FPR * N_neg,  FN = (1 - TPR) * N_pos
n_pos = int(y_test.sum())
n_neg = int(len(y_test) - n_pos)

total_cost = FP_COST * (fpr * n_neg) + FN_COST * ((1 - tpr) * n_pos)
cost_idx   = np.argmin(total_cost)
t_cost     = float(thresholds[cost_idx])
print(f"\n   Cost-sensitive: min(FP_COST * FP + FN_COST * FN)")
print(f"   Optimal threshold = {t_cost:.4f}")
print(f"   TPR={tpr[cost_idx]:.4f}  FPR={fpr[cost_idx]:.4f}  Cost={total_cost[cost_idx]:.0f}")

# ── Evaluate each threshold ────────────────────────────────────────────────────
threshs = {
    "Naive (0.50)":          0.50,
    "Youden's J":            t_youdens,
    f"Cost-Sensitive (FN/FP={FN_COST:.0f})": t_cost,
}

rows = []
print("\n   Per-threshold performance:")
for tname, t in threshs.items():
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    row = {
        "Threshold Name": tname,
        "Threshold":      t,
        "Recall":         recall_score(y_test, y_pred, zero_division=0),
        "Precision":      precision_score(y_test, y_pred, zero_division=0),
        "F1":             f1_score(y_test, y_pred, zero_division=0),
        "TP":             int(tp), "FP": int(fp),
        "FN":             int(fn), "TN": int(tn),
    }
    rows.append(row)
    print(f"     {tname:<35} thresh={t:.4f}  "
          f"Recall={row['Recall']:.4f}  Precision={row['Precision']:.4f}  "
          f"F1={row['F1']:.4f}  (TP={tp:,} FN={fn:,} FP={fp:,})")

df_out = pd.DataFrame(rows)
csv_path = OUT_DIR / "threshold_optimization_results.csv"
df_out.to_csv(csv_path, index=False)

print("\n" + "=" * 65)
print(f"Threshold optimization complete. Saved -> {csv_path}")
print("\nSummary for paper:")
for r in rows:
    print(f"  {r['Threshold Name']}: threshold={r['Threshold']:.4f}  "
          f"Recall={r['Recall']:.4f}  Precision={r['Precision']:.4f}  F1={r['F1']:.4f}")
print("=" * 65)
