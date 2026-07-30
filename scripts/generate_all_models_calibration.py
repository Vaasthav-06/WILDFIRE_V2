"""
scripts/generate_all_models_calibration.py

Generalization of generate_calibration_before_after.py to ALL 7 models.

Addresses reviewer point 4: calibration curves were only shown for the
Stacking Ensemble. This script produces reliability diagrams for all models
using the same methodology (chronological split: calibration fitted on first
half of test set, evaluated on second half).

Outputs:
  paper_figures/fig_calibration_all_models.png

Runtime: ~20-25 min (dominated by loading large .pkl files and predict_proba).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_FEAT  = ROOT / "data" / "processed" / "training_features.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR    = ROOT / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    ("Logistic Regression",  "logistic_regression.pkl", "#4C9BE8"),
    ("Random Forest",        "random_forest.pkl",       "#28A745"),
    ("LightGBM",             "lightgbm.pkl",            "#FD7E14"),
    ("CatBoost",             "catboost.pkl",            "#9B59B6"),
    ("XGBoost",              "xgboost.pkl",             "#E84040"),
    ("Voting Ensemble",      "voting_ensemble.pkl",     "#2E86AB"),
    ("Stacking Ensemble",    "stacking_ensemble.pkl",   "#795548"),
]

N_BINS = 15
DPI    = 180


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error — uniform bins, frequency-weighted."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("All-Model Calibration Curves (Reliability Diagrams)")
print("=" * 65)

# Step 1: Load and split test set
print("\n[1/3] Loading test set ...")
df = pd.read_csv(DATA_FEAT, usecols=["acq_date"] + FEATURE_COLS + [TARGET_COL])
df["acq_date"] = pd.to_datetime(df["acq_date"])
df_test = df[df["acq_date"].dt.year >= 2024].sort_values("acq_date").reset_index(drop=True)
X_test  = df_test[FEATURE_COLS].values
y_test  = df_test[TARGET_COL].values
split_idx = len(df_test) // 2
y_true_eval = y_test[split_idx:]
print(f"   Test rows: {len(df_test):,}  (eval on second half: {len(y_true_eval):,} rows)")

# Step 2: Compute calibration data for each model
print("\n[2/3] Computing calibration curves ...")
calib_data = []

for model_name, pkl_name, color in MODELS:
    pkl_path = MODELS_DIR / pkl_name
    if not pkl_path.exists():
        print(f"   SKIP: {pkl_name} not found")
        calib_data.append(None)
        continue

    mb = pkl_path.stat().st_size / 1e6
    print(f"   Loading {model_name} ({mb:.0f} MB) ...")
    model = joblib.load(pkl_path)

    try:
        y_prob_full = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"   ERROR: {model_name} predict_proba failed: {e}")
        calib_data.append(None)
        continue

    y_prob_fit  = y_prob_full[:split_idx]
    y_true_fit  = y_test[:split_idx]
    y_prob_eval = y_prob_full[split_idx:]

    # Fit calibrator on first half
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob_fit, y_true_fit)
    y_prob_cal = ir.predict(y_prob_eval)

    ece_before = compute_ece(y_true_eval, y_prob_eval)
    ece_after  = compute_ece(y_true_eval, y_prob_cal)
    reduction  = (1 - ece_after / ece_before) * 100 if ece_before > 0 else 0.0

    frac_before, mean_before = calibration_curve(y_true_eval, y_prob_eval, n_bins=N_BINS, strategy="quantile")
    frac_after, mean_after   = calibration_curve(y_true_eval, y_prob_cal,  n_bins=N_BINS, strategy="quantile")

    calib_data.append({
        "name": model_name, "color": color,
        "frac_before": frac_before, "mean_before": mean_before,
        "frac_after":  frac_after,  "mean_after":  mean_after,
        "ece_before":  ece_before,  "ece_after":   ece_after,
        "reduction":   reduction,
    })
    print(f"   {model_name}: ECE {ece_before:.4f} -> {ece_after:.4f}  ({reduction:.1f}% reduction)")

    del model
    import gc; gc.collect()

# Step 3: Generate figure — 4 col × 2 row grid (7 models + 1 legend panel)
print("\n[3/3] Generating figure ...")
valid_data = [d for d in calib_data if d is not None]
n_models   = len(valid_data)
n_cols     = 4
n_rows     = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), sharey=True, sharex=True)
axes_flat = axes.flat if n_rows > 1 else [axes] if n_cols == 1 else list(axes)

for ax_idx, d in enumerate(valid_data):
    ax = axes_flat[ax_idx]
    # Before calibration (lighter)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.plot(d["mean_before"], d["frac_before"], "o--",
            color=d["color"], alpha=0.5, lw=1.5, ms=4, label=f"Raw (ECE={d['ece_before']:.4f})")
    # After calibration
    ax.plot(d["mean_after"], d["frac_after"], "s-",
            color=d["color"], lw=2, ms=5, markeredgecolor="white",
            label=f"Calibrated (ECE={d['ece_after']:.4f})")
    ax.fill_between(d["mean_before"], d["mean_before"], d["frac_before"],
                    alpha=0.07, color=d["color"])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(d["name"], fontsize=10, fontweight="bold")
    ax.text(0.04, 0.88, f"ECE: {d['ece_before']:.4f}→{d['ece_after']:.4f}\n({d['reduction']:.0f}% ↓)",
            transform=ax.transAxes, fontsize=8, color=d["color"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85,
                      edgecolor=d["color"], lw=0.8))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)

    if ax_idx % n_cols == 0:
        ax.set_ylabel("Fraction of Positives", fontsize=9)
    if ax_idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Mean Predicted Probability", fontsize=9)

# Hide unused axes
for ax in list(axes_flat)[n_models:]:
    ax.set_visible(False)

fig.suptitle(
    "Reliability Diagrams: All Models — Before and After Isotonic Regression Calibration\n"
    "Calibrator fitted on chronological first half of 2024–2025 test set; evaluated on second half.",
    fontsize=10.5, fontweight="bold", y=1.01,
)
plt.tight_layout()
out = FIG_DIR / "fig_calibration_all_models.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out.name}")

print("\n" + "=" * 65)
print("All-model calibration complete.")
print(f"  Output: {out}")
print("=" * 65)
