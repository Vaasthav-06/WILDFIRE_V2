"""
scripts/generate_calibration_before_after.py

Fits Isotonic Regression calibration on the first chronological half of the
2024-2025 test set, evaluates on the second half, and generates a side-by-side
before/after reliability diagram.

Saves:
  paper_figures/fig_calibration_before_after_stacking.png

Calibration split rationale:
  - Isotonic regression calibrator is fitted on a dedicated validation slice 
    held out from the training period (October-December 2023).
  - Evaluated on the full, untouched 2024-2025 test set.
    This ensures absolutely no data leakage into the test set.
"""

import sys
import os
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
MODEL_PATH = ROOT / "models" / "stacking_ensemble.pkl"
OUT_DIR    = ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 180
COL_BEFORE = "#E84040"
COL_AFTER  = "#2E86AB"
N_BINS     = 15

FEATURE_COLS = [
    "month_sin", "month_cos",
    "temp", "humidity", "wind", "vpd",
    "evi", "vpd_wind", "elevation_m",
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]
TARGET_COL = "fire_detected"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        DPI,
})


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error -- uniform bins, frequency-weighted."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return ece


# =============================================================================
print("=" * 60)
print("Calibration Before/After -- Stacking Ensemble")
print("=" * 60)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("\n[1/5] Loading data (train + test set) ...")
df = pd.read_csv(
    DATA_FEAT,
    usecols=["acq_date"] + FEATURE_COLS + [TARGET_COL],
)
df["acq_date"] = pd.to_datetime(df["acq_date"])

test_mask = df["acq_date"].dt.year >= 2024
df_test   = df[test_mask].sort_values("acq_date").reset_index(drop=True)
print(f"   Test set: {len(df_test):,} rows (sorted chronologically)")

calib_mask = (df["acq_date"].dt.year == 2023) & (df["acq_date"].dt.month >= 10)
df_calib_fit = df[calib_mask].sort_values("acq_date").reset_index(drop=True)
print(f"   Calib fit set (Oct-Dec 2023): {len(df_calib_fit):,} rows")

# ── Step 2: Load stacking ensemble ────────────────────────────────────────────
gb = MODEL_PATH.stat().st_size / 1e9
print(f"\n[2/5] Loading {MODEL_PATH.name} ({gb:.2f} GB) ...")
model = joblib.load(MODEL_PATH)
print("   Loaded.")

# ── Step 3: Predict on calibration set and fit Isotonic Regression ────────────
print("\n[3/5] Inference on calib fit set and fitting calibrator ...")
X_calib = df_calib_fit[FEATURE_COLS]
y_calib = df_calib_fit[TARGET_COL].values
y_prob_calib = model.predict_proba(X_calib)[:, 1]

ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(y_prob_calib, y_calib)

# ── Step 4: Evaluate on full test set ─────────────────────────────────────────
print("\n[4/5] Inference on full test set ...")
X_test = df_test[FEATURE_COLS]
y_test = df_test[TARGET_COL].values
y_prob_eval = model.predict_proba(X_test)[:, 1]
y_true_eval = y_test

y_prob_cal = ir.predict(y_prob_eval)

ece_before = compute_ece(y_true_eval, y_prob_eval, n_bins=N_BINS)
ece_after  = compute_ece(y_true_eval, y_prob_cal,  n_bins=N_BINS)
reduction  = (1 - ece_after / ece_before) * 100 if ece_before > 0 else 0.0

print(f"   ECE Before : {ece_before:.4f}")
print(f"   ECE After  : {ece_after:.4f}")
print(f"   Reduction  : {reduction:.1f}%")

# ── Step 5: Generate figure ───────────────────────────────────────────────────
print("\n[5/5] Generating before/after figure ...")

frac_before, mean_before = calibration_curve(
    y_true_eval, y_prob_eval, n_bins=N_BINS, strategy="quantile"
)
frac_after, mean_after = calibration_curve(
    y_true_eval, y_prob_cal, n_bins=N_BINS, strategy="quantile"
)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

panels = [
    (axes[0], frac_before, mean_before,
     "Before Calibration\n(Raw Stacking Ensemble)", COL_BEFORE, ece_before),
    (axes[1], frac_after, mean_after,
     "After Calibration\n(Isotonic Regression)", COL_AFTER, ece_after),
]

for ax, frac, mean_pred, title, color, ece in panels:
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7,
            label="Perfect calibration", zorder=2)
    ax.plot(
        mean_pred, frac, "o-", color=color, linewidth=2.5,
        markersize=7, markeredgewidth=1.5, markeredgecolor="white",
        label="Stacking Ensemble", zorder=4,
    )
    ax.fill_between(mean_pred, mean_pred, frac,
                    alpha=0.12, color=color, label="Calibration gap")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.text(
        0.05, 0.90, f"ECE = {ece:.4f}",
        transform=ax.transAxes, fontsize=10.5, color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor=color, linewidth=1.2),
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

axes[0].set_ylabel("Fraction of Positives (Observed Fire Rate)", fontsize=11)

ece_txt = (
    f"ECE reduced from {ece_before:.4f} to {ece_after:.4f}"
    f" ({reduction:.0f}% improvement)"
)
fig.suptitle(
    "Reliability Diagram -- Stacking Ensemble Probability Calibration\n"
    f"Isotonic Regression fitted on Oct--Dec 2023 slice; evaluated on full 2024--2025 test set  |  {ece_txt}",
    fontsize=10.5, fontweight="bold", y=1.02,
)

plt.tight_layout()
out = OUT_DIR / "fig_calibration_before_after_stacking.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out.name}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Done.")
print(f"  ECE Before : {ece_before:.4f}")
print(f"  ECE After  : {ece_after:.4f}")
print(f"  Reduction  : {reduction:.1f}%")
print(f"  Output     : {out}")
print("=" * 60)
