"""
scripts/generate_stacking_paper_figures.py

Generates all publication-quality paper figures for the Stacking Ensemble
deep-dive section (Section 2.3), replacing the XGBoost-specific figures.

Figures generated (saved to paper_figures/):
  fig_stacking_confusion_matrix.png          - Confusion matrix (paper style)
  fig_stacking_learning_curve.png            - Learning curve (pre-generated, copied)
  fig_shap_meta_importance_stacking.png      - Meta-learner SHAP bar (base model trust)
  fig_shap_meta_summary_stacking.png         - Meta-learner SHAP beeswarm
  fig_shap_importance_stacking.png           - PermutationExplainer SHAP bar (original features)
  fig_shap_summary_stacking.png              - PermutationExplainer SHAP beeswarm
  fig_shap_dependence_elevation_stacking.png - Elevation dependence plot
  fig_calibration_curve_stacking.png         - Reliability / calibration diagram

Two SHAP approaches are employed:
  1. Meta-Learner SHAP (LinearExplainer on the LR meta-learner):
       Features = P(fire | BaseModel_k) for each of the 5 base models.
       Answers: "Which base model does the stacking meta-learner trust most?"
  2. PermutationExplainer on the full Stacking Ensemble:
       Features = original 20 physics-informed input features.
       Answers: "Which input variables (elevation, EVI, VPD...) drive fire risk?"
"""

import sys
import os
import shutil
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
from sklearn.metrics import confusion_matrix
import joblib
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_FEAT   = ROOT / "data" / "processed" / "training_features.csv"
MODEL_PATH  = ROOT / "models" / "stacking_ensemble.pkl"
OUT_DIR     = ROOT / "paper_figures"
EXISTING_LC = (
    ROOT / "outputs" / "model_evaluation" / "stacking_ensemble" / "learning_curve.png"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
DPI         = 180
RANDOM_SEED = 42

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        DPI,
})

FIRE_COL   = "#E84040"
NOFIRE_COL = "#4C9BE8"

# ── Feature configuration ──────────────────────────────────────────────────────
FEATURE_COLS = [
    "month_sin", "month_cos",
    "temp", "humidity", "wind", "vpd",
    "evi", "vpd_wind", "elevation_m",
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]
TARGET_COL = "fire_detected"

FEAT_LABELS = {
    "month_sin":             "Month (sin)",
    "month_cos":             "Month (cos)",
    "temp":                  "Temperature",
    "humidity":              "Humidity",
    "wind":                  "Wind Speed",
    "vpd":                   "VPD",
    "evi":                   "EVI",
    "vpd_wind":              "VPD x Wind (FSP)",
    "elevation_m":           "Elevation",
    "lc_tree_cover":         "LC: Tree Cover",
    "lc_shrubland":          "LC: Shrubland",
    "lc_grassland":          "LC: Grassland",
    "lc_cropland":           "LC: Cropland",
    "lc_built_up":           "LC: Built-up",
    "lc_sparse_veg":         "LC: Sparse Veg",
    "lc_snow_ice":           "LC: Snow/Ice",
    "lc_water":              "LC: Water",
    "lc_herbaceous_wetland": "LC: Herbaceous Wetland",
    "lc_mangroves":          "LC: Mangroves",
    "lc_moss_lichen":        "LC: Moss/Lichen",
}
FEAT_DISPLAY = [FEAT_LABELS.get(c, c) for c in FEATURE_COLS]
ELEV_IDX     = FEATURE_COLS.index("elevation_m")


# =============================================================================
print("=" * 60)
print("Generating Stacking Ensemble paper figures")
print("=" * 60)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("\n[1/8] Loading training_features.csv (selected columns) ...")
df = pd.read_csv(
    DATA_FEAT,
    usecols=["acq_date"] + FEATURE_COLS + [TARGET_COL],
)
df["acq_date"] = pd.to_datetime(df["acq_date"])

train_mask = df["acq_date"].dt.year <= 2023
test_mask  = df["acq_date"].dt.year >= 2024

X_train = df.loc[train_mask, FEATURE_COLS].reset_index(drop=True)
y_train = df.loc[train_mask, TARGET_COL].reset_index(drop=True)
X_test  = df.loc[test_mask,  FEATURE_COLS].reset_index(drop=True)
y_test  = df.loc[test_mask,  TARGET_COL].reset_index(drop=True)

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Test fire rate: {y_test.mean()*100:.1f}%")

# ── Step 2: Load model ────────────────────────────────────────────────────────
print(f"\n[2/8] Loading {MODEL_PATH.name} "
      f"({MODEL_PATH.stat().st_size / 1e9:.2f} GB) ...")
model = joblib.load(MODEL_PATH)
print("   Model loaded.")

# ── Step 3: Full test-set inference ──────────────────────────────────────────
print("\n[3/8] Inference on full test set ...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"   Predicted fires: {int(y_pred.sum()):,} / {len(y_pred):,}")


# =============================================================================
# FIGURE 1: Confusion Matrix
# =============================================================================
print("\n[4/8] Confusion Matrix ...")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

cell_labels = [
    [f"TN\n{tn:,}", f"FP\n{fp:,}"],
    [f"FN\n{fn:,}", f"TP\n{tp:,}"],
]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow([[tn, fp], [fn, tp]], cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for i in range(2):
    for j in range(2):
        font_color = "white" if i == j else "#2c3e50"
        ax.text(j, i, cell_labels[i][j],
                ha="center", va="center", fontsize=12,
                fontweight="bold", color=font_color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted: No Fire", "Predicted: Fire"], fontsize=10)
ax.set_yticklabels(["Actual: No Fire", "Actual: Fire"], fontsize=10)
ax.set_title(
    "Confusion Matrix -- Stacking Ensemble (2024-2025 Test Set)",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
plt.tight_layout()

out_cm = OUT_DIR / "fig_stacking_confusion_matrix.png"
fig.savefig(out_cm, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out_cm.name}")
print(f"   TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")


# =============================================================================
# FIGURE 2: Learning Curve (copy pre-generated; avoids 200 ensemble refits)
# =============================================================================
print("\n[5/8] Learning Curve (copying pre-generated artifact) ...")
out_lc = OUT_DIR / "fig_stacking_learning_curve.png"
if EXISTING_LC.exists():
    shutil.copy2(EXISTING_LC, out_lc)
    print(f"   Copied -> {out_lc.name}")
else:
    print(f"   WARNING: {EXISTING_LC} not found -- skipping.")


# =============================================================================
# FIGURES 3 & 4: Meta-Learner SHAP (LinearExplainer on Logistic Regression)
#   Features in this space = P(fire | BaseModel_k) for each of 5 base models.
#   Fix applied: rename "Feature 0..4" to actual base model names at runtime.
# =============================================================================
print("\n[6/8] Meta-Learner SHAP (LinearExplainer) ...")

rng          = np.random.RandomState(RANDOM_SEED)
meta_idx     = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
X_meta_input = X_test.iloc[meta_idx].reset_index(drop=True)

base_names         = list(model.named_estimators_.keys())
meta_feature_names = [f"P(fire|{n})" for n in base_names]
print(f"   Base models detected: {base_names}")

X_meta_arr = model.transform(X_meta_input)
X_meta_df  = pd.DataFrame(X_meta_arr, columns=meta_feature_names)

final_est      = model.final_estimator_
masker_meta    = shap.maskers.Independent(X_meta_df)
explainer_meta = shap.LinearExplainer(final_est, masker_meta)
shap_meta      = explainer_meta.shap_values(X_meta_df)

# -- Meta SHAP bar chart --
fig = plt.figure(figsize=(7, 4))
shap.summary_plot(shap_meta, X_meta_df, plot_type="bar", show=False)
plt.title(
    "SHAP Feature Importance -- Stacking Ensemble Meta-Learner\n"
    "(Trust weight assigned to each base model's probability output)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
out_meta_bar = OUT_DIR / "fig_shap_meta_importance_stacking.png"
fig.savefig(out_meta_bar, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out_meta_bar.name}")

# -- Meta SHAP beeswarm --
fig = plt.figure(figsize=(9, 5))
shap.summary_plot(shap_meta, X_meta_df, show=False)
plt.title(
    "SHAP Summary -- Stacking Ensemble Meta-Learner\n"
    "(How strongly each base model's vote pushes the final prediction)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
out_meta_bee = OUT_DIR / "fig_shap_meta_summary_stacking.png"
fig.savefig(out_meta_bee, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out_meta_bee.name}")


# =============================================================================
# FIGURES 5, 6 & 7: PermutationExplainer SHAP (original input features)
#   Runs the full stacking model as a black box; no TreeExplainer needed.
#   Attributes importance to the original 20 physics-informed features.
# =============================================================================
print("\n[7/8] PermutationExplainer SHAP (original features) ...")
print("   Runs ~41 model passes over explanation set -- please wait ...")

bg_idx   = rng.choice(len(X_train), size=min(300, len(X_train)), replace=False)
X_bg     = X_train.iloc[bg_idx].values

expl_idx = rng.choice(len(X_test), size=min(800, len(X_test)), replace=False)
X_expl   = X_test.iloc[expl_idx].values


def _predict_fn(X_arr: np.ndarray) -> np.ndarray:
    """Black-box wrapper accepted by shap.PermutationExplainer."""
    df_in = pd.DataFrame(X_arr, columns=FEATURE_COLS)
    return model.predict_proba(df_in)[:, 1]


explainer_perm = shap.PermutationExplainer(_predict_fn, X_bg)
shap_perm_expl = explainer_perm(X_expl)
shap_perm_vals = shap_perm_expl.values     # (n_expl, n_features)

X_expl_display = pd.DataFrame(X_expl, columns=FEAT_DISPLAY)

# -- Perm SHAP bar chart --
fig = plt.figure(figsize=(8, 6))
shap.summary_plot(shap_perm_vals, X_expl_display, plot_type="bar", show=False)
plt.title(
    "SHAP Feature Importance -- Stacking Ensemble\n"
    "(PermutationExplainer on original 20 input features)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
out_perm_bar = OUT_DIR / "fig_shap_importance_stacking.png"
fig.savefig(out_perm_bar, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out_perm_bar.name}")

# -- Perm SHAP beeswarm --
fig = plt.figure(figsize=(10, 7))
shap.summary_plot(shap_perm_vals, X_expl_display, show=False)
plt.title(
    "SHAP Summary -- Stacking Ensemble\n"
    "(PermutationExplainer on original 20 input features)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
out_perm_bee = OUT_DIR / "fig_shap_summary_stacking.png"
fig.savefig(out_perm_bee, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out_perm_bee.name}")

# -- Elevation dependence plot --
shap_elev = shap_perm_vals[:, ELEV_IDX]
elev_vals = X_expl[:, ELEV_IDX]

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    elev_vals, shap_elev,
    c=elev_vals, cmap="RdYlBu_r",
    alpha=0.45, s=14, edgecolors="none", zorder=3
)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("Elevation (m)", fontsize=10)
ax.axhline(0, color="#aaa", linestyle="--", linewidth=1, zorder=2)
ax.set_xlabel("Elevation (m)", fontsize=11)
ax.set_ylabel("SHAP value for Elevation\n(impact on predicted fire probability)", fontsize=11)
ax.set_title(
    "SHAP Dependence Plot -- Elevation\n(Stacking Ensemble, PermutationExplainer)",
    fontsize=13, fontweight="bold"
)
ax.grid(True, alpha=0.3, zorder=0)
plt.tight_layout()
out_dep = OUT_DIR / "fig_shap_dependence_elevation_stacking.png"
fig.savefig(out_dep, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out_dep.name}")


# =============================================================================
# FIGURE 8: Calibration Curve (Stacking Ensemble)
# =============================================================================
print("\n[8/8] Calibration Curve ...")

fire_idx_t   = np.where(y_test.values == 1)[0]
nofire_idx_t = np.where(y_test.values == 0)[0]
n_cal        = 25_000
cal_rng      = np.random.RandomState(99)
cal_idx = np.concatenate([
    cal_rng.choice(fire_idx_t,   size=min(n_cal, len(fire_idx_t)),   replace=False),
    cal_rng.choice(nofire_idx_t, size=min(n_cal, len(nofire_idx_t)), replace=False),
])
cal_rng.shuffle(cal_idx)

frac_pos, mean_pred = calibration_curve(
    y_test.values[cal_idx], y_prob[cal_idx],
    n_bins=15, strategy="quantile"
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfectly calibrated", alpha=0.7)
ax.plot(
    mean_pred, frac_pos,
    "o-", color=FIRE_COL, linewidth=2.5,
    markersize=7, markeredgewidth=1.5, markeredgecolor="white",
    label="Stacking Ensemble", zorder=4
)
ax.fill_between(mean_pred, mean_pred, frac_pos,
                alpha=0.12, color=FIRE_COL, label="Calibration gap")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.08)
ax.set_xlabel("Mean Predicted Probability", fontsize=11)
ax.set_ylabel("Fraction of Positives (Observed Fire Rate)", fontsize=11)
ax.set_title(
    "Reliability Diagram -- Stacking Ensemble Fire Probability Calibration",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9, loc="upper left")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
out_cal = OUT_DIR / "fig_calibration_curve_stacking.png"
fig.savefig(out_cal, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out_cal.name}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All Stacking Ensemble figures generated successfully.")
print(f"Output: {OUT_DIR}")
generated = (
    list(OUT_DIR.glob("fig_stacking_*.png"))
    + list(OUT_DIR.glob("fig_shap_*stacking*.png"))
    + [OUT_DIR / "fig_calibration_curve_stacking.png"]
)
for p in sorted(set(generated)):
    if p.exists():
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
print("=" * 60)
