"""
scripts/generate_vpd_evi_interaction.py  (v2)

Uses PermutationExplainer on 300 test samples to get VPD and EVI SHAP values
for the stacking ensemble, then renders a hexbin interaction figure.
Uses tricontour instead of griddata to avoid NaN triangulation errors.

Output: paper_figures/fig_shap_vpd_evi_interaction.png
"""

import os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
import joblib, shap

ROOT      = Path(__file__).resolve().parent.parent
DATA_FEAT = ROOT / "data" / "processed" / "training_features.csv"
MODEL_PKL = ROOT / "models" / "stacking_ensemble.pkl"
OUT_DIR   = ROOT / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "month_sin", "month_cos",
    "temp", "humidity", "wind", "vpd",
    "evi", "vpd_wind", "elevation_m",
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]

DPI  = 180
SEED = 42
N    = 300   # PermutationExplainer: ~300 * 1s = ~60 seconds

print("=" * 60)
print("VPD x EVI SHAP Interaction Figure (PermutationExplainer)")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n[1/3] Loading test set sample ...")
needed = ["acq_date"] + FEATURE_COLS + ["fire_detected"]
df = pd.read_csv(DATA_FEAT, usecols=needed)
df["acq_date"] = pd.to_datetime(df["acq_date"])
df_test = df[df["acq_date"].dt.year >= 2024].reset_index(drop=True)

# Stratified by fire label
fires    = df_test[df_test["fire_detected"] == 1].sample(n=N//2, random_state=SEED)
nonfires = df_test[df_test["fire_detected"] == 0].sample(n=N//2, random_state=SEED)
samp = pd.concat([fires, nonfires], ignore_index=True)
X    = samp[FEATURE_COLS].values
print(f"   {N} samples (stratified): fires={len(fires)}, non-fires={len(nonfires)}")

# ── SHAP via PermutationExplainer ─────────────────────────────────────────────
print(f"\n[2/3] Computing SHAP values ({N} samples, ~{N}s expected) ...")
model = joblib.load(MODEL_PKL)

# Background: 50 random samples
bg_idx = np.random.default_rng(SEED).choice(len(df_test), size=50, replace=False)
X_bg   = df_test.iloc[bg_idx][FEATURE_COLS].values

explainer = shap.PermutationExplainer(
    model.predict_proba, X_bg,
    max_evals=2 * len(FEATURE_COLS) + 1
)
shap_vals = explainer(X)
shap_fire = shap_vals.values[:, :, 1]   # SHAP for fire class
print("   Done.")

vpd_idx  = FEATURE_COLS.index("vpd")
evi_idx  = FEATURE_COLS.index("evi")

vpd_feat  = X[:, vpd_idx]
evi_feat  = X[:, evi_idx]
shap_comb = shap_fire[:, vpd_idx] + shap_fire[:, evi_idx]

# Filter NaNs and Infs
valid = np.isfinite(vpd_feat) & np.isfinite(evi_feat) & np.isfinite(shap_comb)
vpd_feat  = vpd_feat[valid]
evi_feat  = evi_feat[valid]
shap_comb = shap_comb[valid]
print(f"   {valid.sum()} valid samples after filter.")

# ── Figure ─────────────────────────────────────────────────────────────────────
print("\n[3/3] Generating figure ...")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(9, 7))

# Hex-bin
hb = ax.hexbin(evi_feat, vpd_feat, C=shap_comb,
               gridsize=25, cmap="RdYlGn_r",
               reduce_C_function=np.mean,
               vmin=-0.15, vmax=0.15, mincnt=1)
cb = fig.colorbar(hb, ax=ax, pad=0.02)
cb.set_label("Mean SHAP(VPD) + SHAP(EVI)\n"
             "<-- reduces fire risk  |  increases fire risk -->",
             fontsize=10)

# Contour at SHAP = 0 using matplotlib triangulation (NaN-safe)
try:
    triang = mtri.Triangulation(evi_feat, vpd_feat)
    # Mask degenerate triangles
    tri_pts = np.stack([evi_feat[triang.triangles].mean(axis=1),
                        vpd_feat[triang.triangles].mean(axis=1)], axis=1)
    ax.tricontour(triang, shap_comb, levels=[0],
                  colors="black", linewidths=2.5, linestyles="--")
    # label manually
    ax.text(0.52, 0.62, "SHAP = 0", transform=ax.transAxes,
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7))
except Exception as e:
    print(f"   Contour skipped: {e}")

# Quadrant annotations
ax.text(0.04, 0.95, "High EVI, Low VPD\n(Fire suppression)",
        transform=ax.transAxes, fontsize=9, color="#2E7D32", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2E7D32", alpha=0.75))
ax.text(0.96, 0.05, "Low EVI, High VPD\n(Extreme fire risk)",
        transform=ax.transAxes, fontsize=9, color="#C62828", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C62828", alpha=0.75))

ax.set_xlabel("EVI  (Live-Fuel Moisture Proxy:  0 = dry/sparse  |  1 = dense green)",
              fontsize=10)
ax.set_ylabel("VPD -- Vapor Pressure Deficit (kPa)", fontsize=10)
ax.set_title("VPD x EVI Joint SHAP Interaction -- Fire Risk Surface\n"
             "(Stacking Ensemble, PermutationExplainer, n=" + str(int(valid.sum())) + " test samples)\n"
             "Dashed line: combined SHAP = 0 (fire-risk neutral boundary)",
             fontsize=11, fontweight="bold")
plt.tight_layout()

out = OUT_DIR / "fig_shap_vpd_evi_interaction.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"   Saved -> {out.name}")
print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
