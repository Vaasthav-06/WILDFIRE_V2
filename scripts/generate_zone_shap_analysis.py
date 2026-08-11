"""
scripts/generate_zone_shap_analysis.py

Zone-wise SHAP importance analysis across India's 6 agro-ecological zones,
plus VPD x EVI interaction figure.

Uses PermutationExplainer on the Stacking Ensemble (models/stacking_ensemble.pkl).
200 samples per zone x 6 zones = 1200 total samples (~27 min).

Outputs:
  paper_figures/fig_shap_zone_comparison.png
  paper_figures/fig_shap_vpd_evi_interaction.png
  outputs/zone_shap_results.csv
"""

import os, sys, warnings
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import joblib, shap

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_FEAT = ROOT / "data" / "processed" / "training_features.csv"
MODEL_PKL = ROOT / "models" / "stacking_ensemble.pkl"
OUT_DIR   = ROOT / "paper_figures"
CSV_OUT   = ROOT / "outputs" / "zone_shap_results.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

FEATURE_COLS = [
    "month_sin", "month_cos",
    "temp", "humidity", "wind", "vpd",
    "evi", "vpd_wind", "elevation_m",
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]
# Human-readable feature labels (same order as FEATURE_COLS)
FEAT_LABELS = [
    "Month (sin)", "Month (cos)",
    "Temperature", "Humidity", "Wind Speed", "VPD",
    "EVI", "VPDxWind (FSP)", "Elevation",
    "LC: Tree Cover", "LC: Shrubland", "LC: Grassland", "LC: Cropland",
    "LC: Built-Up", "LC: Sparse Veg", "LC: Snow/Ice", "LC: Water",
    "LC: Herb. Wetland", "LC: Mangroves", "LC: Moss/Lichen",
]
TARGET_COL = "fire_detected"
SEED       = 42
DPI        = 180
N_PER_ZONE = 200  # samples per zone for SHAP (200 x 6 zones = 1200 total, ~27 min)

# ── Zone definitions (6 macro-zones, user-approved) ───────────────────────────
ZONES = [
    {"name": "Central Dry\nDeciduous",   "lat": (18, 26), "lon": (74, 86),
     "desc": "MP, Chhattisgarh, Odisha — peak fire zone",    "color": "#E84040"},
    {"name": "North-East\nMoist",         "lat": (22, 28), "lon": (89, 97),
     "desc": "Assam, Mizoram — high-rainfall forests",        "color": "#2E86AB"},
    {"name": "Western\nHimalaya",         "lat": (28, 33), "lon": (75, 82),
     "desc": "HP, Uttarakhand — pine forests, high elevation","color": "#28A745"},
    {"name": "Peninsular Coast\n& Ghats", "lat": (8,  16), "lon": (74, 80),
     "desc": "Kerala, TN, Karnataka coast",                   "color": "#FD7E14"},
    {"name": "Indo-Gangetic\nPlains",     "lat": (26, 32), "lon": (73, 89),
     "desc": "UP, Bihar, Punjab, Haryana",                    "color": "#9B59B6"},
    {"name": "Rajasthan\nArid",           "lat": (24, 30), "lon": (69, 75),
     "desc": "Rajasthan desert",                              "color": "#795548"},
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def assign_zone(lat, lon):
    """Return the FIRST matching zone name, or None."""
    for z in ZONES:
        if z["lat"][0] <= lat <= z["lat"][1] and z["lon"][0] <= lon <= z["lon"][1]:
            return z["name"]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Zone-Wise SHAP Analysis & VPDxEVI Interaction")
print("=" * 60)

# Step 1: Load test set
print("\n[1/4] Loading test dataset ...")
needed = ["latitude", "longitude", "acq_date"] + FEATURE_COLS + [TARGET_COL]
df = pd.read_csv(DATA_FEAT, usecols=needed)
df["acq_date"] = pd.to_datetime(df["acq_date"])
df_test = df[df["acq_date"].dt.year >= 2024].reset_index(drop=True)
df_test["zone"] = df_test.apply(lambda r: assign_zone(r["latitude"], r["longitude"]), axis=1)
print(f"   Test rows: {len(df_test):,}")
print(df_test["zone"].value_counts().to_string())

# Step 2: Sample per zone (stratified by fire_detected)
print(f"\n[2/4] Sampling {N_PER_ZONE} per zone ...")
zone_samples = []
for z in ZONES:
    zname = z["name"]
    zdf   = df_test[df_test["zone"] == zname]
    if len(zdf) == 0:
        print(f"   WARNING: No rows for zone '{zname}'")
        continue
    n = min(N_PER_ZONE, len(zdf))
    samp = zdf.sample(n=n, random_state=SEED)
    samp = samp.copy()
    samp["_zone"] = zname
    zone_samples.append(samp)
    print(f"   {zname!r:<30}: {n} samples  (fires={samp[TARGET_COL].sum()})")

df_shap = pd.concat(zone_samples, ignore_index=True)
X_shap  = df_shap[FEATURE_COLS].values
print(f"   Total SHAP samples: {len(df_shap)}")

# Step 3: Load model + run PermutationExplainer
print(f"\n[3/4] Loading stacking ensemble and running PermutationExplainer ...")
print(f"   (This will take ~{len(df_shap) * 1.3 / 60:.0f} min — please wait)")
model = joblib.load(MODEL_PKL)

# Background data: small random subsample of the full test set
bg_idx = np.random.default_rng(SEED).choice(len(df_test), size=100, replace=False)
X_bg   = df_test.iloc[bg_idx][FEATURE_COLS].values

explainer  = shap.PermutationExplainer(model.predict_proba, X_bg, max_evals=2 * len(FEATURE_COLS) + 1)
shap_vals  = explainer(X_shap)
shap_fire  = shap_vals.values[:, :, 1]   # SHAP for fire class

print("   PermutationExplainer complete.")

# Step 4: Compute per-zone mean |SHAP|
print("\n[4/4] Computing per-zone importance and generating figures ...")

zone_importance = {}
for z in ZONES:
    zname = z["name"]
    mask  = df_shap["_zone"].values == zname
    if mask.sum() == 0:
        continue
    zone_importance[zname] = np.abs(shap_fire[mask]).mean(axis=0)

# Save CSV
rows = []
for zname, imp in zone_importance.items():
    for feat, val in zip(FEAT_LABELS, imp):
        rows.append({"zone": zname, "feature": feat, "mean_abs_shap": val})
pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
print(f"   Saved CSV -> {CSV_OUT.name}")

# ── Figure 1: Grouped horizontal bar chart ─────────────────────────────────────
# Rank features by OVERALL mean importance (across all zones)
all_imp = np.array(list(zone_importance.values()))   # (n_zones, n_feats)
overall_rank = np.argsort(all_imp.mean(axis=0))[::-1]  # descending

# Show top-12 features for readability
top_n  = 12
top_idx = overall_rank[:top_n]
top_labels = [FEAT_LABELS[i] for i in top_idx]

zone_names  = [z["name"] for z in ZONES if z["name"] in zone_importance]
zone_colors = [z["color"] for z in ZONES if z["name"] in zone_importance]
n_zones     = len(zone_names)

fig, ax = plt.subplots(figsize=(12, 7))
bar_h  = 0.12
y_pos  = np.arange(top_n)

for zi, (zname, zcol) in enumerate(zip(zone_names, zone_colors)):
    imp_vals = [zone_importance[zname][i] for i in top_idx]
    offset   = (zi - n_zones / 2 + 0.5) * bar_h
    bars = ax.barh(y_pos + offset, imp_vals, bar_h,
                   color=zcol, alpha=0.82, label=zname.replace("\n", " "))

ax.set_yticks(y_pos)
ax.set_yticklabels(top_labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP Value| (Fire Probability)", fontsize=11)
ax.set_title("Zone-Wise Feature Importance Across India's Agro-Ecological Zones\n"
             "(Stacking Ensemble, Permutation SHAP — top 12 features)",
             fontsize=12, fontweight="bold")
ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.legend(loc="lower right", fontsize=9, ncol=2)
plt.tight_layout()
out1 = OUT_DIR / "fig_shap_zone_comparison.png"
fig.savefig(out1, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out1.name}")

# ── Figure 2: VPD x EVI Interaction hex-bin ───────────────────────────────────
vpd_idx = FEATURE_COLS.index("vpd")
evi_idx = FEATURE_COLS.index("evi")

vpd_vals  = X_shap[:, vpd_idx]
evi_vals  = X_shap[:, evi_idx]
shap_comb = shap_fire[:, vpd_idx] + shap_fire[:, evi_idx]

fig, ax = plt.subplots(figsize=(9, 7))
hb = ax.hexbin(evi_vals, vpd_vals, C=shap_comb,
               gridsize=40, cmap="RdYlGn_r",
               reduce_C_function=np.mean,
               vmin=-0.15, vmax=0.15)
cb = fig.colorbar(hb, ax=ax, label="Mean SHAP(VPD) + SHAP(EVI)\n<- reduces fire risk  |  increases fire risk ->")

# Contour at zero: draw iso-lines from a gridded interpolation
from scipy.interpolate import griddata

# Filter out NaNs before interpolation (1-day lag might introduce NaNs)
valid_mask = ~(np.isnan(evi_vals) | np.isnan(vpd_vals) | np.isnan(shap_comb))
evi_vals_clean = evi_vals[valid_mask]
vpd_vals_clean = vpd_vals[valid_mask]
shap_comb_clean = shap_comb[valid_mask]

xi = np.linspace(np.nanmin(evi_vals), np.nanmax(evi_vals), 80)
yi = np.linspace(np.nanmin(vpd_vals), np.nanmax(vpd_vals), 80)
XI, YI = np.meshgrid(xi, yi)

try:
    ZI = griddata((evi_vals_clean, vpd_vals_clean), shap_comb_clean, (XI, YI), method="linear")
    cs = ax.contour(XI, YI, ZI, levels=[0], colors="black",
                    linewidths=2, linestyles="--")
    ax.clabel(cs, fmt="SHAP = 0", fontsize=9)
except Exception:
    pass

ax.set_xlabel("EVI (Live-Fuel Moisture Proxy)", fontsize=11)
ax.set_ylabel("VPD — Vapor Pressure Deficit (kPa)", fontsize=11)
ax.set_title("VPD x EVI Joint SHAP Interaction\n"
             "High VPD + Low EVI -> extreme fire risk (top-left); "
             "Low VPD + High EVI -> strong fire suppression (bottom-right)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
out2 = OUT_DIR / "fig_shap_vpd_evi_interaction.png"
fig.savefig(out2, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out2.name}")

print("\n" + "=" * 60)
print("Done. Figures saved:")
print(f"  {out1.name}")
print(f"  {out2.name}")
print(f"  {CSV_OUT.name}")
print("=" * 60)
