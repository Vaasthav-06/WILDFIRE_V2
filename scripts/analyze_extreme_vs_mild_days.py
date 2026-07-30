"""
scripts/analyze_extreme_vs_mild_days.py

Static vs. Dynamic Feature Validity — Extreme vs. Mild Fire Days Analysis.

Addresses reviewer point 2: demonstrates the model responds to day-to-day
weather swings, not just static terrain (elevation, land cover).

Methodology:
  - "Extreme fire days": days (in test set) where the median VPD across
    fire-detected cells is in the TOP TERCILE of all fire-day VPD medians.
  - "Mild fire days": same but BOTTOM TERCILE.
  - Thresholds are data-driven and fully reproducible (no hand-picked dates).
  - For each group: compute model Recall, Precision, F1 using XGBoost.
  - Run SHAP PermutationExplainer on 200 stratified samples per group.
  - Compute Static vs. Dynamic SHAP shares per group.
  - Static features: Elevation, all LC columns (11)
  - Dynamic features: Temporal (month_sin/cos), Meteo (temp, humidity, wind, vpd),
                       EVI, VPD_wind interaction

Outputs:
  outputs/extreme_vs_mild_metrics.csv
  paper_figures/fig_extreme_vs_mild_shap.png

Runtime: ~10-15 min (PermutationExplainer on 400 samples, XGBoost).
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
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score
import joblib
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_FEAT = ROOT / "data" / "processed" / "training_features.csv"
MODEL_PATH = ROOT / "models" / "xgboost.pkl"   # use XGBoost (faster SHAP via TreeExplainer)
OUT_DIR   = ROOT / "outputs"
FIG_DIR   = ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
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
SEED = 42

# ── Static vs. Dynamic feature buckets ────────────────────────────────────────
LC_COLS = [
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]
STATIC_FEATURES  = ["elevation_m"] + LC_COLS
DYNAMIC_FEATURES = ["month_sin", "month_cos", "temp", "humidity", "wind",
                     "vpd", "evi", "vpd_wind"]

N_SAMPLES_PER_GROUP = 200   # for PermutationExplainer
DPI = 180

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Extreme vs. Mild Fire Days — Static/Dynamic SHAP Split Analysis")
print("=" * 65)

# Step 1: Load test set
print("\n[1/5] Loading test set ...")
needed = ["acq_date"] + FEATURE_COLS + [TARGET_COL]
df = pd.read_csv(DATA_FEAT, usecols=needed)
df["acq_date"] = pd.to_datetime(df["acq_date"])
df_test = df[df["acq_date"].dt.year >= 2024].reset_index(drop=True)
print(f"   Test set: {len(df_test):,} rows  |  Fire rate: {df_test[TARGET_COL].mean():.4f}")

# Step 2: Compute per-day median VPD across fire cells → tercile thresholds
print("\n[2/5] Computing fire-day VPD tercile thresholds ...")
fire_cells = df_test[df_test[TARGET_COL] == 1].copy()
fire_cells["day"] = fire_cells["acq_date"].dt.date
daily_median_vpd = fire_cells.groupby("day")["vpd"].median()

t33 = daily_median_vpd.quantile(0.333)
t67 = daily_median_vpd.quantile(0.667)
print(f"   Fire-day VPD medians: min={daily_median_vpd.min():.3f}  "
      f"p33={t33:.3f}  p67={t67:.3f}  max={daily_median_vpd.max():.3f}")

extreme_days = set(daily_median_vpd[daily_median_vpd >= t67].index)
mild_days    = set(daily_median_vpd[daily_median_vpd <= t33].index)
print(f"   Extreme fire days (top tercile VPD): {len(extreme_days)}")
print(f"   Mild fire days (bottom tercile VPD): {len(mild_days)}")

df_test["day"] = df_test["acq_date"].dt.date
df_extreme = df_test[df_test["day"].isin(extreme_days)].reset_index(drop=True)
df_mild    = df_test[df_test["day"].isin(mild_days)].reset_index(drop=True)
print(f"   Extreme group size: {len(df_extreme):,}  |  Mild group size: {len(df_mild):,}")

# Step 3: Load XGBoost model
print("\n[3/5] Loading XGBoost model ...")
model = joblib.load(MODEL_PATH)
print("   Loaded.")

# Step 4: Model performance metrics per group
print("\n[4/5] Computing model metrics per group ...")
groups = {"Extreme Fire Days": df_extreme, "Mild Fire Days": df_mild}
metric_rows = []

for gname, gdf in groups.items():
    if len(gdf) == 0:
        print(f"   WARNING: {gname} group is empty — skipping.")
        continue
    X_g = gdf[FEATURE_COLS].values
    y_g = gdf[TARGET_COL].values
    y_prob = model.predict_proba(X_g)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "Group": gname,
        "N_rows": len(gdf),
        "N_fires": int(y_g.sum()),
        "Fire_rate": y_g.mean(),
        "Recall":    recall_score(y_g, y_pred, zero_division=0),
        "Precision": precision_score(y_g, y_pred, zero_division=0),
        "F1":        f1_score(y_g, y_pred, zero_division=0),
    }
    metric_rows.append(metrics)
    print(f"   {gname}: Recall={metrics['Recall']:.4f}  "
          f"Precision={metrics['Precision']:.4f}  F1={metrics['F1']:.4f}")

# Step 5: SHAP analysis per group (TreeExplainer — fast for XGBoost)
print("\n[5/5] Running TreeExplainer SHAP per group ...")
rng = np.random.default_rng(SEED)

static_idx  = [FEATURE_COLS.index(f) for f in STATIC_FEATURES  if f in FEATURE_COLS]
dynamic_idx = [FEATURE_COLS.index(f) for f in DYNAMIC_FEATURES if f in FEATURE_COLS]

explainer = shap.TreeExplainer(model)
shap_results = {}

for gname, gdf in groups.items():
    if len(gdf) == 0:
        continue
    # Stratified sample
    fire_idx   = gdf.index[gdf[TARGET_COL] == 1].tolist()
    nofire_idx = gdf.index[gdf[TARGET_COL] == 0].tolist()
    n_fire   = min(N_SAMPLES_PER_GROUP // 2, len(fire_idx))
    n_nofire = min(N_SAMPLES_PER_GROUP - n_fire, len(nofire_idx))
    chosen   = (rng.choice(fire_idx,   n_fire,   replace=False).tolist() +
                rng.choice(nofire_idx, n_nofire, replace=False).tolist())
    samp = gdf.loc[chosen]
    X_samp = samp[FEATURE_COLS].values

    print(f"   {gname}: computing SHAP for {len(samp)} samples ...")
    sv = explainer.shap_values(X_samp)
    if isinstance(sv, list):
        sv = sv[1]   # fire class

    mean_abs = np.abs(sv).mean(axis=0)
    static_share  = mean_abs[static_idx].sum()
    dynamic_share = mean_abs[dynamic_idx].sum()
    total         = static_share + dynamic_share

    shap_results[gname] = {
        "static_share":   static_share / total if total > 0 else 0,
        "dynamic_share":  dynamic_share / total if total > 0 else 0,
        "static_abs":     static_share,
        "dynamic_abs":    dynamic_share,
        "mean_abs_shap":  mean_abs,
    }
    for row in metric_rows:
        if row["Group"] == gname:
            row["SHAP_static_share"]  = shap_results[gname]["static_share"]
            row["SHAP_dynamic_share"] = shap_results[gname]["dynamic_share"]
            row["SHAP_static_abs"]    = shap_results[gname]["static_abs"]
            row["SHAP_dynamic_abs"]   = shap_results[gname]["dynamic_abs"]
    print(f"     Static SHAP share: {shap_results[gname]['static_share']:.1%}  |  "
          f"Dynamic SHAP share: {shap_results[gname]['dynamic_share']:.1%}")

# Save CSV
df_metrics = pd.DataFrame(metric_rows)
csv_path = OUT_DIR / "extreme_vs_mild_metrics.csv"
df_metrics.to_csv(csv_path, index=False)
print(f"\n   Saved CSV -> {csv_path.name}")

# ── Figure: side-by-side stacked bar ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Panel A: SHAP static vs. dynamic shares
ax = axes[0]
group_names   = list(shap_results.keys())
static_shares = [shap_results[g]["static_share"]  for g in group_names]
dyn_shares    = [shap_results[g]["dynamic_share"]  for g in group_names]

x = np.arange(len(group_names))
bars_s = ax.bar(x, static_shares,  0.5, label="Static (Elevation + Land Cover)",
                color="#795548", alpha=0.82)
bars_d = ax.bar(x, dyn_shares, 0.5, bottom=static_shares,
                label="Dynamic (Meteo + EVI + Temporal)",
                color="#2E86AB", alpha=0.82)

ax.set_xticks(x)
ax.set_xticklabels(group_names, fontsize=10)
ax.set_ylabel("Share of Mean |SHAP| Value", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("Static vs. Dynamic SHAP Attribution\nXGBoost TreeExplainer", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
for bar_s, bar_d, ss, ds in zip(bars_s, bars_d, static_shares, dyn_shares):
    ax.text(bar_s.get_x() + bar_s.get_width()/2, ss/2,
            f"{ss:.0%}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(bar_d.get_x() + bar_d.get_width()/2, ss + ds/2,
            f"{ds:.0%}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

# Panel B: Performance metrics comparison
ax2 = axes[1]
metrics_to_plot = ["Recall", "Precision", "F1"]
n_metrics = len(metrics_to_plot)
width = 0.25
x2 = np.arange(len(group_names))
colors_m = ["#E84040", "#FD7E14", "#28A745"]

for mi, (met, col) in enumerate(zip(metrics_to_plot, colors_m)):
    vals = [df_metrics.loc[df_metrics["Group"] == g, met].values[0] for g in group_names]
    bars = ax2.bar(x2 + mi * width - width, vals, width, label=met, color=col, alpha=0.82)
    ax2.bar_label(bars, fmt="%.3f", padding=2, fontsize=8.5)

ax2.set_xticks(x2)
ax2.set_xticklabels(group_names, fontsize=10)
ax2.set_ylabel("Score", fontsize=11)
ax2.set_ylim(0, 1.08)
ax2.set_title("Model Performance: Extreme vs. Mild Days\n(threshold = 0.5)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)

fig.suptitle(
    "Extreme vs. Mild Fire Days Analysis\n"
    f"Extreme = top-tercile VPD fire-days ({len(extreme_days)} days)  |  "
    f"Mild = bottom-tercile ({len(mild_days)} days)  |  Test set 2024–2025",
    fontsize=10.5, fontweight="bold", y=1.01,
)
plt.tight_layout()
fig_path = FIG_DIR / "fig_extreme_vs_mild_shap.png"
fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved figure -> {fig_path.name}")

print("\n" + "=" * 65)
print("Done.")
print(f"  CSV:    {csv_path}")
print(f"  Figure: {fig_path}")
print("=" * 65)
