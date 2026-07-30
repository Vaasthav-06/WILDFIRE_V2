"""
scripts/run_spatial_blocked_cv.py

Leave-One-Zone-Out Cross-Validation vs. Current Random/Temporal Split.

Addresses reviewer point 1: the paper must EMPIRICALLY demonstrate that
spatial leakage is not occurring, not just argue it conceptually.

Design:
  - Uses the exact ZONES list from generate_zone_shap_analysis.py (no zone drift).
  - Leave-One-Zone-Out CV: for each of 6 zones, train on all other zones' data
    (temporal filter applied within each fold: year <= 2023 = train),
    test on held-out zone's test set (year >= 2024).
  - Also runs the current methodology (temporal random split, same XGBoost) as
    a comparison baseline on the same data.
  - Subsampling: each fold trains on up to SUBSAMPLE_PER_FOLD rows to keep
    runtime manageable (documented; seed=42 for reproducibility).
  - Uses XGBoost with identical hyperparameters to the published model.
    (Stacking Ensemble is 978 MB; reloading it 6 times + retraining each fold
    is impractical. XGBoost as proxy is consistent with the ablation study.)

Outputs:
  outputs/spatial_cv_results.csv
  paper_figures/fig_spatial_cv_comparison.png

Runtime: ~25-35 min total (6 XGBoost folds x ~5 min each).
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
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_FEAT = ROOT / "data" / "processed" / "training_features.csv"
OUT_DIR   = ROOT / "outputs"
FIG_DIR   = ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Import zone definitions directly — no drift from the zone SHAP script ─────
# We use exec/import trick to avoid circular issues from a script-level file.
import importlib.util
_zone_script = ROOT / "scripts" / "generate_zone_shap_analysis.py"

# Parse ZONES list directly from the zone script to avoid running the full script
# (which would load the 978 MB model). We read and eval the ZONES literal only.
import ast, re

with open(_zone_script, "r", encoding="utf-8") as _f:
    _src = _f.read()

# Extract the ZONES list literal from the source
_match = re.search(r"^ZONES\s*=\s*(\[.*?\])", _src, re.DOTALL | re.MULTILINE)
if _match:
    ZONES = ast.literal_eval(_match.group(1))
else:
    # Fallback: hardcode identical zone definitions if parsing fails
    ZONES = [
        {"name": "Central Dry\nDeciduous",   "lat": (18, 26), "lon": (74, 86), "color": "#E84040"},
        {"name": "North-East\nMoist",         "lat": (22, 28), "lon": (89, 97), "color": "#2E86AB"},
        {"name": "Western\nHimalaya",         "lat": (28, 33), "lon": (75, 82), "color": "#28A745"},
        {"name": "Peninsular Coast\n& Ghats", "lat": (8,  16), "lon": (74, 80), "color": "#FD7E14"},
        {"name": "Indo-Gangetic\nPlains",     "lat": (26, 32), "lon": (73, 89), "color": "#9B59B6"},
        {"name": "Rajasthan\nArid",           "lat": (24, 30), "lon": (69, 75), "color": "#795548"},
    ]

FEATURE_COLS = [
    "month_sin", "month_cos",
    "temp", "humidity", "wind", "vpd",
    "evi", "vpd_wind", "elevation_m",
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]
TARGET_COL = "fire_detected"

SEED                = 42
SUBSAMPLE_PER_FOLD  = 500_000   # cap per fold training set (documented)
DPI                 = 180

# XGBoost hyperparameters — identical to the published model (from train_models.py)
XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,   # approximate imbalance weight
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     SEED,
    "n_jobs":           -1,
    "verbosity":        0,
}


def assign_zone(lat: float, lon: float) -> str | None:
    """Return the first matching zone name, or None. Mirrors generate_zone_shap_analysis.py."""
    for z in ZONES:
        if z["lat"][0] <= lat <= z["lat"][1] and z["lon"][0] <= lon <= z["lon"][1]:
            return z["name"]
    return None


def get_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC":       roc_auc_score(y_true, y_prob),
        "Recall":    recall_score(y_true, y_pred,    zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred,        zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Spatially Blocked Cross-Validation (Leave-One-Zone-Out)")
print("=" * 65)

# Step 1: Load dataset
print("\n[1/4] Loading training_features.csv ...")
needed = ["latitude", "longitude", "acq_date"] + FEATURE_COLS + [TARGET_COL]
df = pd.read_csv(DATA_FEAT, usecols=needed)
df["acq_date"] = pd.to_datetime(df["acq_date"])
df["year"] = df["acq_date"].dt.year

# Assign zones (vectorised via apply — ~30 sec on 9.4M rows)
print("   Assigning zones (this takes ~30-60 sec on 9.4M rows) ...")
df["zone"] = df.apply(lambda r: assign_zone(r["latitude"], r["longitude"]), axis=1)

zone_counts = df["zone"].value_counts()
print(f"   Dataset: {len(df):,} rows")
print(f"   Zone coverage:\n{zone_counts.to_string()}")
print(f"   Unassigned rows: {df['zone'].isna().sum():,}")

# Step 2: Current methodology (temporal split) — baseline comparison
print("\n[2/4] Baseline: current temporal split (train<=2023, test>=2024) ...")
from xgboost import XGBClassifier

train_all = df[df["year"] <= 2023]
test_all  = df[df["year"] >= 2024]

# Subsample training for speed and documentation consistency
rng = np.random.default_rng(SEED)
if len(train_all) > SUBSAMPLE_PER_FOLD:
    idx = rng.choice(len(train_all), size=SUBSAMPLE_PER_FOLD, replace=False)
    train_sample = train_all.iloc[idx]
    print(f"   Subsampled train: {SUBSAMPLE_PER_FOLD:,} / {len(train_all):,} rows")
else:
    train_sample = train_all
    print(f"   Train: {len(train_sample):,} rows (no subsampling needed)")

X_tr = train_sample[FEATURE_COLS].values
y_tr = train_sample[TARGET_COL].values
X_te = test_all[FEATURE_COLS].values
y_te = test_all[TARGET_COL].values

print("   Training temporal-split XGBoost ...")
baseline_model = XGBClassifier(**XGBOOST_PARAMS)
baseline_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
baseline_probs = baseline_model.predict_proba(X_te)[:, 1]
baseline_metrics = get_metrics(y_te, baseline_probs)
print(f"   Temporal split — AUC={baseline_metrics['AUC']:.4f}  "
      f"Recall={baseline_metrics['Recall']:.4f}  "
      f"Precision={baseline_metrics['Precision']:.4f}  "
      f"F1={baseline_metrics['F1']:.4f}")

# Step 3: Leave-One-Zone-Out CV
print("\n[3/4] Leave-One-Zone-Out CV (6 folds) ...")
results = []
zone_names = [z["name"] for z in ZONES]

# Append temporal baseline as an "overall" row
results.append({
    "zone": "Overall (Temporal Split)",
    "method": "Random/Temporal Split",
    **baseline_metrics,
})

for zi, held_out_name in enumerate(zone_names):
    zone_label = held_out_name.replace("\n", " ")
    print(f"\n  Fold {zi+1}/6 — Held-out zone: {zone_label}")

    # Identify held-out zone rows
    held_mask = df["zone"] == held_out_name

    # Training: all other zones, year <= 2023
    train_mask = (~held_mask) & (df["year"] <= 2023)
    test_mask  = held_mask    & (df["year"] >= 2024)

    train_zone = df[train_mask]
    test_zone  = df[test_mask]

    if len(test_zone) == 0:
        print(f"    WARNING: No test rows for zone '{zone_label}' in 2024+ — skipping.")
        continue
    if len(train_zone) == 0:
        print(f"    WARNING: No training rows outside zone '{zone_label}' — skipping.")
        continue

    # Subsample training
    if len(train_zone) > SUBSAMPLE_PER_FOLD:
        idx = rng.choice(len(train_zone), size=SUBSAMPLE_PER_FOLD, replace=False)
        train_zone_s = train_zone.iloc[idx]
    else:
        train_zone_s = train_zone

    print(f"    Train: {len(train_zone_s):,} rows | Test: {len(test_zone):,} rows "
          f"| Fire rate: {test_zone[TARGET_COL].mean():.3f}")

    X_tr_z = train_zone_s[FEATURE_COLS].values
    y_tr_z = train_zone_s[TARGET_COL].values
    X_te_z = test_zone[FEATURE_COLS].values
    y_te_z = test_zone[TARGET_COL].values

    if y_te_z.sum() == 0:
        print(f"    WARNING: Zone '{zone_label}' test set has no fire events — skipping.")
        continue

    model_z = XGBClassifier(**XGBOOST_PARAMS)
    model_z.fit(X_tr_z, y_tr_z, verbose=False)
    probs_z = model_z.predict_proba(X_te_z)[:, 1]
    metrics_z = get_metrics(y_te_z, probs_z)

    print(f"    Spatial Block — AUC={metrics_z['AUC']:.4f}  "
          f"Recall={metrics_z['Recall']:.4f}  "
          f"Precision={metrics_z['Precision']:.4f}  "
          f"F1={metrics_z['F1']:.4f}")

    # Also run the temporal-split (already trained) model on this zone's test set
    probs_base_z = baseline_model.predict_proba(X_te_z)[:, 1]
    metrics_base_z = get_metrics(y_te_z, probs_base_z)

    results.append({
        "zone": zone_label,
        "method": "Spatial Block (LOZO-CV)",
        **metrics_z,
    })
    results.append({
        "zone": zone_label,
        "method": "Temporal Split (current)",
        **metrics_base_z,
    })

# Average across all zones for spatial block
sb_rows = [r for r in results if r["method"] == "Spatial Block (LOZO-CV)"]
ts_rows = [r for r in results if r["method"] == "Temporal Split (current)"]

if sb_rows:
    avg_sb = {k: np.mean([r[k] for r in sb_rows]) for k in ["AUC", "Recall", "Precision", "F1"]}
    results.append({"zone": "Average (Spatial Block)", "method": "Spatial Block (LOZO-CV)", **avg_sb})
    print(f"\n  Average Spatial Block AUC = {avg_sb['AUC']:.4f}")

if ts_rows:
    avg_ts = {k: np.mean([r[k] for r in ts_rows]) for k in ["AUC", "Recall", "Precision", "F1"]}
    results.append({"zone": "Average (Temporal Split)", "method": "Temporal Split (current)", **avg_ts})
    print(f"  Average Temporal Split AUC = {avg_ts['AUC']:.4f}")

# Save CSV
df_results = pd.DataFrame(results)
csv_path = OUT_DIR / "spatial_cv_results.csv"
df_results.to_csv(csv_path, index=False)
print(f"\n  Saved CSV -> {csv_path}")

# Step 4: Figure
print("\n[4/4] Generating figure ...")

# Pivot for per-zone grouped bar chart
zone_order = [z["name"].replace("\n", " ") for z in ZONES
              if z["name"].replace("\n", " ") in df_results["zone"].values]
zone_order_with_avg = zone_order + ["Average (Spatial Block)", "Average (Temporal Split)"]

sb_auc = []
ts_auc = []
x_labels = []

for zname in zone_order:
    sb_r = df_results[(df_results["zone"] == zname) & (df_results["method"] == "Spatial Block (LOZO-CV)")]
    ts_r = df_results[(df_results["zone"] == zname) & (df_results["method"] == "Temporal Split (current)")]
    if not sb_r.empty and not ts_r.empty:
        sb_auc.append(float(sb_r["AUC"].iloc[0]))
        ts_auc.append(float(ts_r["AUC"].iloc[0]))
        x_labels.append(zname)

# Add averages
if sb_rows and ts_rows:
    sb_auc.append(avg_sb["AUC"])
    ts_auc.append(avg_ts["AUC"])
    x_labels.append("Overall Average")

x = np.arange(len(x_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width/2, ts_auc, width, label="Temporal Split (current methodology)",
               color="#4C9BE8", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, sb_auc, width, label="Spatial Block / Leave-One-Zone-Out",
               color="#E84040", alpha=0.85, edgecolor="white")

ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8.5)
ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8.5)

# Shade the "Overall Average" bar group differently
if len(x_labels) > len(zone_order):
    avg_idx = len(zone_order)
    ax.axvspan(avg_idx - width - 0.02, avg_idx + width + 0.02, alpha=0.08, color="gray", zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("ROC-AUC", fontsize=11)
ax.set_ylim(max(0, min(sb_auc + ts_auc) - 0.05), 1.02)
ax.set_title(
    "Spatially Blocked CV (Leave-One-Zone-Out) vs. Current Temporal Split\n"
    f"XGBoost  |  Train subsample = {SUBSAMPLE_PER_FOLD:,} rows per fold  |  Test = 2024–2025",
    fontsize=11, fontweight="bold",
)
ax.legend(loc="lower right", fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = FIG_DIR / "fig_spatial_cv_comparison.png"
fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  Saved figure -> {fig_path.name}")

print("\n" + "=" * 65)
print("Spatial Blocked CV complete.")
print(f"  CSV:    {csv_path}")
print(f"  Figure: {fig_path}")
print("=" * 65)
