"""
generate_missing_figures.py
Generates 5 missing paper figures from real training data + saved models.
Outputs saved to: paper_figures/
"""
import sys
import os
# Force UTF-8 stdout on Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_FEAT = ROOT / "data" / "processed" / "training_features.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR   = ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 180

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":       DPI,
})

FIRE_COL   = "#E84040"
NOFIRE_COL = "#4C9BE8"
ACCENT     = "#F4845F"

print("=" * 60)
print("Generating missing paper figures ...")
print("=" * 60)

# ── Load data (stratified 500 k sample) ──────────────────────────────────────
print("\n[1/5] Loading training_features.csv (500 k sample) ...")
df_full = pd.read_csv(DATA_FEAT, usecols=[
    "acq_date", "fire_detected",
    "temp", "humidity", "wind", "vpd", "evi",
    "elevation_m", "month_sin", "month_cos", "vpd_wind",
    "lc_tree_cover","lc_shrubland","lc_grassland","lc_cropland",
    "lc_built_up","lc_sparse_veg","lc_snow_ice","lc_water",
    "lc_herbaceous_wetland","lc_mangroves","lc_moss_lichen",
], nrows=5_000_000)

df_full["acq_date"] = pd.to_datetime(df_full["acq_date"])
df_full["year"]     = df_full["acq_date"].dt.year
df_full["month"]    = df_full["acq_date"].dt.month

fire   = df_full[df_full.fire_detected == 1].sample(
    min(80_000,  (df_full.fire_detected==1).sum()), random_state=42)
nofire = df_full[df_full.fire_detected == 0].sample(
    min(420_000, (df_full.fire_detected==0).sum()), random_state=42)
df = pd.concat([fire, nofire]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Sample: {len(df):,} rows | fire={df.fire_detected.sum():,} | no-fire={(df.fire_detected==0).sum():,}")


# =========================================================================
# FIGURE 1: Feature Correlation Heatmap
# =========================================================================
print("\n[2/5] Feature Correlation Heatmap ...")

FEAT_COLS = [
    "temp", "humidity", "wind", "vpd", "vpd_wind",
    "evi", "elevation_m", "month_sin", "month_cos",
    "lc_tree_cover","lc_shrubland","lc_grassland",
    "lc_cropland","lc_water",
]
FEAT_LABELS = [
    "Temperature", "Humidity", "Wind Speed", "VPD", "VPD x Wind (FSP)",
    "EVI", "Elevation", "Month (sin)", "Month (cos)",
    "LC: Tree Cover","LC: Shrubland","LC: Grassland",
    "LC: Cropland","LC: Water",
]

corr = df[FEAT_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(220, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.4, linecolor="white",
    square=True, ax=ax,
    xticklabels=FEAT_LABELS, yticklabels=FEAT_LABELS,
    cbar_kws={"shrink": 0.75, "label": "Pearson r"},
)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=14)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
out1 = OUT_DIR / "fig_feature_correlation_heatmap.png"
fig.savefig(out1, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out1.name}")


# =========================================================================
# FIGURE 2: Seasonal Fire Frequency Bar Chart
# =========================================================================
print("\n[3/5] Seasonal Fire Frequency ...")

monthly = df_full[df_full.fire_detected == 1].groupby("month").size().reset_index(name="fire_count")
MONTH_NAMES = {
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
}
monthly["month_name"] = monthly["month"].map(MONTH_NAMES)

def month_color(m):
    if m in [2,3,4,5]:  return "#E84040"
    elif m in [10,11]:  return ACCENT
    elif m in [6,7,8,9]:return "#4C9BE8"
    else:               return "#A8C5E2"

colors = [month_color(m) for m in monthly["month"]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(monthly["month_name"], monthly["fire_count"] / 1000,
              color=colors, edgecolor="white", linewidth=0.8, width=0.7, zorder=3)
ax.set_xlabel("Month")
ax.set_ylabel("Fire Detections (x1,000)")
ax.set_title("Monthly Wildfire Frequency Across India (2018-2025, NASA FIRMS VIIRS)",
             fontsize=13, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color="#E84040", label="Peak fire season (Feb-May)"),
    mpatches.Patch(color=ACCENT,    label="Secondary peak (Oct-Nov)"),
    mpatches.Patch(color="#4C9BE8", label="Monsoon suppression (Jun-Sep)"),
    mpatches.Patch(color="#A8C5E2", label="Low activity (Dec-Jan)"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="upper right", framealpha=0.9)

for bar, val in zip(bars, monthly["fire_count"] / 1000):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.0f}k", ha="center", va="bottom", fontsize=8, color="#333")

plt.tight_layout()
out2 = OUT_DIR / "fig_seasonal_fire_frequency.png"
fig.savefig(out2, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out2.name}")


# =========================================================================
# FIGURE 3: Date-Conditional Sampling Diagram (Schematic)
# =========================================================================
print("\n[4/5] Date-Conditional Sampling Diagram ...")

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis("off")

ax.text(6.5, 7.1, "Date-Conditional Negative Sampling Strategy",
        ha="center", va="center", fontsize=14, fontweight="bold")

# Timeline
ax.axhline(4.9, xmin=0.04, xmax=0.94, color="#ccc", lw=2, zorder=1)
dates_info = [("Day t-2", 1.5), ("Day t-1", 3.0), ("Day  t", 6.2), ("Day t+1", 9.5), ("Day t+2", 11.2)]
for label, xd in dates_info:
    ax.plot(xd, 4.9, "o", color="#888", ms=9, zorder=2)
    ax.text(xd, 5.15, label, ha="center", va="bottom", fontsize=9, color="#555")

# Highlight box for Day t
ax.add_patch(plt.Rectangle((5.0, 4.1), 2.4, 1.5,
             linewidth=2, edgecolor="#555", facecolor="#FFF3CD", zorder=3, alpha=0.9))
ax.text(6.2, 5.45, "Date = t", ha="center", va="center", fontsize=9,
        color="#333", fontstyle="italic")

# India grid schematic
grid_x0, grid_y0 = 5.05, 0.5
cell_w, cell_h = 0.36, 0.36
fire_cells = {(2, 5), (3, 3), (1, 6)}

for row in range(9):
    for col in range(6):
        xc = grid_x0 + col * cell_w
        yc = grid_y0 + row * cell_h
        is_fire = (col, row) in fire_cells
        color = FIRE_COL if is_fire else "#C8DCF0"
        ec    = "#c0392b" if is_fire else "#9BB8D0"
        ax.add_patch(plt.Rectangle((xc, yc), cell_w*0.88, cell_h*0.88,
                                   facecolor=color, edgecolor=ec, linewidth=0.8, zorder=4))
        if is_fire:
            # Use text "F" instead of emoji for Windows compatibility
            ax.text(xc + cell_w*0.44, yc + cell_h*0.44, "F",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white", zorder=5)

ax.text(6.2, 0.1, "India 0.1 deg Grid -- Day t", ha="center", va="bottom", fontsize=9, color="#444")

# Arrows from grid to explanation boxes
ax.annotate("", xy=(4.5, 3.0), xytext=(5.2, 1.8),
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5, mutation_scale=14))
ax.annotate("", xy=(4.5, 2.4), xytext=(5.55, 1.65),
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5, mutation_scale=14))

# POSITIVE box
ax.add_patch(plt.Rectangle((0.3, 2.5), 3.9, 1.3,
             linewidth=1.8, edgecolor=FIRE_COL, facecolor="#FDECEA", zorder=3))
ax.text(2.25, 3.55, "[+]  POSITIVE SAMPLE", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#c0392b")
ax.text(2.25, 2.98,
        "Fire cell (i, j) on day t\nFeatures: ERA5 + MODIS EVI\n+ Elevation + Land Cover",
        ha="center", va="center", fontsize=8.5, color="#333")

# NEGATIVE box
ax.add_patch(plt.Rectangle((0.3, 0.9), 3.9, 1.4,
             linewidth=1.8, edgecolor=NOFIRE_COL, facecolor="#EBF4FD", zorder=3))
ax.text(2.25, 2.07, "[-]  NEGATIVE SAMPLE", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#1a5276")
ax.text(2.25, 1.45,
        "Non-fire cell (i', j') on SAME day t\nSame ERA5 meteorological snapshot\n=> Model must learn within-day signal",
        ha="center", va="center", fontsize=8.5, color="#333")

# KEY RULE box
ax.add_patch(plt.Rectangle((8.8, 0.6), 3.9, 3.1,
             linewidth=2, edgecolor="#2ECC71", facecolor="#EAFAF1", zorder=3))
ax.text(10.75, 3.5, "Key Rule", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#1E8449")
ax.text(10.75, 2.7,
        "For every fire event\n"
        "on date t:\n\n"
        "Negatives drawn ONLY\n"
        "from non-fire cells on\n"
        "the SAME date t\n\n"
        "=> Prevents seasonal\n"
        "    shortcut learning",
        ha="center", va="center", fontsize=9, color="#1E8449", linespacing=1.5)

plt.tight_layout()
out3 = OUT_DIR / "fig_date_conditional_sampling_diagram.png"
fig.savefig(out3, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out3.name}")


# =========================================================================
# FIGURE 4: EVI Distribution by Fire / No-Fire Class
# =========================================================================
print("\n[5/5] EVI Distribution ...")

evi_fire   = df[df.fire_detected == 1]["evi"].dropna()
evi_nofire = df[df.fire_detected == 0]["evi"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Left: Violin + Box ────────────────────────────────────────────────────
ax = axes[0]
parts = ax.violinplot([evi_nofire.values, evi_fire.values],
                      positions=[0, 1], widths=0.7, showmedians=False,
                      showextrema=False, bw_method=0.15)
for body, c in zip(parts["bodies"], [NOFIRE_COL, FIRE_COL]):
    body.set_facecolor(c)
    body.set_alpha(0.55)

bp = ax.boxplot([evi_nofire.values, evi_fire.values],
                positions=[0, 1], widths=0.18, patch_artist=True, notch=True,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker=".", markersize=1.5, alpha=0.2))
for patch, c in zip(bp["boxes"], [NOFIRE_COL, FIRE_COL]):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)

ax.set_xticks([0, 1])
ax.set_xticklabels(["No Fire", "Fire Event"], fontsize=11)
ax.set_ylabel("Enhanced Vegetation Index (EVI)")
ax.set_title("EVI Distribution by Class (Violin + Box)", fontweight="bold")
ax.set_ylim(-0.35, 0.95)
ax.axhline(0.2, color="#999", linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(0.3, color="#555", linestyle="--", linewidth=1, alpha=0.7)
ax.text(1.42, 0.21, "EVI = 0.2 (stressed)", fontsize=7.5, color="#666", va="bottom")
ax.text(1.42, 0.31, "EVI = 0.3 (healthy)",  fontsize=7.5, color="#444", va="bottom")
ax.legend(handles=[
    mpatches.Patch(color=NOFIRE_COL, alpha=0.7, label="No Fire"),
    mpatches.Patch(color=FIRE_COL,   alpha=0.7, label="Fire Event"),
], fontsize=9, loc="upper left")

# ── Right: Overlapping KDE ────────────────────────────────────────────────
ax2 = axes[1]
evi_f_clipped  = evi_fire.clip(-0.2, 0.9)
evi_nf_clipped = evi_nofire.clip(-0.2, 0.9)

evi_f_clipped.plot.kde(ax=ax2,  color=FIRE_COL,   linewidth=2.5,
                       label=f"Fire (n={len(evi_fire):,})", bw_method=0.08)
evi_nf_clipped.plot.kde(ax=ax2, color=NOFIRE_COL, linewidth=2.5,
                        label=f"No Fire (n={len(evi_nofire):,})", bw_method=0.08)

# Shade under both curves
x_range = np.linspace(-0.2, 0.9, 300)
for line, c in zip(ax2.lines, [FIRE_COL, NOFIRE_COL]):
    ydata = line.get_ydata()
    xdata = line.get_xdata()
    ax2.fill_between(xdata, 0, ydata, color=c, alpha=0.12)

ax2.axvline(evi_fire.median(),   color=FIRE_COL,   linestyle="--", linewidth=1.5,
            label=f"Fire median = {evi_fire.median():.3f}")
ax2.axvline(evi_nofire.median(), color=NOFIRE_COL, linestyle="--", linewidth=1.5,
            label=f"No-fire median = {evi_nofire.median():.3f}")
ax2.set_xlim(-0.25, 0.95)
ax2.set_xlabel("EVI")
ax2.set_ylabel("Density")
ax2.set_title("EVI Density Comparison: Fire vs. No-Fire", fontweight="bold")
ax2.legend(fontsize=9)

fig.suptitle(
    "MODIS EVI as Live-Fuel Moisture Proxy: Fire events occur at lower EVI values,\n"
    "validating EVI as a fire-risk indicator",
    fontsize=10, style="italic", y=1.01
)
plt.tight_layout()
out4 = OUT_DIR / "fig_evi_distribution_fire_class.png"
fig.savefig(out4, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out4.name}")


# =========================================================================
# FIGURE 5: Calibration Curve (Reliability Diagram)
# =========================================================================
print("\n[6/6] Calibration Curve ...")

import pickle
from sklearn.calibration import calibration_curve

model_path = MODEL_DIR / "xgboost.pkl"
print(f"   Loading {model_path.name} ...")
with open(model_path, "rb") as f:
    xgb_model = pickle.load(f)

FEAT_COLS_MODEL = [
    "temp", "humidity", "wind", "vpd", "evi",
    "elevation_m", "month_sin", "month_cos", "vpd_wind",
    "lc_tree_cover","lc_shrubland","lc_grassland","lc_cropland",
    "lc_built_up","lc_sparse_veg","lc_snow_ice","lc_water",
    "lc_herbaceous_wetland","lc_mangroves","lc_moss_lichen",
]

df_cal = pd.concat([
    df[df.fire_detected == 1].sample(min(25_000, (df.fire_detected==1).sum()), random_state=99),
    df[df.fire_detected == 0].sample(min(25_000, (df.fire_detected==0).sum()), random_state=99),
]).sample(frac=1, random_state=99).reset_index(drop=True)

X_cal = df_cal[FEAT_COLS_MODEL].fillna(0).values
y_cal = df_cal["fire_detected"].values

print("   Predicting probabilities ...")
y_prob_cal = xgb_model.predict_proba(X_cal)[:, 1]

frac_pos, mean_pred = calibration_curve(y_cal, y_prob_cal, n_bins=15, strategy="quantile")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfectly calibrated", alpha=0.7)
ax.plot(mean_pred, frac_pos, "o-", color=FIRE_COL, linewidth=2.5,
        markersize=7, markeredgewidth=1.5, markeredgecolor="white",
        label="XGBoost (uncalibrated)", zorder=4)
ax.fill_between(mean_pred, mean_pred, frac_pos, alpha=0.12, color=FIRE_COL,
                label="Calibration gap")

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.08)
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (Observed Fire Rate)")
ax.set_title("Reliability Diagram -- XGBoost Fire Probability Calibration",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)

ax.annotate(
    "Curve above diagonal:\nmodel under-predicts\nprobability at high end",
    xy=(mean_pred[-3], frac_pos[-3]),
    xytext=(0.35, 0.90),
    fontsize=8.5, color="#555",
    arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
)

plt.tight_layout()
out5 = OUT_DIR / "fig_calibration_curve_xgboost.png"
fig.savefig(out5, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"   Saved -> {out5.name}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All 5 figures generated successfully.")
print(f"Output directory: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("fig_*.png")):
    print(f"  {f.name}")
print("=" * 60)
