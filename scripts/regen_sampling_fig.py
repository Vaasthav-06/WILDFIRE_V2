"""
scripts/regen_sampling_fig.py

Regenerates fig_sampling_comparison_metrics.png using the updated
hardcoded metric values from the paper revision.

New values:
  DC:  AUC=0.9613  Recall=0.9606  Precision=0.8602  F1=0.9076  Corr=-0.082
  RU:  AUC=0.9485  Recall=0.9421  Precision=0.8490  F1=0.8931  Corr=-0.215
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Updated metric values ──────────────────────────────────────────────────────
res_dc = {"AUC": 0.9613, "Recall": 0.9606, "Precision": 0.8602, "F1": 0.9076}
res_ru = {"AUC": 0.9485, "Recall": 0.9421, "Precision": 0.8490, "F1": 0.8931}

corr_dc = -0.082
corr_ru = -0.215

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

COL_DC = "#2E86AB"
COL_RU = "#E84040"
metrics_keys = ["AUC", "Recall", "Precision", "F1"]

fig, ax = plt.subplots(figsize=(9, 5))
x    = np.arange(len(metrics_keys))
w    = 0.32
vals_dc = [res_dc[k] for k in metrics_keys]
vals_ru = [res_ru[k] for k in metrics_keys]

bars_dc = ax.bar(x - w/2, vals_dc, w, color=COL_DC,
                 label="Date-Conditional (DC, ours)",
                 alpha=0.88, edgecolor="white")
bars_ru = ax.bar(x + w/2, vals_ru, w, color=COL_RU,
                 label="Random-Uniform (RU, baseline)",
                 alpha=0.88, edgecolor="white")

for bar in bars_dc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, color=COL_DC, fontweight="bold")
for bar in bars_ru:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, color=COL_RU, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics_keys, fontsize=11)
ax.set_ylabel("Score")
ax.set_ylim(0.75, 1.03)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.legend(fontsize=10)
ax.set_title(
    "Overall Performance: Date-Conditional vs. Random-Uniform Negative Sampling\n"
    f"DC month\u2013prob correlation: r={corr_dc:+.3f}  |  "
    f"RU month\u2013prob correlation: r={corr_ru:+.3f}",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()

out1 = OUT_DIR / "fig_sampling_comparison_metrics.png"
fig.savefig(out1, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved -> {out1}")
