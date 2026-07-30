# WILDFIRE PAPER — AGENT HANDOFF DOCUMENT
# Read this first in any new conversation before touching the paper.

---

## Project Location
```
c:\dev\wildfire_v2\
Main paper: c:\dev\wildfire_v2\wildfire_paper_with_figures.tex   (604 lines, compiles to PDF)
PDF output: c:\dev\wildfire_v2\wildfire_paper_with_figures.pdf
Paper figures: c:\dev\wildfire_v2\paper_figures\
Models: c:\dev\wildfire_v2\models\
Scripts: c:\dev\wildfire_v2\scripts\
Main dataset: c:\dev\wildfire_v2\data\processed\training_features.csv  (2.2 GB, 9.4M rows)
```

---

## What Has Already Been Completed (DO NOT REDO)

### Phase 1 — Stacking Ensemble Deep-Dive (COMPLETE)
- Section 2.3 of the paper was fully replaced: confusion matrix, learning curve,
  Meta-Learner SHAP, Permutation SHAP, and calibration — all now show the
  Stacking Ensemble (not XGBoost).
- Figures generated via: `scripts/generate_stacking_paper_figures.py`
- All figures are in `paper_figures/`:
  - fig_stacking_confusion_matrix.png
  - fig_stacking_learning_curve.png
  - fig_shap_meta_importance_stacking.png
  - fig_shap_meta_summary_stacking.png
  - fig_shap_importance_stacking.png
  - fig_shap_summary_stacking.png
  - fig_shap_dependence_elevation_stacking.png
  - fig_calibration_curve_stacking.png

### Phase 2 — Calibration Elevation (S1) — COMPLETE
- Isotonic Regression post-calibrator was fitted and evaluated.
- **Result: ECE 0.0281 → 0.0091 (67.7% reduction)**
- New figure: `paper_figures/fig_calibration_before_after_stacking.png`
- Script: `scripts/generate_calibration_before_after.py`
- LaTeX changes applied:
  - Section 1.5: Pipeline updated from 6 → 7 stages; Stage 6 = "Probability Calibration"
  - TikZ workflow diagram: new blue-tinted calibration node added
  - Section 2.5: Fully rewritten with ECE equation, before/after figure, 67.7% result

### Phase 3 — Ablation Study (S2) — COMPLETE
- Feature group ablation using mean-substitution on XGBoost (~1 min runtime)
- Script: `scripts/run_feature_ablation.py`
- Results CSV: `outputs/ablation_results.csv`
- Key results:
  - Meteorology removal: ΔAUC = -0.0703 (most critical by AUC)
  - Elevation removal: ΔRecall = -0.1317 (most critical by recall)
  - Fuel Stack (EVI + Elev): ΔAUC = -0.1157 (largest combined drop)
  - Land Cover: ΔAUC ≈ 0 (minimal contribution)
  - KBDI: documented as rejected candidate (†) in table footnote
- LaTeX change: Line ~209 in paper — ablation sentence replaced with full Table 2

---

## What Is PENDING (The Next Agent Must Do This)

### R1 — Date-Conditional Negative Sampling as Formal Contribution

**R1-A: LaTeX only (no code)**
In Section 1.3 of the paper (around line 244–267), AFTER the current narrative
on naive sampling approaches, add:
- A **formal 3-way comparison table** of sampling strategies with equations:
  - Random Uniform: $N^{RU}$ = all non-fire rows regardless of date
  - Spatial-Only: $N^{SO}$ = non-fire rows within same month
  - Date-Conditional (ours): $N^{DC}$ = non-fire rows on exact same date $t$
- A brief "Temporal Leakage Theorem" paragraph showing RU allows seasonal shortcuts.

**R1-B: Mini-experiment (requires new script)**
Script to write: `scripts/compare_sampling_strategies.py`
What it does:
1. Load training_features.csv (year ≤ 2023 = train, year ≥ 2024 = test)
2. Simulate a Random Uniform negative sampling: take all fire positives
   from training set + sample equal number of non-fire rows randomly from
   any date (not date-matched). Same total size as the real training set.
3. Train XGBoost on this RU dataset (identical hyperparameters to existing xgboost.pkl)
4. Evaluate BOTH models on the same test set:
   - Overall: AUC, Recall, Precision
   - Per-month: recall by calendar month (Jan–Dec) — the key insight
   - Temporal bias: correlation between month and predicted probability
5. Generate:
   - `paper_figures/fig_sampling_comparison_metrics.png` — bar chart: DC vs RU overall
   - `paper_figures/fig_sampling_monthly_recall.png` — line: per-month recall DC vs RU

The existing `models/xgboost.pkl` is the DC model — reuse it directly.
Only the RU model needs retraining (~10-15 min).

**R1-C: Title and Abstract update (LaTeX only)**
- Line 33: Update title to:
  "Robust Wildfire Risk Prediction Across India via Date-Conditional Negative
   Sampling and Explainable Stacking Ensembles"
- Lines 42–51: Rewrite abstract to LEAD with DCNS contribution.
  USER PREFERENCE: Keep accessible/informal tone (not formal academic register).
  Just restructure to mention DCNS more prominently at the top.

---

### R2 — Spatial Autocorrelation Defense (LaTeX only, no code)

Add a new subsubsection after the temporal validation protocol in Section 1.4
(around line 365–370, inside the "System Architecture" subsection):

`\subsubsection{Spatial Autocorrelation and Data Leakage Considerations}`

Key arguments to make (~300 words):
1. Acknowledge Tobler's First Law of Geography
2. ERA5 is at 0.25° (~28km) but the grid is 0.1° (~11km) → adjacent cells
   share the same ERA5 weather value BY DESIGN — the model cannot exploit
   hyper-local weather autocorrelation because it doesn't exist at 11km.
3. Static features (elevation SRTM ~90m, ESA WorldCover ~10m) don't change
   year-to-year → no temporal leakage from spatial proximity.
4. Date-conditional sampling already makes within-date spatial comparison
   the primary learning signal — which is the correct target.
5. Honest limitation: spatial block cross-validation would be ideal for future work.

No figures needed. Text-only edit.

---

### R3 — Operationalizing SHAP: Zones and VPD×EVI Interaction

**R3-A: Zone-based SHAP script**
Script to write: `scripts/generate_zone_shap_analysis.py`

**6 zones** (user approved: add Indo-Gangetic Plains + Rajasthan Arid to original 4):

| Zone Label | Description | Lat range | Lon range |
|---|---|---|---|
| Central Dry Deciduous | MP, CG, Odisha | 18–26°N | 74–86°E |
| North-East Moist | Assam, Mizoram, NE states | 22–28°N | 89–97°E |
| Western Himalaya | HP, Uttarakhand | 28–33°N | 75–82°E |
| Peninsular Coast & Ghats | Kerala, TN, Karnataka coast | 8–16°N | 74–80°E |
| Indo-Gangetic Plains | UP, Bihar, Punjab, Haryana | 26–32°N | 73–89°E |
| Rajasthan Arid | Rajasthan desert | 24–30°N | 69–75°E |

Steps:
1. Load training_features.csv with lat, lon, all FEATURE_COLS (test set, year ≥ 2024)
2. Assign each row to a zone based on lat/lon bounding box.
   - Note: zones may overlap at boundaries — use the FIRST match in order above
3. For each zone, sample 500 rows stratified by fire_detected
4. Use `shap.PermutationExplainer` on the stacking ensemble (`models/stacking_ensemble.pkl`)
5. Compute mean absolute SHAP values per feature per zone
6. Generate:
   - `paper_figures/fig_shap_zone_comparison.png`
     → Grouped horizontal bar chart: 6 zone groups, features ranked by overall importance,
       colors per zone. This is the key novelty figure.
   - Also extract SHAP values for VPD and EVI columns for R3-B below.

**R3-B: VPD×EVI interaction figure**
From the same PermutationExplainer results (or a 1000-sample full-test run):
- Create a 2D hex-bin scatter:
  - X-axis: EVI feature value (0–1 range)
  - Y-axis: VPD feature value
  - Color: SHAP(VPD) + SHAP(EVI) combined (total fire-risk contribution)
  - Contour at combined_SHAP = 0
- Save as: `paper_figures/fig_shap_vpd_evi_interaction.png`

**R3-C: LaTeX updates**
In the SHAP section (around line ~555), add two new subsubsections:
1. `\subsubsection{Zone-Wise Feature Importance and Ecological Interpretation}`
   - Include fig_shap_zone_comparison.png
   - 2–3 sentences per zone interpreting dominant features vs. fire ecology
2. `\subsubsection{VPD--EVI Feature Interaction and Fire-Ecology Threshold}`
   - Include fig_shap_vpd_evi_interaction.png
   - Explain the joint threshold effect (high VPD + low EVI = extreme risk)

**Runtime: ~50 min (background task)**

---

## Model Files Available

| File | Size | Purpose |
|---|---|---|
| `models/xgboost.pkl` | 1.4 MB | Base XGBoost model (DC-trained) |
| `models/stacking_ensemble.pkl` | 978 MB | Production stacking ensemble |
| `models/production_model.pkl` | 978 MB | Same as stacking_ensemble.pkl |
| `models/lightgbm.pkl` | 1 MB | LightGBM base model |
| `models/random_forest.pkl` | 975 MB | Random Forest base model |

---

## Dataset Details

File: `data/processed/training_features.csv`
- Size: 2.2 GB, 9.4M rows
- Columns: latitude, longitude, acq_date, temp, humidity, wind, vpd, evi,
  vpd_wind, elevation_m, month_sin, month_cos, fire_detected,
  lc_tree_cover, lc_shrubland, lc_grassland, lc_cropland, lc_built_up,
  lc_sparse_veg, lc_snow_ice, lc_water, lc_herbaceous_wetland,
  lc_mangroves, lc_moss_lichen
- Train/test split: acq_date year ≤ 2023 = train | year ≥ 2024 = test
- Train size: 7,250,763 rows | Test size: 2,168,509 rows

---

## Environment

Python virtual environment at: `c:\dev\wildfire_v2\venv\`
Activate with: (auto-activated by the venv detection in the project)
Key packages: scikit-learn, xgboost, lightgbm, shap, pandas, numpy, matplotlib, joblib

LaTeX: pdflatex (available on PATH)
Compile command: `pdflatex -interaction=nonstopmode wildfire_paper_with_figures.tex`
Run from: `c:\dev\wildfire_v2\`

---

## Paper Section Map (Current Line Numbers — approximate, may shift)

| Section | Lines |
|---|---|
| Abstract | 42–51 |
| Introduction | 52–77 |
| 1.1 Data Sources | 78–140 |
| 1.2 Feature Engineering | 141–242 |
| 1.2.4 Ablation Table (Table 2) | ~209–233 |
| 1.3 Date-Conditional Negative Sampling | 244–267 |
| 1.4 System Architecture | ~310–408 |
| 1.4 Pipeline Stage 6 (Calibration) | ~345–360 |
| 1.5 Performance Metrics | ~410–440 |
| 2.1 Model Comparison & Selection | ~442–510 |
| 2.2 Overfitting Diagnostic | ~510–522 |
| 2.3 Stacking Ensemble Detailed Eval | ~522–545 |
| 2.4 SHAP Explainability | ~545–575 |
| 2.5 Probability Calibration | ~566–604 |

---

## User Preferences (Important)

- Abstract style: Keep accessible/informal tone — just lead with DCNS more prominently
- SHAP zones: 6 zones (4 original + Indo-Gangetic Plains + Rajasthan Arid)
- Ablation method: Mean-substitution (already done)
- All work uses strict temporal split: train ≤ 2023, test ≥ 2024

---

## Priority Order for Next Session

1. R1-B (mini-experiment) — Launch as background task FIRST (longest runtime ~15 min)
2. R3-A/B (zone SHAP + VPD×EVI) — Launch as background task SECOND (~50 min)
3. R2 (spatial autocorrelation text) — Write while both scripts run (~10 min)
4. R1-A (formal equations in LaTeX) — Write while scripts run (~15 min)
5. R1-C (title + abstract) — Write while scripts run (~10 min)
6. Integrate all figures into LaTeX after scripts finish
7. Recompile PDF (2 passes)
