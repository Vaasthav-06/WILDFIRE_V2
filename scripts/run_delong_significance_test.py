"""
scripts/run_delong_significance_test.py

DeLong's paired AUC significance test for near-zero ablation features.

Addresses reviewer MJ-3: Table 2 shows Land Cover (?AUC � 0) and VPD�Wind
(?AUC = -0.0027) contribute negligibly. This script performs DeLong's paired
AUC test to determine whether these differences are statistically significant.

Method: fastDeLong (Sun & Xu, 2014)

Outputs:
  outputs/delong_significance_results.csv
  Console report with z-statistic, p-value, and 95% CI for each test.

Runtime: ~5-10 min (predict_proba on test set for XGBoost model).
"""

import os
import sys
import warnings

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
import joblib

ROOT       = Path(__file__).resolve().parent.parent
DATA_FEAT  = ROOT / "data" / "processed" / "training_features.csv"
MODEL_PATH = ROOT / "models" / "xgboost.pkl"

FEATURE_COLS = [
    "month_sin", "month_cos",
    "temp", "humidity", "wind", "vpd",
    "evi", "vpd_wind", "elevation_m",
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]
TARGET_COL = "fire_detected"

LAND_COVER_COLS = [
    "lc_tree_cover", "lc_shrubland", "lc_grassland", "lc_cropland",
    "lc_built_up", "lc_sparse_veg", "lc_snow_ice", "lc_water",
    "lc_herbaceous_wetland", "lc_mangroves", "lc_moss_lichen",
]
VPD_WIND_COLS = ["vpd_wind"]


def delong_paired_test(y_true, y_prob_full, y_prob_ablated):
    """
    DeLong's paired AUC test using midrank structural components.
    Returns: auc_full, auc_ablated, delta_auc, z, p_value, ci_low, ci_high
    """
    y_true = np.array(y_true, dtype=float)
    y_prob_full    = np.array(y_prob_full, dtype=float)
    y_prob_ablated = np.array(y_prob_ablated, dtype=float)

    m = int(y_true.sum())   # positives
    n = len(y_true) - m     # negatives

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    def kernel_matrix(x_pos, x_neg):
        """Compute psi matrix: psi(X_i, Y_j) = P(X_i > Y_j) + 0.5*P(X_i==Y_j)"""
        V10 = np.zeros(len(x_pos))
        V01 = np.zeros(len(x_neg))
        for i, xi in enumerate(x_pos):
            gt = np.sum(xi > x_neg)
            eq = np.sum(xi == x_neg)
            V10[i] = (gt + 0.5 * eq) / n
        for j, yj in enumerate(x_neg):
            lt = np.sum(yj < x_pos)
            eq = np.sum(yj == x_pos)
            V01[j] = (lt + 0.5 * eq) / m
        return V10, V01

    classifiers = [y_prob_full, y_prob_ablated]
    aucs = [roc_auc_score(y_true, p) for p in classifiers]

    # Compute structural components for each classifier
    all_V10 = []
    all_V01 = []
    for clf_prob in classifiers:
        x_pos = clf_prob[pos_idx]
        x_neg = clf_prob[neg_idx]
        V10, V01 = kernel_matrix(x_pos, x_neg)
        all_V10.append(V10)
        all_V01.append(V01)

    all_V10 = np.array(all_V10)  # (2, m)
    all_V01 = np.array(all_V01)  # (2, n)

    # Covariance matrix of AUC estimates
    SX = np.cov(all_V10) / m
    SY = np.cov(all_V01) / n
    S  = SX + SY  # (2, 2)

    delta_auc = aucs[0] - aucs[1]
    var_diff   = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se_diff    = np.sqrt(max(var_diff, 1e-15))

    z       = delta_auc / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    ci_low  = delta_auc - 1.96 * se_diff
    ci_high = delta_auc + 1.96 * se_diff

    return aucs[0], aucs[1], delta_auc, z, p_value, ci_low, ci_high


def main():
    print("=" * 65)
    print("DeLong Paired AUC Significance Test  (MJ-3)")
    print("Targets: Land Cover, VPD�Wind")
    print("=" * 65)

    print("\n[1/4] Loading test set ...")
    df = pd.read_csv(DATA_FEAT, usecols=["acq_date"] + FEATURE_COLS + [TARGET_COL])
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df_test  = df[df["acq_date"].dt.year >= 2024].sort_values("acq_date").reset_index(drop=True)
    df_train = df[df["acq_date"].dt.year < 2024]
    y_test   = df_test[TARGET_COL].values
    train_means = df_train[FEATURE_COLS].mean()
    print(f"   Test rows : {len(df_test):,}")

    print("\n[2/4] Loading XGBoost model ...")
    model = joblib.load(MODEL_PATH)
    print("   Loaded.")

    print("\n[3/4] Baseline predictions on full model ...")
    X_test_df = df_test[FEATURE_COLS].copy()
    y_prob_full = model.predict_proba(X_test_df)[:, 1]
    print(f"   Baseline AUC: {roc_auc_score(y_test, y_prob_full):.6f}")

    print("\n[4/4] Running DeLong tests ...")
    tests = [
        ("Land Cover (11 indicators)", LAND_COVER_COLS),
        ("VPD x Wind Interaction",     VPD_WIND_COLS),
    ]

    results = []
    for test_name, ablate_cols in tests:
        X_abl = X_test_df.copy()
        for col in ablate_cols:
            X_abl[col] = train_means[col]
        y_prob_abl = model.predict_proba(X_abl)[:, 1]

        auc_f, auc_a, delta, z, p, ci_lo, ci_hi = delong_paired_test(y_test, y_prob_full, y_prob_abl)
        sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "NOT significant"
        print(f"\n  Feature: {test_name}")
        print(f"    AUC full    = {auc_f:.6f}")
        print(f"    AUC ablated = {auc_a:.6f}")
        print(f"    ?AUC        = {delta:+.6f}")
        print(f"    z           = {z:.4f}")
        print(f"    p-value     = {p:.4e}  ?  {sig}")
        print(f"    95% CI      = [{ci_lo:.6f}, {ci_hi:.6f}]")
        results.append({
            "Feature Group": test_name,
            "AUC Full": round(auc_f, 6),
            "AUC Ablated": round(auc_a, 6),
            "DeltaAUC": round(delta, 6),
            "z": round(z, 4),
            "p-value": round(p, 6),
            "Significant p<0.05": p < 0.05,
            "CI 95% low": round(ci_lo, 6),
            "CI 95% high": round(ci_hi, 6),
        })

    out_path = ROOT / "outputs" / "delong_significance_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for r in results:
        tag = "? significant" if r["Significant p<0.05"] else "? NOT significant"
        print(f"  {r['Feature Group']:35s}  ?AUC={r['DeltaAUC']:+.6f}  p={r['p-value']:.2e}  {tag}")
    print(f"\nSaved ? {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
