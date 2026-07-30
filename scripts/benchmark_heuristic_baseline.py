"""
scripts/benchmark_heuristic_baseline.py

Benchmarks the ML models against heuristic and physics-based baselines
on the 2024-2025 test set:
  1. Canadian Forest Fire Weather Index (FWI) — van Wagner (1987)
     PRIMARY credible operational baseline. The FWI is used operationally
     by NFMS India and many international fire agencies.
  2. Two-feature Logistic Regression (temp, wind) — lower bound baseline
  3. Simple Meteorological Rule-based Threshold — lower bound baseline

IMPORTANT NOTE ON FWI COMPUTATION:
FWI requires day-over-day state per grid cell. Computing it for all 9.4M rows
in training_features.csv would take hours. This script runs FWI on a
representative SUBSAMPLE of unique grid cells from the test set only.
Results are reported with this caveat.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, accuracy_score,
                              precision_score, recall_score, f1_score, roc_curve)
import joblib

ROOT      = Path(__file__).resolve().parent.parent
DATA_FEAT = ROOT / "data" / "processed" / "training_features.csv"
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FWI subsample: number of unique (lat, lon) cells to evaluate
# (full test set has ~thousands of unique cells; 500 gives stable statistics)
FWI_N_CELLS = 500
SEED = 42


def get_metrics(y_true, y_prob, y_pred=None, threshold: float = 0.5):
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    auc_roc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    pr_auc  = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0
    brier   = brier_score_loss(y_true, y_prob) if y_prob is not None else 0.5
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    return {"AUC": auc_roc, "PR-AUC": pr_auc, "Brier": brier,
            "Acc": acc, "Prec": prec, "Rec": rec, "F1": f1}


def main():
    print("=" * 65)
    print("Heuristic Baseline Benchmarks on Test Set (2024-2025)")
    print("=" * 65)

    # 1. Load Data
    print("\n[1/4] Loading features ...")
    cols = ["acq_date", "latitude", "longitude",
            "temp", "wind", "humidity", "vpd", "fire_detected"]
    df = pd.read_csv(DATA_FEAT, usecols=cols)
    df["acq_date"] = pd.to_datetime(df["acq_date"])

    train_df = df[df["acq_date"].dt.year < 2024].reset_index(drop=True)
    test_df  = df[df["acq_date"].dt.year >= 2024].reset_index(drop=True)

    y_train = train_df["fire_detected"].values
    y_test  = test_df["fire_detected"].values
    print(f"   Train: {len(train_df):,}  |  Test: {len(test_df):,}  "
          f"|  Test fire rate: {y_test.mean():.4f}")

    results = {}

    # 2. Two-Feature LR
    print("\n[2/4] Training 2-feature LR baseline (temp, wind) ...")
    X_train_2f = train_df[["temp", "wind"]].values
    X_test_2f  = test_df[["temp", "wind"]].values
    lr_2f = LogisticRegression(random_state=SEED, max_iter=500)
    lr_2f.fit(X_train_2f, y_train)
    lr_prob = lr_2f.predict_proba(X_test_2f)[:, 1]
    lr_pred = lr_2f.predict(X_test_2f)
    results["2-Feature LR (Temp + Wind)"] = get_metrics(y_test, lr_prob, lr_pred)
    m = results["2-Feature LR (Temp + Wind)"]
    print(f"   AUC={m['AUC']:.4f}  PR-AUC={m['PR-AUC']:.4f}  "
          f"Recall={m['Rec']:.4f}  Precision={m['Prec']:.4f}")

    # 3. Rule-based threshold
    print("\n[3/4] Evaluating rule-based threshold baseline ...")
    temp_mean = test_df["temp"].mean()
    is_kelvin = temp_mean > 150
    if is_kelvin:
        rule_mask = ((test_df["temp"] > (30 + 273.15)) &
                     (test_df["humidity"] < 35) &
                     (test_df["vpd"] > 2.0))
    else:
        rule_mask = ((test_df["temp"] > 30) &
                     (test_df["humidity"] < 35) &
                     (test_df["vpd"] > 2.0))
    rule_pred = rule_mask.astype(int).values
    results["Threshold Heuristic (Temp>30°C, RH<35%, VPD>2kPa)"] = get_metrics(
        y_test, rule_pred.astype(float), rule_pred)
    m = results["Threshold Heuristic (Temp>30°C, RH<35%, VPD>2kPa)"]
    print(f"   AUC={m['AUC']:.4f}  Recall={m['Rec']:.4f}  Precision={m['Prec']:.4f}")

    # 4. FWI Baseline
    print(f"\n[4/4] Computing FWI baseline on {FWI_N_CELLS} randomly sampled cells ...")
    print("   (FWI requires day-over-day recurrence per cell — computing on subsample)")

    # Add FWI module to path
    sys.path.insert(0, str(ROOT))
    from src.baselines.fwi import compute_fwi_for_dataframe

    # Sample unique cells
    rng = np.random.default_rng(SEED)
    unique_cells = test_df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    n_sample = min(FWI_N_CELLS, len(unique_cells))
    sample_cells = unique_cells.sample(n=n_sample, random_state=SEED)
    # Faster: merge-based cell mask
    sample_cells["_in_sample"] = True
    test_sample = test_df.merge(sample_cells, on=["latitude", "longitude"], how="inner")
    test_sample = test_sample.sort_values(["latitude", "longitude", "acq_date"]).reset_index(drop=True)
    print(f"   Sampled test subset: {len(test_sample):,} rows across {n_sample} cells")

    # Compute FWI (precip column absent from training_features.csv — using 0mm assumption)
    print("   Running FWI computation (no precip column available — using 0mm/day assumption) ...")
    test_fwi = compute_fwi_for_dataframe(
        test_sample,
        lat_col="latitude", lon_col="longitude",
        date_col="acq_date", temp_col="temp",
        rh_col="humidity", wind_col="wind",
        precip_col=None,  # Not available in training_features.csv
    )
    print(f"   FWI computed. Range: [{test_fwi['fwi'].min():.1f}, {test_fwi['fwi'].max():.1f}]")

    y_fwi_true = test_fwi["fire_detected"].values
    fwi_scores = test_fwi["fwi"].values

    if y_fwi_true.sum() == 0:
        print("   WARNING: No fire events in sampled cells — cannot compute AUC. Try larger FWI_N_CELLS.")
        results["FWI (van Wagner 1987)"] = {k: float("nan") for k in
                                             ["AUC", "PR-AUC", "Brier", "Acc", "Prec", "Rec", "F1"]}
    else:
        # Threshold-optimize FWI using Youden's J (same as Task 4, for fairness)
        fpr, tpr, threshs = roc_curve(y_fwi_true, fwi_scores)
        best_idx = np.argmax(tpr - fpr)
        best_t   = float(threshs[best_idx])
        fwi_pred = (fwi_scores >= best_t).astype(int)
        print(f"   FWI Youden's J threshold: {best_t:.2f}")

        # Min-Max normalize FWI to [0,1] for probabilistic metrics (PR-AUC, Brier)
        fwi_min, fwi_max = fwi_scores.min(), fwi_scores.max()
        fwi_norm = (fwi_scores - fwi_min) / (fwi_max - fwi_min + 1e-8)

        results["FWI (van Wagner 1987, Youden threshold)"] = get_metrics(
            y_fwi_true, fwi_norm, fwi_pred)
        m = results["FWI (van Wagner 1987, Youden threshold)"]
        print(f"   FWI AUC={m['AUC']:.4f}  PR-AUC={m['PR-AUC']:.4f}  "
              f"Recall={m['Rec']:.4f}  Precision={m['Prec']:.4f}  Brier={m['Brier']:.4f}")
        print(f"   (Note: {FWI_N_CELLS}-cell subsample, no precipitation data — "
              f"FWI slightly overestimated on wet days)")

    # Save results
    rows = [{"Model": k, **v} for k, v in results.items()]
    df_out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "baseline_benchmark_results.csv"
    df_out.to_csv(csv_path, index=False)

    print("\n" + "=" * 65)
    print("RESULTS TABLE")
    print("=" * 65)
    print(df_out.to_string(index=False))
    print(f"\nSaved -> {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
