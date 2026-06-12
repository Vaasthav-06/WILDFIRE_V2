"""
scripts/train_models.py

Full model training and evaluation pipeline.
"""

import json
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.rule import Rule

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
    FEATURE_COLS,
    TARGET_COL,
    RANDOM_SEED,
)
from src.models.train import train_all_models, build_ensemble, compute_metrics
from src.models.evaluate import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
    plot_reliability_diagram,
    plot_shap_importance,
    run_ablation_study,
    plot_model_comparison,
    run_error_analysis,
)
from src.models.cross_validation import SpatialBlockCV

import lightgbm as lgb

logger.add("logs/train_models.log", rotation="50 MB", level="INFO")
console = Console()


def phase(title: str):
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))


def main():
    # ── Load data ────────────────────────────────────────────
    path = DATA_PROCESSED / "training_features.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows | {len(FEATURE_COLS)} features")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # ── Train all models with spatial CV ─────────────────────
    phase("Phase 4A: Training all models")
    results = train_all_models(df, cv_sample_size=200_000)

    results_path = RESULTS_DIR / "metrics" / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"CV results saved → {results_path}")

    # ── Collect predictions for plots ────────────────────────
    phase("Phase 4B: Generating evaluation plots")
    model_probs = {}
    for name in results:
        model = joblib.load(MODELS_DIR / f"{name}.pkl")
        if hasattr(model, "predict_proba"):
            model_probs[name] = model.predict_proba(X)[:, 1]

    # ── Build and evaluate ensemble ──────────────────────────
    phase("Phase 4C: Building ensemble")
    ensemble = build_ensemble(results)
    ensemble.fit(X, y)
    model_probs["ensemble"] = ensemble.predict_proba(X)[:, 1]
    joblib.dump(ensemble, MODELS_DIR / "ensemble.pkl")

    best_name = max(results, key=lambda n: results[n]["cv"]["roc_auc_mean"])
    best_model = joblib.load(MODELS_DIR / f"{best_name}.pkl")
    best_threshold = results[best_name]["threshold"]
    logger.info(f"Best model: {best_name} (threshold={best_threshold})")

    # ── Evaluation plots ─────────────────────────────────────
    plot_roc_curves(model_probs, y.values)
    plot_pr_curves(model_probs, y.values)
    plot_reliability_diagram(model_probs, y.values)
    plot_model_comparison(results)

    y_pred_best = (model_probs[best_name] >= best_threshold).astype(int)
    plot_confusion_matrix(y.values, y_pred_best, best_name, best_threshold)

    shap_sample = X.sample(5000, random_state=RANDOM_SEED)
    plot_shap_importance(best_model, shap_sample, best_name)

    # ── Ablation study ───────────────────────────────────────
    phase("Phase 4D: Ablation study")
    ablation_sample = df.sample(min(100_000, len(df)), random_state=RANDOM_SEED)
    y_abl = ablation_sample[TARGET_COL]

    ablation_df = run_ablation_study(
        model_class=lgb.LGBMClassifier,
        df=ablation_sample,
        y=y_abl,
        model_kwargs={"n_estimators": 200, "random_state": RANDOM_SEED, "verbose": -1},
    )

    # ── Error analysis ───────────────────────────────────────
    phase("Phase 4E: Error analysis")
    run_error_analysis(
        df=df,
        y_true=y.values,
        y_prob=model_probs[best_name],
        threshold=best_threshold,
        model_name=best_name,
    )

    # ── Save production model ─────────────────────────────────
    production = {
        "model": ensemble,
        "threshold": best_threshold,
        "features": FEATURE_COLS,
        "best_single": best_name,
        "cv_results": results,
    }
    joblib.dump(production, MODELS_DIR / "production_model.pkl")
    phase("Final Summary")

    logger.info("FINAL MODEL COMPARISON")
    for name, res in sorted(
        results.items(), key=lambda x: x[1]["cv"]["roc_auc_mean"], reverse=True
    ):
        cv = res["cv"]
        logger.info(
            f"{name:<25} AUC={cv['roc_auc_mean']:.4f}±{cv['roc_auc_std']:.4f} F1={cv['f1_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
