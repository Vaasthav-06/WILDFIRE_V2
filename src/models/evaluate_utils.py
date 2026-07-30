"""
src/models/evaluate_utils.py

Helper functions to calculate ML metrics and plot ROC/PR/SHAP/learning-curve charts,
plus a multi-model comparison artifact generator.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for script environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
    auc, precision_recall_curve, roc_curve,
    average_precision_score, brier_score_loss,
)
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.ensemble import VotingClassifier, StackingClassifier
import shap
from loguru import logger


# ── Shared style ─────────────────────────────────────────────────────────────
_PALETTE_BASE = "#4C9BE8"
_PALETTE_ENS  = "#F4845F"
_PALETTE_PROD = "#2ECC71"
_FIG_DPI = 150   # lower than 300 for speed; still sharp for reports


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    """Return accuracy / precision / recall / F1 / ROC-AUC / PR-AUC / Brier Score as a flat dict.

    Extended with PR-AUC (average_precision_score) and Brier Score (brier_score_loss)
    to address reviewer point 4: these are essential for imbalanced classification tasks
    like wildfire detection, where class imbalance makes AUC alone insufficient.
    """
    return {
        "Accuracy":   accuracy_score(y_true, y_pred),
        "Precision":  precision_score(y_true, y_pred, zero_division=0),
        "Recall":     recall_score(y_true, y_pred, zero_division=0),
        "F1 Score":   f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":    roc_auc_score(y_true, y_prob),
        "PR-AUC":     average_precision_score(y_true, y_prob),
        "Brier":      brier_score_loss(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-model evaluation charts (confusion matrix, ROC, PR)
# ─────────────────────────────────────────────────────────────────────────────

def generate_evaluation_artifacts(y_true, y_pred, y_prob, out_dir):
    """Save confusion-matrix, ROC-curve, and PR-curve PNGs to *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=_FIG_DPI)
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=_FIG_DPI)
    plt.close()

    # PR Curve
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=_FIG_DPI)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Learning curve (per-model overfitting visual diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def generate_learning_curve(model, X_train, y_train, out_dir, model_name: str = ""):
    """
    Compute and save a learning curve plot (train score vs. CV score across
    increasing training set sizes) using a time-respecting TimeSeriesSplit.

    A widening gap between train and val curves → overfitting.
    Both curves plateau low → underfitting.

    NOTE: learning_curve() works on the *training set only* — the held-out
    test set (X_test / y_test) never enters this computation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    MAX_CV_ROWS = 1500000
    if len(X_train) > MAX_CV_ROWS:
        logger.info(f"Learning curve for {model_name}: Subsampling from {len(X_train):,} to {MAX_CV_ROWS:,} rows for memory safety.")
        X_train_cv = X_train.tail(MAX_CV_ROWS)
        y_train_cv = y_train.tail(MAX_CV_ROWS)
    else:
        X_train_cv = X_train
        y_train_cv = y_train

    tscv = TimeSeriesSplit(n_splits=5)
    
    # Temporarily force model's internal n_jobs to 1 during learning_curve if applicable to avoid thread explosion
    original_n_jobs = getattr(model, "n_jobs", None)
    if hasattr(model, "set_params") and "n_jobs" in model.get_params():
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass
    if hasattr(model, "steps"):
        for step_name, step_obj in model.steps:
            if hasattr(step_obj, "set_params") and "n_jobs" in step_obj.get_params():
                try:
                    step_obj.set_params(n_jobs=1)
                except Exception:
                    pass

    # Reduce train_sizes for RandomForest to save memory/time
    is_rf = "Random Forest" in model_name or "rf" in model_name.lower()
    train_sizes_arr = np.linspace(0.1, 1.0, 4) if is_rf else np.linspace(0.1, 1.0, 8)
    total_fits = len(train_sizes_arr) * 5
    logger.info(f"Learning curve for {model_name}: Expecting {total_fits} fits across {len(train_sizes_arr)} train sizes.")

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train_cv,
            y_train_cv,
            cv=tscv,
            scoring="roc_auc",
            train_sizes=train_sizes_arr,
            n_jobs=1,        # 1 job to avoid thread explosion
            shuffle=False,   # preserve temporal order
        )
    except (MemoryError, Exception) as e:
        if isinstance(e, MemoryError) or "MemoryError" in str(type(e)):
            logger.warning(f"Learning curve skipped for {model_name} due to memory constraints: {e}")
        else:
            logger.error(f"Learning curve failed for {model_name}: {e}")
        return
    finally:
        if original_n_jobs is not None and hasattr(model, "set_params") and "n_jobs" in model.get_params():
            model.set_params(n_jobs=original_n_jobs)
        if hasattr(model, "steps"):
            for step_name, step_obj in model.steps:
                if step_name == "rf" or step_name == "lr":
                    if hasattr(step_obj, "set_params") and "n_jobs" in step_obj.get_params():
                        step_obj.set_params(n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#4C9BE8")
    ax.fill_between(train_sizes, val_mean - val_std,   val_mean + val_std,   alpha=0.1, color="#F4845F")
    ax.plot(train_sizes, train_mean, "o-", color="#4C9BE8", label="Train AUC")
    ax.plot(train_sizes, val_mean,   "s-", color="#F4845F", label="CV Val AUC (TimeSeriesSplit)")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Learning Curve — {model_name}" if model_name else "Learning Curve")
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve.png", dpi=_FIG_DPI)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP artifacts (per-model, with ensemble routing)
# ─────────────────────────────────────────────────────────────────────────────

def _shap_for_voting_classifier(model: VotingClassifier, X_sample) -> Optional[np.ndarray]:
    """
    Average SHAP values across all base estimators inside a VotingClassifier.
    Each base model uses the appropriate explainer (Tree or Linear).
    Returns averaged shap_values array or None on failure.
    NOTE: shap.TreeExplainer does not use GPU — this is expected behavior and
    not a regression; SHAP runs on CPU only regardless of model training device.
    """
    shap_arrays = []
    for name, estimator in model.named_estimators_.items():
        try:
            est_type = type(estimator).__name__.lower()
            if hasattr(estimator, "named_steps"):
                # Pipeline — unwrap to the final step
                inner = list(estimator.named_steps.values())[-1]
                est_type = type(inner).__name__.lower()
            else:
                inner = estimator

            if any(t in est_type for t in ["lgbm", "xgb", "catboost", "forest", "tree"]):
                explainer = shap.TreeExplainer(inner)
                sv = explainer.shap_values(X_sample)
                if isinstance(sv, list):
                    sv = sv[1]
            else:
                explainer = shap.LinearExplainer(inner, shap.maskers.Independent(X_sample))
                sv = explainer.shap_values(X_sample)

            shap_arrays.append(sv)
        except Exception as e:
            logger.warning(f"SHAP failed for VotingClassifier member '{name}': {e}")

    if not shap_arrays:
        return None
    return np.mean(shap_arrays, axis=0)


def _shap_for_stacking_classifier(model: StackingClassifier, X_sample) -> Optional[np.ndarray]:
    """
    SHAP for StackingClassifier: use LinearExplainer on the final meta-learner
    (LogisticRegression) operating on the stacked probability outputs.
    NOTE: shap.TreeExplainer does not use GPU — expected, not a regression.
    Falls back to a bar-chart-only result if the meta-learner SHAP fails.
    """
    try:
        final_est = model.final_estimator_
        # Transform X_sample through the base layer to get the stacked feature matrix
        X_meta = model.transform(X_sample)
        explainer = shap.LinearExplainer(final_est, shap.maskers.Independent(X_meta))
        sv = explainer.shap_values(X_meta)
        # Map shap values back to base-model names as feature names
        return sv, X_meta
    except Exception as e:
        logger.warning(f"SHAP failed for StackingClassifier meta-learner: {e}")
        return None, None


def generate_shap_artifacts(model, X_sample, model_name: str, out_dir):
    """
    Generate SHAP bar + summary plots for any supported model type.
    Supports: tree-based models, Pipeline-wrapped models, VotingClassifier,
    StackingClassifier, and LogisticRegression (via LinearExplainer).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Routing ──────────────────────────────────────────────────────────
        if isinstance(model, VotingClassifier):
            shap_values = _shap_for_voting_classifier(model, X_sample)
            if shap_values is None:
                logger.warning(f"Skipping SHAP plots for {model_name}: no members succeeded.")
                return
            X_plot = X_sample

        elif isinstance(model, StackingClassifier):
            shap_values, X_meta = _shap_for_stacking_classifier(model, X_sample)
            if shap_values is None:
                logger.warning(f"Skipping SHAP plots for {model_name}: stacking transform failed.")
                return
            # X_meta has columns = base estimator proba outputs, not original features
            X_plot = X_meta

        else:
            # Single model or Pipeline
            clf = model
            if hasattr(model, "named_steps"):
                clf = list(model.named_steps.values())[-1]

            model_type = type(clf).__name__.lower()

            if any(t in model_type for t in ["lgbm", "xgb", "catboost", "forest", "tree"]):
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # positive class
            else:
                explainer = shap.LinearExplainer(clf, shap.maskers.Independent(X_sample))
                shap_values = explainer.shap_values(X_sample)

            X_plot = X_sample

        # ── Plots ─────────────────────────────────────────────────────────────
        # Bar plot
        plt.figure()
        shap.summary_plot(shap_values, X_plot, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_bar.png", dpi=_FIG_DPI, bbox_inches="tight")
        plt.close()

        # Summary (beeswarm) plot
        plt.figure()
        shap.summary_plot(shap_values, X_plot, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=_FIG_DPI, bbox_inches="tight")
        plt.close()

    except Exception as e:
        logger.warning(f"Could not generate SHAP plots for {model_name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-model comparison artifacts (called once after all models evaluated)
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_artifacts(
    results: Dict[str, dict],
    y_true,
    out_dir,
    production_model_name: str = "",
) -> None:
    """
    Generate 5 cross-model comparison PNGs into *out_dir*.

    Parameters
    ----------
    results : dict
        Mapping of model_name -> result dict with keys:
          y_pred, y_prob, metrics (test), train_metrics,
          cv_scores (array), train_time_s, device, is_ensemble
    y_true : array-like
        Shared ground-truth labels (same test split for all models).
    out_dir : Path-like
        Output directory — will be created if absent.
    production_model_name : str
        Name of the model selected for production (will be starred in charts).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names        = list(results.keys())
    is_ensemble  = [results[n]["is_ensemble"] for n in names]
    test_metrics = [results[n]["metrics"] for n in names]
    train_auc    = [results[n]["train_metrics"]["ROC-AUC"] for n in names]
    test_auc     = [m["ROC-AUC"] for m in test_metrics]
    # Bug 1 fix: use stored cv_auc_mean when present (handles None sentinel for
    # skipped CV correctly).  Fall back to computing from cv_scores array only
    # when the stored value is absent or the array is non-empty.
    def _cv_mean_for(r: dict) -> float:
        stored = r.get("cv_auc_mean")
        if stored is not None:
            return float(stored)
        cv = r["cv_scores"]
        return float(np.mean(cv)) if len(cv) > 0 else 0.0
    cv_mean = [_cv_mean_for(results[n]) for n in names]

    bar_colors   = [_PALETTE_ENS if e else _PALETTE_BASE for e in is_ensemble]

    # ── (a) Grouped metrics bar chart ────────────────────────────────────────
    # Include PR-AUC and Brier Score if present in the metrics dict
    base_keys   = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    extra_keys  = [k for k in ["PR-AUC", "Brier"] if all(k in m for m in test_metrics)]
    metric_keys = base_keys + extra_keys
    n_models    = len(names)
    n_metrics   = len(metric_keys)
    x           = np.arange(n_models)
    width       = max(0.10, 0.80 / n_metrics)

    fig, ax = plt.subplots(figsize=(max(14, n_models * 2.0), 6))
    for i, mk in enumerate(metric_keys):
        vals = [test_metrics[j].get(mk, 0.0) for j in range(n_models)]
        ax.bar(x + i * width - (n_metrics - 1) * width / 2, vals, width, label=mk)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Metrics Comparison — All Candidates (incl. PR-AUC & Brier Score)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_comparison.png", dpi=_FIG_DPI)
    plt.close()

    # ── (b) Overlaid ROC curves ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    # Bug 5 fix: plt.cm.get_cmap(name, n) was removed in recent matplotlib.
    # Use matplotlib.colormaps[name] and pre-compute per-model colours instead.
    _tab10 = plt.colormaps["tab10"]
    colors = [_tab10(i / max(n_models - 1, 1)) for i in range(n_models)]
    for idx, name in enumerate(names):
        y_prob = results[name]["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        ls = "--" if is_ensemble[idx] else "-"
        ax.plot(fpr, tpr, ls, color=colors[idx], lw=1.6,
                label=f"{name} (AUC={roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Candidates")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves_overlay.png", dpi=_FIG_DPI)
    plt.close()

    # ── (c) Overlaid PR curves ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, name in enumerate(names):
        y_prob = results[name]["y_prob"]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = auc(rec, prec)
        ls = "--" if is_ensemble[idx] else "-"
        ax.plot(rec, prec, ls, color=colors[idx], lw=1.6,
                label=f"{name} (PR-AUC={pr_auc_val:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Candidates")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves_overlay.png", dpi=_FIG_DPI)
    plt.close()

    # ── (d) Horizontal AUC ranking bar chart ─────────────────────────────────
    sort_idx     = np.argsort(test_auc)
    sorted_names = [names[i] for i in sort_idx]
    sorted_auc   = [test_auc[i] for i in sort_idx]
    sorted_colors= [bar_colors[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(9, max(4, n_models * 0.55)))
    bars = ax.barh(sorted_names, sorted_auc, color=sorted_colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    for i, (sn, sa) in enumerate(zip(sorted_names, sorted_auc)):
        if sn == production_model_name:
            ax.text(sa + 0.005, i, " ★ PRODUCTION", va="center", fontsize=8,
                    color=_PALETTE_PROD, fontweight="bold")
    ax.set_xlabel("Test ROC-AUC")
    ax.set_title("Model AUC Ranking")
    ax.set_xlim(0, min(1.12, max(sorted_auc) + 0.1))
    ax.grid(True, axis="x", alpha=0.3)
    legend_patches = [
        mpatches.Patch(color=_PALETTE_BASE, label="Base model"),
        mpatches.Patch(color=_PALETTE_ENS,  label="Ensemble"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "auc_ranking.png", dpi=_FIG_DPI)
    plt.close()

    # ── (e) Train vs. Test AUC comparison (paired bars) ─────────────────────
    x      = np.arange(n_models)
    width  = 0.35
    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.4), 6))
    ax.bar(x - width / 2, train_auc, width, label="Train AUC", color="#4C9BE8", alpha=0.85)
    ax.bar(x + width / 2, test_auc,  width, label="Test AUC",  color="#F4845F", alpha=0.85)
    # Annotate gap where overfitting is visible
    for i in range(n_models):
        gap = train_auc[i] - test_auc[i]
        if gap > 0.05:
            ax.annotate(f"⚠ Δ{gap:.2f}", xy=(x[i], max(train_auc[i], test_auc[i]) + 0.01),
                        ha="center", fontsize=7, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.15)
    ax.set_title("Train vs. Test AUC — Overfitting Diagnostic")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_vs_test_auc.png", dpi=_FIG_DPI)
    plt.close()

    logger.info(f"Comparison artifacts saved to {out_dir}")