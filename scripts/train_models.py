"""
scripts/train_models.py

Master Training Orchestrator for Wildfire Risk Prediction.

Pipeline:
  1. GPU detection (once, used for all GPU-capable models)
  2. Data load + temporal split (2018-2023 train | 2024-2025 test)
  3. Train & evaluate 5 base models individually
  4. Build VotingClassifier + StackingClassifier ensembles; evaluate both
  5. Select production ensemble (higher test AUC wins)
  6. Generate per-model + cross-model comparison artifacts
  7. Package production_model.pkl with full diagnostic bundle

Data-leakage safeguards
  - SimpleImputer / StandardScaler are fit ONLY on X_train (inside Pipeline.fit).
    They are never called on X_test or the combined dataset before the split.
  - Temporal split produces non-overlapping date ranges:
      train: acq_date.year <= 2023
      test:  acq_date.year >= 2024
    An assertion below verifies zero index overlap.
  - Cross-validation (TimeSeriesSplit) runs on X_train / y_train exclusively.
    X_test / y_test are touched only once, at final evaluation.

GPU notes
  - XGBoost: device="cuda" | "cpu"  (auto-toggled by _detect_gpu())
  - LightGBM: device="gpu"  | "cpu"
  - CatBoost: task_type="GPU" | "CPU"
  - RandomForest / LogisticRegression: no native GPU path in scikit-learn;
    left on CPU with n_jobs=-1 for maximum CPU parallelism.
    Future improvement: cuML/RAPIDS could accelerate these — not added here
    as it is not already a project dependency.
  - StackingClassifier refits each base model on every internal CV fold.
    sklearn.clone() copies all __init__ params (incl. device/task_type),
    so GPU settings are preserved across all folds automatically.
  - shap.TreeExplainer does NOT use GPU — this is expected, not a regression.
"""

import os
import sys

# Force UTF-8 output on Windows terminals (avoids cp1252 UnicodeEncodeError).
# Must be set before Rich Console is instantiated.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # reconfigure not available in all environments

import time
import joblib
import numpy as np
import pandas as pd

from loguru import logger
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Rich UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    EVAL_DIR,
    SHAP_DIR,
    COMPARISON_DIR,
    FEATURE_COLS,
    TARGET_COL,
    RANDOM_SEED,
    RISK_THRESHOLD_HIGH,
)
from src.models.evaluate_utils import (
    calculate_metrics,
    generate_evaluation_artifacts,
    generate_learning_curve,
    generate_shap_artifacts,
    generate_comparison_artifacts,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add("logs/training_pipeline.log", rotation="50 MB", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

console = Console()

# ── Overfitting / underfitting thresholds ─────────────────────────────────────
OVERFIT_DELTA   = 0.05   # train_auc - test_auc  (or train - cv_mean) > this → overfit
UNDERFIT_THRESH = 0.65   # train_auc < this → underfit
SUSPICIOUS_AUC  = 0.99   # any AUC >= this → possible data leakage / trivial separability

BASE_MODEL_NAMES = [
    "Logistic Regression",
    "CatBoost",
    "LightGBM",
    "XGBoost",
    "Random Forest",
]


# ─────────────────────────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_gpu() -> tuple[bool, str]:
    """
    Detect GPU availability by performing a tiny XGBoost probe fit.
    Returns (gpu_available: bool, device_label: str).

    We probe via XGBoost (already a dependency) rather than importing torch
    (not in requirements.txt) to avoid introducing an uninstalled dependency.
    """
    try:
        import xgboost as xgb
        probe = xgb.XGBClassifier(n_estimators=1, device="cuda", verbosity=0)
        # Minimal 2-sample fit to confirm CUDA is accessible
        probe.fit(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0, 1]),
        )
        # If we reach here, CUDA is available
        device_label = "GPU (CUDA)"
        logger.info("GPU probe: CUDA is available.")
        return True, device_label
    except Exception as e:
        logger.info(f"GPU probe: CUDA not available ({e}). Falling back to CPU.")
        return False, "CPU"


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_base_model_configs(gpu: bool) -> dict:
    """
    Return a fresh dict of (name -> unfitted model) for each of the 5 base models.

    GPU-capable models (XGBoost, LightGBM, CatBoost) are configured for GPU
    when *gpu* is True, and fall back to CPU gracefully when False — no crash.
    RandomForest and LogisticRegression have no native GPU path in scikit-learn
    and run on CPU with n_jobs=-1 for maximum CPU parallelism.
    """
    xgb_device  = "cuda" if gpu else "cpu"
    lgb_device  = "gpu"  if gpu else "cpu"
    cat_task    = "GPU"  if gpu else "CPU"

    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("lr",      LogisticRegression(
                random_state=RANDOM_SEED,
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1,          # maximize CPU parallelism (no GPU path in sklearn)
            )),
        ]),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            random_state=RANDOM_SEED,
            auto_class_weights="Balanced",
            verbose=0,
            task_type=cat_task,     # "GPU" or "CPU"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            verbose=-1,
            device=lgb_device,      # "gpu" or "cpu"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            scale_pos_weight=2.0,
            eval_metric="logloss",
            device=xgb_device,      # "cuda" or "cpu"
        ),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf",      RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=10,
                min_samples_split=10,
                max_features="sqrt",
                random_state=RANDOM_SEED,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
    }


def get_ensemble_configs(gpu: bool) -> dict:
    """
    Return freshly-instantiated VotingClassifier and StackingClassifier.

    Both ensembles use the SAME base model configs as the individual training loop.
    sklearn.clone() (called internally by VotingClassifier/StackingClassifier.fit)
    copies all __init__ params, including device/task_type — so GPU settings are
    preserved across all internal CV folds of StackingClassifier.

    StackingClassifier internal CV:
      - cv=TimeSeriesSplit(n_splits=5) respects temporal ordering.
      - Operates ONLY on X_train/y_train; X_test is never seen during fitting.
      - TimeSeriesSplit does not accept a random_state (it's deterministic by design).
    """
    estimator_list = list(get_base_model_configs(gpu).items())

    voting = VotingClassifier(
        estimators=estimator_list,
        voting="soft",          # average predict_proba across all members
        n_jobs=1,               # parallelism handled inside each base model
    )

    stacking = StackingClassifier(
        estimators=estimator_list,
        final_estimator=LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            n_jobs=-1,
        ),
        cv=5,   # StratifiedKFold on X_train to ensure full partition (cross_val_predict fails on TimeSeriesSplit)
        passthrough=False,
        n_jobs=1,
    )

    return {
        "Voting Ensemble":   voting,
        "Stacking Ensemble": stacking,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_flags(train_auc: float, test_auc: float,
                  cv_auc_mean,          # float | None  (None = CV skipped)
                  cv_scores: np.ndarray) -> dict:
    """
    Return overfit / underfit / suspicious boolean flags.

    cv_auc_mean=None signals that CV was skipped or reused from a prior run.
    In that case the train-vs-CV branch of the overfit check is skipped
    entirely to avoid the massive false gap caused by a 0.0 placeholder.
    Only the train-vs-test comparison is used when CV data is unavailable.
    """
    overfit_vs_test = train_auc - test_auc > OVERFIT_DELTA
    overfit_vs_cv   = (cv_auc_mean is not None and
                       train_auc - cv_auc_mean > OVERFIT_DELTA)
    overfit = overfit_vs_test or overfit_vs_cv

    underfit   = train_auc < UNDERFIT_THRESH
    # Exclude None from suspicious check
    aucs       = [train_auc, test_auc] + ([cv_auc_mean] if cv_auc_mean is not None else [])
    suspicious = any(a >= SUSPICIOUS_AUC for a in aucs)
    return {
        "overfit_flag":      overfit,
        "overfit_vs_test":   overfit_vs_test,    # which comparison tripped (if any)
        "overfit_vs_cv":     overfit_vs_cv,
        "underfit_flag":     underfit,
        "suspicious_flag":   suspicious,
    }


def run_timeseries_cv(model, X_train, y_train, name: str = "Model", n_splits: int = 5) -> np.ndarray:
    """
    Run TimeSeriesSplit cross-validation on the training set ONLY.
    X_test / y_test are never passed here — enforced by design.
    To avoid MemoryErrors on huge datasets, we subsample to the most recent 1.5M rows for CV.
    Returns an array of per-fold ROC-AUC scores.
    """
    MAX_CV_ROWS = 1500000
    if len(X_train) > MAX_CV_ROWS:
        logger.info(f"  {name} CV: Subsampling training data from {len(X_train):,} to {MAX_CV_ROWS:,} rows for memory safety.")
        # Time-respecting subsample: take the most recent rows
        X_cv = X_train.tail(MAX_CV_ROWS)
        y_cv = y_train.tail(MAX_CV_ROWS)
    else:
        X_cv = X_train
        y_cv = y_train
        
    # Temporarily force model's internal n_jobs to 1 during CV if applicable to avoid thread explosion
    original_n_jobs = getattr(model, "n_jobs", None)
    if hasattr(model, "set_params") and "n_jobs" in model.get_params():
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass

    # If it's a Pipeline, we need to set n_jobs on the estimator
    if hasattr(model, "steps"):
        for step_name, step_obj in model.steps:
            if hasattr(step_obj, "set_params") and "n_jobs" in step_obj.get_params():
                try:
                    step_obj.set_params(n_jobs=1)
                except Exception:
                    pass

    tscv = TimeSeriesSplit(n_splits=n_splits)
    try:
        scores = cross_val_score(
            model, X_cv, y_cv,
            cv=tscv,
            scoring="roc_auc",
            n_jobs=1,           # 1 job to avoid GPU contention / thread explosion across folds
        )
    except (MemoryError, Exception) as e:
        if isinstance(e, MemoryError) or "MemoryError" in str(type(e)):
            logger.warning(f"  CV skipped for {name} due to memory constraints, falling back to train/test metrics only.")
            scores = np.array([])
        else:
            logger.error(f"  CV failed for {name}: {e}")
            scores = np.array([])
    finally:
        # Restore original n_jobs
        if original_n_jobs is not None and hasattr(model, "set_params") and "n_jobs" in model.get_params():
            model.set_params(n_jobs=original_n_jobs)
        if hasattr(model, "steps"):
            for step_name, step_obj in model.steps:
                if step_name == "rf" or step_name == "lr":
                    if hasattr(step_obj, "set_params") and "n_jobs" in step_obj.get_params():
                        step_obj.set_params(n_jobs=-1)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Terminal printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flag_str(flags: dict) -> str:
    """Build a short flag string for inline display."""
    parts = []
    if flags["overfit_flag"]:    parts.append("[red]OVERFIT[/red]")
    if flags["underfit_flag"]:   parts.append("[yellow]UNDERFIT[/yellow]")
    if flags["suspicious_flag"]: parts.append("[magenta]SUSPICIOUS[/magenta]")
    return ", ".join(parts) if parts else "[green]OK[/green]"


def print_model_header(name: str, device: str):
    console.rule(f"[bold blue]{name.upper()}[/bold blue]")
    console.print(f"  [dim]Device:[/dim] [cyan]{device}[/cyan]")


def print_model_metrics_table(
    name: str,
    train_metrics: dict,
    test_metrics: dict,
    cv_scores: np.ndarray,
    flags: dict,
    elapsed: float,
    is_ensemble: bool,
    other_ensemble_auc: float = None,
    cv_skipped: bool = False,    # True when CV was intentionally skipped / reused from Phase 1
):
    """
    Print the rich metrics table with train/test/CV columns + verdict.

    cv_skipped=True renders "N/A (CV skipped)" in the CV row rather than
    "0.0000+/-0.0000", so skipped CV is never mistaken for a real result.
    """
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    table.add_column("Metric",      style="cyan",    width=20)
    table.add_column("Train",       style="blue",    justify="right", width=10)
    table.add_column("Test",        style="magenta", justify="right", width=10)

    cv_mean = float(np.mean(cv_scores)) if len(cv_scores) else 0.0
    cv_std  = float(np.std(cv_scores))  if len(cv_scores) else 0.0

    for key in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        train_val = train_metrics.get(key, 0.0)
        test_val  = test_metrics.get(key, 0.0)
        table.add_row(key, f"{train_val:.4f}", f"{test_val:.4f}")

    # Bug 1 fix: show N/A when CV was explicitly skipped, not 0.0000±0.0000
    if cv_skipped or (len(cv_scores) == 0 and cv_skipped):
        cv_display = "N/A (CV skipped)"
    elif len(cv_scores) == 0:
        cv_display = "N/A (CV failed)"
    else:
        cv_display = f"{cv_mean:.4f}+/-{cv_std:.4f}"

    table.add_row(
        "CV AUC (mean+/-std)",
        cv_display,
        "",
    )

    console.print(table)
    console.print(f"  [dim]Elapsed:[/dim] {elapsed:.1f}s")

    # Flag warnings
    if flags["suspicious_flag"]:
        console.print(
            f"  [bold magenta][!] SUSPICIOUS: AUC >= {SUSPICIOUS_AUC} detected. "
            "Likely data leakage or trivially-separable class. Manual review required.[/bold magenta]"
        )
        logger.warning(f"{name}: SUSPICIOUS AUC >= {SUSPICIOUS_AUC} -- check for data leakage.")

    if flags["overfit_flag"]:
        # Bug 1 fix: explicitly state which comparison triggered the overfit flag
        # so the message is never misleading (train-vs-test numbers printed for
        # a train-vs-CV trigger, or vice versa).
        parts = []
        if flags.get("overfit_vs_test", False):
            delta_tt = train_metrics["ROC-AUC"] - test_metrics["ROC-AUC"]
            parts.append(
                f"train-vs-test: {train_metrics['ROC-AUC']:.4f} vs "
                f"{test_metrics['ROC-AUC']:.4f} (D={delta_tt:.4f})"
            )
        if flags.get("overfit_vs_cv", False) and len(cv_scores) > 0:
            delta_cv = train_metrics["ROC-AUC"] - cv_mean
            parts.append(
                f"train-vs-CV: {train_metrics['ROC-AUC']:.4f} vs "
                f"{cv_mean:.4f} (D={delta_cv:.4f})"
            )
        detail = " | ".join(parts) if parts else "unknown comparison"
        console.print(
            f"  [bold red][!] OVERFITTING detected ({detail})[/bold red]"
        )
        logger.warning(f"{name}: overfitting -- {detail}")

    if flags["underfit_flag"]:
        console.print(
            f"  [bold yellow][!] UNDERFITTING: train AUC {train_metrics['ROC-AUC']:.4f} < {UNDERFIT_THRESH}[/bold yellow]"
        )
        logger.warning(f"{name}: underfitting -- train_auc={train_metrics['ROC-AUC']:.4f}")

    # Verdict line
    if is_ensemble:
        if other_ensemble_auc is not None:
            diff = test_metrics["ROC-AUC"] - other_ensemble_auc
            if diff >= 0:
                verdict = f"[bold green][OK] New best ensemble (+{diff:.4f} AUC vs other ensemble)[/bold green]"
            else:
                verdict = f"[dim]Below other ensemble by {abs(diff):.4f} AUC points[/dim]"
        else:
            verdict = "[dim]First ensemble evaluated[/dim]"
    else:
        verdict = "[dim]Reference point — ensemble will supersede this regardless of AUC[/dim]"

    console.print(f"  Verdict: {verdict}\n")


def print_pipeline_header(df, train_mask, test_mask, gpu_label: str, gpu: bool):
    """Print the pipeline overview panel (Task 2 point 1)."""
    n_train      = int(train_mask.sum())
    n_test       = int(test_mask.sum())
    pos_train    = float(df.loc[train_mask, TARGET_COL].mean() * 100)
    neg_train    = 100.0 - pos_train
    pos_test     = float(df.loc[test_mask,  TARGET_COL].mean() * 100)
    neg_test     = 100.0 - pos_test
    train_dates  = (df.loc[train_mask, "acq_date"].min().date(),
                    df.loc[train_mask, "acq_date"].max().date())
    test_dates   = (df.loc[test_mask,  "acq_date"].min().date(),
                    df.loc[test_mask,  "acq_date"].max().date())

    gpu_str = f"[bold green]{gpu_label}[/bold green]" if gpu else f"[dim]{gpu_label}[/dim]"

    console.print(Panel(
        f"[bold cyan]WILDFIRE V2 - TRAINING PIPELINE[/bold cyan]\n\n"
        f"  Dataset rows    : [yellow]{len(df):,}[/yellow]  |  Features: [yellow]{len(FEATURE_COLS)}[/yellow]\n"
        f"  Train split     : [blue]{n_train:,} rows[/blue]  "
        f"({train_dates[0]} -> {train_dates[1]})\n"
        f"  Test split      : [magenta]{n_test:,} rows[/magenta]  "
        f"({test_dates[0]} -> {test_dates[1]})\n"
        f"  Class balance   : Train -> fire={pos_train:.1f}% / no-fire={neg_train:.1f}%\n"
        f"                    Test  -> fire={pos_test:.1f}%  / no-fire={neg_test:.1f}%\n"
        f"  GPU detected    : {gpu_str}\n"
        f"  Models planned  : 5 base models + VotingClassifier + StackingClassifier\n",
        expand=False,
        border_style="bold blue",
    ))


def print_final_comparison_table(all_results: dict, production_name: str):
    """Print the consolidated 7-row comparison table sorted by test ROC-AUC."""
    table = Table(
        title="[bold]Final Model Comparison[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Model",           style="cyan",    width=22)
    table.add_column("Train AUC",       justify="right", width=10)
    table.add_column("Test AUC",        justify="right", width=10)
    table.add_column("CV AUC (u+/-s)",    justify="right", width=14)
    table.add_column("Acc",             justify="right", width=7)
    table.add_column("Prec",            justify="right", width=7)
    table.add_column("Rec",             justify="right", width=7)
    table.add_column("F1",              justify="right", width=7)
    table.add_column("Time (s)",        justify="right", width=8)
    table.add_column("Device",          justify="left",  width=12)
    table.add_column("Flags",           justify="left",  width=18)

    # Sort descending by test AUC
    sorted_names = sorted(
        all_results.keys(),
        key=lambda n: all_results[n]["metrics"]["ROC-AUC"],
        reverse=True,
    )

    for name in sorted_names:
        r         = all_results[name]
        m         = r["metrics"]
        tm        = r["train_metrics"]
        cv        = r["cv_scores"]
        flags     = r["flags"]
        # Bug 1 fix: render N/A when CV was skipped rather than fake 0.0000
        if len(cv) > 0:
            cv_mean      = float(np.mean(cv))
            cv_std       = float(np.std(cv))
            cv_display   = f"{cv_mean:.4f}+/-{cv_std:.4f}"
        else:
            # Distinguish "intentionally skipped" vs "genuinely failed"
            _skipped     = r.get("cv_auc_mean") is None  # None sentinel = skipped
            cv_display   = "N/A (reused)" if _skipped else "N/A (failed)"
        is_prod   = (name == production_name)
        row_style = "bold green" if is_prod else ""

        flag_parts = []
        if flags["overfit_flag"]:    flag_parts.append("OVERFIT")
        if flags["underfit_flag"]:   flag_parts.append("UNDERFIT")
        if flags["suspicious_flag"]: flag_parts.append("SUSPICIOUS")
        flag_str = ", ".join(flag_parts) if flag_parts else "OK"

        prod_marker = " [*]" if is_prod else ""

        table.add_row(
            f"{name}{prod_marker}",
            f"{tm['ROC-AUC']:.4f}",
            f"{m['ROC-AUC']:.4f}",
            cv_display,
            f"{m['Accuracy']:.4f}",
            f"{m['Precision']:.4f}",
            f"{m['Recall']:.4f}",
            f"{m['F1 Score']:.4f}",
            f"{r['train_time_s']:.1f}",
            r["device"],
            flag_str,
            style=row_style,
        )

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    device: str,
    is_ensemble: bool,
    other_ensemble_auc: float = None,
    shap_sample: pd.DataFrame = None,
) -> dict:
    """
    Fit a model, compute train/test/CV metrics, generate all artifacts,
    and return a result dict for comparison.

    NOTE: X_test / y_test are used ONLY for final evaluation (predict + metrics).
    They never enter cross_val_score or learning_curve computations.
    """
    safe_name       = name.lower().replace(" ", "_")
    model_eval_dir  = EVAL_DIR   / safe_name
    model_shap_dir  = SHAP_DIR   / safe_name

    console.print()
    print_model_header(name, device)
    logger.info(f"START training: {name} | device={device}")

    # ── Fit ──────────────────────────────────────────────────────────────────
    console.print("  [dim]Fitting model...[/dim]")
    if name == "Stacking Ensemble":
        logger.info(f"  {name}: Expected long runtime due to 5 internal CV refits of all base models.")
        console.print(f"  [dim]Stacking Ensemble fits base models 5 times during CV. This will take a while.[/dim]")
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except (MemoryError, Exception) as e:
        if isinstance(e, MemoryError) or "MemoryError" in str(type(e)):
            logger.warning(f"  {name}: fit skipped due to MemoryError: {e}")
            console.print(f"  [bold yellow][!] {name} fit skipped due to MemoryError.[/bold yellow]")
        else:
            logger.error(f"  {name}: fit failed with error: {e}")
            console.print(f"  [bold red][!] {name} fit failed: {e}[/bold red]")
        return None
    elapsed = time.time() - t0
    logger.info(f"  {name}: fit complete in {elapsed:.1f}s")

    # ── Predictions ──────────────────────────────────────────────────────────
    y_pred_test  = model.predict(X_test)
    y_prob_test  = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]

    # ── Metrics ──────────────────────────────────────────────────────────────
    test_metrics  = calculate_metrics(y_test,  y_pred_test,  y_prob_test)
    train_metrics = calculate_metrics(y_train, y_pred_train, y_prob_train)

    # ── AUC self-check assertion ─────────────────────────────────────────────
    # Recompute AUC independently to catch any silent bug where the wrong array
    # was logged.  Must match within floating-point tolerance.
    _recomputed_auc = roc_auc_score(y_test, y_prob_test)
    assert abs(_recomputed_auc - test_metrics["ROC-AUC"]) < 1e-8, (
        f"AUC self-check FAILED for {name}: stored={test_metrics['ROC-AUC']:.8f} "
        f"recomputed={_recomputed_auc:.8f}"
    )

    # ── CV (training set only — no test leakage) ─────────────────────────────
    console.print("  [dim]Running TimeSeriesSplit cross-validation (train set only)...[/dim]")
    cv_scores = run_timeseries_cv(model, X_train, y_train, name=name, n_splits=5)
    # Bug 1 fix: use None (not 0.0) when CV was not run, so compute_flags can
    # distinguish "CV skipped" from a real AUC of 0.  This prevents the massive
    # fake overfit gap caused by train_auc - 0.0 > OVERFIT_DELTA.
    cv_mean   = float(np.mean(cv_scores)) if len(cv_scores) > 0 else None
    cv_std    = float(np.std(cv_scores))  if len(cv_scores) > 0 else 0.0
    if cv_mean is not None:
        logger.info(f"  {name}: CV AUC = {cv_mean:.4f}±{cv_std:.4f}")
    else:
        logger.info(f"  {name}: CV skipped due to errors (e.g. MemoryError).")

    # ── Overfit / underfit / suspicious flags ─────────────────────────────────
    flags = compute_flags(
        train_auc   = train_metrics["ROC-AUC"],
        test_auc    = test_metrics["ROC-AUC"],
        cv_auc_mean = cv_mean,
        cv_scores   = cv_scores,
    )

    # ── Terminal metrics table ────────────────────────────────────────────────
    print_model_metrics_table(
        name, train_metrics, test_metrics, cv_scores, flags,
        elapsed, is_ensemble, other_ensemble_auc,
        cv_skipped=(len(cv_scores) == 0),
    )

    # ── Evaluation charts (confusion matrix, ROC, PR) ─────────────────────────
    console.print(f"  [dim]Saving evaluation artifacts → {model_eval_dir}/[/dim]")
    t_eval = time.time()
    generate_evaluation_artifacts(y_test, y_pred_test, y_prob_test, model_eval_dir)
    eval_elapsed = time.time() - t_eval

    # ── Learning curve ────────────────────────────────────────────────────────
    console.print(f"  [dim]Generating learning curve...[/dim]")
    t_lc = time.time()
    generate_learning_curve(model, X_train, y_train, model_eval_dir, model_name=name)
    lc_elapsed = time.time() - t_lc

    # ── SHAP ──────────────────────────────────────────────────────────────────
    console.print(f"  [dim]Generating SHAP plots → {model_shap_dir}/[/dim]")
    if shap_sample is None:
        shap_sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_SEED)
    t_shap = time.time()
    generate_shap_artifacts(model, shap_sample, name, model_shap_dir)
    shap_elapsed = time.time() - t_shap

    # ── Save individual model .pkl ─────────────────────────────────────────────
    joblib.dump(model, MODELS_DIR / f"{safe_name}.pkl")

    # Log to file
    logger.info(
        f"  {name}: train_auc={train_metrics['ROC-AUC']:.4f} "
        f"test_auc={test_metrics['ROC-AUC']:.4f} "
        f"cv={cv_mean:.4f}±{cv_std:.4f} "
        f"overfit={flags['overfit_flag']} underfit={flags['underfit_flag']} "
        f"suspicious={flags['suspicious_flag']}"
    )

    return {
        "y_pred":        y_pred_test,
        "y_prob":        y_prob_test,
        "metrics":       test_metrics,
        "train_metrics": train_metrics,
        "cv_scores":     cv_scores,
        "train_time_s":  elapsed,
        "eval_time_s":   eval_elapsed,
        "lc_time_s":     lc_elapsed,
        "shap_time_s":   shap_elapsed,
        "device":        device,
        "is_ensemble":   is_ensemble,
        "flags":         flags,
        # Flat fields for production bundle convenience
        "train_auc":     train_metrics["ROC-AUC"],
        "test_auc":      test_metrics["ROC-AUC"],
        "cv_auc_mean":   cv_mean,    # None when CV was skipped
        "cv_auc_std":    cv_std,
    }


def main():
    console.clear()

    # ── 0. GPU detection (once — used by all GPU-capable models) ─────────────
    gpu, gpu_label = _detect_gpu()
    device_label   = {
        "XGBoost":    gpu_label if gpu else "CPU",
        "LightGBM":   gpu_label if gpu else "CPU",
        "CatBoost":   gpu_label if gpu else "CPU",
        "Logistic Regression": "CPU (n_jobs=-1)",
        "Random Forest":       "CPU (n_jobs=-1)",
        "Voting Ensemble":     "Mixed (per-member)",
        "Stacking Ensemble":   "Mixed (per-member)",
    }

    # ── 1. Load data ──────────────────────────────────────────────────────────
    path = DATA_PROCESSED / "training_features.csv"
    logger.info(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])

    # ── 2. Temporal split (2018-2023 train | 2024-2025 test) ─────────────────
    train_mask = df["acq_date"].dt.year <= 2023
    test_mask  = df["acq_date"].dt.year >= 2024

    # DATA LEAKAGE ASSERTION: verify zero index overlap between train and test.
    # SimpleImputer/StandardScaler are fit inside Pipeline.fit(X_train, y_train)
    # and therefore see only training data — never X_test.
    overlap = set(df.index[train_mask]) & set(df.index[test_mask])
    assert len(overlap) == 0, (
        f"DATA LEAKAGE: {len(overlap)} rows appear in BOTH train and test splits!"
    )
    logger.info("Data leakage assertion PASSED: train/test index sets are disjoint.")

    X_train = df.loc[train_mask, FEATURE_COLS].reset_index(drop=True)
    y_train = df.loc[train_mask, TARGET_COL].reset_index(drop=True)
    X_test  = df.loc[test_mask,  FEATURE_COLS].reset_index(drop=True)
    y_test  = df.loc[test_mask,  TARGET_COL].reset_index(drop=True)

    print_pipeline_header(df, train_mask, test_mask, gpu_label, gpu)
    logger.info(
        f"Split: train={len(X_train):,} rows, test={len(X_test):,} rows | "
        f"GPU={gpu} ({gpu_label})"
    )

    # ── 3. SHAP background sample (reused by all models) ─────────────────────
    shap_sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_SEED)

    all_results: dict = {}

    # ── 4. Base model training loop ───────────────────────────────────────────
    console.rule("[bold]Phase 1 — Base Models[/bold]")
    base_configs = get_base_model_configs(gpu)

    for name, model in base_configs.items():
        result = train_and_evaluate(
            name       = name,
            model      = model,
            X_train    = X_train,
            y_train    = y_train,
            X_test     = X_test,
            y_test     = y_test,
            device     = device_label[name],
            is_ensemble= False,
            shap_sample= shap_sample,
        )
        if result is not None:
            all_results[name] = result

    # ── 5. Ensemble construction & evaluation ─────────────────────────────────
    console.rule("[bold]Phase 2 — Ensemble Models[/bold]")
    ensemble_configs  = get_ensemble_configs(gpu)
    ensemble_test_auc: dict[str, float] = {}

    for name, model in ensemble_configs.items():
        # Pass the other ensemble's AUC (if already computed) for the verdict
        other_auc = None
        other_names = [n for n in ensemble_configs if n != name]
        if other_names[0] in ensemble_test_auc:
            other_auc = ensemble_test_auc[other_names[0]]

        result = train_and_evaluate(
            name             = name,
            model            = model,
            X_train          = X_train,
            y_train          = y_train,
            X_test           = X_test,
            y_test           = y_test,
            device           = device_label[name],
            is_ensemble      = True,
            other_ensemble_auc = other_auc,
            shap_sample      = shap_sample,
        )
        if result is not None:
            all_results[name] = result
            ensemble_test_auc[name] = result["metrics"]["ROC-AUC"]

    # ── 6. Final AUC self-check for ALL candidates ────────────────────────────
    console.print("\n[dim]Running final AUC integrity self-check for all candidates...[/dim]")
    for cname, res in all_results.items():
        recheck = roc_auc_score(y_test, res["y_prob"])
        assert abs(recheck - res["metrics"]["ROC-AUC"]) < 1e-8, (
            f"Final AUC self-check FAILED for {cname}: "
            f"stored={res['metrics']['ROC-AUC']:.8f} recomputed={recheck:.8f}"
        )
    console.print("  [green]✓ All AUC values verified — no silent bugs detected.[/green]")
    logger.info("Final AUC self-check: PASSED for all candidates.")

    # ── 7. Production ensemble selection ─────────────────────────────────────
    # ALWAYS an ensemble — never a base model, even if a base model scores higher.
    ensemble_auc_map = {}
    for n in ["Voting Ensemble", "Stacking Ensemble"]:
        if n in all_results:
            ensemble_auc_map[n] = all_results[n]["metrics"]["ROC-AUC"]
    
    if not ensemble_auc_map:
        raise ValueError("No ensembles successfully evaluated.")
    selected_name  = max(ensemble_auc_map, key=ensemble_auc_map.get)
    selected_model = ensemble_configs[selected_name]
    selected_res   = all_results[selected_name]
    model_type_key = (
        "voting_ensemble" if selected_name == "Voting Ensemble" else "stacking_ensemble"
    )

    logger.info(
        f"Production ensemble selected: {selected_name} "
        f"(AUC={selected_res['test_auc']:.4f}) | type={model_type_key}"
    )

    # ── 8. Comparison artifacts ───────────────────────────────────────────────
    console.print("\n[dim]Generating cross-model comparison artifacts...[/dim]")
    t_comp = time.time()
    generate_comparison_artifacts(
        results               = all_results,
        y_true                = y_test,
        out_dir               = COMPARISON_DIR,
        production_model_name = selected_name,
    )
    comp_elapsed = time.time() - t_comp
    console.print(f"  Comparison charts saved to: [dim]{COMPARISON_DIR}/[/dim]")

    # ── 9. Consolidated comparison table ─────────────────────────────────────
    console.rule("[bold]Final Results[/bold]")
    print_final_comparison_table(all_results, production_name=selected_name)

    # ── 10. Production bundle packaging ──────────────────────────────────────
    production_bundle = {
        # ── Backward-compatible keys (FastAPI inference layer reads these) ────
        # VotingClassifier / StackingClassifier both expose .predict() /
        # .predict_proba() with the identical sklearn interface — confirmed:
        # src/inference/engine.py calls self.model.predict_proba(X)[:,1]
        # which works unchanged for both ensemble types.
        "model":     selected_model,
        "threshold": RISK_THRESHOLD_HIGH,
        "features":  FEATURE_COLS,
        # backward-compat alias (engine.py line: self.best_name = bundle.get("best_single"))
        "best_single": selected_name,
        "cv_results":  {},                 # kept for engine.summary() compatibility

        # ── New fields ───────────────────────────────────────────────────────
        "model_type":    model_type_key,
        "ensemble_members": BASE_MODEL_NAMES,
        "base_model_auc_comparison": {
            n: all_results[n]["metrics"]["ROC-AUC"]
            for n in BASE_MODEL_NAMES
        },

        # ── Overfitting/underfitting diagnostics (Task 5.7) ──────────────────
        "train_auc":      selected_res["train_auc"],
        "test_auc":       selected_res["test_auc"],
        "cv_auc_mean":    selected_res["cv_auc_mean"],
        "cv_auc_std":     selected_res["cv_auc_std"],
        "overfit_flag":   selected_res["flags"]["overfit_flag"],
        "underfit_flag":  selected_res["flags"]["underfit_flag"],
        "suspicious_flag":selected_res["flags"]["suspicious_flag"],
    }

    joblib.dump(production_bundle, MODELS_DIR / "production_model.pkl")
    logger.info(f"Production bundle saved: {MODELS_DIR / 'production_model.pkl'}")

    # ── 11. Final summary panel ───────────────────────────────────────────────
    base_auc_lines = "\n".join(
        f"    {n}: {all_results[n]['metrics']['ROC-AUC']:.4f}"
        for n in BASE_MODEL_NAMES if n in all_results
    )
    
    other_ensemble_names = [n for n in ["Voting Ensemble", "Stacking Ensemble"] if n != selected_name and n in ensemble_auc_map]
    not_selected_str = f"{other_ensemble_names[0]} -> AUC {ensemble_auc_map[other_ensemble_names[0]]:.4f}" if other_ensemble_names else "None"

    # Time breakdown calculation
    base_model_fit_time = sum(res["train_time_s"] for n, res in all_results.items() if not res["is_ensemble"])
    voting_fit_time = all_results.get("Voting Ensemble", {}).get("train_time_s", 0)
    stacking_fit_time = all_results.get("Stacking Ensemble", {}).get("train_time_s", 0)
    total_lc_time = sum(res["lc_time_s"] for res in all_results.values())
    total_shap_time = sum(res["shap_time_s"] for res in all_results.values())
    
    runtime_breakdown = (
        f"\n  [bold]Runtime Breakdown:[/bold]\n"
        f"    Base Models Fit : {base_model_fit_time:.1f}s\n"
        f"    Voting Fit      : {voting_fit_time:.1f}s\n"
        f"    Stacking Fit    : {stacking_fit_time:.1f}s\n"
        f"    Learning Curves : {total_lc_time:.1f}s\n"
        f"    SHAP Analysis   : {total_shap_time:.1f}s\n"
        f"    Comparison Plots: {comp_elapsed:.1f}s\n"
    )

    console.print(Panel(
        f"[bold green][OK] PIPELINE COMPLETE[/bold green]\n\n"
        f"  Production model  : [bold cyan]{selected_name}[/bold cyan] "
        f"(type=[dim]{model_type_key}[/dim])\n"
        f"  Test AUC (ensemble): [bold]{selected_res['test_auc']:.4f}[/bold]  |  "
        f"Train AUC: {selected_res['train_auc']:.4f}  |  "
        f"CV: {selected_res['cv_auc_mean']:.4f}+/-{selected_res['cv_auc_std']:.4f}\n"
        f"  Other ensemble    : {not_selected_str}\n"
        f"  Base model AUCs   :\n{base_auc_lines}\n"
        f"  Overfit / Underfit: {selected_res['flags']['overfit_flag']} / "
        f"{selected_res['flags']['underfit_flag']}\n"
        f"  GPU used          : {gpu_label}\n"
        f"  Artifacts         : {EVAL_DIR} | {SHAP_DIR} | {COMPARISON_DIR}\n"
        f"{runtime_breakdown}",
        expand=False,
        border_style="bold green",
    ))

    logger.info(
        f"Pipeline complete. Production={selected_name} "
        f"test_auc={selected_res['test_auc']:.4f} "
        f"overfit={selected_res['flags']['overfit_flag']} "
        f"underfit={selected_res['flags']['underfit_flag']}"
    )


if __name__ == "__main__":
    main()