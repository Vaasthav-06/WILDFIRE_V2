"""
scripts/fix_stacking_and_finalize.py

Standalone Fix Script — Stacking Ensemble + Finalization
=========================================================

Context
-------
Phase 1 of scripts/train_models.py completed successfully:
  LogisticRegression, CatBoost, LightGBM, XGBoost, RandomForest,
  and VotingClassifier all trained, evaluated, and saved to models/*.pkl.

Only StackingClassifier failed ("cross_val_predict only works for partitions",
TimeSeriesSplit incompatibility), and a downstream KeyError crashed the final
comparison/selection step before production_model.pkl could be finalized.

What this script does
---------------------
Step 1 — Reconstruct the exact same data split (no retraining, just reloading).
         Imports the same data-loading/split logic and config constants used in
         train_models.py so the split is guaranteed identical.

Step 2 — Load the 5 already-fitted base model .pkl files + voting_ensemble.pkl
         and recompute train/test metrics via INFERENCE ONLY (.predict /
         .predict_proba).  No .fit() is called on these 6 loaded objects.
         Existing evaluation artifacts on disk are confirmed and reused.

Step 3 — Fix and fit the StackingClassifier with the corrected cv strategy:
         StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
         which IS a true partition that cross_val_predict can use (unlike
         TimeSeriesSplit).  Evaluate, generate artifacts, save to disk.

Step 4 — Rebuild the final comparison output with all 7 candidates and write
         the production_model.pkl bundle with the full schema.

Re-runnable
-----------
Running this script again is safe and idempotent.  It does not touch or re-fit
any of the 5 base models or the VotingClassifier.

Imports from
------------
  scripts/train_models.py   — get_base_model_configs(), compute_flags(),
                              run_timeseries_cv(), print_model_header(),
                              print_model_metrics_table(),
                              print_final_comparison_table(), _detect_gpu(),
                              OVERFIT_DELTA, UNDERFIT_THRESH, SUSPICIOUS_AUC,
                              BASE_MODEL_NAMES

  src/config.py             — DATA_PROCESSED, MODELS_DIR, EVAL_DIR, SHAP_DIR,
                              COMPARISON_DIR, FEATURE_COLS, TARGET_COL,
                              RANDOM_SEED, RISK_THRESHOLD_HIGH

  src/models/evaluate_utils — calculate_metrics(), generate_evaluation_artifacts(),
                              generate_learning_curve(), generate_shap_artifacts(),
                              generate_comparison_artifacts()
"""

import os
import sys

# Force UTF-8 output on Windows terminals (avoids cp1252 UnicodeEncodeError).
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
import joblib
import numpy as np
import pandas as pd

from loguru import logger
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from rich.console import Console
from rich.panel import Panel
from rich import box

# ── Project config ────────────────────────────────────────────────────────────
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

# ── Evaluation utilities ──────────────────────────────────────────────────────
from src.models.evaluate_utils import (
    calculate_metrics,
    generate_evaluation_artifacts,
    generate_learning_curve,
    generate_shap_artifacts,
    generate_comparison_artifacts,
)

# ── Import shared logic from the main pipeline (no duplication) ───────────────
# We add the project root to sys.path so train_models can be imported as a module
# regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train_models import (
    get_base_model_configs,
    compute_flags,
    run_timeseries_cv,
    print_model_header,
    print_model_metrics_table,
    print_final_comparison_table,
    _detect_gpu,
    OVERFIT_DELTA,
    UNDERFIT_THRESH,
    SUSPICIOUS_AUC,
    BASE_MODEL_NAMES,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    "logs/fix_stacking_and_finalize.log",
    rotation="50 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

console = Console()

# ── Name -> safe filename mapping (must match train_models.py convention) ──────
_SAFE_NAME = {
    "Logistic Regression": "logistic_regression",
    "CatBoost":            "catboost",
    "LightGBM":            "lightgbm",
    "XGBoost":             "xgboost",
    "Random Forest":       "random_forest",
    "Voting Ensemble":     "voting_ensemble",
    "Stacking Ensemble":   "stacking_ensemble",
}

# Expected artifact files per model directory (confirmed -> reuse, missing -> warn)
_EXPECTED_EVAL_FILES = [
    "confusion_matrix.png",
    "roc_curve.png",
    "pr_curve.png",
    "learning_curve.png",
]
_EXPECTED_SHAP_FILES = ["shap_bar.png", "shap_summary.png"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: verify existing artifacts on disk
# ─────────────────────────────────────────────────────────────────────────────

def _confirm_existing_artifacts(name: str) -> bool:
    """
    Check that the Phase 1 evaluation and SHAP artifacts exist for *name*.

    Bug 2 fix: prints the exact absolute file path for every missing file so
    the mismatch is immediately visible without a manual audit.

    SHAP files are treated as a soft-warn only: SHAP generation can fail
    legitimately for some model types (e.g. VotingClassifier with missing
    named_estimators_) and the missing shap_bar.png / shap_summary.png
    does NOT block the inference-only reload workflow.

    The safe_name mapping uses the same rule as train_models.py:
        safe_name = name.lower().replace(" ", "_")
    Both scripts derive it the same way so there is no drift between them.
    """
    # Derive safe_name exactly as train_models.py does in train_and_evaluate():
    #   safe_name = name.lower().replace(" ", "_")
    # _SAFE_NAME dict encodes the same transformation; verified byte-for-byte
    # identical by inspection of train_models.py line:
    #   safe_name = name.lower().replace(" ", "_")
    safe = _SAFE_NAME[name]
    eval_dir = EVAL_DIR / safe
    shap_dir = SHAP_DIR / safe

    eval_missing = []
    for fname in _EXPECTED_EVAL_FILES:
        p = eval_dir / fname
        if not p.exists():
            eval_missing.append(p)
            # Bug 2 fix: log the exact absolute path so mismatch is visible
            logger.warning(f"  {name}: Eval artifact missing: {p.resolve()}")

    shap_missing = []
    for fname in _EXPECTED_SHAP_FILES:
        p = shap_dir / fname
        if not p.exists():
            shap_missing.append(p)
            # Bug 2 fix: log exact absolute path
            logger.warning(f"  {name}: SHAP artifact missing: {p.resolve()}")

    eval_ok = len(eval_missing) == 0
    # SHAP is soft: a missing SHAP file does not block inference-only reload
    shap_ok = len(shap_missing) == 0

    if eval_ok and shap_ok:
        msg = (
            f"Reusing existing evaluation artifacts for {name}, "
            "skipping regeneration"
        )
        logger.info(f"  {msg}")
        console.print(f"  [dim]{msg}[/dim]")
    elif eval_ok and not shap_ok:
        # Core eval charts exist; SHAP is optional
        console.print(
            f"  [yellow][!] {name}: SHAP artifacts missing on disk "
            f"(soft-warn, not blocking):\n"
            + "\n".join(f"    {p.resolve()}" for p in shap_missing)
            + "\n  Reusing existing eval artifacts (confusion matrix, ROC, PR, "
            "learning curve); SHAP was skipped or failed in Phase 1.[/yellow]"
        )
        logger.info(
            f"  Reusing existing evaluation artifacts for {name}, "
            "skipping regeneration (SHAP soft-warn only)."
        )
    else:
        # Eval charts missing — this is a real problem worth flagging loudly
        console.print(
            f"  [bold red][!] {name}: Core eval artifacts missing on disk:\n"
            + "\n".join(f"    {p.resolve()}" for p in eval_missing)
            + "\n  Proceeding with inference-only metrics; "
            "charts cannot be reused.[/bold red]"
        )

    return eval_ok  # only eval_ok is the binding gate for reuse decision


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load a pre-fitted model and compute inference-only metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_and_evaluate_fitted(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    device: str,
    is_ensemble: bool,
) -> dict:
    """
    Load a pre-fitted model .pkl, run INFERENCE ONLY (no .fit() call),
    compute train/test metrics, reconfirm existing artifacts, and return a
    result dict compatible with the all_results schema used in train_models.py.

    Safety: we explicitly assert that no .fit attribute is called by never
    calling it.  The loaded object's weights are read-only in this context.
    """
    safe = _SAFE_NAME[name]
    pkl_path = MODELS_DIR / f"{safe}.pkl"

    console.print()
    print_model_header(name, device)
    logger.info(f"LOAD (inference only): {name} from {pkl_path}")

    if not pkl_path.exists():
        logger.error(f"  {name}: model file not found at {pkl_path}. Skipping.")
        console.print(
            f"  [bold red][!] {name}: {pkl_path} not found. Skipping.[/bold red]"
        )
        return None

    t0 = time.time()
    model = joblib.load(pkl_path)
    load_elapsed = time.time() - t0
    logger.info(f"  {name}: loaded in {load_elapsed:.2f}s")
    console.print(f"  [dim]Loaded {pkl_path.name} in {load_elapsed:.2f}s[/dim]")

    # ── Inference on train and test sets ──────────────────────────────────────
    # IMPORTANT: .fit() is NOT called on this loaded model anywhere below.
    t_infer = time.time()
    y_pred_test  = model.predict(X_test)
    y_prob_test  = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    infer_elapsed = time.time() - t_infer

    # ── Metrics ──────────────────────────────────────────────────────────────
    test_metrics  = calculate_metrics(y_test,  y_pred_test,  y_prob_test)
    train_metrics = calculate_metrics(y_train, y_pred_train, y_prob_train)

    # AUC self-check (matches pattern in train_models.py)
    _recomputed_auc = roc_auc_score(y_test, y_prob_test)
    assert abs(_recomputed_auc - test_metrics["ROC-AUC"]) < 1e-8, (
        f"AUC self-check FAILED for {name}: "
        f"stored={test_metrics['ROC-AUC']:.8f} recomputed={_recomputed_auc:.8f}"
    )

    # ── CV skipped for loaded Phase 1 models ─────────────────────────────────
    # Phase 1 CV artifacts are authoritative and already on disk.
    # We do not re-run run_timeseries_cv() here -- that would refit clones,
    # which would be semantically confusing and add unnecessary runtime.
    # Bug 1 fix: use None (not 0.0) so compute_flags() skips the train-vs-CV
    # overfit check, preventing a massive false gap from train_auc - 0.0.
    cv_scores = np.array([])
    cv_mean   = None    # None = CV skipped / reused from Phase 1 (Bug 1 fix)
    cv_std    = 0.0
    logger.info(
        f"  {name}: CV diagnostic skipped for loaded model "
        "(artifacts from Phase 1 are authoritative). "
        "Inference-only metrics used."
    )
    console.print(
        f"  [dim]CV diagnostic skipped (Phase 1 artifacts authoritative). "
        f"Inference elapsed: {infer_elapsed:.2f}s[/dim]"
    )

    # ── Flags ────────────────────────────────────────────────────────────────
    flags = compute_flags(
        train_auc   = train_metrics["ROC-AUC"],
        test_auc    = test_metrics["ROC-AUC"],
        cv_auc_mean = cv_mean,
        cv_scores   = cv_scores,
    )

    # ── Print metrics table ───────────────────────────────────────────────────
    print_model_metrics_table(
        name, train_metrics, test_metrics, cv_scores, flags,
        elapsed=infer_elapsed,
        is_ensemble=is_ensemble,
        other_ensemble_auc=None,
        cv_skipped=True,    # Bug 1 fix: renders "N/A (CV skipped)" in table
    )

    # ── Confirm existing artifacts (log note, no regeneration) ────────────────
    _confirm_existing_artifacts(name)

    logger.info(
        f"  {name}: train_auc={train_metrics['ROC-AUC']:.4f} "
        f"test_auc={test_metrics['ROC-AUC']:.4f} "
        f"infer={infer_elapsed:.2f}s (no retraining)"
    )

    return {
        "y_pred":        y_pred_test,
        "y_prob":        y_prob_test,
        "metrics":       test_metrics,
        "train_metrics": train_metrics,
        "cv_scores":     cv_scores,
        "train_time_s":  0.0,    # no fit time -- loaded from disk
        "eval_time_s":   0.0,    # no re-evaluation time
        "lc_time_s":     0.0,
        "shap_time_s":   0.0,
        "device":        device,
        "is_ensemble":   is_ensemble,
        "flags":         flags,
        # Flat fields for production bundle convenience.
        # Bug 1 fix: cv_auc_mean=None (not 0.0) when CV was skipped, so
        # downstream consumers can render "N/A" rather than a fake zero.
        "train_auc":     train_metrics["ROC-AUC"],
        "test_auc":      test_metrics["ROC-AUC"],
        "cv_auc_mean":   None,   # None = CV skipped / reused from Phase 1
        "cv_auc_std":    0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Fix and fit the StackingClassifier (new work)
# ─────────────────────────────────────────────────────────────────────────────

def fit_stacking_ensemble(
    gpu: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    shap_sample: pd.DataFrame,
    device: str,
) -> dict:
    """
    Build and fit a fresh StackingClassifier with the corrected cv strategy:
      StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    This IS a true partition that cross_val_predict can use, unlike
    TimeSeriesSplit(n_splits=5), which caused the original
    "cross_val_predict only works for partitions" error.

    The 5 base estimator instances are recreated from get_base_model_configs()
    (imported directly from train_models.py) to guarantee exact hyperparameter
    parity with Phase 1.  No fitted Phase 1 model objects are passed here.

    On success: saves stacking_ensemble.pkl and all evaluation artifacts.
    On failure: logs clearly and returns None so Step 4 can continue safely.

    CONCEPTUAL NOTE on the two separate CV uses:
      INTERNAL CV (used by StackingClassifier.fit internally):
          StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
          -> generates the meta-features for the final_estimator.
      DIAGNOSTIC CV (run after .fit() succeeds, separately below):
          TimeSeriesSplit(n_splits=5) via run_timeseries_cv()
          -> measures temporal generalization of the FITTED stacking model.
    These are two separate, conceptually distinct uses of cross-validation.
    """
    name     = "Stacking Ensemble"
    safe     = _SAFE_NAME[name]
    eval_dir = EVAL_DIR / safe
    shap_dir = SHAP_DIR / safe

    console.print()
    print_model_header(name, device)
    logger.info(f"START fitting (new): {name} | device={device}")

    # ── Reconstruct fresh (unfitted) base estimators ─────────────────────────
    # Import get_base_model_configs() directly from train_models.py to guarantee
    # exact hyperparameter parity -- no hardcoding in this script.
    #
    # Bug 4 fix: override LightGBM to device="cpu" for StackingClassifier use.
    # Root cause: LightGBM's GPU histogram tree-learner crashes on small /
    # uneven data slices ("Check failed: best_split_info.right_count > 0")
    # which appear in StratifiedKFold folds during StackingClassifier.fit().
    # This is a known LightGBM issue on GPU with minority-class imbalance on
    # small fold sub-slices.  The workaround is to force CPU for the internal
    # stacking CV folds only.  The Phase 1 standalone LightGBM model (already
    # fitted and saved as models/lightgbm.pkl) is completely UNTOUCHED.
    #
    # PREVENTIVE NOTE FOR VotingClassifier: if VotingClassifier is ever rebuilt
    # (it succeeded in Phase 1 and does NOT need to be rebuilt now), apply the
    # same device="cpu" override to its LightGBM member for the same reason.
    # VotingClassifier uses cross_val_predict internally on the same fold slices.
    raw_configs = get_base_model_configs(gpu)
    stacking_estimators = []
    for est_name, estimator in raw_configs.items():
        if est_name == "LightGBM":
            # Bug 4 fix: force CPU for stacking's internal CV folds.
            # LightGBM GPU tree-learner crashes on small fold slices.
            # This does NOT affect the standalone lightgbm.pkl from Phase 1.
            try:
                estimator = estimator.set_params(device="cpu")
                logger.info(
                    "  Stacking Ensemble: LightGBM overridden to device='cpu' "
                    "(GPU tree-learner workaround for small StratifiedKFold slices)."
                )
            except Exception as e:
                logger.warning(
                    f"  Stacking Ensemble: Could not override LightGBM device to cpu: {e}. "
                    "Proceeding with original config (may crash on GPU fold slices)."
                )
        stacking_estimators.append((est_name, estimator))
    estimator_list = stacking_estimators

    # ── Build StackingClassifier with corrected cv ────────────────────────────
    # FIX: Use StratifiedKFold (a true partition) instead of TimeSeriesSplit.
    # TimeSeriesSplit is NOT a partition -- it does not cover all samples
    # simultaneously across folds, which is why cross_val_predict rejected it.
    # StratifiedKFold with shuffle=True + random_state gives reproducible splits
    # and preserves class balance across folds.
    stacking = StackingClassifier(
        estimators = estimator_list,
        final_estimator = LogisticRegression(
            random_state = RANDOM_SEED,
            max_iter     = 1000,
            n_jobs       = -1,
        ),
        cv          = StratifiedKFold(
            n_splits     = 5,
            shuffle      = True,
            random_state = RANDOM_SEED,
        ),
        passthrough = False,
        n_jobs      = 1,    # parallelism handled inside each base model
    )

    # ── Fit ──────────────────────────────────────────────────────────────────
    console.print(
        "  [dim]Stacking Ensemble fits base models 5 times during internal "
        "StratifiedKFold CV. This will take a while.[/dim]"
    )
    logger.info(
        f"  {name}: Expected long runtime due to 5 internal CV refits of all "
        "base models (StratifiedKFold, corrected from TimeSeriesSplit)."
    )
    t0 = time.time()
    try:
        stacking.fit(X_train, y_train)
    except Exception as e:
        elapsed_fail = time.time() - t0
        is_mem = isinstance(e, MemoryError) or "MemoryError" in str(type(e))
        if is_mem:
            logger.warning(
                f"  {name}: fit skipped after {elapsed_fail:.1f}s "
                f"due to MemoryError: {e}"
            )
            console.print(
                f"  [bold yellow][!] {name} fit skipped due to MemoryError "
                f"after {elapsed_fail:.1f}s. Continuing to Step 4 with "
                "Stacking marked as FAILED.[/bold yellow]"
            )
        else:
            logger.error(
                f"  {name}: fit FAILED after {elapsed_fail:.1f}s with error: {e}"
            )
            console.print(
                f"  [bold red][!] {name} fit FAILED after {elapsed_fail:.1f}s: "
                f"{e}\nContinuing to Step 4 with Stacking marked as FAILED.[/bold red]"
            )
        return None

    elapsed = time.time() - t0
    logger.info(f"  {name}: fit complete in {elapsed:.1f}s")
    console.print(f"  [green]Fit complete in {elapsed:.1f}s[/green]")

    # ── Predictions (inference only on the newly-fitted stacking model) ───────
    y_pred_test  = stacking.predict(X_test)
    y_prob_test  = stacking.predict_proba(X_test)[:, 1]
    y_pred_train = stacking.predict(X_train)
    y_prob_train = stacking.predict_proba(X_train)[:, 1]

    # ── Metrics ──────────────────────────────────────────────────────────────
    test_metrics  = calculate_metrics(y_test,  y_pred_test,  y_prob_test)
    train_metrics = calculate_metrics(y_train, y_pred_train, y_prob_train)

    # AUC self-check
    _recomputed_auc = roc_auc_score(y_test, y_prob_test)
    assert abs(_recomputed_auc - test_metrics["ROC-AUC"]) < 1e-8, (
        f"AUC self-check FAILED for {name}: "
        f"stored={test_metrics['ROC-AUC']:.8f} recomputed={_recomputed_auc:.8f}"
    )

    # ── DIAGNOSTIC CV AUC (TimeSeriesSplit on training set only) ─────────────
    # CONCEPTUALLY DISTINCT from the internal StratifiedKFold CV above.
    # This diagnostic CV measures temporal generalization of the FITTED stacking
    # model and is labeled separately in output to avoid confusion.
    console.print(
        "  [dim]Running DIAGNOSTIC TimeSeriesSplit CV (training set only, "
        "separate from the internal StratifiedKFold used during .fit())...[/dim]"
    )
    logger.info(
        f"  {name}: Running DIAGNOSTIC TimeSeriesSplit CV "
        "(distinct from internal StratifiedKFold used during .fit())."
    )
    cv_scores = run_timeseries_cv(stacking, X_train, y_train, name=name, n_splits=5)
    cv_mean   = float(np.mean(cv_scores)) if len(cv_scores) > 0 else 0.0
    cv_std    = float(np.std(cv_scores))  if len(cv_scores) > 0 else 0.0
    if len(cv_scores) > 0:
        logger.info(f"  {name}: Diagnostic CV AUC = {cv_mean:.4f}+/-{cv_std:.4f}")
    else:
        logger.info(f"  {name}: Diagnostic CV skipped (MemoryError or other error).")

    # ── Flags ────────────────────────────────────────────────────────────────
    flags = compute_flags(
        train_auc   = train_metrics["ROC-AUC"],
        test_auc    = test_metrics["ROC-AUC"],
        cv_auc_mean = cv_mean,
        cv_scores   = cv_scores,
    )

    # ── Terminal metrics table ────────────────────────────────────────────────
    print_model_metrics_table(
        name, train_metrics, test_metrics, cv_scores, flags,
        elapsed=elapsed,
        is_ensemble=True,
        other_ensemble_auc=None,
    )

    # ── Evaluation charts ─────────────────────────────────────────────────────
    console.print(f"  [dim]Saving evaluation artifacts -> {eval_dir}/[/dim]")
    t_eval = time.time()
    generate_evaluation_artifacts(y_test, y_pred_test, y_prob_test, eval_dir)
    eval_elapsed = time.time() - t_eval

    # ── Learning curve ────────────────────────────────────────────────────────
    console.print("  [dim]Generating learning curve (may take a few minutes)...[/dim]")
    t_lc = time.time()
    generate_learning_curve(stacking, X_train, y_train, eval_dir, model_name=name)
    lc_elapsed = time.time() - t_lc

    # ── SHAP ─────────────────────────────────────────────────────────────────
    console.print(f"  [dim]Generating SHAP plots -> {shap_dir}/[/dim]")
    t_shap = time.time()
    generate_shap_artifacts(stacking, shap_sample, name, shap_dir)
    shap_elapsed = time.time() - t_shap

    # ── Save model .pkl ───────────────────────────────────────────────────────
    pkl_path = MODELS_DIR / f"{safe}.pkl"
    joblib.dump(stacking, pkl_path)
    logger.info(f"  {name}: saved to {pkl_path}")
    console.print(f"  [green]Saved -> {pkl_path}[/green]")

    logger.info(
        f"  {name}: train_auc={train_metrics['ROC-AUC']:.4f} "
        f"test_auc={test_metrics['ROC-AUC']:.4f} "
        f"cv={cv_mean:.4f}+/-{cv_std:.4f} "
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
        "is_ensemble":   True,
        "flags":         flags,
        "train_auc":     train_metrics["ROC-AUC"],
        "test_auc":      test_metrics["ROC-AUC"],
        "cv_auc_mean":   cv_mean,
        "cv_auc_std":    cv_std,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.clear()

    console.print(Panel(
        "[bold cyan]WILDFIRE V2 -- FIX STACKING ENSEMBLE & FINALIZE[/bold cyan]\n\n"
        "  This script does NOT re-run the full training pipeline.\n"
        "  It reuses 5 base models + VotingClassifier via INFERENCE ONLY,\n"
        "  fits ONLY the StackingClassifier (which failed in Phase 1),\n"
        "  and rebuilds the final comparison table + production_model.pkl.\n",
        expand=False,
        border_style="bold blue",
    ))

    # ── 0. GPU detection (needed to recreate unfitted base model instances) ───
    gpu, gpu_label = _detect_gpu()
    device_label = {
        "Logistic Regression": "CPU (n_jobs=-1)",
        "CatBoost":            gpu_label if gpu else "CPU",
        "LightGBM":            gpu_label if gpu else "CPU",
        "XGBoost":             gpu_label if gpu else "CPU",
        "Random Forest":       "CPU (n_jobs=-1)",
        "Voting Ensemble":     "Mixed (per-member)",
        "Stacking Ensemble":   "Mixed (per-member)",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 -- Reconstruct the exact same data split (no retraining)
    # ─────────────────────────────────────────────────────────────────────────
    console.rule("[bold]Step 1 -- Reconstruct Data Split (identical to Phase 1)[/bold]")

    path = DATA_PROCESSED / "training_features.csv"
    logger.info(f"Loading dataset: {path}")
    console.print(f"  [dim]Loading {path}...[/dim]")
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])

    # Temporal split -- IDENTICAL to train_models.py main()
    train_mask = df["acq_date"].dt.year <= 2023
    test_mask  = df["acq_date"].dt.year >= 2024

    # DATA LEAKAGE ASSERTION -- same check as in train_models.py
    overlap = set(df.index[train_mask]) & set(df.index[test_mask])
    assert len(overlap) == 0, (
        f"DATA LEAKAGE: {len(overlap)} rows appear in BOTH train and test splits!"
    )
    logger.info("Data leakage assertion PASSED: train/test index sets are disjoint.")

    X_train = df.loc[train_mask, FEATURE_COLS].reset_index(drop=True)
    y_train = df.loc[train_mask, TARGET_COL].reset_index(drop=True)
    X_test  = df.loc[test_mask,  FEATURE_COLS].reset_index(drop=True)
    y_test  = df.loc[test_mask,  TARGET_COL].reset_index(drop=True)

    n_train  = int(train_mask.sum())
    n_test   = int(test_mask.sum())
    pos_train = float(y_train.mean() * 100)
    pos_test  = float(y_test.mean() * 100)

    console.print(
        f"  [green]Split confirmed:[/green] "
        f"train={n_train:,} rows (fire={pos_train:.1f}%) | "
        f"test={n_test:,} rows (fire={pos_test:.1f}%)"
    )
    logger.info(f"Split: train={n_train:,} rows, test={n_test:,} rows")

    # SHAP background sample -- same seed as train_models.py
    shap_sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_SEED)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 -- Load and evaluate the 6 already-fitted models (inference only)
    # ─────────────────────────────────────────────────────────────────────────
    console.rule(
        "[bold]Step 2 -- Load Phase 1 Models (Inference Only -- No .fit() Calls)[/bold]"
    )
    console.print(
        "  [dim]Loading 5 base models + VotingClassifier via joblib. "
        "Only .predict()/.predict_proba() will be called -- no fitting.[/dim]"
    )

    # Models to load from Phase 1 (base models + voting ensemble)
    # is_ensemble value controls display/legend coloring in comparison charts
    phase1_models = {
        "Logistic Regression": False,
        "CatBoost":            False,
        "LightGBM":            False,
        "XGBoost":             False,
        "Random Forest":       False,
        "Voting Ensemble":     True,
    }

    all_results: dict = {}

    t_step2_start = time.time()
    for name, is_ensemble in phase1_models.items():
        result = load_and_evaluate_fitted(
            name        = name,
            X_train     = X_train,
            y_train     = y_train,
            X_test      = X_test,
            y_test      = y_test,
            device      = device_label[name],
            is_ensemble = is_ensemble,
        )
        if result is not None:
            all_results[name] = result

    step2_elapsed = time.time() - t_step2_start
    console.print(
        f"\n  [green]Step 2 complete in {step2_elapsed:.1f}s "
        f"({len(all_results)} models loaded -- no retraining occurred).[/green]"
    )
    logger.info(
        f"Step 2 complete: {len(all_results)} models loaded via inference only "
        f"in {step2_elapsed:.1f}s."
    )

    # Bug 3 fix: remove the hardcoded 60s wall-time threshold which was
    # miscalibrated for large datasets (Random Forest alone takes >60s to
    # run inference on millions of rows).  Instead, we simply report the
    # Step 2 elapsed time so the user can judge for themselves.  A genuine
    # sign of retraining would be total Step 2 time close to Phase 1 fit
    # time (hours), not a fixed absolute value.
    logger.info(
        f"Step 2 total inference time: {step2_elapsed:.1f}s "
        "(expected seconds-to-low-minutes for pure inference on this dataset size)."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 -- Fix and fit the Stacking Ensemble (new work)
    # ─────────────────────────────────────────────────────────────────────────
    console.rule("[bold]Step 3 -- Fix & Fit StackingClassifier (New Work)[/bold]")
    console.print(
        "  [dim]Using StratifiedKFold(n_splits=5, shuffle=True) as the internal "
        "CV strategy -- this is a true partition and resolves the original "
        "'cross_val_predict only works for partitions' error.[/dim]"
    )

    stacking_result = fit_stacking_ensemble(
        gpu         = gpu,
        X_train     = X_train,
        y_train     = y_train,
        X_test      = X_test,
        y_test      = y_test,
        shap_sample = shap_sample,
        device      = device_label["Stacking Ensemble"],
    )

    if stacking_result is not None:
        all_results["Stacking Ensemble"] = stacking_result
        console.print(
            f"\n  [bold green][OK] Stacking Ensemble fitted successfully. "
            f"Test AUC = {stacking_result['test_auc']:.4f}[/bold green]"
        )
        logger.info(
            f"Step 3 complete: Stacking Ensemble fitted. "
            f"test_auc={stacking_result['test_auc']:.4f}"
        )
    else:
        console.print(
            "\n  [bold red][!] Stacking Ensemble fit FAILED again. "
            "Proceeding to Step 4 with Stacking marked as FAILED. "
            "Production model will be selected from available ensembles.[/bold red]"
        )
        logger.error("Step 3: Stacking Ensemble fit FAILED. Continuing to Step 4.")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 -- Rebuild final comparison output with all available candidates
    # ─────────────────────────────────────────────────────────────────────────
    console.rule("[bold]Step 4 -- Rebuild Final Comparison & Production Bundle[/bold]")

    if len(all_results) == 0:
        console.print("[bold red]No models in all_results -- cannot proceed.[/bold red]")
        logger.error("Step 4 aborted: all_results is empty.")
        return

    # ── 4a. Final AUC self-check for ALL candidates in all_results ────────────
    console.print(
        "\n[dim]Running final AUC integrity self-check for all candidates...[/dim]"
    )
    for cname, res in all_results.items():
        recheck = roc_auc_score(y_test, res["y_prob"])
        assert abs(recheck - res["metrics"]["ROC-AUC"]) < 1e-8, (
            f"Final AUC self-check FAILED for {cname}: "
            f"stored={res['metrics']['ROC-AUC']:.8f} recomputed={recheck:.8f}"
        )
    console.print("  [green]All AUC values verified -- no silent bugs detected.[/green]")
    logger.info("Final AUC self-check: PASSED for all candidates.")

    # ── 4b. Production ensemble selection ─────────────────────────────────────
    # Only compare ensembles that actually succeeded (defensive -- no hardcoded list).
    ensemble_candidates = {
        n: all_results[n]["metrics"]["ROC-AUC"]
        for n in ["Voting Ensemble", "Stacking Ensemble"]
        if n in all_results
    }

    if not ensemble_candidates:
        console.print(
            "[bold red][!] No ensemble candidates in all_results. "
            "Cannot select production model.[/bold red]"
        )
        logger.error("Step 4: No ensemble candidates available. Aborting.")
        return

    selected_name = max(ensemble_candidates, key=ensemble_candidates.get)
    selected_res  = all_results[selected_name]
    model_type_key = (
        "voting_ensemble" if selected_name == "Voting Ensemble"
        else "stacking_ensemble"
    )

    # Load the winner's fitted model object for the bundle
    selected_model = joblib.load(MODELS_DIR / f"{_SAFE_NAME[selected_name]}.pkl")

    logger.info(
        f"Production ensemble selected: {selected_name} "
        f"(AUC={selected_res['test_auc']:.4f}) | type={model_type_key}"
    )
    console.print(
        f"\n  [bold green]Production model selected: {selected_name} "
        f"(Test AUC = {selected_res['test_auc']:.4f})[/bold green]"
    )

    # ── 4c. Comparison artifacts (overwrite incomplete Phase 1 versions) ──────
    # Defensive: only reference candidates present in all_results.
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
    logger.info(f"Comparison artifacts regenerated in {comp_elapsed:.1f}s.")

    # ── 4d. Final comparison table ────────────────────────────────────────────
    console.rule("[bold]Final Results[/bold]")
    print_final_comparison_table(all_results, production_name=selected_name)

    # ── 4e. Production bundle packaging (full schema) ─────────────────────────
    # base_model_auc_comparison only includes models present in all_results
    base_auc_comparison = {
        n: all_results[n]["metrics"]["ROC-AUC"]
        for n in BASE_MODEL_NAMES
        if n in all_results
    }

    production_bundle = {
        # ── Backward-compatible keys (FastAPI inference layer reads these) ────
        "model":     selected_model,
        "threshold": RISK_THRESHOLD_HIGH,
        "features":  FEATURE_COLS,
        # backward-compat alias (engine.py: self.best_name = bundle.get("best_single"))
        "best_single": selected_name,
        "cv_results":  {},    # kept for engine.summary() compatibility

        # ── New fields ───────────────────────────────────────────────────────
        "model_type":    model_type_key,
        "ensemble_members": BASE_MODEL_NAMES,
        "base_model_auc_comparison": base_auc_comparison,

        # ── Overfitting/underfitting diagnostics ────────────────────────────
        "train_auc":       selected_res["train_auc"],
        "test_auc":        selected_res["test_auc"],
        "cv_auc_mean":     selected_res["cv_auc_mean"],
        "cv_auc_std":      selected_res["cv_auc_std"],
        "overfit_flag":    selected_res["flags"]["overfit_flag"],
        "underfit_flag":   selected_res["flags"]["underfit_flag"],
        "suspicious_flag": selected_res["flags"]["suspicious_flag"],

        # ── Comparison between the two ensembles (whichever succeeded) ──────
        "ensemble_auc_comparison": {
            n: all_results[n]["metrics"]["ROC-AUC"]
            for n in ["Voting Ensemble", "Stacking Ensemble"]
            if n in all_results
        },
    }

    prod_path = MODELS_DIR / "production_model.pkl"
    joblib.dump(production_bundle, prod_path)
    logger.info(f"Production bundle saved: {prod_path}")
    console.print(f"  [green]production_model.pkl saved -> {prod_path}[/green]")

    # ── 4f. Final summary panel ───────────────────────────────────────────────
    base_auc_lines = "\n".join(
        f"    {n}: {all_results[n]['metrics']['ROC-AUC']:.4f}"
        for n in BASE_MODEL_NAMES
        if n in all_results
    )

    other_ensemble_names = [
        n for n in ["Voting Ensemble", "Stacking Ensemble"]
        if n != selected_name and n in all_results
    ]
    not_selected_str = (
        f"{other_ensemble_names[0]} -> AUC {ensemble_candidates[other_ensemble_names[0]]:.4f}"
        if other_ensemble_names
        else "None (only one ensemble succeeded)"
    )

    stacking_status = (
        f"SUCCESS (Test AUC = {all_results['Stacking Ensemble']['test_auc']:.4f})"
        if "Stacking Ensemble" in all_results
        else "FAILED -- excluded from comparison"
    )

    stacking_fit_s = (
        stacking_result["train_time_s"]
        if stacking_result is not None
        else 0.0
    )

    # Bug 1 fix: cv_auc_mean may be None when CV was skipped (Voting Ensemble reloaded
    # from Phase 1).  Format it as "N/A (reused)" to avoid TypeError and to clearly
    # communicate that no CV was re-run for this model in the fix script.
    _cv_display_prod = (
        f"{selected_res['cv_auc_mean']:.4f}+/-{selected_res['cv_auc_std']:.4f}"
        if selected_res["cv_auc_mean"] is not None
        else "N/A (reused from Phase 1)"
    )

    console.print(Panel(
        f"[bold green][OK] FIX SCRIPT COMPLETE[/bold green]\n\n"
        f"  Production model   : [bold cyan]{selected_name}[/bold cyan] "
        f"(type=[dim]{model_type_key}[/dim])\n"
        f"  Test AUC (ensemble): [bold]{selected_res['test_auc']:.4f}[/bold]  |  "
        f"Train AUC: {selected_res['train_auc']:.4f}  |  "
        f"CV: {_cv_display_prod}\n"
        f"  Other ensemble     : {not_selected_str}\n"
        f"  Stacking status    : {stacking_status}\n"
        f"  Base model AUCs    :\n{base_auc_lines}\n"
        f"  Overfit / Underfit : {selected_res['flags']['overfit_flag']} / "
        f"{selected_res['flags']['underfit_flag']}\n"
        f"  GPU used           : {gpu_label}\n"
        f"  Artifacts          : {EVAL_DIR} | {SHAP_DIR} | {COMPARISON_DIR}\n\n"
        f"  [bold]Runtime Breakdown:[/bold]\n"
        f"    Step 2 (Phase 1 inference reload): {step2_elapsed:.1f}s\n"
        f"    Step 3 (Stacking fit)            : {stacking_fit_s:.1f}s\n"
        f"    Step 4 (comparison plots)        : {comp_elapsed:.1f}s\n",
        expand=False,
        border_style="bold green",
    ))

    logger.info(
        f"Fix script complete. Production={selected_name} "
        f"test_auc={selected_res['test_auc']:.4f} "
        f"overfit={selected_res['flags']['overfit_flag']} "
        f"underfit={selected_res['flags']['underfit_flag']}"
    )


if __name__ == "__main__":
    main()
