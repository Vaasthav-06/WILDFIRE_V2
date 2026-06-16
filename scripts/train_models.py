"""
scripts/train_models.py

Master Training Orchestrator for Wildfire Risk Prediction.
Handles model training, metrics generation, artifact plotting, and strict terminal logging.
"""

import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

# Rich UI for terminal
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    EVAL_DIR,
    SHAP_DIR,
    FEATURE_COLS,
    TARGET_COL,
    RANDOM_SEED,
)
from src.models.evaluate_utils import calculate_metrics, generate_evaluation_artifacts, generate_shap_artifacts

# Setup logging
logger.remove()
logger.add("logs/training_pipeline.log", rotation="50 MB", level="INFO")
console = Console()

def get_models():
    """Define the exact models to be trained."""
    return {
        "Logistic Regression": Pipeline([
            ('imputer', SimpleImputer(strategy='median')), # <-- ADDED IMPUTER
            ('scaler', StandardScaler()), 
            ('lr', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced'))
        ]),
        "Random Forest": Pipeline([
            ('imputer', SimpleImputer(strategy='median')), # <-- ADDED IMPUTER PIPELINE
            ('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1))
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300, random_state=RANDOM_SEED, scale_pos_weight=2.0, eval_metric="logloss", device="cuda"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, device="gpu"
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, random_state=RANDOM_SEED, auto_class_weights='Balanced', verbose=0, task_type="GPU"
        )
    }

def print_metrics_table(metrics: dict):
    """Generate a clean terminal table for evaluation metrics."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=15)
    table.add_column("Value", style="magenta", justify="right")
    
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}")
    
    console.print("-" * 34)
    console.print(table)
    console.print("-" * 34)

def main():
    console.clear()
    console.print(Panel.fit("[bold green]WILDFIRE V2: MODEL TRAINING PIPELINE[/bold green]"))

    # ── 1. Load Data ─────────────────────────────────────────
    path = DATA_PROCESSED / "training_features.csv"
    logger.info(f"Loading {path}")
    console.print(f"[bold yellow][INFO][/bold yellow] Loading dataset from: {path.name}...")
    
    df = pd.read_csv(path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # ── Strict Temporal Split (Train: 2018-2023 | Test: 2024-2025) ──
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    
    train_mask = df["acq_date"].dt.year <= 2023
    test_mask = df["acq_date"].dt.year >= 2024

    X_train = df.loc[train_mask, FEATURE_COLS]
    y_train = df.loc[train_mask, TARGET_COL]

    X_test = df.loc[test_mask, FEATURE_COLS]
    y_test = df.loc[test_mask, TARGET_COL]
    
    # Optional: Delete the train_test_split import at the top of the file if you like a clean workspace
    
    console.print(f"[bold yellow][INFO][/bold yellow] Features: {len(FEATURE_COLS)} | Train rows: {len(X_train):,} | Test rows: {len(X_test):,}\n")

    models = get_models()
    best_auc = 0
    best_model_name = ""
    
    # ── 2. Train & Evaluate Loop ─────────────────────────────
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        model_eval_dir = EVAL_DIR / safe_name
        model_shap_dir = SHAP_DIR / safe_name
        
        console.print(f"{'='*50}\n[bold blue]TRAINING MODEL: {name.upper()}[/bold blue]\n{'='*50}")
        
        # Train
        console.print("[bold yellow][INFO][/bold yellow] Training started...")
        model.fit(X_train, y_train)
        console.print("[bold yellow][INFO][/bold yellow] Training completed.\n")

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluate
        console.print("[bold]Evaluation Results:[/bold]")
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        print_metrics_table(metrics)
        
        if metrics["ROC-AUC"] > best_auc:
            best_auc = metrics["ROC-AUC"]
            best_model_name = name

        # Generate Artifacts
        console.print(f"[bold yellow][INFO][/bold yellow] Generating evaluation plots...")
        generate_evaluation_artifacts(y_test, y_pred, y_prob, model_eval_dir)
        console.print(f"       Saved metrics plots: [dim]{model_eval_dir}/[/dim]")
        
        console.print(f"[bold yellow][INFO][/bold yellow] Generating SHAP analysis...")
        # Use a background sample to speed up SHAP calculation
        shap_sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_SEED)
        generate_shap_artifacts(model, shap_sample, name, model_shap_dir)
        console.print(f"       Saved SHAP analysis: [dim]{model_shap_dir}/[/dim]")

        # Save Model
        joblib.dump(model, MODELS_DIR / f"{safe_name}.pkl")
        console.print("\n")

    # ── 3. Pipeline Conclusion ───────────────────────────────
    console.print(Panel(
        f"[bold green]PIPELINE COMPLETE[/bold green]\n"
        f"All models trained and artifacts generated.\n"
        f"Best Performer: [bold cyan]{best_model_name}[/bold cyan] (AUC: {best_auc:.4f})",
        expand=False
    ))

if __name__ == "__main__":
    main()