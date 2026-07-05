"""
src/models/evaluate_utils.py
Helper functions to calculate ML metrics and plot ROC/PR/SHAP charts.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
import shap
from loguru import logger
from pathlib import Path

def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

def generate_evaluation_artifacts(y_true, y_pred, y_prob, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    
    # ROC Curve
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curve.png', dpi=300)
    plt.close()
    
    # PR Curve
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(out_dir / 'pr_curve.png', dpi=300)
    plt.close()

def generate_shap_artifacts(model, X_sample, model_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        clf = model
        # Unwrap pipeline if necessary
        if hasattr(model, 'named_steps'):
            clf = list(model.named_steps.values())[-1]
            
        model_type = type(clf).__name__.lower()
        
        # Use TreeExplainer for boosting models
        if any(t in model_type for t in ['lgbm', 'xgb', 'catboost', 'forest', 'tree']):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Keep positive class
        else:
            explainer = shap.LinearExplainer(clf, shap.maskers.Independent(X_sample))
            shap_values = explainer.shap_values(X_sample)

        # Bar plot
        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not generate SHAP plots for {model_name}: {e}")