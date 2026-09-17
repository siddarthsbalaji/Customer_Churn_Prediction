"""
Model Evaluation & Cost-Sensitive Threshold Calibration for Churn Prediction.
Calculates ROC-AUC, PR-AUC, F-beta scores, and expected financial loss.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float=0.5
) -> Dict[str, float]:
    """Computes comprehensive classification metrics at a given decision threshold."""
    y_pred=(y_prob>=threshold).astype(int)
    tn, fp, fn, tp=confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics={
        "threshold": float(round(threshold, 4)),
        "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "precision": float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
        "f1": float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
        "f2": float(round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_true, y_prob), 4)),
        "pr_auc": float(round(average_precision_score(y_true, y_prob), 4)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    return metrics
def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float=500.0,
    cost_fp: float=35.0,
    threshold_range: Tuple[float, float]=(0.1, 0.9),
    steps: int=81
) -> Dict[str, Any]:
    """
    Sweeps decision thresholds to determine the cost-minimizing and F2-maximizing thresholds.
    Business cost formula:
        Total Cost = (False Negatives * CLV Loss) + (False Positives * Incentive Cost)
    """
    thresholds=np.linspace(threshold_range[0], threshold_range[1], steps)
    best_f2 = -1.0
    best_f2_threshold=0.5
    min_cost=float("inf")
    best_cost_threshold=0.5
    threshold_curve=[]
    for t in thresholds:
        y_pred=(y_prob>=t).astype(int)
        tn, fp, fn, tp=confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        f2=fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        total_loss=(fn*cost_fn)+(fp*cost_fp)
        if f2>best_f2:
            best_f2=f2
            best_f2_threshold=t
        if total_loss<min_cost:
            min_cost=total_loss
            best_cost_threshold=t
        threshold_curve.append({
            "threshold": round(float(t), 3),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "f2": round(float(f2), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "expected_financial_cost": round(float(total_loss), 2)
        })
    return {
        "best_f2_threshold": round(float(best_f2_threshold), 3),
        "max_f2_score": round(float(best_f2), 4),
        "best_cost_threshold": round(float(best_cost_threshold), 3),
        "min_expected_loss": round(float(min_cost), 2),
        "cost_assumptions": {"cost_fn": cost_fn, "cost_fp": cost_fp},
        "threshold_curve": threshold_curve
    }
def evaluate_pipeline_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int=5,
    random_state: int=42
) -> Dict[str, Any]:
    """
    Performs out-of-fold stratified cross-validation predictions and calculates metrics.
    """
    skf=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_prob=cross_val_predict(
        pipeline,
        X,
        y,
        cv=skf,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]
    metrics_default=compute_metrics_at_threshold(y.values, y_prob, threshold=0.5)
    threshold_search=find_optimal_threshold(y.values, y_prob)
    metrics_optimal=compute_metrics_at_threshold(
        y.values,
        y_prob,
        threshold=threshold_search["best_f2_threshold"]
    )
    return {
        "cv_folds": n_splits,
        "default_threshold_metrics": metrics_default,
        "optimal_threshold": threshold_search["best_f2_threshold"],
        "optimal_threshold_metrics": metrics_optimal,
        "cost_analysis": {
            "best_cost_threshold": threshold_search["best_cost_threshold"],
            "min_expected_loss": threshold_search["min_expected_loss"]
        },
        "y_prob_oof": y_prob
    }
