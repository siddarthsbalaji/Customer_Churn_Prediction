"""
Model Training & Tournament Orchestrator for Customer Churn Decision Engine.
Trains Baseline (Logistic Regression) vs. Champion (Random Forest),
optimizes decision threshold, and serializes production pipeline artifacts.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.config import (
    MODELS_DIR,
    PIPELINE_ARTIFACT_PATH,
    METADATA_ARTIFACT_PATH,
    RAW_DATA_DIR,
)
from src.data.preprocessor import (
    get_feature_names,
    load_raw_dataset,
    split_data,
)
from src.models.evaluate import (
    compute_metrics_at_threshold,
    evaluate_pipeline_cv,
    find_optimal_threshold,
)
from src.models.pipeline import (
    build_churn_model_pipeline,
    save_pipeline_artifacts,
)
def get_candidate_models() -> Dict[str, Any]:
    """Returns candidate estimators for tournament evaluation."""
    return {
        "LogisticRegression_Baseline": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ),
        "RandomForest_Champion": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }
def train_and_evaluate_tournament(
    csv_path: Path | str | None=None,
    test_size: float=0.2,
    cv_folds: int=5,
    random_state: int=42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Executes model tournament:
    1. Loads data and splits into train/test sets.
    2. Runs 5-Fold Stratified CV for each candidate.
    3. Evaluates test set performance and determines optimal business threshold.
    4. Serializes champion pipeline and metadata.
    """
    print("=" * 65)
    print("  Customer Churn Decision Engine: Model Training & Evaluation")
    print("=" * 65)
    X, y=load_raw_dataset(csv_path)
    print(f"Dataset Loaded: {len(X):,} records. Positive Churn: {y.mean() * 100:.2f}%")
    X_train, X_test, y_train, y_test=split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Splits: Train = {len(X_train):,} rows | Test = {len(X_test):,} rows")
    candidates=get_candidate_models()
    tournament_results={}
    fitted_pipelines={}
    print("\n--- Running 5-Fold Stratified Cross-Validation Tournament ---")
    for name, clf in candidates.items():
        print(f"\nEvaluating candidate: {name} ...")
        pipeline=build_churn_model_pipeline(classifier=clf)
        cv_results=evaluate_pipeline_cv(
            pipeline, X_train, y_train, n_splits=cv_folds, random_state=random_state
        )
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name]=pipeline
        y_test_prob=pipeline.predict_proba(X_test)[:, 1]
        test_metrics_default=compute_metrics_at_threshold(y_test.values, y_test_prob, threshold=0.5)
        train_opt_search=find_optimal_threshold(y_train.values, cv_results["y_prob_oof"])
        opt_thresh=train_opt_search["best_f2_threshold"]
        test_metrics_optimal=compute_metrics_at_threshold(y_test.values, y_test_prob, threshold=opt_thresh)
        tournament_results[name]={
            "cv_metrics_default": cv_results["default_threshold_metrics"],
            "cv_metrics_optimal": cv_results["optimal_threshold_metrics"],
            "test_metrics_default": test_metrics_default,
            "test_metrics_optimal": test_metrics_optimal,
            "calibrated_threshold": opt_thresh,
            "cost_optimal_threshold": train_opt_search["best_cost_threshold"]
        }
        print(f"  CV ROC-AUC:      {cv_results['default_threshold_metrics']['roc_auc']:.4f}")
        print(f"  CV PR-AUC:       {cv_results['default_threshold_metrics']['pr_auc']:.4f}")
        print(f"  Test Recall (0.5): {test_metrics_default['recall']:.4f} | Test Precision: {test_metrics_default['precision']:.4f}")
        print(f"  Calibrated Threshold (τ*): {opt_thresh:.3f}")
        print(f"  Test Recall (τ*):  {test_metrics_optimal['recall']:.4f} | Test F2: {test_metrics_optimal['f2']:.4f}")
    champion_name = "RandomForest_Champion"
    champion_pipeline=fitted_pipelines[champion_name]
    champion_eval=tournament_results[champion_name]
    calibrated_threshold=champion_eval["calibrated_threshold"]
    feature_names=get_feature_names(fitted_pipelines["RandomForest_Champion"].named_steps["preprocessor"])
    rf_eval=tournament_results["RandomForest_Champion"]
    lr_eval=tournament_results["LogisticRegression_Baseline"]
    metadata={
        "default_model": "random_forest",
        "model_name": "RandomForest_Champion",
        "model_type": "RandomForestClassifier",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_records": len(X_train),
        "test_records": len(X_test),
        "optimal_threshold": rf_eval["calibrated_threshold"],
        "cost_optimal_threshold": rf_eval["cost_optimal_threshold"],
        "champion_test_metrics": rf_eval["test_metrics_optimal"],
        "models": {
            "random_forest": {
                "key": "random_forest",
                "model_name": "RandomForest_Champion",
                "display_name": "Random Forest (Champion Ensemble)",
                "model_type": "RandomForestClassifier",
                "pipeline_file": "random_forest_pipeline.joblib",
                "optimal_threshold": rf_eval["calibrated_threshold"],
                "cost_optimal_threshold": rf_eval["cost_optimal_threshold"],
                "metrics_default": rf_eval["test_metrics_default"],
                "metrics_calibrated": rf_eval["test_metrics_optimal"]
            },
            "logistic_regression": {
                "key": "logistic_regression",
                "model_name": "LogisticRegression_Baseline",
                "display_name": "Logistic Regression (Linear Baseline)",
                "model_type": "LogisticRegression",
                "pipeline_file": "logistic_regression_pipeline.joblib",
                "optimal_threshold": lr_eval["calibrated_threshold"],
                "cost_optimal_threshold": lr_eval["cost_optimal_threshold"],
                "metrics_default": lr_eval["test_metrics_default"],
                "metrics_calibrated": lr_eval["test_metrics_optimal"]
            }
        },
        "tournament_comparison": {
            k: {
                "test_roc_auc": v["test_metrics_default"]["roc_auc"],
                "test_pr_auc": v["test_metrics_default"]["pr_auc"],
                "default_recall": v["test_metrics_default"]["recall"],
                "calibrated_threshold": v["calibrated_threshold"],
                "calibrated_recall": v["test_metrics_optimal"]["recall"],
                "calibrated_f2": v["test_metrics_optimal"]["f2"]
            }
            for k, v in tournament_results.items()
        },
        "encoded_feature_count": len(feature_names),
        "encoded_feature_names": feature_names
    }
    print(f"\n--- Serializing Multi-Model Pipelines & Metadata ---")
    from src.config import RF_PIPELINE_PATH, LR_PIPELINE_PATH
    save_pipeline_artifacts(
        pipeline=fitted_pipelines["RandomForest_Champion"],
        metadata=metadata,
        pipeline_path=RF_PIPELINE_PATH,
        metadata_path=METADATA_ARTIFACT_PATH
    )
    save_pipeline_artifacts(
        pipeline=fitted_pipelines["RandomForest_Champion"],
        metadata=metadata,
        pipeline_path=PIPELINE_ARTIFACT_PATH,
        metadata_path=METADATA_ARTIFACT_PATH
    )
    save_pipeline_artifacts(
        pipeline=fitted_pipelines["LogisticRegression_Baseline"],
        metadata=metadata,
        pipeline_path=LR_PIPELINE_PATH,
        metadata_path=METADATA_ARTIFACT_PATH
    )
    print(f"Random Forest Pipeline saved to:     {RF_PIPELINE_PATH}")
    print(f"Logistic Regression Pipeline saved to: {LR_PIPELINE_PATH}")
    print(f"Master Metadata saved to:              {METADATA_ARTIFACT_PATH}")
    print("=" * 65)
    return fitted_pipelines, metadata
if __name__ == "__main__":
    train_and_evaluate_tournament()
