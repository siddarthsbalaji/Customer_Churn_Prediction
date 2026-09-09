"""
SHAP Explainability Service for Customer Churn Decision Engine.
Supports TreeExplainer (Random Forest) and LinearExplainer (Logistic Regression),
feature humanization, local waterfall diagnostics, and cross-model comparison.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import (
    LR_EXPLAINER_PATH,
    RF_EXPLAINER_PATH,
    SHAP_EXPLAINER_ARTIFACT_PATH,
)


def humanize_feature_name(feature_col: str) -> str:
    """Translates raw one-hot encoded and engineered feature names into executive-friendly labels."""
    mapping = {
        "tenure": "Tenure Duration (Months)",
        "MonthlyCharges": "Monthly Billing Rate ($)",
        "TotalCharges": "Cumulative Lifetime Spend ($)",
        "ServiceCount": "Digital Add-on Density (Count)",
        "ChargeRatio": "Charge-to-Spend Velocity",
        "Contract_Month-to-month": "Month-to-month Contract (High Risk)",
        "Contract_One year": "1-Year Fixed Contract",
        "Contract_Two year": "2-Year Fixed Contract",
        "PaymentMethod_Electronic check": "Electronic Check Billing",
        "PaymentMethod_Bank transfer (automatic)": "Automatic Bank Direct Debit",
        "PaymentMethod_Credit card (automatic)": "Automatic Credit Card Billing",
        "PaymentMethod_Mailed check": "Mailed Paper Check",
        "InternetService_Fiber optic": "Fiber Optic Broadband",
        "InternetService_DSL": "DSL Broadband",
        "InternetService_No": "No Internet Subscription",
        "TechSupport_No": "Lacks Technical Support",
        "TechSupport_Yes": "Active VIP Tech Support",
        "OnlineSecurity_No": "Lacks Online Security Protection",
        "OnlineSecurity_Yes": "Active Online Security Suite",
        "OnlineBackup_No": "Lacks Cloud Backup",
        "OnlineBackup_Yes": "Active Cloud Backup",
        "DeviceProtection_No": "Lacks Device Warranty",
        "DeviceProtection_Yes": "Active Device Protection",
        "PaperlessBilling_Yes": "Paperless Billing Enabled",
        "PaperlessBilling_No": "Paper Invoicing",
        "SeniorCitizen_1": "Senior Citizen Account",
        "SeniorCitizen_0": "Non-Senior Account",
        "HasHighRiskCombo_Yes": "High-Risk Contract & Payment Pairing",
        "HasHighRiskCombo_No": "Standard Contract/Billing Pairing",
        "TenureCohort_0-12m": "Early-Life Cohort (0-12 Months)",
        "TenureCohort_49-72m": "Mature Tenured Cohort (4+ Years)"
    }
    return mapping.get(feature_col, feature_col.replace("_", " "))


def build_and_save_all_explainers(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    background_samples: int = 50
) -> Dict[str, Any]:
    """
    Constructs and serializes explainers for both Random Forest (TreeExplainer)
    and Logistic Regression (LinearExplainer).
    """
    explainers = {}
    rf_pipe = pipelines.get("random_forest")
    lr_pipe = pipelines.get("logistic_regression")

    if rf_pipe is None and "RandomForest_Champion" in pipelines:
        rf_pipe = pipelines["RandomForest_Champion"]
    if lr_pipe is None and "LogisticRegression_Baseline" in pipelines:
        lr_pipe = pipelines["LogisticRegression_Baseline"]

    # Transform X_train once using RF preprocessor
    cleaner = rf_pipe.named_steps["cleaner"]
    engineer = rf_pipe.named_steps["engineer"]
    preprocessor = rf_pipe.named_steps["preprocessor"]

    X_clean = cleaner.transform(X_train)
    X_eng = engineer.transform(X_clean)
    X_trans = preprocessor.transform(X_eng)
    background_sample = X_trans[:background_samples]

    # 1. Random Forest TreeExplainer
    print("Building Random Forest TreeExplainer...")
    rf_clf = rf_pipe.named_steps["classifier"]
    rf_explainer = shap.TreeExplainer(rf_clf)
    explainers["random_forest"] = rf_explainer

    RF_EXPLAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"explainer": rf_explainer, "background_sample": background_sample}, RF_EXPLAINER_PATH, compress=3)
    joblib.dump({"explainer": rf_explainer, "background_sample": background_sample}, SHAP_EXPLAINER_ARTIFACT_PATH, compress=3)

    # 2. Logistic Regression LinearExplainer
    print("Building Logistic Regression LinearExplainer...")
    lr_clf = lr_pipe.named_steps["classifier"]
    lr_explainer = shap.LinearExplainer(lr_clf, background_sample)
    explainers["logistic_regression"] = lr_explainer

    LR_EXPLAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"explainer": lr_explainer, "background_sample": background_sample}, LR_EXPLAINER_PATH, compress=3)

    print(f"Random Forest explainer saved to:     {RF_EXPLAINER_PATH}")
    print(f"Logistic Regression explainer saved to: {LR_EXPLAINER_PATH}")

    return explainers


def build_and_save_shap_explainer(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    background_samples: int = 50,
    save_path: Optional[Path | str] = None
) -> Any:
    """Backward-compatible helper to serialize default champion TreeExplainer."""
    if save_path is None:
        save_path = SHAP_EXPLAINER_ARTIFACT_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cleaner = pipeline.named_steps["cleaner"]
    engineer = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_clean = cleaner.transform(X_train)
    X_eng = engineer.transform(X_clean)
    X_trans = preprocessor.transform(X_eng)

    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)

    bundle = {
        "explainer": explainer,
        "background_sample": X_trans[:background_samples]
    }
    joblib.dump(bundle, save_path, compress=3)
    return explainer


def load_shap_explainer(
    save_path: Optional[Path | str] = None
) -> Any:
    """Loads serialized SHAP explainer bundle."""
    if save_path is None:
        save_path = SHAP_EXPLAINER_ARTIFACT_PATH

    bundle = joblib.load(save_path)
    return bundle["explainer"]


def extract_shap_values_and_base(shap_output: Any) -> Tuple[np.ndarray, float]:
    """Normalizes output shapes across TreeExplainer (3D/2D) and LinearExplainer (2D/1D)."""
    if len(shap_output.values.shape) == 3:
        values = shap_output.values[0, :, 1]
        base_val = float(shap_output.base_values[0, 1])
    elif len(shap_output.values.shape) == 2:
        values = shap_output.values[0]
        b = shap_output.base_values[0]
        base_val = float(b[1]) if hasattr(b, "__len__") and len(b) > 1 else float(b)
    else:
        values = shap_output.values
        base_val = float(shap_output.base_values)
    return values, base_val


def explain_single_customer(
    pipeline: Pipeline,
    explainer: Any,
    customer_raw_df: pd.DataFrame,
    feature_names: List[str],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Transforms a single customer row and calculates individual SHAP attributions.
    Compatible with both Random Forest and Logistic Regression explainers.
    """
    cleaner = pipeline.named_steps["cleaner"]
    engineer = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Transform record
    X_clean = cleaner.transform(customer_raw_df)
    X_eng = engineer.transform(X_clean)
    X_trans = preprocessor.transform(X_eng)

    # Compute SHAP values
    shap_output = explainer(X_trans)
    values, base_val = extract_shap_values_and_base(shap_output)

    prob_score = float(pipeline.predict_proba(customer_raw_df)[0][1])

    # Rank features by contribution
    features_ranked = []
    for idx, shap_val in enumerate(values):
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        features_ranked.append({
            "feature": feat_name,
            "display_name": humanize_feature_name(feat_name),
            "shap_value": float(round(shap_val, 4)),
            "feature_value": float(round(X_trans[0, idx], 3)) if idx < X_trans.shape[1] else None
        })

    # Partition into positive risk drivers and negative protective drivers
    risk_drivers = sorted([f for f in features_ranked if f["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
    protective_drivers = sorted([f for f in features_ranked if f["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

    return {
        "base_value": round(base_val, 4),
        "prediction_probability": round(prob_score, 4),
        "risk_drivers": risk_drivers,
        "protective_factors": protective_drivers,
        "raw_shap_values": values,
        "transformed_features": X_trans[0]
    }


def render_customer_waterfall_figure(
    pipeline: Pipeline,
    explainer: Any,
    customer_raw_df: pd.DataFrame,
    feature_names: List[str],
    max_display: int = 10,
    customer_id: str = "Target Customer",
    model_name: str = "Model"
) -> plt.Figure:
    """
    Renders a formatted Matplotlib Waterfall plot for a selected customer row.
    """
    cleaner = pipeline.named_steps["cleaner"]
    engineer = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_clean = cleaner.transform(customer_raw_df)
    X_eng = engineer.transform(X_clean)
    X_trans = preprocessor.transform(X_eng)

    shap_output = explainer(X_trans)
    values, base_val = extract_shap_values_and_base(shap_output)

    humanized_names = [humanize_feature_name(f) for f in feature_names]

    explanation = shap.Explanation(
        values=values,
        base_values=base_val,
        data=X_trans[0],
        feature_names=humanized_names
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.title(
        f"SHAP Root-Cause Diagnostics ({model_name}) — Account {customer_id}",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    plt.tight_layout()

    return fig


def compare_customer_models(
    pipelines: Dict[str, Pipeline],
    explainers: Dict[str, Any],
    customer_raw_df: pd.DataFrame,
    feature_names: List[str],
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """
    Generates a head-to-head comparative diagnosis of a customer between
    Random Forest and Logistic Regression.
    """
    rf_pipe = pipelines["random_forest"]
    lr_pipe = pipelines["logistic_regression"]
    rf_exp = explainers["random_forest"]
    lr_exp = explainers["logistic_regression"]

    rf_diag = explain_single_customer(rf_pipe, rf_exp, customer_raw_df, feature_names, top_k=3)
    lr_diag = explain_single_customer(lr_pipe, lr_exp, customer_raw_df, feature_names, top_k=3)

    rf_prob = rf_diag["prediction_probability"]
    lr_prob = lr_diag["prediction_probability"]
    delta_prob = round(rf_prob - lr_prob, 4)

    rf_thresh = thresholds.get("random_forest", 0.210)
    lr_thresh = thresholds.get("logistic_regression", 0.310)

    rf_tier = "CRITICAL" if rf_prob >= 0.70 else ("MODERATE" if rf_prob >= 0.40 else "LOW")
    lr_tier = "CRITICAL" if lr_prob >= 0.70 else ("MODERATE" if lr_prob >= 0.40 else "LOW")

    # Driver agreement
    rf_driver_names = set(d["display_name"] for d in rf_diag["risk_drivers"])
    lr_driver_names = set(d["display_name"] for d in lr_diag["risk_drivers"])
    agreed_drivers = list(rf_driver_names.intersection(lr_driver_names))

    return {
        "random_forest": {
            "churn_probability": rf_prob,
            "risk_tier": rf_tier,
            "is_at_risk": rf_prob >= rf_thresh,
            "top_drivers": rf_diag["risk_drivers"]
        },
        "logistic_regression": {
            "churn_probability": lr_prob,
            "risk_tier": lr_tier,
            "is_at_risk": lr_prob >= lr_thresh,
            "top_drivers": lr_diag["risk_drivers"]
        },
        "probability_delta": delta_prob,
        "tier_agreement": rf_tier == lr_tier,
        "agreed_risk_drivers": agreed_drivers
    }
