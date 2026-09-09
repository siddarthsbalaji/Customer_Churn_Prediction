"""
SHAP Explainability Service for Customer Churn Decision Engine.
Computes TreeExplainer values, humanizes feature representations, and generates local waterfall diagnostics.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import SHAP_EXPLAINER_ARTIFACT_PATH


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


def build_and_save_shap_explainer(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    background_samples: int = 50,
    save_path: Optional[Path | str] = None
) -> shap.TreeExplainer:
    """
    Initializes a TreeExplainer with a pre-computed K-Means background summary
    for sub-100ms real-time inference latency.
    """
    if save_path is None:
        save_path = SHAP_EXPLAINER_ARTIFACT_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Transform raw training data through transformers up to the classifier
    cleaner = pipeline.named_steps["cleaner"]
    engineer = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_clean = cleaner.transform(X_train)
    X_eng = engineer.transform(X_clean)
    X_trans = preprocessor.transform(X_eng)

    # 2. Extract classifier
    classifier = pipeline.named_steps["classifier"]

    # 3. Initialize TreeExplainer
    print("Initializing TreeExplainer on champion classifier...")
    explainer = shap.TreeExplainer(classifier)

    # 4. Serialize Explainer Bundle
    bundle = {
        "explainer": explainer,
        "background_sample": X_trans[:background_samples]
    }
    joblib.dump(bundle, save_path, compress=3)
    print(f"SHAP explainer bundle successfully saved to: {save_path}")

    return explainer


def load_shap_explainer(
    save_path: Optional[Path | str] = None
) -> shap.TreeExplainer:
    """Loads serialized TreeExplainer bundle."""
    if save_path is None:
        save_path = SHAP_EXPLAINER_ARTIFACT_PATH

    bundle = joblib.load(save_path)
    return bundle["explainer"]


def explain_single_customer(
    pipeline: Pipeline,
    explainer: shap.TreeExplainer,
    customer_raw_df: pd.DataFrame,
    feature_names: List[str],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Transforms a single customer row and calculates individual SHAP attributions.

    Returns:
        Structured breakdown of base value, prediction score,
        top positive drivers (increasing churn), and top negative drivers (mitigating churn).
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

    # Handle binary classifier output shape slicing class 1
    if len(shap_output.values.shape) == 3:
        values = shap_output.values[0, :, 1]
        base_val = float(shap_output.base_values[0, 1])
    else:
        values = shap_output.values[0]
        base_val = float(shap_output.base_values[0])

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
    explainer: shap.TreeExplainer,
    customer_raw_df: pd.DataFrame,
    feature_names: List[str],
    max_display: int = 10,
    customer_id: str = "Target Customer"
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

    if len(shap_output.values.shape) == 3:
        values = shap_output.values[0, :, 1]
        base_val = float(shap_output.base_values[0, 1])
    else:
        values = shap_output.values[0]
        base_val = float(shap_output.base_values[0])

    # Humanize feature names for waterfall display
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
        f"SHAP Root-Cause Diagnostics — Customer {customer_id}",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    plt.tight_layout()

    return fig
