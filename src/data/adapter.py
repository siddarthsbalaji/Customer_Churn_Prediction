"""
Custom Dataset Adapter & Alignment Engine.
Transforms arbitrary customer datasets with novel metrics into model-compatible
DataFrames using AI classification results while preserving custom domain attributes (e.g. nationality).
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.ai.metric_classifier import AIMetricClassificationResult
from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CANONICAL_STATISTICAL_DEFAULTS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def normalize_categorical_value(feature: str, val: Any) -> Any:
    """Normalizes arbitrary categorical strings to standard canonical categories."""
    if pd.isna(val) or val is None:
        return CANONICAL_STATISTICAL_DEFAULTS.get(feature, "No")

    s = str(val).strip()

    if feature == "Contract":
        s_lower = s.lower()
        if any(term in s_lower for term in ["two", "2 yr", "2-yr", "24m", "2 year"]):
            return "Two year"
        elif any(term in s_lower for term in ["one", "1 yr", "1-yr", "12m", "1 year", "annual"]):
            return "One year"
        return "Month-to-month"

    if feature == "PaymentMethod":
        s_lower = s.lower()
        if "electronic" in s_lower or "e-check" in s_lower:
            return "Electronic check"
        elif "mail" in s_lower or "check" in s_lower:
            return "Mailed check"
        elif "bank" in s_lower or "transfer" in s_lower:
            return "Bank transfer (automatic)"
        elif "card" in s_lower or "credit" in s_lower:
            return "Credit card (automatic)"
        return "Electronic check"

    if feature == "InternetService":
        s_lower = s.lower()
        if "fiber" in s_lower:
            return "Fiber optic"
        elif "dsl" in s_lower or "broadband" in s_lower or "cable" in s_lower:
            return "DSL"
        elif "no" in s_lower or "none" in s_lower:
            return "No"
        return "DSL"

    if feature == "gender":
        s_lower = s.lower()
        if s_lower in ["m", "male", "man", "1"]:
            return "Male"
        return "Female"

    # Generic Yes/No fields
    if feature in ["Partner", "Dependents", "PhoneService", "MultipleLines",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]:
        s_lower = s.lower()
        if s_lower in ["1", "true", "yes", "y", "t", "active"]:
            return "Yes"
        elif s_lower in ["0", "false", "no", "n", "f", "inactive"]:
            return "No"
        elif feature in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"] and "no internet" in s_lower:
            return "No internet service"
        elif feature == "MultipleLines" and "no phone" in s_lower:
            return "No phone service"
        return "No"

    if feature == "SeniorCitizen":
        s_lower = s.lower()
        if s_lower in ["1", "true", "yes", "y"]:
            return 1
        return 0

    return s


def adapt_custom_dataset(
    raw_df: pd.DataFrame,
    classification: AIMetricClassificationResult,
    user_mapping_override: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Adapts an arbitrary dataset using AI classification and statistical default imputation:
    1. Renames mapped columns to canonical features.
    2. Retains custom domain metrics (like 'nationality', 'credit_score') for UI context.
    3. Normalizes categorical values.
    4. Auto-imputes missing core/optional features with statistical defaults.
    5. Coerces numeric features.
    6. Generates synthetic customerIDs if missing.

    Returns:
        (aligned_df, adapter_metadata)
    """
    aligned_df = raw_df.copy()
    warnings: List[str] = []
    imputed_features: Dict[str, Any] = {}

    # Merge AI mapping with user overrides if provided
    active_mapping: Dict[str, str] = dict(classification.mapped_canonical)
    if user_mapping_override:
        for canonical, original in user_mapping_override.items():
            if original and original in raw_df.columns:
                active_mapping[canonical] = original

    # Track preserved custom attributes (e.g. nationality, credit_score, balance)
    mapped_original_cols = set(active_mapping.values())
    preserved_custom_cols = [c for c in raw_df.columns if c not in mapped_original_cols]

    # Apply column renaming (original -> canonical)
    rename_dict = {orig: canon for canon, orig in active_mapping.items()}
    aligned_df = aligned_df.rename(columns=rename_dict)

    # 1. Ensure Customer ID exists
    if ID_COLUMN not in aligned_df.columns:
        aligned_df[ID_COLUMN] = [f"ACCOUNT-{i+1:05d}" for i in range(len(aligned_df))]
        warnings.append(f"Generated synthetic {ID_COLUMN} for records.")

    # 2. Value mapping & normalization for mapped features
    for col in aligned_df.columns:
        if col in CATEGORICAL_FEATURES or col == "SeniorCitizen":
            aligned_df[col] = aligned_df[col].apply(lambda v, c=col: normalize_categorical_value(c, v))

    # 3. Auto-impute missing canonical features using statistical defaults
    for feat in ALL_FEATURES:
        if feat not in aligned_df.columns:
            default_val = CANONICAL_STATISTICAL_DEFAULTS.get(feat, "No")
            aligned_df[feat] = default_val
            imputed_features[feat] = default_val

    if imputed_features:
        warnings.append(
            f"Auto-imputed {len(imputed_features)} missing canonical features with statistical defaults: "
            f"{list(imputed_features.keys())}"
        )

    # 4. Numeric hygiene
    for num_col in NUMERIC_FEATURES:
        if num_col in aligned_df.columns:
            numeric_series = pd.to_numeric(
                aligned_df[num_col].astype(str).str.strip().replace(r"^\s*$", np.nan, regex=True),
                errors="coerce"
            )
            median_val = CANONICAL_STATISTICAL_DEFAULTS.get(num_col, 0.0)
            null_count = int(numeric_series.isna().sum())
            if null_count > 0:
                numeric_series = numeric_series.fillna(median_val)
                warnings.append(f"Filled {null_count} null/non-numeric values in '{num_col}' with median {median_val}.")
            aligned_df[num_col] = numeric_series

    # If TotalCharges was imputed or missing, approximate from tenure * MonthlyCharges if feasible
    if "TotalCharges" in imputed_features and "tenure" in aligned_df.columns and "MonthlyCharges" in aligned_df.columns:
        aligned_df["TotalCharges"] = (aligned_df["tenure"] * aligned_df["MonthlyCharges"]).round(2)
        warnings.append("Synthesized 'TotalCharges' as (tenure × MonthlyCharges).")

    adapter_metadata = {
        "provider_used": classification.provider_used,
        "status_message": classification.status_message,
        "mapped_columns": active_mapping,
        "preserved_custom_columns": preserved_custom_cols,
        "imputed_features": imputed_features,
        "total_records": len(aligned_df),
        "warnings": warnings,
    }

    return aligned_df, adapter_metadata
