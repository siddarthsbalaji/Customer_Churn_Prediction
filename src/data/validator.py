"""
Schema Validation and Column Alias Resolution for Dynamic User CSV Uploads.
Ensures arbitrary customer datasets conform to model pipeline expectations.
"""
import re
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    COLUMN_ALIASES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
CORE_MANDATORY_COLUMNS=[
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "PaymentMethod"
]
def normalize_column_name(name: str) -> str:
    """Normalizes column name by stripping whitespace, underscores, dashes, and casing."""
    return re.sub(r"[_\s\-]+", "", str(name)).lower()
def build_alias_lookup() -> Dict[str, str]:
    """Builds a flat lookup mapping normalized alias strings to canonical column names."""
    lookup={}
    for canonical, aliases in COLUMN_ALIASES.items():
        lookup[normalize_column_name(canonical)]=canonical
        for alias in aliases:
            lookup[normalize_column_name(alias)]=canonical
    return lookup
def validate_and_align_dataset(
    df: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """
    Validates uploaded DataFrame, maps aliased columns, checks for required fields,
    coerces numeric types, and fills missing optional features with imputed defaults.
    Returns:
        (aligned_df, missing_required_columns, warnings)
    """
    warnings: List[str]=[]
    missing_required: List[str]=[]
    aligned_df=df.copy()
    alias_lookup=build_alias_lookup()
    rename_mapping: Dict[str, str]={}
    normalized_incoming={normalize_column_name(c): c for c in aligned_df.columns}
    for norm_alias, canonical in alias_lookup.items():
        if norm_alias in normalized_incoming:
            original_col=normalized_incoming[norm_alias]
            rename_mapping[original_col]=canonical
    aligned_df=aligned_df.rename(columns=rename_mapping)
    for mandatory_col in CORE_MANDATORY_COLUMNS:
        if mandatory_col not in aligned_df.columns:
            missing_required.append(mandatory_col)
    if missing_required:
        return None, missing_required, warnings
    for feat in ALL_FEATURES:
        if feat not in aligned_df.columns:
            if feat in NUMERIC_FEATURES:
                aligned_df[feat]=np.nan
                warnings.append(f"Missing numerical column '{feat}' was auto-populated with NaN for imputer.")
            else:
                aligned_df[feat] = "No" if feat not in ["Contract", "PaymentMethod"] else "Missing"
                warnings.append(f"Missing categorical column '{feat}' was auto-populated with default placeholder.")
    for num_col in NUMERIC_FEATURES:
        if num_col in aligned_df.columns:
            initial_nans=aligned_df[num_col].isna().sum()
            numeric_series=pd.to_numeric(
                aligned_df[num_col].astype(str).str.strip().replace(r"^\s*$", np.nan, regex=True),
                errors="coerce"
            )
            new_nans=numeric_series.isna().sum()-initial_nans
            if new_nans>0:
                warnings.append(f"Column '{num_col}' contained {new_nans} non-numeric entries coerced to NaN.")
            aligned_df[num_col]=numeric_series
    if ID_COLUMN not in aligned_df.columns:
        aligned_df[ID_COLUMN]=[f"UPLOAD-{i+1:05d}" for i in range(len(aligned_df))]
        warnings.append(f"No '{ID_COLUMN}' detected; synthetic customer IDs generated.")
    return aligned_df, [], warnings
