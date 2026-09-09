"""
Custom Data Cleaner Transformer for IBM Telco Customer Churn dataset.
Handles missing/whitespace entries in TotalCharges, type coercion, and target encoding.
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import ID_COLUMN, TARGET_COLUMN


class TelcoDataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans raw customer records:
    1. Trims whitespace from all string/object columns.
    2. Coerces 'TotalCharges' whitespace strings to numeric, filling zero-tenure rows with 0.0.
    3. Handles binary target 'Churn' conversion ('Yes' -> 1, 'No' -> 0) if present.
    """
    def __init__(self, drop_id: bool = True):
        self.drop_id = drop_id

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        # 1. Trim whitespace on string-like columns
        for col in X_df.select_dtypes(include=["object", "string", "str"]).columns:
            X_df[col] = X_df[col].astype(str).str.strip()

        # 2. Fix TotalCharges whitespace anomaly
        if "TotalCharges" in X_df.columns:
            # Replace empty strings or whitespace with NaN, then convert to numeric
            X_df["TotalCharges"] = pd.to_numeric(
                X_df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True),
                errors="coerce"
            )
            # If tenure == 0, TotalCharges is legitimately 0.0
            if "tenure" in X_df.columns:
                zero_tenure_mask = (X_df["tenure"] == 0) & (X_df["TotalCharges"].isna())
                X_df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

            # Median fill any remaining residual NaNs
            X_df["TotalCharges"] = X_df["TotalCharges"].fillna(0.0)

        # 3. Ensure numeric columns are properly typed
        if "tenure" in X_df.columns:
            X_df["tenure"] = pd.to_numeric(X_df["tenure"], errors="coerce").fillna(0).astype(int)

        if "MonthlyCharges" in X_df.columns:
            X_df["MonthlyCharges"] = pd.to_numeric(X_df["MonthlyCharges"], errors="coerce").fillna(0.0)

        # 4. Optional removal of customerID from feature matrix
        if self.drop_id and ID_COLUMN in X_df.columns:
            X_df = X_df.drop(columns=[ID_COLUMN])

        return X_df


def encode_target_series(y: pd.Series) -> pd.Series:
    """Safely converts target Churn series ('Yes'/'No') to 1/0 integers."""
    if y is None:
        return None
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)
    return y.astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int)
