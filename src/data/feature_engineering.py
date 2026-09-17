"""
Feature Engineering Transformer for Telco Customer Churn.
Derives service density scores, charge ratios, tenure cohorts, and interaction risk flags.
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
ONLINE_SERVICES=[
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]
class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives actionable business and interaction features:
    1. ServiceCount: Number of value-added digital services subscribed.
    2. ChargeRatio: Velocity of monthly charges relative to cumulative spend.
    3. TenureCohort: Categorical tenure bands (0-12m, 13-24m, 25-48m, 49-72m).
    4. HasHighRiskCombo: Flag for Month-to-month contract + Electronic check payment.
    5. HasFamilySupport: Indicates whether customer has a partner or dependents.
    """
    def __init__(self, create_cohorts: bool=True):
        self.create_cohorts=create_cohorts
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X_df=pd.DataFrame(X)
        else:
            X_df=X.copy()
        available_services=[s for s in ONLINE_SERVICES if s in X_df.columns]
        if available_services:
            X_df["ServiceCount"]=(X_df[available_services] == "Yes").sum(axis=1).astype(int)
        else:
            X_df["ServiceCount"]=0
        if "MonthlyCharges" in X_df.columns and "TotalCharges" in X_df.columns:
            X_df["ChargeRatio"]=(X_df["MonthlyCharges"]/(X_df["TotalCharges"]+1.0)).round(5)
        else:
            X_df["ChargeRatio"]=0.0
        if self.create_cohorts and "tenure" in X_df.columns:
            bins=[-1, 12, 24, 48, 72, np.inf]
            labels=["0-12m", "13-24m", "25-48m", "49-72m", "72m+"]
            X_df["TenureCohort"]=pd.cut(
                X_df["tenure"],
                bins=bins,
                labels=labels
            ).astype(str)
        if "Contract" in X_df.columns and "PaymentMethod" in X_df.columns:
            is_m2m=X_df["Contract"] == "Month-to-month"
            is_echeck=X_df["PaymentMethod"] == "Electronic check"
            X_df["HasHighRiskCombo"]=np.where(is_m2m & is_echeck, "Yes", "No")
        if "Partner" in X_df.columns and "Dependents" in X_df.columns:
            has_partner=X_df["Partner"] == "Yes"
            has_dependents=X_df["Dependents"] == "Yes"
            X_df["HasFamilySupport"]=np.where(has_partner | has_dependents, "Yes", "No")
        return X_df
