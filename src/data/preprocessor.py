"""
Unified Preprocessor Pipeline for Customer Churn Decision Engine.
Integrates cleaning, feature engineering, and Scikit-Learn ColumnTransformer.
"""
from typing import Tuple, List, Optional, Union
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)
from src.data.cleaner import TelcoDataCleaner, encode_target_series
from src.data.feature_engineering import TelcoFeatureEngineer

# Post-engineering feature subsets
EXTENDED_NUMERIC_FEATURES = NUMERIC_FEATURES + ["ServiceCount", "ChargeRatio"]
EXTENDED_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + [
    "TenureCohort",
    "HasHighRiskCombo",
    "HasFamilySupport"
]


def create_column_transformer() -> ColumnTransformer:
    """Creates Scikit-Learn ColumnTransformer for numeric and categorical subsets."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, EXTENDED_NUMERIC_FEATURES),
            ("cat", categorical_transformer, EXTENDED_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return column_transformer


def create_preprocessor_pipeline(drop_id: bool = True) -> Pipeline:
    """
    Creates an end-to-end data transformation pipeline:
    TelcoDataCleaner -> TelcoFeatureEngineer -> ColumnTransformer
    """
    return Pipeline([
        ("cleaner", TelcoDataCleaner(drop_id=drop_id)),
        ("engineer", TelcoFeatureEngineer(create_cohorts=True)),
        ("preprocessor", create_column_transformer())
    ])


def get_feature_names(col_transformer: ColumnTransformer) -> List[str]:
    """Extracts post-encoding feature names from fitted ColumnTransformer."""
    feature_names = []
    for name, transformer, cols in col_transformer.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names = list(transformer.get_feature_names_out(cols))
            feature_names.extend(names)
        elif hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            names = list(transformer.named_steps["onehot"].get_feature_names_out(cols))
            feature_names.extend(names)
        else:
            feature_names.extend(cols)
    return feature_names


def load_raw_dataset(csv_path: Optional[Path | str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw IBM Telco dataset and separates features X and encoded target y.
    """
    if csv_path is None:
        csv_path = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = pd.read_csv(csv_path)

    if TARGET_COLUMN in df.columns:
        y = encode_target_series(df[TARGET_COLUMN])
        X = df.drop(columns=[TARGET_COLUMN])
    else:
        y = None
        X = df

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Performs stratified train/test split to preserve churn class ratio."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
