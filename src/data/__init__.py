"""
Data processing package for Customer Churn Decision Engine.
"""
from src.data.cleaner import TelcoDataCleaner, encode_target_series
from src.data.feature_engineering import TelcoFeatureEngineer
from src.data.preprocessor import (
    create_preprocessor_pipeline,
    create_column_transformer,
    get_feature_names,
    load_raw_dataset,
    split_data,
    EXTENDED_NUMERIC_FEATURES,
    EXTENDED_CATEGORICAL_FEATURES,
)
from src.data.validator import validate_and_align_dataset, CORE_MANDATORY_COLUMNS
from src.data.adapter import adapt_custom_dataset

__all__=[
    "TelcoDataCleaner",
    "encode_target_series",
    "TelcoFeatureEngineer",
    "create_preprocessor_pipeline",
    "create_column_transformer",
    "get_feature_names",
    "load_raw_dataset",
    "split_data",
    "EXTENDED_NUMERIC_FEATURES",
    "EXTENDED_CATEGORICAL_FEATURES",
    "validate_and_align_dataset",
    "CORE_MANDATORY_COLUMNS",
    "adapt_custom_dataset",
]
