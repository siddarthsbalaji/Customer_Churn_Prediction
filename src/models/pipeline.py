"""
Model Pipeline Composition & Serialization for Customer Churn Decision Engine.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.config import PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH
from src.data.cleaner import TelcoDataCleaner
from src.data.feature_engineering import TelcoFeatureEngineer
from src.data.preprocessor import create_column_transformer
def build_churn_model_pipeline(
    classifier: Optional[BaseEstimator]=None,
    drop_id: bool=True
) -> Pipeline:
    """
    Constructs an end-to-end unified Scikit-Learn Pipeline:
    [TelcoDataCleaner] -> [TelcoFeatureEngineer] -> [ColumnTransformer] -> [Classifier]
    """
    if classifier is None:
        classifier=RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    pipeline=Pipeline([
        ("cleaner", TelcoDataCleaner(drop_id=drop_id)),
        ("engineer", TelcoFeatureEngineer(create_cohorts=True)),
        ("preprocessor", create_column_transformer()),
        ("classifier", classifier)
    ])
    return pipeline
def save_pipeline_artifacts(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    pipeline_path: Optional[Path | str]=None,
    metadata_path: Optional[Path | str]=None
) -> None:
    """Serializes pipeline via joblib and saves metadata dictionary as JSON or joblib bundle."""
    if pipeline_path is None:
        pipeline_path=PIPELINE_ARTIFACT_PATH
    if metadata_path is None:
        metadata_path=METADATA_ARTIFACT_PATH
    pipeline_path=Path(pipeline_path)
    metadata_path=Path(metadata_path)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pipeline_path, compress=3)
    import json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
def load_pipeline_artifacts(
    pipeline_path: Optional[Path | str]=None,
    metadata_path: Optional[Path | str]=None
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Loads serialized pipeline and operational metadata."""
    if pipeline_path is None:
        pipeline_path=PIPELINE_ARTIFACT_PATH
    if metadata_path is None:
        metadata_path=METADATA_ARTIFACT_PATH
    pipeline=joblib.load(pipeline_path)
    import json
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata=json.load(f)
    return pipeline, metadata
