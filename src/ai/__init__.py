"""
AI Metric Classification & Schema Understanding Package.
"""
from src.ai.metric_classifier import (
    MetricClassification,
    AIMetricClassificationResult,
    classify_dataset,
    inspect_dataset_schema,
)

__all__ = [
    "MetricClassification",
    "AIMetricClassificationResult",
    "classify_dataset",
    "inspect_dataset_schema",
]
