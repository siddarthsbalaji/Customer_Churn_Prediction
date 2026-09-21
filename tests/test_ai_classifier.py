"""
Unit & Integration Tests for AI-Powered Metric Classification & Dataset Adapter.
"""
import unittest
import pandas as pd
import numpy as np

from src.ai.metric_classifier import (
    classify_dataset,
    heuristic_classify_metrics,
    inspect_dataset_schema,
    _extract_json_payload,
    AIMetricClassificationResult,
)
from src.data.adapter import adapt_custom_dataset, normalize_categorical_value
from src.config import (
    ALL_FEATURES,
    CANONICAL_STATISTICAL_DEFAULTS,
    ID_COLUMN,
    RF_PIPELINE_PATH,
    SAMPLE_UPLOADS_DIR,
)
from src.models.pipeline import load_pipeline_artifacts
from src.config import METADATA_ARTIFACT_PATH


class TestAIMetricClassifier(unittest.TestCase):
    """Test suite for schema inspection, heuristic classification, and JSON extraction."""

    def setUp(self):
        self.banking_csv = SAMPLE_UPLOADS_DIR / "custom_banking_churn.csv"
        self.saas_csv = SAMPLE_UPLOADS_DIR / "custom_saas_churn.csv"

    def test_schema_inspection(self):
        df = pd.DataFrame({
            "nationality": ["France", "Spain", "Germany"],
            "balance": [1000.0, 2500.5, 0.0],
            "client_id": ["A1", "A2", "A3"]
        })
        summary = inspect_dataset_schema(df)
        self.assertEqual(len(summary), 3)
        col_names = [s["column_name"] for s in summary]
        self.assertIn("nationality", col_names)
        self.assertIn("balance", col_names)
        self.assertIn("client_id", col_names)

    def test_extract_json_payload(self):
        raw_markdown = """Here is the schema mapping:
```json
{
  "metrics": [
    {
      "original_name": "nationality",
      "role": "DEMOGRAPHIC",
      "canonical_mapping": null,
      "confidence": 0.95,
      "reasoning": "Customer country."
    }
  ]
}
```
"""
        parsed = _extract_json_payload(raw_markdown)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["metrics"][0]["original_name"], "nationality")

    def test_heuristic_classification_banking_metrics(self):
        df = pd.read_csv(self.banking_csv)
        report = classify_dataset(df)

        self.assertIn(report.provider_used, ["heuristic_fallback", "ai_gemini", "ai_openai"])
        self.assertIn("nationality", report.custom_attributes)
        self.assertEqual(report.mapped_canonical.get("tenure"), "account_age_months")
        self.assertEqual(report.mapped_canonical.get("MonthlyCharges"), "monthly_fee")
        self.assertEqual(report.mapped_canonical.get("Contract"), "plan_duration")
        self.assertEqual(report.mapped_canonical.get("PaymentMethod"), "payment_channel")

    def test_heuristic_classification_saas_metrics(self):
        df = pd.read_csv(self.saas_csv)
        report = classify_dataset(df)

        self.assertIn("country", report.custom_attributes)
        self.assertEqual(report.mapped_canonical.get("MonthlyCharges"), "mrr")
        self.assertEqual(report.mapped_canonical.get("tenure"), "months_active")


class TestCustomDatasetAdapter(unittest.TestCase):
    """Test suite for dataset adapter, normalization, and statistical imputation."""

    def setUp(self):
        self.banking_df = pd.read_csv(SAMPLE_UPLOADS_DIR / "custom_banking_churn.csv")

    def test_normalize_categorical_value(self):
        self.assertEqual(normalize_categorical_value("Contract", "Annual"), "One year")
        self.assertEqual(normalize_categorical_value("Contract", "24m"), "Two year")
        self.assertEqual(normalize_categorical_value("Contract", "Monthly"), "Month-to-month")
        self.assertEqual(normalize_categorical_value("PaymentMethod", "e-check"), "Electronic check")
        self.assertEqual(normalize_categorical_value("PaymentMethod", "bank wire"), "Bank transfer (automatic)")
        self.assertEqual(normalize_categorical_value("SeniorCitizen", "1"), 1)
        self.assertEqual(normalize_categorical_value("Partner", "Yes"), "Yes")

    def test_adapt_custom_dataset(self):
        report = classify_dataset(self.banking_df)
        aligned_df, meta = adapt_custom_dataset(self.banking_df, report)

        # 1. Verify all required pipeline features exist
        for feat in ALL_FEATURES:
            self.assertIn(feat, aligned_df.columns, f"Missing canonical feature: {feat}")

        # 2. Verify custom attribute 'nationality' is preserved
        self.assertIn("nationality", aligned_df.columns)
        self.assertIn("nationality", meta["preserved_custom_columns"])
        self.assertEqual(aligned_df["nationality"].iloc[0], "France")

        # 3. Verify ID column exists
        self.assertIn(ID_COLUMN, aligned_df.columns)

        # 4. Verify statistical defaults were imputed
        self.assertIn("InternetService", meta["imputed_features"])
        self.assertEqual(aligned_df["InternetService"].iloc[0], CANONICAL_STATISTICAL_DEFAULTS["InternetService"])

    def test_end_to_end_model_inference_on_custom_dataset(self):
        report = classify_dataset(self.banking_df)
        aligned_df, _ = adapt_custom_dataset(self.banking_df, report)

        pipeline, metadata = load_pipeline_artifacts(RF_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
        probabilities = pipeline.predict_proba(aligned_df)[:, 1]

        self.assertEqual(len(probabilities), len(self.banking_df))
        for p in probabilities:
            self.assertTrue(0.0 <= p <= 1.0)
            self.assertFalse(np.isnan(p))


if __name__ == "__main__":
    unittest.main()
