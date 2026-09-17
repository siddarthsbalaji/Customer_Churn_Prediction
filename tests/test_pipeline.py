"""
Comprehensive Unit Tests for Telco Preprocessing Pipeline and Schema Validator.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, SAMPLE_UPLOADS_DIR
from src.data.cleaner import TelcoDataCleaner, encode_target_series
from src.data.feature_engineering import TelcoFeatureEngineer
from src.data.preprocessor import (
    create_preprocessor_pipeline,
    load_raw_dataset,
    split_data,
    get_feature_names,
)
from src.data.validator import validate_and_align_dataset
class TestTelcoCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner=TelcoDataCleaner(drop_id=True)
    def test_totalcharges_whitespace_handling(self):
        df_dirty=pd.DataFrame({
            "customerID": ["CUST1", "CUST2"],
            "tenure": [0, 5],
            "MonthlyCharges": [25.0, 50.0],
            "TotalCharges": [" ", " 250.5 "]
        })
        df_clean=self.cleaner.transform(df_dirty)
        self.assertNotIn("customerID", df_clean.columns)
        self.assertEqual(df_clean.loc[0, "TotalCharges"], 0.0)
        self.assertEqual(df_clean.loc[1, "TotalCharges"], 250.5)
    def test_target_encoding(self):
        s=pd.Series(["Yes", "No", "Yes", "No"])
        encoded=encode_target_series(s)
        self.assertListEqual(encoded.tolist(), [1, 0, 1, 0])
class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.engineer=TelcoFeatureEngineer(create_cohorts=True)
    def test_feature_derivations(self):
        df=pd.DataFrame({
            "tenure": [6, 36],
            "MonthlyCharges": [70.0, 80.0],
            "TotalCharges": [420.0, 2880.0],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"]
        })
        transformed=self.engineer.transform(df)
        self.assertIn("ServiceCount", transformed.columns)
        self.assertEqual(transformed.loc[0, "ServiceCount"], 2)
        self.assertEqual(transformed.loc[1, "ServiceCount"], 2)
        self.assertIn("ChargeRatio", transformed.columns)
        self.assertGreater(transformed.loc[0, "ChargeRatio"], 0)
        self.assertIn("TenureCohort", transformed.columns)
        self.assertEqual(transformed.loc[0, "TenureCohort"], "0-12m")
        self.assertEqual(transformed.loc[1, "TenureCohort"], "25-48m")
        self.assertIn("HasHighRiskCombo", transformed.columns)
        self.assertEqual(transformed.loc[0, "HasHighRiskCombo"], "Yes")
        self.assertEqual(transformed.loc[1, "HasHighRiskCombo"], "No")
class TestSchemaValidator(unittest.TestCase):
    def test_alias_resolution_on_messy_sample(self):
        sample_path=SAMPLE_UPLOADS_DIR / "messy_aliased_sample.csv"
        if sample_path.exists():
            df_messy=pd.read_csv(sample_path)
            aligned_df, missing_cols, warnings=validate_and_align_dataset(df_messy)
            self.assertEqual(len(missing_cols), 0, f"Unexpected missing columns: {missing_cols}")
            self.assertIn("tenure", aligned_df.columns)
            self.assertIn("MonthlyCharges", aligned_df.columns)
            self.assertIn("Contract", aligned_df.columns)
    def test_missing_mandatory_columns(self):
        df_invalid=pd.DataFrame({
            "gender": ["Female"],
            "SeniorCitizen": [0]
        })
        aligned_df, missing_cols, warnings=validate_and_align_dataset(df_invalid)
        self.assertIsNone(aligned_df)
        self.assertIn("Contract", missing_cols)
        self.assertIn("MonthlyCharges", missing_cols)
class TestFullPipeline(unittest.TestCase):
    def test_pipeline_on_raw_dataset(self):
        raw_csv=RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not raw_csv.exists():
            self.skipTest("Raw dataset not found.")
        X, y=load_raw_dataset(raw_csv)
        self.assertEqual(len(X), 7043)
        self.assertEqual(len(y), 7043)
        self.assertAlmostEqual(y.mean(), 0.265, delta=0.01)
        X_train, X_test, y_train, y_test=split_data(X, y, test_size=0.2, random_state=42)
        pipeline=create_preprocessor_pipeline(drop_id=True)
        X_train_trans=pipeline.fit_transform(X_train)
        X_test_trans=pipeline.transform(X_test)
        self.assertFalse(np.isnan(X_train_trans).any())
        self.assertFalse(np.isnan(X_test_trans).any())
        self.assertEqual(X_train_trans.shape[0], len(X_train))
        self.assertEqual(X_test_trans.shape[0], len(X_test))
        feature_names=get_feature_names(pipeline.named_steps["preprocessor"])
        self.assertEqual(len(feature_names), X_train_trans.shape[1])
    def test_pipeline_serialization(self):
        import tempfile
        from sklearn.linear_model import LogisticRegression
        from src.models.pipeline import build_churn_model_pipeline, save_pipeline_artifacts, load_pipeline_artifacts
        raw_csv=RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        X, y=load_raw_dataset(raw_csv)
        X_sub, y_sub=X.head(100), y.head(100)
        pipe=build_churn_model_pipeline(classifier=LogisticRegression(max_iter=200))
        pipe.fit(X_sub, y_sub)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path=Path(tmpdir) / "test_pipeline.joblib"
            meta_path=Path(tmpdir) / "metadata.json"
            metadata={"model_name": "LogisticRegression", "threshold": 0.45}
            save_pipeline_artifacts(pipe, metadata, model_path, meta_path)
            loaded_pipe, loaded_meta=load_pipeline_artifacts(model_path, meta_path)
            preds=loaded_pipe.predict(X_sub.head(5))
            self.assertEqual(len(preds), 5)
            self.assertEqual(loaded_meta["model_name"], "LogisticRegression")
if __name__ == "__main__":
    unittest.main()
