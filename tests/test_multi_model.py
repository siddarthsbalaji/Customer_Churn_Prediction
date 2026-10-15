"""
Unit and Integration Tests for Dual-Model Selection and Comparison.
"""
import unittest
from fastapi.testclient import TestClient
import pandas as pd
from app.api.main import app
from src.config import (
    XGB_EXPLAINER_PATH,
    XGB_PIPELINE_PATH,
    METADATA_ARTIFACT_PATH,
    RF_EXPLAINER_PATH,
    RF_PIPELINE_PATH,
    SAMPLE_UPLOADS_DIR,
)
from src.explainability.shap_service import (
    compare_customer_models,
    explain_single_customer,
    load_shap_explainer,
)
from src.models.pipeline import load_pipeline_artifacts
class TestMultiModelArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rf_pipe, cls.metadata=load_pipeline_artifacts(RF_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
        cls.xgb_pipe, _=load_pipeline_artifacts(XGB_PIPELINE_PATH, METADATA_ARTIFACT_PATH)
        cls.rf_exp=load_shap_explainer(RF_EXPLAINER_PATH)
        cls.xgb_exp=load_shap_explainer(XGB_EXPLAINER_PATH)
        cls.sample_df=pd.read_csv(SAMPLE_UPLOADS_DIR / "valid_sample.csv")
    def test_artifacts_exist(self):
        self.assertTrue(RF_PIPELINE_PATH.exists())
        self.assertTrue(XGB_PIPELINE_PATH.exists())
        self.assertTrue(RF_EXPLAINER_PATH.exists())
        self.assertTrue(XGB_EXPLAINER_PATH.exists())
        self.assertIn("models", self.metadata)
        self.assertIn("random_forest", self.metadata["models"])
        self.assertIn("xgboost", self.metadata["models"])
    def test_multi_model_predictions(self):
        single_row=self.sample_df.iloc[[0]]
        rf_prob=float(self.rf_pipe.predict_proba(single_row)[0][1])
        xgb_prob=float(self.xgb_pipe.predict_proba(single_row)[0][1])
        self.assertTrue(0.0<=rf_prob<=1.0)
        self.assertTrue(0.0<=xgb_prob<=1.0)
    def test_both_explainers_execute(self):
        single_row=self.sample_df.iloc[[0]]
        feature_names=self.metadata["encoded_feature_names"]
        rf_exp_res=explain_single_customer(
            self.rf_pipe, self.rf_exp, single_row, feature_names, top_k=3
        )
        xgb_exp_res=explain_single_customer(
            self.xgb_pipe, self.xgb_exp, single_row, feature_names, top_k=3
        )
        self.assertIn("risk_drivers", rf_exp_res)
        self.assertIn("risk_drivers", xgb_exp_res)
        self.assertEqual(len(rf_exp_res["risk_drivers"]), 3)
        self.assertEqual(len(xgb_exp_res["risk_drivers"]), 3)
    def test_compare_customer_models_function(self):
        single_row=self.sample_df.iloc[[0]]
        feature_names=self.metadata["encoded_feature_names"]
        thresholds={"random_forest": 0.210, "xgboost": 0.240}
        comparison=compare_customer_models(
            pipelines={"random_forest": self.rf_pipe, "xgboost": self.xgb_pipe},
            explainers={"random_forest": self.rf_exp, "xgboost": self.xgb_exp},
            customer_raw_df=single_row,
            feature_names=feature_names,
            thresholds=thresholds
        )
        self.assertIn("random_forest", comparison)
        self.assertIn("xgboost", comparison)
        self.assertIn("probability_delta", comparison)
        self.assertIn("tier_agreement", comparison)
        self.assertIsInstance(comparison["tier_agreement"], bool)
class TestAPIMultiModelRouting(unittest.TestCase):
    def setUp(self):
        self.payload={
            "customerID": "TEST-MULTI-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 45.0,
            "TotalCharges": 90.0
        }
    def test_predict_with_model_selection(self):
        with TestClient(app) as client:
            res_rf=client.post("/predict?model=random_forest", json=self.payload)
            self.assertEqual(res_rf.status_code, 200)
            data_rf=res_rf.json()
            self.assertEqual(data_rf["model_used"], "random_forest")
            self.assertEqual(data_rf["decision_threshold_applied"], 0.210)
            res_xgb=client.post("/predict?model=xgboost", json=self.payload)
            self.assertEqual(res_xgb.status_code, 200)
            data_xgb=res_xgb.json()
            self.assertEqual(data_xgb["model_used"], "xgboost")
            self.assertEqual(data_xgb["decision_threshold_applied"], 0.240)
    def test_explain_with_model_selection(self):
        with TestClient(app) as client:
            res_xgb=client.post("/explain?model=xgboost", json=self.payload)
            self.assertEqual(res_xgb.status_code, 200)
            data_xgb=res_xgb.json()
            self.assertEqual(data_xgb["model_used"], "xgboost")
            self.assertIn("risk_drivers", data_xgb)
    def test_compare_endpoint(self):
        with TestClient(app) as client:
            res=client.post("/compare", json=self.payload)
            self.assertEqual(res.status_code, 200)
            data=res.json()
            self.assertIn("random_forest", data)
            self.assertIn("xgboost", data)
            self.assertIn("probability_delta", data)
            self.assertIn("tier_agreement", data)
            self.assertEqual(data["random_forest"]["model_used"], "random_forest")
            self.assertEqual(data["xgboost"]["model_used"], "xgboost")
if __name__ == "__main__":
    unittest.main()
