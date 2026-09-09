"""
Integration Tests for FastAPI REST Endpoints.
Verifies health, single prediction, SHAP explanation, batch upload, and CSV export.
"""
import io
import unittest
from fastapi.testclient import TestClient
import pandas as pd

from app.api.main import app
from src.config import SAMPLE_UPLOADS_DIR


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use TestClient with lifespan context
        cls.client = TestClient(app)

    def test_health_check(self):
        with TestClient(app) as client:
            res = client.get("/health")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["model_loaded"])
            self.assertTrue(data["explainer_loaded"])

    def test_metadata_endpoint(self):
        with TestClient(app) as client:
            res = client.get("/metadata")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("model_name", data)
            self.assertIn("optimal_threshold", data)
            self.assertIn("encoded_feature_names", data)

    def test_single_predict_endpoint(self):
        payload = {
            "customerID": "TEST-CUST-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 75.0,
            "TotalCharges": 150.0
        }
        with TestClient(app) as client:
            res = client.post("/predict", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["customer_id"], "TEST-CUST-001")
            self.assertTrue(0.0 <= data["churn_probability"] <= 1.0)
            self.assertIn(data["risk_tier"], ["CRITICAL", "MODERATE", "LOW"])
            self.assertIsInstance(data["is_at_risk"], bool)

    def test_single_explain_endpoint(self):
        payload = {
            "customerID": "TEST-CUST-002",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0
        }
        with TestClient(app) as client:
            res = client.post("/explain", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["customer_id"], "TEST-CUST-002")
            self.assertIn("risk_drivers", data)
            self.assertIn("protective_factors", data)
            self.assertIn("prescribed_action", data)
            self.assertIn("action_title", data["prescribed_action"])

    def test_batch_risk_valid_csv(self):
        csv_path = SAMPLE_UPLOADS_DIR / "valid_sample.csv"
        with open(csv_path, "rb") as f:
            with TestClient(app) as client:
                res = client.post(
                    "/batch-risk",
                    files={"file": ("valid_sample.csv", f, "text/csv")}
                )
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertGreater(data["total_records"], 0)
                self.assertIn("results", data)
                self.assertGreaterEqual(data["at_risk_count"], 0)
                self.assertGreaterEqual(data["total_mrr_at_risk"], 0.0)

    def test_batch_risk_aliased_csv(self):
        csv_path = SAMPLE_UPLOADS_DIR / "messy_aliased_sample.csv"
        with open(csv_path, "rb") as f:
            with TestClient(app) as client:
                res = client.post(
                    "/batch-risk",
                    files={"file": ("messy_aliased_sample.csv", f, "text/csv")}
                )
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["total_records"], 5)

    def test_batch_risk_invalid_csv_schema(self):
        invalid_csv = "gender,SeniorCitizen\nFemale,0\n"
        with TestClient(app) as client:
            res = client.post(
                "/batch-risk",
                files={"file": ("bad_schema.csv", io.BytesIO(invalid_csv.encode("utf-8")), "text/csv")}
            )
            self.assertEqual(res.status_code, 422)

    def test_batch_csv_export(self):
        csv_path = SAMPLE_UPLOADS_DIR / "valid_sample.csv"
        with open(csv_path, "rb") as f:
            with TestClient(app) as client:
                res = client.post(
                    "/batch-risk/export-csv",
                    files={"file": ("valid_sample.csv", f, "text/csv")}
                )
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.headers["content-type"], "text/csv; charset=utf-8")
                self.assertIn("attachment", res.headers.get("content-disposition", ""))
                self.assertIn("churn_probability", res.text)


if __name__ == "__main__":
    unittest.main()
