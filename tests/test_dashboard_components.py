"""
Unit Tests for Streamlit Dashboard Components Logic.
"""
import unittest
import pandas as pd
from src.config import (
    METADATA_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
    SAMPLE_UPLOADS_DIR,
)
from src.models.pipeline import load_pipeline_artifacts
from src.decision_engine.rules import prescribe_retention_action
class TestDashboardLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.metadata=load_pipeline_artifacts(
            PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH
        )
        cls.sample_df=pd.read_csv(SAMPLE_UPLOADS_DIR / "valid_sample.csv")
    def test_priority_matrix_logic(self):
        threshold=self.metadata.get("optimal_threshold", 0.210)
        df=self.sample_df.copy()
        probs=self.pipeline.predict_proba(df)[:, 1]
        df["churn_probability"]=probs.round(4)
        df["is_at_risk"]=df["churn_probability"]>=threshold
        def assign_tier(p: float) -> str:
            if p>=0.70:
                return "CRITICAL"
            elif p>=0.40:
                return "MODERATE"
            return "LOW"
        df["risk_tier"]=df["churn_probability"].apply(assign_tier)
        self.assertEqual(len(df), len(self.sample_df))
        self.assertIn("CRITICAL", df["risk_tier"].values)
        self.assertTrue(any(df["is_at_risk"]))
    def test_simulator_counterfactual_logic(self):
        base_series=self.sample_df.iloc[0].copy()
        base_df=pd.DataFrame([base_series.to_dict()])
        sim_df=base_df.copy()
        sim_df["Contract"] = "Two year"
        sim_df["TechSupport"] = "Yes"
        sim_df["PaymentMethod"] = "Credit card (automatic)"
        base_prob=float(self.pipeline.predict_proba(base_df)[0][1])
        sim_prob=float(self.pipeline.predict_proba(sim_df)[0][1])
        self.assertLess(sim_prob, base_prob)
if __name__ == "__main__":
    unittest.main()
