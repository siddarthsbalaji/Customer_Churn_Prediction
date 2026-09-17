"""
Unit Tests for SHAP Explainability Service and Prescriptive Decision Engine.
"""
import unittest
import matplotlib.pyplot as plt
import pandas as pd
from src.config import (
    METADATA_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
    SHAP_EXPLAINER_ARTIFACT_PATH,
    SAMPLE_UPLOADS_DIR,
)
from src.decision_engine.rules import PLAYBOOKS, prescribe_retention_action
from src.explainability.shap_service import (
    explain_single_customer,
    humanize_feature_name,
    load_shap_explainer,
    render_customer_waterfall_figure,
)
from src.models.pipeline import load_pipeline_artifacts
class TestPrescriptiveRules(unittest.TestCase):
    def test_low_risk_playbook(self):
        action=prescribe_retention_action(
            churn_prob=0.25,
            top_shap_drivers=[]
        )
        self.assertEqual(action["action_code"], PLAYBOOKS["ORGANIC_NURTURE"]["action_code"])
        self.assertEqual(action["priority"], "LOW")
    def test_critical_contract_playbook(self):
        drivers=[
            {"feature": "Contract_Month-to-month", "display_name": "Month-to-month Contract", "shap_value": 0.12},
            {"feature": "tenure", "display_name": "Tenure Duration", "shap_value": 0.08}
        ]
        action=prescribe_retention_action(
            churn_prob=0.82,
            top_shap_drivers=drivers
        )
        self.assertEqual(action["action_code"], PLAYBOOKS["CONTRACT_UPGRADE"]["action_code"])
        self.assertEqual(action["priority"], "CRITICAL")
    def test_moderate_support_playbook(self):
        drivers=[
            {"feature": "TechSupport_No", "display_name": "Lacks Tech Support", "shap_value": 0.09},
            {"feature": "MonthlyCharges", "display_name": "Monthly Billing Rate", "shap_value": 0.05}
        ]
        action=prescribe_retention_action(
            churn_prob=0.55,
            top_shap_drivers=drivers
        )
        self.assertEqual(action["action_code"], PLAYBOOKS["TECH_SUPPORT_VIP"]["action_code"])
class TestShapService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.metadata=load_pipeline_artifacts(
            PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH
        )
        cls.explainer=load_shap_explainer(SHAP_EXPLAINER_ARTIFACT_PATH)
        cls.feature_names=cls.metadata["encoded_feature_names"]
        cls.sample_df=pd.read_csv(SAMPLE_UPLOADS_DIR / "valid_sample.csv")
    def test_humanize_feature_name(self):
        self.assertEqual(
            humanize_feature_name("Contract_Month-to-month"),
            "Month-to-month Contract (High Risk)"
        )
        self.assertEqual(
            humanize_feature_name("PaymentMethod_Electronic check"),
            "Electronic Check Billing"
        )
    def test_explain_single_customer(self):
        single_row=self.sample_df.iloc[[0]]
        explanation=explain_single_customer(
            pipeline=self.pipeline,
            explainer=self.explainer,
            customer_raw_df=single_row,
            feature_names=self.feature_names,
            top_k=5
        )
        self.assertIn("base_value", explanation)
        self.assertIn("prediction_probability", explanation)
        self.assertIn("risk_drivers", explanation)
        self.assertIn("protective_factors", explanation)
        self.assertLessEqual(len(explanation["risk_drivers"]), 5)
        self.assertTrue(0.0<=explanation["prediction_probability"]<=1.0)
    def test_render_waterfall_figure(self):
        single_row=self.sample_df.iloc[[0]]
        fig=render_customer_waterfall_figure(
            pipeline=self.pipeline,
            explainer=self.explainer,
            customer_raw_df=single_row,
            feature_names=self.feature_names,
            max_display=8,
            customer_id="TEST-ROW-0"
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
if __name__ == "__main__":
    unittest.main()
