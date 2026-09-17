"""
Unit test to verify configuration constants, paths, and feature schema definitions.
Compatible with both standard unittest and pytest.
"""
import unittest
from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    COLUMN_ALIASES,
    DEFAULT_DECISION_THRESHOLD,
    ID_COLUMN,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    RISK_TIERS,
    TARGET_COLUMN,
)
class TestConfig(unittest.TestCase):
    def test_feature_definitions(self):
        self.assertEqual(ID_COLUMN, "customerID")
        self.assertEqual(TARGET_COLUMN, "Churn")
        self.assertEqual(len(NUMERIC_FEATURES), 3)
        self.assertIn("tenure", NUMERIC_FEATURES)
        self.assertIn("MonthlyCharges", NUMERIC_FEATURES)
        self.assertIn("TotalCharges", NUMERIC_FEATURES)
        self.assertEqual(len(CATEGORICAL_FEATURES), 16)
        self.assertEqual(len(ALL_FEATURES), len(NUMERIC_FEATURES)+len(CATEGORICAL_FEATURES))
    def test_risk_tiers(self):
        self.assertGreater(RISK_TIERS["CRITICAL"], RISK_TIERS["MODERATE"])
        self.assertGreaterEqual(DEFAULT_DECISION_THRESHOLD, RISK_TIERS["MODERATE"])
    def test_column_aliases(self):
        self.assertIn("MonthlyCharges", COLUMN_ALIASES)
        self.assertIn("monthly_charges", COLUMN_ALIASES["MonthlyCharges"])
        self.assertIn("Contract", COLUMN_ALIASES)
if __name__ == "__main__":
    unittest.main()
