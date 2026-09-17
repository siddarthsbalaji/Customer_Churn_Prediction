"""
Unit Tests for Model Evaluation Routines, Threshold Optimization, and Serialized Artifacts.
"""
import json
import unittest
import numpy as np
import pandas as pd
from src.config import (
    METADATA_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
    SAMPLE_UPLOADS_DIR,
)
from src.models.evaluate import (
    compute_metrics_at_threshold,
    find_optimal_threshold,
)
from src.models.pipeline import load_pipeline_artifacts
class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.y_true=np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        self.y_prob=np.array([0.1, 0.2, 0.35, 0.75, 0.8, 0.45, 0.15, 0.9, 0.3, 0.65])
    def test_compute_metrics_at_threshold(self):
        metrics=compute_metrics_at_threshold(self.y_true, self.y_prob, threshold=0.5)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("f2", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertGreaterEqual(metrics["roc_auc"], 0.8)
    def test_threshold_optimization(self):
        opt=find_optimal_threshold(self.y_true, self.y_prob, cost_fn=500.0, cost_fp=35.0)
        self.assertIn("best_f2_threshold", opt)
        self.assertIn("best_cost_threshold", opt)
        self.assertGreater(opt["max_f2_score"], 0.0)
        self.assertGreaterEqual(opt["min_expected_loss"], 0.0)
        self.assertTrue(0.1<=opt["best_f2_threshold"]<=0.9)
class TestSerializedModelArtifacts(unittest.TestCase):
    def test_artifacts_exist_and_load(self):
        self.assertTrue(PIPELINE_ARTIFACT_PATH.exists(), "Model pipeline artifact not found.")
        self.assertTrue(METADATA_ARTIFACT_PATH.exists(), "Model metadata artifact not found.")
        pipeline, metadata=load_pipeline_artifacts(
            PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH
        )
        self.assertIsNotNone(pipeline)
        self.assertIn("optimal_threshold", metadata)
        self.assertIn("champion_test_metrics", metadata)
        self.assertIn("encoded_feature_names", metadata)
    def test_pipeline_inference_on_sample(self):
        pipeline, metadata=load_pipeline_artifacts(
            PIPELINE_ARTIFACT_PATH, METADATA_ARTIFACT_PATH
        )
        sample_path=SAMPLE_UPLOADS_DIR / "valid_sample.csv"
        df=pd.read_csv(sample_path)
        probs=pipeline.predict_proba(df)[:, 1]
        self.assertEqual(len(probs), len(df))
        self.assertTrue(all(0.0<=p<=1.0 for p in probs))
        threshold=metadata["optimal_threshold"]
        preds=(probs>=threshold).astype(int)
        self.assertIn(preds[0], [0, 1])
if __name__ == "__main__":
    unittest.main()
