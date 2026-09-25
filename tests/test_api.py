"""
tests/test_api.py
-----------------
Automated tests for the Flask Backend API.
Verifies health endpoint, sample loader, prediction endpoint,
and DiCE counterfactual explanation endpoint.
"""

import os
import sys
import json
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

class TestThyroidAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_health_endpoint(self):
        """Verifies GET /api/health returns 200 OK and correct model metrics."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model_name"], "Random Forest Classifier")
        self.assertEqual(data["metrics"]["accuracy"], "95.86%")
        self.assertEqual(data["total_input_features"], 29)
        print("\n[PASS] test_01_health_endpoint: Health check OK, metrics verified.")

    def test_02_samples_endpoint(self):
        """Verifies GET /api/samples returns real test patient profiles."""
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("positive_case", data["samples"])
        self.assertIn("negative_case", data["samples"])
        print("[PASS] test_02_samples_endpoint: Sample patient presets loaded.")

    def test_03_predict_positive_case(self):
        """Verifies POST /api/predict correctly identifies thyroid disease sample."""
        samples_res = self.client.get("/api/samples")
        pos_sample = samples_res.get_json()["samples"]["positive_case"]["data"]

        response = self.client.post("/api/predict", json=pos_sample)
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()["data"]
        self.assertEqual(res_data["prediction"], 1)
        self.assertEqual(res_data["label"], "Thyroid Disease Detected")
        self.assertGreater(res_data["probability_disease"], 0.70)
        print(f"[PASS] test_03_predict_positive_case: Disease detected with {res_data['confidence_percent']}% confidence.")

    def test_04_predict_negative_case(self):
        """Verifies POST /api/predict correctly identifies healthy normal sample."""
        samples_res = self.client.get("/api/samples")
        neg_sample = samples_res.get_json()["samples"]["negative_case"]["data"]

        response = self.client.post("/api/predict", json=neg_sample)
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()["data"]
        self.assertEqual(res_data["prediction"], 0)
        self.assertEqual(res_data["label"], "Normal / No Disease")
        self.assertGreater(res_data["probability_healthy"], 0.70)
        print(f"[PASS] test_04_predict_negative_case: Normal patient detected with {res_data['confidence_percent']}% confidence.")

    def test_05_explain_endpoint(self):
        """Verifies POST /api/explain generates real counterfactuals with disclaimer."""
        samples_res = self.client.get("/api/samples")
        pos_sample = samples_res.get_json()["samples"]["positive_case"]["data"]

        payload = {
            "features": pos_sample,
            "desired_class": 0,
            "total_cfs": 2
        }
        response = self.client.post("/api/explain", json=payload)
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()["data"]

        self.assertGreaterEqual(res_data["total_counterfactuals_found"], 1)
        self.assertIn("explanation_disclaimer", res_data)
        self.assertIn("not a medical recommendation", res_data["explanation_disclaimer"].lower())

        first_cf = res_data["counterfactuals"][0]
        self.assertEqual(first_cf["prediction"], 0)
        self.assertGreater(first_cf["feature_changes_count"], 0)
        print(f"[PASS] test_05_explain_endpoint: {res_data['total_counterfactuals_found']} counterfactuals generated.")
        print(f"       Disclaimer: {res_data['explanation_disclaimer'][:65]}...")

if __name__ == "__main__":
    unittest.main()
