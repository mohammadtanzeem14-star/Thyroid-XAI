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

    def test_06_frontend_static_serving(self):
        """Verifies frontend HTML, CSS, and JavaScript are served correctly."""
        res_html = self.client.get("/")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn(b"ThyroCare XAI", res_html.data)

        res_css = self.client.get("/styles.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn(b":root", res_css.data)

        res_js = self.client.get("/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn(b"DOMContentLoaded", res_js.data)
        print("[PASS] test_06_frontend_static_serving: index.html, styles.css, app.js verified.")

    def test_07_admin_dataset_info(self):
        """Verifies Module 1: Upload / Inspect dataset endpoint."""
        res = self.client.get("/api/admin/dataset-info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_rows"], 9172)
        self.assertEqual(data["total_columns"], 32)
        self.assertIn("preview", data)
        self.assertEqual(len(data["preview"]), 5)
        print("[PASS] test_07_admin_dataset_info: Dataset stats and preview verified.")

    def test_08_admin_preprocess(self):
        """Verifies Module 2: Preprocess dataset endpoint."""
        res = self.client.post("/api/admin/preprocess")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["raw_feature_count"], 29)
        self.assertEqual(data["transformed_feature_count"], 55)
        self.assertEqual(data["train_samples"], 7337)
        self.assertEqual(data["test_samples"], 1835)
        print("[PASS] test_08_admin_preprocess: Preprocessing pipeline (29 -> 55 features, 80/20 split) verified.")

    def test_09_admin_train_logistic_regression(self):
        """Verifies Module 3a: Apply Logistic Regression with real metrics."""
        res = self.client.post("/api/admin/train", json={"algorithm": "logistic_regression"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]
        self.assertEqual(data["algorithm"], "Logistic Regression")
        self.assertGreater(data["metrics"]["accuracy"], 70.0)
        self.assertGreater(data["metrics"]["roc_auc"], 0.75)
        self.assertIn("confusion_matrix", data)
        print(f"[PASS] test_09_admin_train_logistic_regression: LR Acc={data['metrics']['accuracy']}%, ROC={data['metrics']['roc_auc']}.")

    def test_10_admin_train_random_forest(self):
        """Verifies Module 3b: Apply Random Forest with real metrics."""
        res = self.client.post("/api/admin/train", json={"algorithm": "random_forest"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]
        self.assertEqual(data["algorithm"], "Random Forest Classifier")
        self.assertGreater(data["metrics"]["accuracy"], 95.0)
        self.assertGreater(data["metrics"]["roc_auc"], 0.99)
        self.assertIn("confusion_matrix", data)
        print(f"[PASS] test_10_admin_train_random_forest: RF Acc={data['metrics']['accuracy']}%, ROC={data['metrics']['roc_auc']}.")

    def test_11_admin_comparison(self):
        """Verifies Module 4: View Algorithm Comparison Graph endpoint."""
        res = self.client.get("/api/admin/comparison")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]
        self.assertEqual(len(data["algorithms"]), 2)
        names = [a["name"] for a in data["algorithms"]]
        self.assertIn("Logistic Regression", names)
        self.assertIn("Random Forest", names)
        self.assertIn("conclusion", data)
        print("[PASS] test_11_admin_comparison: Comparison data with real metrics verified.")

if __name__ == "__main__":
    unittest.main()
