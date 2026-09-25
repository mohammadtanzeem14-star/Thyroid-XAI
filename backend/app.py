"""
backend/app.py
--------------
Flask REST API for Thyroid Disease Diagnosis & Counterfactual Explainable AI.

Endpoints:
- GET  /api/health       : System status and model performance metrics.
- GET  /api/samples      : Real test patient presets for 1-click viva testing.
- POST /api/predict      : Model prediction and probability breakdown.
- POST /api/explain      : DiCE Counterfactual Explanation with minimal feature changes.
- GET  /                 : Serves Frontend UI.
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to sys.path to enable clean imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.xai_service import XAIService, FEATURE_COLUMNS, NUMERICAL_COLS, CATEGORICAL_COLS
from backend.admin_service import AdminService

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# Initialize service singletons on startup
service = XAIService.get_instance(base_dir=BASE_DIR)
admin_service = AdminService(base_dir=BASE_DIR)

@app.route("/", methods=["GET"])
def serve_index():
    """Serves the frontend application single-page interface."""
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({
        "message": "Thyroid XAI Backend API is running.",
        "docs": "Frontend interface will be available after Stage 3.",
        "endpoints": ["/api/health", "/api/samples", "/api/predict", "/api/explain"]
    }), 200

@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    """Serves static assets (CSS, JS, images) from frontend/."""
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    return jsonify({"error": f"File not found: {path}"}), 404

@app.route("/api/health", methods=["GET"])
def health():
    """Returns system status, active model, and verified evaluation metrics."""
    return jsonify({
        "status": "healthy",
        "project": "Enhancing Thyroid Disease Diagnosis With Machine Learning and Counterfactual Explainable AI",
        "model_name": "Random Forest Classifier",
        "metrics": {
            "accuracy": "95.86%",
            "roc_auc": "0.9937",
            "precision": "0.88",
            "recall": "0.97",
            "f1_score": "0.92"
        },
        "total_input_features": len(FEATURE_COLUMNS),
        "xai_engine": "DiCE (Diverse Counterfactual Explanations)",
        "features": {
            "numerical": NUMERICAL_COLS,
            "categorical": CATEGORICAL_COLS
        }
    }), 200

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns real test samples for instant testing in the UI or API."""
    try:
        samples = service.get_sample_profiles()
        return jsonify({
            "status": "success",
            "samples": samples
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts patient features, runs through preprocessor and Random Forest model,
    returns prediction and class probabilities.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        result = service.predict(data)
        return jsonify({
            "status": "success",
            "data": result
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/explain", methods=["POST"])
def explain():
    """
    Generates real DiCE counterfactual explanations.
    Returns changed features, original vs counterfactual values, and model explanation disclaimer.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        features = data.get("features", data)
        desired_class = data.get("desired_class", None)
        total_cfs = int(data.get("total_cfs", 2))

        result = service.explain(features, desired_class=desired_class, total_cfs=total_cfs)
        return jsonify({
            "status": "success",
            "data": result
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================================
# ADMIN DASHBOARD MODULES
# =====================================================================

@app.route("/api/admin/dataset-info", methods=["GET"])
@app.route("/api/admin/upload-dataset", methods=["POST"])
def upload_dataset():
    """
    Module 1: Upload / Inspect Dataset.
    Accepts an uploaded CSV file (multipart or raw text) or inspects current active dataset.
    Never overwrites original thyroid0387_cleaned.csv.
    """
    try:
        file_content = None
        filename = None

        if request.method == "POST":
            if "file" in request.files:
                uploaded_file = request.files["file"]
                filename = uploaded_file.filename
                file_content = uploaded_file.read()
            elif request.is_json and "csv_data" in request.json:
                file_content = request.json["csv_data"]
                filename = request.json.get("filename", "uploaded_data.csv")

        info = admin_service.get_dataset_info(file_content=file_content, filename=filename)
        return jsonify(info), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/admin/preprocess", methods=["POST"])
def preprocess_data():
    """
    Module 2: Preprocess Dataset.
    Executes median/mode imputation, standard scaling, one-hot encoding, and 80/20 train/test split.
    """
    try:
        prep_summary = admin_service.preprocess_dataset()
        return jsonify(prep_summary), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/admin/train", methods=["POST"])
def train_model():
    """
    Module 3: Apply Algorithm.
    Trains Logistic Regression or Random Forest in-memory and calculates real metrics.
    Does NOT overwrite production model files.
    """
    try:
        payload = request.get_json(silent=True) or {}
        algorithm = payload.get("algorithm", "random_forest")
        result = admin_service.train_algorithm(algorithm_name=algorithm)
        return jsonify({"status": "success", "data": result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/admin/comparison", methods=["GET"])
def get_algorithm_comparison():
    """
    Module 4: View Algorithm Comparison Graph.
    Returns calculated comparative performance metrics for Logistic Regression vs Random Forest.
    """
    try:
        comparison = admin_service.get_comparison()
        return jsonify({"status": "success", "data": comparison}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = 5000
    print(f"\n=======================================================")
    print(f" Thyroid Disease Diagnosis & XAI Backend API Running")
    print(f" Access URL: http://localhost:{port}")
    print(f" Health check: http://localhost:{port}/api/health")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
