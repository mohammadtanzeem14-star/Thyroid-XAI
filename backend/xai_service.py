"""
backend/xai_service.py
----------------------
XAI Service encapsulating the pre-trained Random Forest model,
preprocessor, and DiCE counterfactual explanation engine.

IMPORTANT:
- Operates strictly in inference mode on existing models.
- Does NOT retrain or alter any existing model, metric, or dataset files.
- Counterfactual outputs are decision boundary explanations, NOT medical advice.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import dice_ml

FEATURE_COLUMNS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
    "referral_source"
]

NUMERICAL_COLS = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
CATEGORICAL_COLS = [c for c in FEATURE_COLUMNS if c not in NUMERICAL_COLS]

EXPLANATION_DISCLAIMER = (
    "Model decision explanation only, not a medical recommendation. "
    "Counterfactuals show minimal algorithmic feature changes that would "
    "cause the Random Forest model to alter its classification outcome."
)

class XAIService:
    _instance = None

    @classmethod
    def get_instance(cls, base_dir=None):
        if cls._instance is None:
            cls._instance = cls(base_dir=base_dir)
        return cls._instance

    def __init__(self, base_dir=None):
        if base_dir is None:
            # Assumes project root is parent directory of backend/
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_dir = base_dir

        self.model_dir = os.path.join(self.base_dir, "model")
        self.dataset_dir = os.path.join(self.base_dir, "dataset")

        self.preprocessor = None
        self.rf_model = None
        self.pipeline = None
        self.dice_exp = None
        self.sample_profiles = {}
        self.default_values = {}

        self._load_artifacts()
        self._initialize_dice()

    def _load_artifacts(self):
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
        rf_path = os.path.join(self.model_dir, "random_forest_model.pkl")

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"Random Forest model not found: {rf_path}")

        print(f"[XAIService] Loading preprocessor: {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)

        print(f"[XAIService] Loading Random Forest model: {rf_path}")
        self.rf_model = joblib.load(rf_path)

        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", self.rf_model)
        ])

    def _initialize_dice(self):
        dataset_path = os.path.join(self.dataset_dir, "thyroid0387_cleaned.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        X = df[FEATURE_COLUMNS]
        y = df["target"]

        # Stratified 80/20 train/test split matching random_state=42
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        num_imputer = self.preprocessor.transformers_[0][1].named_steps["imputer"]
        cat_imputer = self.preprocessor.transformers_[1][1].named_steps["imputer"]

        # Compute default fallback values based on median & mode
        for i, col in enumerate(NUMERICAL_COLS):
            self.default_values[col] = float(num_imputer.statistics_[i])
        for i, col in enumerate(CATEGORICAL_COLS):
            self.default_values[col] = str(cat_imputer.statistics_[i])

        # Prepare background reference for DiCE
        X_train_imp = X_train.copy()
        X_train_imp[NUMERICAL_COLS] = num_imputer.transform(X_train[NUMERICAL_COLS])
        X_train_imp[CATEGORICAL_COLS] = cat_imputer.transform(X_train[CATEGORICAL_COLS])

        dice_train_df = X_train_imp.copy()
        dice_train_df["target"] = y_train.values

        print("[XAIService] Initializing DiCE explainer...")
        d = dice_ml.Data(
            dataframe=dice_train_df,
            continuous_features=NUMERICAL_COLS,
            outcome_name="target"
        )
        m = dice_ml.Model(model=self.pipeline, backend="sklearn")
        self.dice_exp = dice_ml.Dice(d, m, method="random")

        # Pre-select representative real test samples for 1-click viva testing
        X_test_imp = X_test.copy()
        X_test_imp[NUMERICAL_COLS] = num_imputer.transform(X_test[NUMERICAL_COLS])
        X_test_imp[CATEGORICAL_COLS] = cat_imputer.transform(X_test[CATEGORICAL_COLS])

        test_preds = self.pipeline.predict(X_test_imp)

        # Real positive sample (ID: 8434 or 1421)
        pos_indices = np.where((y_test.values == 1) & (test_preds == 1))[0]
        neg_indices = np.where((y_test.values == 0) & (test_preds == 0))[0]

        if len(pos_indices) > 0:
            pos_row = X_test_imp.iloc[pos_indices[0]]
            self.sample_profiles["positive_case"] = {
                "name": "Patient A — Confirmed Thyroid Disease",
                "description": "Real test sample with elevated TSH and abnormal thyroid hormone panel",
                "expected_class": 1,
                "data": {col: (float(pos_row[col]) if col in NUMERICAL_COLS else str(pos_row[col])) for col in FEATURE_COLUMNS}
            }

        if len(neg_indices) > 0:
            neg_row = X_test_imp.iloc[neg_indices[0]]
            self.sample_profiles["negative_case"] = {
                "name": "Patient B — Normal / Euthyroid",
                "description": "Real test sample with normal hormone levels and no clinical symptoms",
                "expected_class": 0,
                "data": {col: (float(neg_row[col]) if col in NUMERICAL_COLS else str(neg_row[col])) for col in FEATURE_COLUMNS}
            }

        print("[XAIService] Initialization complete.")

    def format_input_dataframe(self, input_dict):
        """Converts arbitrary input dictionary into a single-row DataFrame with all 29 features."""
        row = {}
        for col in FEATURE_COLUMNS:
            val = input_dict.get(col, None)
            if val is None or val == "" or str(val).lower() == "nan":
                val = self.default_values[col]
            
            if col in NUMERICAL_COLS:
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = self.default_values[col]
            else:
                row[col] = str(val).strip()
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, input_dict):
        """Generates Random Forest prediction and probability breakdown."""
        df = self.format_input_dataframe(input_dict)
        pred = int(self.pipeline.predict(df)[0])
        proba = self.pipeline.predict_proba(df)[0]

        prob_0 = round(float(proba[0]), 4)
        prob_1 = round(float(proba[1]), 4)
        confidence = round(float(max(proba) * 100), 2)

        return {
            "prediction": pred,
            "label": "Thyroid Disease Detected" if pred == 1 else "Normal / No Disease",
            "probability_healthy": prob_0,
            "probability_disease": prob_1,
            "confidence_percent": confidence,
            "input_features": df.iloc[0].to_dict()
        }

    def explain(self, input_dict, desired_class=None, total_cfs=2):
        """
        Generates real DiCE counterfactual explanations.
        Identifies minimal feature changes required to flip model prediction.
        """
        df = self.format_input_dataframe(input_dict)
        pred = int(self.pipeline.predict(df)[0])
        proba = self.pipeline.predict_proba(df)[0]

        if desired_class is None:
            desired_class = 0 if pred == 1 else 1

        cf_obj = self.dice_exp.generate_counterfactuals(
            df,
            total_CFs=total_cfs,
            desired_class=desired_class
        )

        cfs_df = cf_obj.cf_examples_list[0].final_cfs_df
        cfs_list = []

        if cfs_df is not None and len(cfs_df) > 0:
            for i, (_, cf_row) in enumerate(cfs_df.iterrows(), 1):
                cf_feature_df = pd.DataFrame([cf_row[FEATURE_COLUMNS]])
                cf_pred = int(self.pipeline.predict(cf_feature_df)[0])
                cf_proba = self.pipeline.predict_proba(cf_feature_df)[0]

                changes = []
                for col in FEATURE_COLUMNS:
                    orig_v = df[col].values[0]
                    cf_v = cf_row[col]

                    is_diff = False
                    delta = None
                    if col in NUMERICAL_COLS:
                        if abs(float(orig_v) - float(cf_v)) > 1e-4:
                            is_diff = True
                            delta = round(float(cf_v) - float(orig_v), 4)
                    else:
                        if str(orig_v) != str(cf_v):
                            is_diff = True
                            delta = f"{orig_v} -> {cf_v}"

                    if is_diff:
                        changes.append({
                            "feature": col,
                            "original_value": float(orig_v) if col in NUMERICAL_COLS else str(orig_v),
                            "counterfactual_value": float(cf_v) if col in NUMERICAL_COLS else str(cf_v),
                            "difference": delta
                        })

                cfs_list.append({
                    "cf_index": i,
                    "prediction": cf_pred,
                    "label": "Thyroid Disease Detected" if cf_pred == 1 else "Normal / No Disease",
                    "probability_healthy": round(float(cf_proba[0]), 4),
                    "probability_disease": round(float(cf_proba[1]), 4),
                    "confidence_percent": round(float(max(cf_proba) * 100), 2),
                    "feature_changes_count": len(changes),
                    "changed_features": changes,
                    "all_features": {c: (float(cf_row[c]) if c in NUMERICAL_COLS else str(cf_row[c])) for c in FEATURE_COLUMNS}
                })

        return {
            "original_prediction": {
                "prediction": pred,
                "label": "Thyroid Disease Detected" if pred == 1 else "Normal / No Disease",
                "probability_healthy": round(float(proba[0]), 4),
                "probability_disease": round(float(proba[1]), 4),
                "confidence_percent": round(float(max(proba) * 100), 2)
            },
            "desired_class": desired_class,
            "desired_label": "Thyroid Disease Detected" if desired_class == 1 else "Normal / No Disease",
            "total_counterfactuals_found": len(cfs_list),
            "counterfactuals": cfs_list,
            "explanation_disclaimer": EXPLANATION_DISCLAIMER
        }

    def get_sample_profiles(self):
        return self.sample_profiles
