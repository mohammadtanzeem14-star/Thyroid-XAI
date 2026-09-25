"""
generate_counterfactuals.py
---------------------------
Generates real Counterfactual Explanations using DiCE and the existing,
pre-trained Random Forest model and preprocessor.

IMPORTANT:
- Does NOT retrain or alter any existing model, dataset, or metric files.
- Uses strictly the existing preprocessor.pkl and random_forest_model.pkl.
- Counterfactual changes represent algorithmic decision boundary explanations
  (what minimal input changes would cause the machine learning model to alter its
  prediction), and are NOT clinical or medical treatment recommendations.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import dice_ml

def main():
    print("=" * 70)
    print("Stage 1: Counterfactual XAI Evaluation & Export using DiCE")
    print("=" * 70)

    # 1. Load dataset (Read-only)
    dataset_path = os.path.join("dataset", "thyroid0387_cleaned.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"[+] Loaded cleaned dataset: {dataset_path} ({df.shape[0]} records, {df.shape[1]} columns)")

    # 2. Separate features and target
    feature_cols = [c for c in df.columns if c not in ["diagnosis", "record_id", "target"]]
    X = df[feature_cols]
    y = df["target"]

    # 3. Stratified 80/20 split matching verified training seed (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Recreated test split: {len(X_test)} samples (random_state=42, stratify=y)")

    # 4. Load existing saved preprocessor and Random Forest model (Read-only)
    preprocessor_path = os.path.join("model", "preprocessor.pkl")
    rf_model_path = os.path.join("model", "random_forest_model.pkl")

    preprocessor = joblib.load(preprocessor_path)
    rf_model = joblib.load(rf_model_path)
    print(f"[+] Successfully loaded existing preprocessor: {preprocessor_path}")
    print(f"[+] Successfully loaded existing Random Forest model: {rf_model_path}")

    # Build full pipeline for inference
    full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf_model)])

    # Extract numerical and categorical columns from the fitted preprocessor
    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])
    num_imputer = preprocessor.transformers_[0][1].named_steps["imputer"]
    cat_imputer = preprocessor.transformers_[1][1].named_steps["imputer"]

    # Impute missing values using the existing fitted imputers so DiCE has a valid data space
    X_train_imp = X_train.copy()
    X_train_imp[num_cols] = num_imputer.transform(X_train[num_cols])
    X_train_imp[cat_cols] = cat_imputer.transform(X_train[cat_cols])

    X_test_imp = X_test.copy()
    X_test_imp[num_cols] = num_imputer.transform(X_test[num_cols])
    X_test_imp[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    train_data_for_dice = X_train_imp.copy()
    train_data_for_dice["target"] = y_train.values

    # 5. Initialize DiCE
    print("[+] Initializing DiCE framework...")
    dice_data = dice_ml.Data(
        dataframe=train_data_for_dice,
        continuous_features=num_cols,
        outcome_name="target"
    )
    dice_model = dice_ml.Model(model=full_pipeline, backend="sklearn")
    exp = dice_ml.Dice(dice_data, dice_model, method="random")

    # 6. Select real test samples
    # We select 5 confirmed positive samples (Thyroid Disease -> desired 0)
    # and 5 confirmed negative samples (Normal -> desired 1)
    test_preds = full_pipeline.predict(X_test)
    pos_mask = (y_test.values == 1) & (test_preds == 1)
    neg_mask = (y_test.values == 0) & (test_preds == 0)

    pos_indices = np.where(pos_mask)[0][:5]
    neg_indices = np.where(neg_mask)[0][:5]

    selected_samples = []
    for idx in pos_indices:
        selected_samples.append({
            "split_idx": int(idx),
            "original_index": int(X_test.index[idx]),
            "expected_label": 1,
            "desired_class": 0,
            "category": "Thyroid Disease (Class 1 -> Class 0)"
        })
    for idx in neg_indices:
        selected_samples.append({
            "split_idx": int(idx),
            "original_index": int(X_test.index[idx]),
            "expected_label": 0,
            "desired_class": 1,
            "category": "Healthy / Normal (Class 0 -> Class 1)"
        })

    print(f"[+] Selected {len(selected_samples)} real test samples ({len(pos_indices)} positive, {len(neg_indices)} negative).")

    all_cf_rows = []
    summary_records = []
    total_cfs_count = 0

    print("\n[+] Generating counterfactuals with DiCE...")
    for i, meta in enumerate(selected_samples, 1):
        idx = meta["split_idx"]
        orig_row = X_test_imp.iloc[[idx]]
        orig_idx_val = meta["original_index"]

        orig_pred = int(full_pipeline.predict(orig_row)[0])
        orig_proba = full_pipeline.predict_proba(orig_row)[0].tolist()

        desired = meta["desired_class"]
        sample_label = f"Sample_{i}_ID_{orig_idx_val}"

        print(f"  [{i}/{len(selected_samples)}] Processing {sample_label} (Pred: {orig_pred}, Desired: {desired})...")

        # Record original sample in tabular export
        orig_export_row = {
            "sample_label": sample_label,
            "sample_type": "Original Test Sample",
            "sample_id": orig_idx_val,
            "predicted_class": orig_pred,
            "prob_class_0": round(orig_proba[0], 4),
            "prob_class_1": round(orig_proba[1], 4),
            "explanation_disclaimer": "Model decision explanation only, not a medical recommendation"
        }
        for col in feature_cols:
            orig_export_row[col] = orig_row[col].values[0]
        all_cf_rows.append(orig_export_row)

        # Generate Counterfactuals using DiCE
        cf_exp = exp.generate_counterfactuals(
            orig_row,
            total_CFs=2,
            desired_class=desired
        )

        cfs_df = cf_exp.cf_examples_list[0].final_cfs_df
        cfs_list = []

        if cfs_df is not None and len(cfs_df) > 0:
            for cf_idx, (_, cf_row) in enumerate(cfs_df.iterrows(), 1):
                total_cfs_count += 1
                cf_feature_df = pd.DataFrame([cf_row[feature_cols]])
                cf_pred = int(full_pipeline.predict(cf_feature_df)[0])
                cf_proba = full_pipeline.predict_proba(cf_feature_df)[0].tolist()

                # Find differences
                feature_changes = []
                for col in feature_cols:
                    orig_val = orig_row[col].values[0]
                    cf_val = cf_row[col]

                    # Compare numeric or string
                    is_changed = False
                    diff_val = None
                    if col in num_cols:
                        if abs(float(orig_val) - float(cf_val)) > 1e-4:
                            is_changed = True
                            diff_val = round(float(cf_val) - float(orig_val), 4)
                    else:
                        if str(orig_val) != str(cf_val):
                            is_changed = True
                            diff_val = f"{orig_val} -> {cf_val}"

                    if is_changed:
                        feature_changes.append({
                            "feature": col,
                            "original_value": float(orig_val) if col in num_cols else str(orig_val),
                            "counterfactual_value": float(cf_val) if col in num_cols else str(cf_val),
                            "difference": diff_val
                        })

                cf_record = {
                    "counterfactual_id": f"{sample_label}_CF_{cf_idx}",
                    "predicted_class": cf_pred,
                    "prob_class_0": round(cf_proba[0], 4),
                    "prob_class_1": round(cf_proba[1], 4),
                    "feature_changes_count": len(feature_changes),
                    "changed_features": feature_changes
                }
                cfs_list.append(cf_record)

                # Add to CSV row list
                cf_export_row = {
                    "sample_label": sample_label,
                    "sample_type": f"Counterfactual Explanation #{cf_idx}",
                    "sample_id": orig_idx_val,
                    "predicted_class": cf_pred,
                    "prob_class_0": round(cf_proba[0], 4),
                    "prob_class_1": round(cf_proba[1], 4),
                    "explanation_disclaimer": "Model decision explanation only, not a medical recommendation"
                }
                for col in feature_cols:
                    cf_export_row[col] = cf_row[col]
                all_cf_rows.append(cf_export_row)

        summary_records.append({
            "sample_label": sample_label,
            "sample_id": orig_idx_val,
            "scenario": meta["category"],
            "original_prediction": {
                "class": orig_pred,
                "label": "Thyroid Disease" if orig_pred == 1 else "Normal / No Disease",
                "prob_class_0": round(orig_proba[0], 4),
                "prob_class_1": round(orig_proba[1], 4)
            },
            "counterfactuals_generated": len(cfs_list),
            "counterfactuals": cfs_list
        })

    # 7. Save outputs to model/
    csv_out_path = os.path.join("model", "counterfactual_samples.csv")
    pd.DataFrame(all_cf_rows).to_csv(csv_out_path, index=False)
    print(f"\n[+] Saved tabular counterfactuals to: {csv_out_path} ({len(all_cf_rows)} rows)")

    json_out_path = os.path.join("model", "counterfactual_summary.json")
    final_json = {
        "metadata": {
            "model_type": "RandomForestClassifier",
            "model_path": "model/random_forest_model.pkl",
            "preprocessor_path": "model/preprocessor.pkl",
            "dataset_source": "dataset/thyroid0387_cleaned.csv",
            "test_split_strategy": "stratified 80/20 (random_state=42)",
            "xai_framework": "DiCE (Diverse Counterfactual Explanations)",
            "xai_method": "random",
            "legal_and_medical_disclaimer": (
                "IMPORTANT: Counterfactual changes generated by DiCE represent algorithmic "
                "decision boundary explanations showing what feature modifications would cause the "
                "machine learning model to alter its classification output. They are strictly "
                "computational model explanations and MUST NOT be interpreted or used as clinical, "
                "medical, or therapeutic treatment recommendations."
            )
        },
        "total_test_samples_evaluated": len(selected_samples),
        "total_counterfactuals_generated": total_cfs_count,
        "samples": summary_records
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)
    print(f"[+] Saved structured summary to: {json_out_path}")

    print("\n" + "=" * 70)
    print("Stage 1 Completed Successfully!")
    print(f"- Total Test Samples: {len(selected_samples)}")
    print(f"- Total Counterfactual Examples: {total_cfs_count}")
    print(f"- CSV: {csv_out_path}")
    print(f"- JSON: {json_out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
