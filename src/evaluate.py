#!/usr/bin/env python3
"""
SageMaker Evaluation script: computes metrics and writes evaluation.json
Input model:    /opt/ml/processing/input/model/model.tar.gz
Input test data:/opt/ml/processing/input/test/test.csv
Output:         /opt/ml/processing/output/evaluation/evaluation.json
"""
import os
import json
import tarfile
import tempfile
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

MODEL_TAR_PATH = "/opt/ml/processing/input/model/model.tar.gz"
TEST_DATA_PATH = "/opt/ml/processing/input/test/test.csv"
OUTPUT_DIR = "/opt/ml/processing/output/evaluation/"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "evaluation.json")
MODEL_FILENAME = "xgboost-model"
CLASSIFICATION_THRESHOLD = 0.5


def extract_model(tar_path, extract_dir):
    """Extract model.tar.gz and return path to the XGBoost model file."""
    print(f"Extracting model from {tar_path} to {extract_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    model_path = os.path.join(extract_dir, MODEL_FILENAME)
    if not os.path.exists(model_path):
        # Search recursively in case it was nested
        for root, dirs, files in os.walk(extract_dir):
            if MODEL_FILENAME in files:
                model_path = os.path.join(root, MODEL_FILENAME)
                break
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find '{MODEL_FILENAME}' in extracted archive at {extract_dir}"
        )
    print(f"Found model at {model_path}")
    return model_path


def evaluate():
    # --- Load model ---
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = extract_model(MODEL_TAR_PATH, tmpdir)
        bst = xgb.Booster()
        bst.load_model(model_path)
        print("Model loaded successfully.")

        # --- Load test data ---
        print(f"Loading test data from {TEST_DATA_PATH}")
        test_df = pd.read_csv(TEST_DATA_PATH, header=0)
        print(f"Test data shape: {test_df.shape}")

        # First column is the label (Churn), remaining are features
        y_true = test_df.iloc[:, 0].values
        X_test = test_df.iloc[:, 1:].values

        dtest = xgb.DMatrix(X_test)

        # --- Predict ---
        y_prob = bst.predict(dtest)
        y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)

        # --- Metrics ---
        auc = float(roc_auc_score(y_true, y_prob))
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        print(f"\nAUC:       {auc:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]))

        # --- Write evaluation.json ---
        evaluation = {
            "classification_metrics": {
                "auc": {"value": round(auc, 4), "standard_deviation": 0.0},
                "accuracy": {"value": round(accuracy, 4)},
                "precision": {"value": round(precision, 4)},
                "recall": {"value": round(recall, 4)},
                "f1": {"value": round(f1, 4)},
            }
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(evaluation, f, indent=2)
        print(f"\nEvaluation report saved to {OUTPUT_PATH}")
        print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    evaluate()
