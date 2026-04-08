#!/usr/bin/env python3
"""
SageMaker Processing script: cleans Telco CSV and outputs train/test splits.
Input:  /opt/ml/processing/input/data/telco_churn.csv
Output: /opt/ml/processing/output/train/train.csv
        /opt/ml/processing/output/test/test.csv
        /opt/ml/processing/output/feature_names.json
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_PATH = "/opt/ml/processing/input/data/telco_churn.csv"
OUTPUT_TRAIN = "/opt/ml/processing/output/train/train.csv"
OUTPUT_TEST = "/opt/ml/processing/output/test/test.csv"
OUTPUT_FEATURES = "/opt/ml/processing/output/feature_names.json"


def preprocess():
    print(f"Reading data from {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Raw data shape: {df.shape}")

    # Drop customer ID — not a predictive feature
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric (some rows contain whitespace strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total_charges = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)
    print(f"Filled {df['TotalCharges'].isna().sum()} NaN values in TotalCharges with median {median_total_charges:.2f}")

    # Encode target: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")

    # One-hot encode all remaining categorical (object) columns
    df = pd.get_dummies(df, drop_first=True)
    print(f"Shape after one-hot encoding: {df.shape}")

    # Save feature names (all columns except the target)
    feature_names = [col for col in df.columns if col != "Churn"]
    os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)
    with open(OUTPUT_FEATURES, "w") as f:
        json.dump(feature_names, f)
    print(f"Saved {len(feature_names)} feature names to {OUTPUT_FEATURES}")

    # Stratified 80/20 train/test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Reorder columns so Churn is first (XGBoost built-in CSV requirement)
    cols = ["Churn"] + [c for c in df.columns if c != "Churn"]
    train_df = train_df[cols]
    test_df = test_df[cols]

    # Write outputs
    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_TEST), exist_ok=True)
    train_df.to_csv(OUTPUT_TRAIN, index=False, header=True)
    test_df.to_csv(OUTPUT_TEST, index=False, header=True)
    print(f"Saved train data to {OUTPUT_TRAIN}")
    print(f"Saved test data  to {OUTPUT_TEST}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
