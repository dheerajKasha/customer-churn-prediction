#!/usr/bin/env python3
"""
SageMaker Training script for XGBoost churn classifier.
Reads hyperparameters from /opt/ml/input/config/hyperparameters.json
Saves model to /opt/ml/model/xgboost-model
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb

HP_PATH = "/opt/ml/input/config/hyperparameters.json"
TRAIN_DATA_DIR = "/opt/ml/input/data/train/"
VAL_DATA_DIR = "/opt/ml/input/data/validation/"
MODEL_DIR = "/opt/ml/model/"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost-model")

DEFAULT_HYPERPARAMETERS = {
    "max_depth": 6,
    "eta": 0.1,
    "num_round": 100,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}


def load_hyperparameters():
    """Load hyperparameters from SageMaker config, falling back to defaults."""
    params = DEFAULT_HYPERPARAMETERS.copy()
    if os.path.exists(HP_PATH):
        with open(HP_PATH, "r") as f:
            hp = json.load(f)
        print(f"Loaded hyperparameters from {HP_PATH}: {hp}")
        # SageMaker passes all HP values as strings — cast to appropriate types
        if "max_depth" in hp:
            params["max_depth"] = int(hp["max_depth"])
        if "eta" in hp:
            params["eta"] = float(hp["eta"])
        if "num_round" in hp:
            params["num_round"] = int(hp["num_round"])
        if "objective" in hp:
            params["objective"] = hp["objective"]
        if "eval_metric" in hp:
            params["eval_metric"] = hp["eval_metric"]
        if "subsample" in hp:
            params["subsample"] = float(hp["subsample"])
        if "colsample_bytree" in hp:
            params["colsample_bytree"] = float(hp["colsample_bytree"])
        if "min_child_weight" in hp:
            params["min_child_weight"] = int(hp["min_child_weight"])
    else:
        print(f"No hyperparameters file found at {HP_PATH}. Using defaults: {params}")
    return params


def load_csv_from_dir(data_dir):
    """Load the first CSV file found in data_dir."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    filepath = os.path.join(data_dir, csv_files[0])
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, header=0)
    return df


def df_to_dmatrix(df):
    """Split first column as label and remaining as features, return DMatrix."""
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return xgb.DMatrix(X, label=y)


def train():
    hp = load_hyperparameters()
    num_round = hp.pop("num_round")

    print("Loading training data...")
    train_df = load_csv_from_dir(TRAIN_DATA_DIR)
    print(f"Train shape: {train_df.shape}")

    print("Loading validation data...")
    val_df = load_csv_from_dir(VAL_DATA_DIR)
    print(f"Validation shape: {val_df.shape}")

    dtrain = df_to_dmatrix(train_df)
    dval = df_to_dmatrix(val_df)

    watchlist = [(dtrain, "train"), (dval, "validation")]

    print(f"Starting XGBoost training — num_round={num_round}, params={hp}")
    evals_result = {}
    bst = xgb.train(
        params=hp,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=watchlist,
        evals_result=evals_result,
        early_stopping_rounds=10,
        verbose_eval=10,
    )

    train_auc = evals_result["train"]["auc"][-1]
    val_auc = evals_result["validation"]["auc"][-1]
    print(f"Final Train AUC:      {train_auc:.4f}")
    print(f"Final Validation AUC: {val_auc:.4f}")
    print(f"Best iteration:       {bst.best_iteration}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    bst.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
