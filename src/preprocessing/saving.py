"""
saving.py

Save all processed datasets and reports.
"""

import os
import json
import numpy as np
import pandas as pd

from .config import PROCESSED_PATH


def save_dataset(
        merged_df,
        processed_df,
        X_train,
        X_test,
        y_train,
        y_test):

    print("\n" + "=" * 80)
    print("SAVING DATASETS")
    print("=" * 80)

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # ============================================================
    # Save Merged Dataset
    # ============================================================

    merged_path = os.path.join(
        PROCESSED_PATH,
        "merged_dataset.csv"
    )

    merged_df.to_csv(
        merged_path,
        index=False
    )

    print("[OK] merged_dataset.csv")

    # ============================================================
    # Save Processed Dataset
    # ============================================================

    processed_path = os.path.join(
        PROCESSED_PATH,
        "processed_dataset.csv"
    )

    processed_df.to_csv(
        processed_path,
        index=False
    )

    print("[OK] processed_dataset.csv")

    # ============================================================
    # Save Training Dataset
    # ============================================================

    train_df = X_train.copy()

    train_df["Label"] = y_train.values

    train_path = os.path.join(
        PROCESSED_PATH,
        "train_processed.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    
    print("[OK] train_processed.csv")


    # ============================================================
    # Save Testing Dataset
    # ============================================================

    test_df = X_test.copy()

    test_df["Label"] = y_test.values

    test_path = os.path.join(
        PROCESSED_PATH,
        "test_processed.csv"
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print("[OK] test_processed.csv")

    # ============================================================
    # Save NumPy Arrays
    # ============================================================

    np.save(
        os.path.join(PROCESSED_PATH, "X_train.npy"),
        X_train.to_numpy()
    )

    np.save(
        os.path.join(PROCESSED_PATH, "X_test.npy"),
        X_test.to_numpy()
    )

    np.save(
        os.path.join(PROCESSED_PATH, "y_train.npy"),
        y_train.to_numpy()
    )

    np.save(
        os.path.join(PROCESSED_PATH, "y_test.npy"),
        y_test.to_numpy()
    )

    print("[OK] X_train.npy")
    print("[OK] X_test.npy")
    print("[OK] y_train.npy")
    print("[OK] y_test.npy")

    # ============================================================
    # Save Feature Names
    # ============================================================

    feature_names = list(X_train.columns)

    feature_path = os.path.join(
        PROCESSED_PATH,
        "feature_names.json"
    )

    with open(feature_path, "w") as file:

        json.dump(
            feature_names,
            file,
            indent=4
        )

    print("[OK] feature_names.json")

    # ============================================================
    # Save Summary
    # ============================================================

    summary = {

        "Merged Dataset Shape":
            list(merged_df.shape),

        "Processed Dataset Shape":
            list(processed_df.shape),

        "Training Dataset Shape":
            list(train_df.shape),

        "Testing Dataset Shape":
            list(test_df.shape),

        "Training Samples":
            int(len(train_df)),

        "Testing Samples":
            int(len(test_df)),

        "Number of Features":
            int(X_train.shape[1]),

        "Training Classes":
            int(y_train.nunique()),

        "Testing Classes":
            int(y_test.nunique())

    }

    summary_path = os.path.join(
        PROCESSED_PATH,
        "preprocessing_summary.json"
    )

    with open(summary_path, "w") as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("[OK] preprocessing_summary.json")

    print("\n" + "=" * 80)
    print("ALL FILES SAVED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated Files")

    print("------------------------------------")

    print("merged_dataset.csv")

    print("processed_dataset.csv")

    print("train_processed.csv")

    print("test_processed.csv")

    print("X_train.npy")

    print("X_test.npy")

    print("y_train.npy")

    print("y_test.npy")

    print("feature_names.json")

    print("preprocessing_summary.json")

    print("------------------------------------")

    print("\nTraining Shape :", train_df.shape)

    print("Testing Shape  :", test_df.shape)

    print("\nDataset Saving Completed Successfully.")