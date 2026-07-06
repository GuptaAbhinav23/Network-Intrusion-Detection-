"""
feature_importance.py

Feature Importance Analysis using Random Forest

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from .config import (
    TABLE_PATH,
    FIGURE_PATH
)


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

def create_directories():

    os.makedirs(TABLE_PATH, exist_ok=True)

    os.makedirs(FIGURE_PATH, exist_ok=True)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance_analysis(
        df: pd.DataFrame,
        n_estimators=200,
        random_state=42):
    """
    Compute Feature Importance using Random Forest.
    """

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    create_directories()

    # ------------------------------------------------------
    # Check Label Column
    # ------------------------------------------------------

    if "Label" not in df.columns:

        raise Exception("Label column not found.")

    # ------------------------------------------------------
    # Features and Target
    # ------------------------------------------------------

    X = df.drop(columns=["Label"])

    y = df["Label"]

    print(f"\nTraining Samples : {len(X):,}")
    print(f"Features         : {X.shape[1]}")

    # ------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=n_estimators,

        random_state=random_state,

        n_jobs=-1,

        class_weight="balanced"

    )

    print("\nTraining Random Forest...")

    model.fit(X, y)

    print("Training Completed.")

    # ------------------------------------------------------
    # Importance
    # ------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(

        by="Importance",

        ascending=False

    ).reset_index(drop=True)

    importance_df["Cumulative Importance"] = (

        importance_df["Importance"]

        .cumsum()

    )

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    csv_path = os.path.join(

        TABLE_PATH,

        "feature_importance.csv"

    )

    importance_df.to_csv(

        csv_path,

        index=False

    )

    # ------------------------------------------------------
    # Top 20 Features
    # ------------------------------------------------------

    top20 = importance_df.head(20)

    top20.to_csv(

        os.path.join(

            TABLE_PATH,

            "top20_feature_importance.csv"

        ),

        index=False

    )

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    model_path = os.path.join(

        TABLE_PATH,

        "feature_importance_model.pkl"

    )

    joblib.dump(

        model,

        model_path

    )

    # ------------------------------------------------------
    # JSON Summary
    # ------------------------------------------------------

    summary = {

        "Total Features":

            int(len(importance_df)),

        "Most Important Feature":

            str(

                importance_df.iloc[0]["Feature"]

            ),

        "Highest Importance":

            float(

                importance_df.iloc[0]["Importance"]

            ),

        "Random Forest Trees":

            int(n_estimators)

    }

    with open(

        os.path.join(

            TABLE_PATH,

            "feature_importance_summary.json"

        ),

        "w"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4

        )

    # ------------------------------------------------------
    # Top 20 Plot
    # ------------------------------------------------------

    plt.figure(figsize=(12,8))

    plt.barh(

        top20["Feature"],

        top20["Importance"]

    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")

    plt.ylabel("Features")

    plt.title("Top 20 Feature Importance")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "top20_feature_importance.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Cumulative Importance
    # ------------------------------------------------------

    plt.figure(figsize=(10,6))

    plt.plot(

        importance_df.index + 1,

        importance_df["Cumulative Importance"],

        linewidth=2

    )

    plt.grid(True)

    plt.xlabel("Number of Features")

    plt.ylabel("Cumulative Importance")

    plt.title("Cumulative Feature Importance")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "cumulative_feature_importance.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print("\nTop 20 Important Features\n")

    print(top20)

    print("\nGenerated Files")

    print("---------------------------------------------")

    print("feature_importance.csv")

    print("top20_feature_importance.csv")

    print("feature_importance_summary.json")

    print("feature_importance_model.pkl")

    print("top20_feature_importance.png")

    print("cumulative_feature_importance.png")

    print("---------------------------------------------")

    print("\nFeature Importance Analysis Completed Successfully.")

    return importance_df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from loader import load_dataset
    from config import PROCESSED_DATASET

    df = load_dataset(PROCESSED_DATASET)

    feature_importance_analysis(df)