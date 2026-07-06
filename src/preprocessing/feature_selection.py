import os
import glob
import joblib
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from .config import (
    RAW_DATASET_PATH,
    PROCESSED_PATH,
    SCALER_PATH,
    ENCODER_PATH
)

# =====================================================
# REMOVE CONSTANT & QUASI-CONSTANT FEATURES
# =====================================================

def remove_constant_quasi_constant_features(df,
                                            constant_threshold=1,
                                            quasi_threshold=0.99):
    """
    Remove constant and quasi-constant features.

    Parameters
    ----------
    df : pandas.DataFrame

    constant_threshold :
        Number of unique values allowed for a constant feature.
        Default = 1

    quasi_threshold :
        If one value occupies >=99% of the rows,
        the feature is considered quasi-constant.
    """

    print("\n" + "=" * 70)
    print("REMOVING CONSTANT & QUASI-CONSTANT FEATURES")
    print("=" * 70)

    total_columns_before = df.shape[1]

    constant_features = []

    quasi_constant_features = []

    # Ignore Label column
    feature_columns = [col for col in df.columns if col != "Label"]

    # -------------------------------------------------
    # Constant Features
    # -------------------------------------------------

    for column in feature_columns:

        if df[column].nunique(dropna=False) <= constant_threshold:

            constant_features.append(column)

    # -------------------------------------------------
    # Quasi Constant Features
    # -------------------------------------------------

    for column in feature_columns:

        if column in constant_features:
            continue

        dominant_ratio = (
            df[column]
            .value_counts(normalize=True, dropna=False)
            .values[0]
        )

        if dominant_ratio >= quasi_threshold:

            quasi_constant_features.append(column)

    # -------------------------------------------------
    # Remove Features
    # -------------------------------------------------

    features_to_remove = constant_features + quasi_constant_features

    if len(features_to_remove) > 0:

        df.drop(columns=features_to_remove,
                inplace=True)

    # -------------------------------------------------
    # Report
    # -------------------------------------------------

    print(f"Total Features Before : {total_columns_before}")

    print(f"Constant Features Removed : {len(constant_features)}")

    if len(constant_features) > 0:

        for feature in constant_features:

            print(f"   Constant : {feature}")

    print(f"\nQuasi-Constant Features Removed : {len(quasi_constant_features)}")

    if len(quasi_constant_features) > 0:

        for feature in quasi_constant_features:

            print(f"   Quasi : {feature}")

    print(f"\nTotal Features After : {df.shape[1]}")

    print("\nFeature Selection Completed Successfully.")

    return df


# =====================================================
# REMOVE HIGHLY CORRELATED FEATURES
# =====================================================

import json

def remove_highly_correlated_features(
        df,
        correlation_threshold=0.95,
        save_report=True):
    """
    Remove highly correlated features.

    Parameters
    ----------
    df : pandas.DataFrame

    correlation_threshold : float
        Correlation threshold.
        Default = 0.95
    """

    print("\n" + "=" * 70)
    print("REMOVING HIGHLY CORRELATED FEATURES")
    print("=" * 70)

    total_before = df.shape[1]

    # ----------------------------------------
    # Separate Features
    # ----------------------------------------

    X = df.drop(columns=["Label"])

    y = df["Label"]

    # ----------------------------------------
    # Correlation Matrix
    # ----------------------------------------

    correlation_matrix = X.corr().abs()

    # ----------------------------------------
    # Upper Triangle
    # ----------------------------------------

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    # ----------------------------------------
    # Find Correlated Features
    # ----------------------------------------

    correlated_features = []

    correlation_pairs = []

    for column in upper_triangle.columns:

        high_corr = upper_triangle[column] > correlation_threshold

        if high_corr.any():

            correlated_features.append(column)

            correlated_columns = upper_triangle.index[high_corr].tolist()

            for item in correlated_columns:

                correlation_pairs.append({

                    "Remove": column,

                    "Correlated_With": item,

                    "Correlation":
                    round(
                        upper_triangle.loc[item, column],
                        4
                    )

                })

    correlated_features = sorted(
        list(set(correlated_features))
    )

    # ----------------------------------------
    # Remove Features
    # ----------------------------------------

    X.drop(
        columns=correlated_features,
        inplace=True
    )

    processed_df = X.copy()

    processed_df["Label"] = y

    # ----------------------------------------
    # Report
    # ----------------------------------------

    print(f"Features Before : {total_before}")

    print(f"Highly Correlated Features Removed : {len(correlated_features)}")

    print(f"Features After : {processed_df.shape[1]}")

    if len(correlated_features) > 0:

        print("\nRemoved Features:\n")

        for feature in correlated_features:

            print(feature)

    # ----------------------------------------
    # Save Report
    # ----------------------------------------

    if save_report:

        report = {

            "threshold": correlation_threshold,

            "removed_features": correlated_features,

            "correlation_pairs": correlation_pairs

        }

        report_path = os.path.join(
            PROCESSED_PATH,
            "correlation_report.json"
        )

        with open(report_path, "w") as file:

            json.dump(
                report,
                file,
                indent=4
            )

        print("\nCorrelation report saved.")

    print("\nCorrelation Analysis Completed Successfully.")

    return processed_df