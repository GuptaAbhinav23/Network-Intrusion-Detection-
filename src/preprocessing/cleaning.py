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
# CLEAN COLUMN NAMES
# =====================================================

import re

def clean_column_names(df):
    """
    Clean and standardize column names.
    """

    print("\n" + "=" * 70)
    print("CLEANING COLUMN NAMES")
    print("=" * 70)

    original_columns = df.columns.tolist()

    cleaned_columns = []

    for col in original_columns:

        # Convert to string
        col = str(col)

        # Remove leading/trailing spaces
        col = col.strip()

        # Replace multiple spaces with single underscore
        col = re.sub(r"\s+", "_", col)

        # Remove brackets
        col = col.replace("(", "")
        col = col.replace(")", "")

        # Replace slash
        col = col.replace("/", "_")

        # Replace dash
        col = col.replace("-", "_")

        # Replace dot
        col = col.replace(".", "_")

        # Remove commas
        col = col.replace(",", "")

        # Remove colon
        col = col.replace(":", "")

        # Remove percentage symbol
        col = col.replace("%", "Percent")

        # Remove any remaining special characters
        col = re.sub(r"[^A-Za-z0-9_]", "", col)

        # Remove repeated underscores
        col = re.sub(r"_+", "_", col)

        # Remove underscore at beginning/end
        col = col.strip("_")

        cleaned_columns.append(col)

    # Check duplicate names
    duplicates = pd.Series(cleaned_columns).duplicated()

    if duplicates.any():

        print("\nDuplicate column names found.")

        seen = {}

        new_columns = []

        for col in cleaned_columns:

            if col not in seen:

                seen[col] = 0
                new_columns.append(col)

            else:

                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")

        cleaned_columns = new_columns

    df.columns = cleaned_columns

    print(f"Total Columns : {len(cleaned_columns)}")
    print("Column names cleaned successfully.")

    return df


# =====================================================
# REMOVE DUPLICATE ROWS
# =====================================================

def remove_duplicates(df):
    """
    Remove duplicate records from the dataset.
    """

    print("\n" + "=" * 70)
    print("REMOVING DUPLICATE RECORDS")
    print("=" * 70)

    total_rows_before = len(df)

    duplicate_rows = df.duplicated().sum()

    print(f"Total Rows Before Cleaning : {total_rows_before:,}")
    print(f"Duplicate Rows Found       : {duplicate_rows:,}")

    if duplicate_rows > 0:

        df = df.drop_duplicates(keep="first").reset_index(drop=True)

    total_rows_after = len(df)

    removed = total_rows_before - total_rows_after

    print(f"Rows Removed               : {removed:,}")
    print(f"Total Rows After Cleaning  : {total_rows_after:,}")

    print("Duplicate removal completed successfully.")

    return df

# =====================================================
# REPLACE ±INFINITY VALUES
# =====================================================

def replace_infinity(df):
    """
    Replace positive and negative infinity values with NaN.
    """

    print("\n" + "=" * 70)
    print("REPLACING INFINITE VALUES")
    print("=" * 70)

    # Count infinite values before replacement
    positive_inf = np.isposinf(df.select_dtypes(include=[np.number])).sum().sum()
    negative_inf = np.isneginf(df.select_dtypes(include=[np.number])).sum().sum()

    total_inf = positive_inf + negative_inf

    print(f"Positive Infinity Values : {positive_inf:,}")
    print(f"Negative Infinity Values : {negative_inf:,}")
    print(f"Total Infinity Values    : {total_inf:,}")

    # Replace ±Infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Infinity values successfully replaced with NaN.")

    return df

# =====================================================
# CONVERT FEATURES TO NUMERIC
# =====================================================

def convert_features_to_numeric(df):
    """
    Convert all feature columns to numeric data types.
    The 'Label' column is excluded.
    Invalid values are converted to NaN.
    """

    print("\n" + "=" * 70)
    print("CONVERTING FEATURES TO NUMERIC")
    print("=" * 70)

    total_columns = len(df.columns)

    converted_columns = 0
    failed_columns = []

    for column in df.columns:

        # Skip label column
        if column == "Label":
            continue

        try:

            # Remove leading/trailing spaces
            df[column] = df[column].astype(str).str.strip()

            # Convert to numeric
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            converted_columns += 1

        except Exception:

            failed_columns.append(column)

    print(f"Total Columns           : {total_columns}")
    print(f"Converted Feature Count : {converted_columns}")

    if len(failed_columns) > 0:

        print("\nColumns with Conversion Issues:")

        for col in failed_columns:
            print(" -", col)

    else:

        print("All feature columns successfully converted.")

    # Count NaN values created during conversion
    nan_count = df.isna().sum().sum()

    print(f"\nNaN Values After Conversion : {nan_count:,}")

    return df



# =====================================================
# HANDLE MISSING VALUES
# =====================================================

def handle_missing_values(df, threshold=0.40):
    """
    Handle missing values using a research-grade strategy.

    Parameters
    ----------
    df : pandas.DataFrame
    threshold : float
        Maximum allowed percentage of missing values in a column.
        Columns above this threshold are removed.

    Returns
    -------
    pandas.DataFrame
    """

    print("\n" + "=" * 70)
    print("HANDLING MISSING VALUES")
    print("=" * 70)

    rows_before = df.shape[0]
    cols_before = df.shape[1]

    print(f"Dataset Shape : {df.shape}")

    # -------------------------------------------------
    # Missing values before cleaning
    # -------------------------------------------------

    missing = df.isnull().sum()

    total_missing = missing.sum()

    print(f"\nTotal Missing Values : {total_missing:,}")

    if total_missing == 0:

        print("No missing values found.")

        return df

    # -------------------------------------------------
    # Remove columns having too many missing values
    # -------------------------------------------------

    percentage = (missing / len(df)) * 100

    remove_columns = percentage[percentage > threshold * 100].index.tolist()

    if len(remove_columns) > 0:

        print(f"\nRemoving {len(remove_columns)} columns "
              f"having more than {threshold*100:.0f}% missing values.")

        for col in remove_columns:
            print(f" - {col}")

        df.drop(columns=remove_columns, inplace=True)

    else:

        print("\nNo columns exceeded missing-value threshold.")

    # -------------------------------------------------
    # Fill Numeric Columns
    # -------------------------------------------------

    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:

        median = df[col].median()

        df[col].fillna(median, inplace=True)

    print(f"\nFilled {len(numeric_columns)} numeric columns using Median.")

    # -------------------------------------------------
    # Fill Categorical Columns
    # -------------------------------------------------

    categorical_columns = df.select_dtypes(exclude=np.number).columns

    for col in categorical_columns:

        mode = df[col].mode()

        if len(mode) > 0:

            df[col].fillna(mode[0], inplace=True)

    print(f"Filled {len(categorical_columns)} categorical columns using Mode.")

    # -------------------------------------------------
    # Final Check
    # -------------------------------------------------

    remaining_missing = df.isnull().sum().sum()

    print("\nFinal Dataset Shape :", df.shape)

    print(f"Remaining Missing Values : {remaining_missing}")

    print(f"Rows : {rows_before:,} -> {df.shape[0]:,}")

    print(f"Columns : {cols_before} -> {df.shape[1]}")

    print("\nMissing Value Handling Completed Successfully.")

    return df


# =====================================================
# HANDLE OUTLIERS (IQR CLIPPING)
# =====================================================

def handle_outliers(df):
    """
    Handle outliers using IQR clipping.

    Extreme values are clipped instead of removing rows.
    This preserves attack traffic for intrusion detection.
    """

    print("\n" + "=" * 70)
    print("HANDLING OUTLIERS")
    print("=" * 70)

    feature_columns = [c for c in df.columns if c != "Label"]

    total_outliers = 0

    processed_columns = 0

    for column in feature_columns:

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        Q1 = df[column].quantile(0.25)

        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        outlier_count = (
            ((df[column] < lower) |
             (df[column] > upper))
        ).sum()

        total_outliers += outlier_count

        # Clip values instead of deleting rows
        df[column] = df[column].clip(
            lower=lower,
            upper=upper
        )

        processed_columns += 1

    print(f"Processed Numeric Features : {processed_columns}")

    print(f"Outlier Values Clipped     : {total_outliers:,}")

    print("Outlier handling completed successfully.")

    return df