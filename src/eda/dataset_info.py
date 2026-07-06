"""
dataset_info.py

Generate basic information about the processed dataset.

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import json
import pandas as pd


from .config import TABLE_PATH


# ==========================================================
# CREATE DIRECTORY
# ==========================================================

def create_directory():

    os.makedirs(TABLE_PATH, exist_ok=True)


# ==========================================================
# DATASET INFORMATION
# ==========================================================

def dataset_information(df: pd.DataFrame):
    """
    Generate complete dataset information.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
    """

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    create_directory()

    # ------------------------------------------------------
    # Basic Information
    # ------------------------------------------------------

    rows = df.shape[0]
    columns = df.shape[1]

    features = columns - 1 if "Label" in df.columns else columns

    memory_mb = round(
        df.memory_usage(deep=True).sum() / (1024 ** 2),
        2
    )

    numerical_columns = list(
        df.select_dtypes(include="number").columns
    )

    categorical_columns = list(
        df.select_dtypes(exclude="number").columns
    )

    # ------------------------------------------------------
    # Data Types
    # ------------------------------------------------------

    datatype_df = pd.DataFrame({

        "Column": df.columns,
        "Datatype": df.dtypes.astype(str)

    })

    datatype_path = os.path.join(
        TABLE_PATH,
        "column_datatypes.csv"
    )

    datatype_df.to_csv(
        datatype_path,
        index=False
    )

    # ------------------------------------------------------
    # Dataset Summary
    # ------------------------------------------------------

    summary = {

        "Rows": int(rows),

        "Columns": int(columns),

        "Features": int(features),

        "Numerical Features":
            len(numerical_columns),

        "Categorical Features":
            len(categorical_columns),

        "Memory Usage (MB)":
            memory_mb,

        "Duplicate Rows":
            int(df.duplicated().sum()),

        "Missing Values":
            int(df.isnull().sum().sum()),

        "Label Classes":
            int(df["Label"].nunique())
            if "Label" in df.columns else 0

    }

    summary_path = os.path.join(
        TABLE_PATH,
        "dataset_summary.json"
    )

    with open(summary_path, "w") as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ------------------------------------------------------
    # Data Type Counts
    # ------------------------------------------------------

    dtype_count = (
        df.dtypes
        .astype(str)
        .value_counts()
        .reset_index()
    )

    dtype_count.columns = [

        "Datatype",
        "Count"

    ]

    dtype_count_path = os.path.join(
        TABLE_PATH,
        "datatype_count.csv"
    )

    dtype_count.to_csv(
        dtype_count_path,
        index=False
    )

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(f"\nRows                 : {rows:,}")
    print(f"Columns              : {columns}")
    print(f"Features             : {features}")
    print(f"Numerical Features   : {len(numerical_columns)}")
    print(f"Categorical Features : {len(categorical_columns)}")
    print(f"Memory Usage         : {memory_mb:.2f} MB")
    print(f"Duplicate Rows       : {summary['Duplicate Rows']}")
    print(f"Missing Values       : {summary['Missing Values']}")

    if "Label" in df.columns:

        print(f"Attack Classes       : {summary['Label Classes']}")

    print("\nDatatype Distribution")

    print(dtype_count)

    print("\nFiles Generated")

    print("------------------------------------")
    print("column_datatypes.csv")
    print("datatype_count.csv")
    print("dataset_summary.json")
    print("------------------------------------")

    print("\nDataset Information Generated Successfully.")

    return summary