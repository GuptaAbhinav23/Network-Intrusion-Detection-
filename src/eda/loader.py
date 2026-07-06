"""
loader.py

Load the processed dataset for Exploratory Data Analysis (EDA).

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import pandas as pd


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the processed dataset.

    Parameters
    ----------
    dataset_path : str
        Path of processed_dataset.csv

    Returns
    -------
    pandas.DataFrame
    """

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    # ------------------------------------------------------
    # Check Dataset Exists
    # ------------------------------------------------------

    if not os.path.exists(dataset_path):

        raise FileNotFoundError(
            f"\nDataset not found:\n{dataset_path}"
        )

    print(f"\nDataset Found")
    print(dataset_path)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    try:

        df = pd.read_csv(
            dataset_path,
            low_memory=False
        )

    except Exception as e:

        raise Exception(
            f"Unable to load dataset.\n{e}"
        )

    # ------------------------------------------------------
    # Empty Dataset Check
    # ------------------------------------------------------

    if df.empty:

        raise ValueError(
            "Loaded dataset is empty."
        )

    # ------------------------------------------------------
    # Remove Duplicate Columns
    # ------------------------------------------------------

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()

    if len(duplicate_columns) > 0:

        print("\nDuplicate Columns Found")

        for col in duplicate_columns:
            print(" -", col)

        df = df.loc[:, ~df.columns.duplicated()]

        print("\nDuplicate columns removed.")

    # ------------------------------------------------------
    # Check Label Column
    # ------------------------------------------------------

    if "Label" not in df.columns:

        raise Exception(
            "Label column not found in dataset."
        )

    # ------------------------------------------------------
    # Basic Information
    # ------------------------------------------------------

    print("\nDataset Loaded Successfully\n")

    print(f"Rows              : {df.shape[0]:,}")
    print(f"Columns           : {df.shape[1]}")
    print(f"Features          : {df.shape[1]-1}")
    print(f"Label Column      : Label")

    memory = (
        df.memory_usage(deep=True)
        .sum() / (1024 ** 2)
    )

    print(f"Memory Usage      : {memory:.2f} MB")

    print("\nData Types\n")

    print(df.dtypes.value_counts())

    print("\nFirst Five Rows\n")

    print(df.head())

    print("\nDataset Loading Completed Successfully.")

    return df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    DATASET = "../../dataset/processed/processed_dataset.csv"

    dataframe = load_dataset(DATASET)