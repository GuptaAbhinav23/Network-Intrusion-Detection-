"""
statistical_summary.py

Generate descriptive statistics for all numerical features.

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import json
import numpy as np
import pandas as pd

from .config import TABLE_PATH


# ==========================================================
# CREATE DIRECTORY
# ==========================================================

def create_directory():

    os.makedirs(TABLE_PATH, exist_ok=True)


# ==========================================================
# STATISTICAL SUMMARY
# ==========================================================

def statistical_summary(df: pd.DataFrame):
    """
    Generate descriptive statistics.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    create_directory()

    # ------------------------------------------------------
    # Select Numerical Columns
    # ------------------------------------------------------

    numerical_df = df.select_dtypes(include=np.number)

    if numerical_df.empty:

        raise Exception("No numerical columns found.")

    # ------------------------------------------------------
    # Generate Statistics
    # ------------------------------------------------------

    statistics = []

    for column in numerical_df.columns:

        series = numerical_df[column]

        q1 = series.quantile(0.25)
        q2 = series.quantile(0.50)
        q3 = series.quantile(0.75)

        statistics.append({

            "Feature": column,

            "Count": int(series.count()),

            "Missing Values": int(series.isnull().sum()),

            "Unique Values": int(series.nunique()),

            "Mean": float(series.mean()),

            "Median": float(series.median()),

            "Mode":
                float(series.mode().iloc[0])
                if not series.mode().empty else np.nan,

            "Minimum": float(series.min()),

            "Maximum": float(series.max()),

            "Range":
                float(series.max() - series.min()),

            "Variance": float(series.var()),

            "Standard Deviation":
                float(series.std()),

            "Q1": float(q1),

            "Q2": float(q2),

            "Q3": float(q3),

            "IQR": float(q3 - q1),

            "Skewness":
                float(series.skew()),

            "Kurtosis":
                float(series.kurt())

        })

    summary_df = pd.DataFrame(statistics)

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    csv_path = os.path.join(
        TABLE_PATH,
        "statistical_summary.csv"
    )

    summary_df.to_csv(
        csv_path,
        index=False
    )

    # ------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------

    json_path = os.path.join(
        TABLE_PATH,
        "statistical_summary.json"
    )

    with open(json_path, "w") as file:

        json.dump(
            statistics,
            file,
            indent=4
        )

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(f"\nNumerical Features : {len(summary_df)}")

    print("\nFirst Five Features\n")

    print(summary_df.head())

    print("\nFiles Generated")

    print("--------------------------------------")
    print("statistical_summary.csv")
    print("statistical_summary.json")
    print("--------------------------------------")

    print("\nStatistical Summary Completed Successfully.")

    return summary_df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from loader import load_dataset
    from config import PROCESSED_DATASET

    dataframe = load_dataset(PROCESSED_DATASET)

    statistical_summary(dataframe)