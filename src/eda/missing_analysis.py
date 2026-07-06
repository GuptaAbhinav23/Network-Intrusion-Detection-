"""
missing_analysis.py

Perform Missing Value Analysis for the processed dataset.

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import TABLE_PATH, FIGURE_PATH


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

def create_directories():

    os.makedirs(TABLE_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)


# ==========================================================
# MISSING VALUE ANALYSIS
# ==========================================================

def missing_value_analysis(df: pd.DataFrame):
    """
    Analyze missing values in the dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    create_directories()

    total_rows = len(df)

    # ------------------------------------------------------
    # Missing Count
    # ------------------------------------------------------

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / total_rows
    ) * 100

    report = pd.DataFrame({

        "Feature": df.columns,
        "Missing_Count": missing_count.values,
        "Missing_Percentage": missing_percentage.values

    })

    report = report.sort_values(
        by="Missing_Count",
        ascending=False
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    csv_path = os.path.join(
        TABLE_PATH,
        "missing_value_report.csv"
    )

    report.to_csv(
        csv_path,
        index=False
    )

    # ------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------

    summary = {

        "Total Features": int(df.shape[1]),

        "Features With Missing Values":
            int((missing_count > 0).sum()),

        "Total Missing Values":
            int(missing_count.sum()),

        "Maximum Missing Percentage":
            float(round(missing_percentage.max(), 4))

    }

    json_path = os.path.join(
        TABLE_PATH,
        "missing_value_summary.json"
    )

    with open(json_path, "w") as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ------------------------------------------------------
    # Missing Bar Plot
    # ------------------------------------------------------

    non_zero = report[
        report["Missing_Count"] > 0
    ]

    if len(non_zero) > 0:

        plt.figure(figsize=(14, 6))

        plt.bar(
            non_zero["Feature"],
            non_zero["Missing_Count"]
        )

        plt.xticks(rotation=90)

        plt.xlabel("Features")

        plt.ylabel("Missing Values")

        plt.title("Missing Values Per Feature")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURE_PATH,
                "missing_values_barplot.png"
            ),
            dpi=300
        )

        plt.close()

    # ------------------------------------------------------
    # Missing Heatmap
    # ------------------------------------------------------

    sample_size = min(1000, len(df))

    sampled_df = df.sample(
        sample_size,
        random_state=42
    )

    plt.figure(figsize=(14, 6))

    plt.imshow(
        sampled_df.isnull(),
        aspect="auto",
        interpolation="nearest"
    )

    plt.title("Missing Value Heatmap (Sampled Data)")

    plt.xlabel("Features")

    plt.ylabel("Samples")

    plt.colorbar(label="Missing")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            FIGURE_PATH,
            "missing_value_heatmap.png"
        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(f"\nTotal Features              : {df.shape[1]}")
    print(f"Total Samples               : {len(df):,}")
    print(f"Total Missing Values        : {summary['Total Missing Values']}")
    print(f"Features With Missing       : {summary['Features With Missing Values']}")
    print(f"Maximum Missing Percentage  : {summary['Maximum Missing Percentage']:.4f}%")

    print("\nTop Missing Features\n")

    print(report.head(20))

    print("\nFiles Generated")

    print("------------------------------------------")
    print("missing_value_report.csv")
    print("missing_value_summary.json")
    print("missing_values_barplot.png")
    print("missing_value_heatmap.png")
    print("------------------------------------------")

    print("\nMissing Value Analysis Completed Successfully.")

    return report