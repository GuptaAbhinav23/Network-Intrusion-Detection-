"""
outlier_analysis.py

Outlier Analysis using IQR Method

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

    os.makedirs(

        os.path.join(
            FIGURE_PATH,
            "boxplots"
        ),

        exist_ok=True

    )


# ==========================================================
# OUTLIER ANALYSIS
# ==========================================================

def outlier_analysis(
        df: pd.DataFrame,
        sample_size=10000):
    """
    Analyze outliers using IQR.

    Parameters
    ----------
    df : DataFrame

    sample_size : int
        Used only for boxplots.
    """

    print("\n" + "=" * 70)
    print("OUTLIER ANALYSIS")
    print("=" * 70)

    create_directories()

    numerical_df = df.select_dtypes(include=np.number)

    if "Label" in numerical_df.columns:

        numerical_df = numerical_df.drop(columns=["Label"])

    report = []

    # ------------------------------------------------------
    # Outlier Count
    # ------------------------------------------------------

    for feature in numerical_df.columns:

        series = numerical_df[feature].dropna()

        if len(series) == 0:
            continue

        Q1 = series.quantile(0.25)

        Q3 = series.quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        outliers = (

            (series < lower) |
            (series > upper)

        )

        outlier_count = outliers.sum()

        outlier_percentage = (

            outlier_count /
            len(series)

        ) * 100

        report.append({

            "Feature": feature,

            "Minimum": float(series.min()),

            "Maximum": float(series.max()),

            "Q1": float(Q1),

            "Q3": float(Q3),

            "IQR": float(IQR),

            "Lower_Bound": float(lower),

            "Upper_Bound": float(upper),

            "Outlier_Count": int(outlier_count),

            "Outlier_Percentage":

                round(outlier_percentage, 4)

        })

    report_df = pd.DataFrame(report)

    report_df = report_df.sort_values(

        by="Outlier_Count",

        ascending=False

    )

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    report_df.to_csv(

        os.path.join(

            TABLE_PATH,

            "outlier_report.csv"

        ),

        index=False

    )

    # ------------------------------------------------------
    # JSON Summary
    # ------------------------------------------------------

    summary = {

        "Total Features":

            int(len(report_df)),

        "Features With Outliers":

            int(

                (report_df["Outlier_Count"] > 0).sum()

            ),

        "Maximum Outlier Percentage":

            float(

                report_df["Outlier_Percentage"].max()

            )

    }

    with open(

        os.path.join(

            TABLE_PATH,

            "outlier_summary.json"

        ),

        "w"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4

        )

    # ------------------------------------------------------
    # Sample for Boxplots
    # ------------------------------------------------------

    if len(numerical_df) > sample_size:

        sampled_df = numerical_df.sample(

            sample_size,

            random_state=42

        )

    else:

        sampled_df = numerical_df.copy()

    # ------------------------------------------------------
    # Boxplots
    # ------------------------------------------------------

    for feature in sampled_df.columns:

        plt.figure(figsize=(8,2))

        plt.boxplot(

            sampled_df[feature].dropna(),

            vert=False

        )

        plt.title(feature)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                FIGURE_PATH,

                "boxplots",

                f"{feature}.png"

            ),

            dpi=300

        )

        plt.close()

    # ------------------------------------------------------
    # Top 20 Outliers
    # ------------------------------------------------------

    top20 = report_df.head(20)

    top20.to_csv(

        os.path.join(

            TABLE_PATH,

            "top20_outlier_features.csv"

        ),

        index=False

    )

    # ------------------------------------------------------
    # Bar Plot
    # ------------------------------------------------------

    plt.figure(figsize=(12,6))

    plt.bar(

        top20["Feature"],

        top20["Outlier_Count"]

    )

    plt.xticks(

        rotation=60,

        ha="right"

    )

    plt.ylabel("Outlier Count")

    plt.title("Top 20 Features with Maximum Outliers")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "top20_outliers.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(f"\nFeatures Analysed : {len(report_df)}")

    print(f"Features Having Outliers : {summary['Features With Outliers']}")

    print("\nTop Outlier Features\n")

    print(top20)

    print("\nGenerated Files")

    print("----------------------------------------")

    print("outlier_report.csv")

    print("top20_outlier_features.csv")

    print("outlier_summary.json")

    print("top20_outliers.png")

    print("Boxplots Folder")

    print("----------------------------------------")

    print("\nOutlier Analysis Completed Successfully.")

    return report_df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from loader import load_dataset
    from config import PROCESSED_DATASET

    df = load_dataset(PROCESSED_DATASET)

    outlier_analysis(df)