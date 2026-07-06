"""
correlation_analysis.py

Correlation Analysis for Network Intrusion Detection Dataset

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
# CORRELATION ANALYSIS
# ==========================================================

def correlation_analysis(
        df: pd.DataFrame,
        threshold=0.90):
    """
    Perform correlation analysis.

    Parameters
    ----------
    df : pandas.DataFrame

    threshold : float
        Correlation threshold.
    """

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    create_directories()

    # ------------------------------------------------------
    # Numerical Features
    # ------------------------------------------------------

    numerical_df = df.select_dtypes(include=np.number)

    if "Label" in numerical_df.columns:

        numerical_df = numerical_df.drop(columns=["Label"])

    # ------------------------------------------------------
    # Correlation Matrix
    # ------------------------------------------------------

    correlation_matrix = numerical_df.corr()

    correlation_matrix.to_csv(

        os.path.join(

            TABLE_PATH,
            "correlation_matrix.csv"

        )

    )

    # ------------------------------------------------------
    # Highly Correlated Features
    # ------------------------------------------------------

    upper_triangle = correlation_matrix.where(

        np.triu(

            np.ones(correlation_matrix.shape),

            k=1

        ).astype(bool)

    )

    correlated_pairs = []

    for column in upper_triangle.columns:

        high_corr = upper_triangle[column].abs() >= threshold

        correlated_columns = upper_triangle.index[high_corr]

        for feature in correlated_columns:

            correlated_pairs.append({

                "Feature_1": feature,

                "Feature_2": column,

                "Correlation":

                    round(
                        upper_triangle.loc[
                            feature,
                            column
                        ],
                        4
                    )

            })

    correlated_df = pd.DataFrame(correlated_pairs)

    correlated_df = correlated_df.sort_values(

        by="Correlation",

        ascending=False

    )

    correlated_df.to_csv(

        os.path.join(

            TABLE_PATH,

            "highly_correlated_features.csv"

        ),

        index=False

    )

    # ------------------------------------------------------
    # Top Correlations
    # ------------------------------------------------------

    top20 = correlated_df.head(20)

    top20.to_csv(

        os.path.join(

            TABLE_PATH,

            "top20_correlations.csv"

        ),

        index=False

    )

    # ------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------

    plt.figure(figsize=(18,15))

    plt.imshow(

        correlation_matrix,

        cmap="coolwarm",

        interpolation="nearest",

        aspect="auto",

        vmin=-1,

        vmax=1

    )

    plt.colorbar()

    plt.xticks(

        range(len(correlation_matrix.columns)),

        correlation_matrix.columns,

        rotation=90,

        fontsize=7

    )

    plt.yticks(

        range(len(correlation_matrix.columns)),

        correlation_matrix.columns,

        fontsize=7

    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "correlation_heatmap.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Correlation Distribution
    # ------------------------------------------------------

    correlation_values = correlation_matrix.values.flatten()

    correlation_values = correlation_values[

        correlation_values != 1

    ]

    plt.figure(figsize=(8,5))

    plt.hist(

        correlation_values,

        bins=50

    )

    plt.title("Correlation Distribution")

    plt.xlabel("Correlation")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "correlation_distribution.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = {

        "Total Features":

            int(numerical_df.shape[1]),

        "Correlation Threshold":

            float(threshold),

        "Highly Correlated Pairs":

            int(len(correlated_df))

    }

    with open(

        os.path.join(

            TABLE_PATH,

            "correlation_summary.json"

        ),

        "w"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4

        )

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(f"\nNumerical Features : {numerical_df.shape[1]}")

    print(f"Threshold          : {threshold}")

    print(f"Highly Correlated  : {len(correlated_df)}")

    print("\nTop Correlations\n")

    print(top20)

    print("\nGenerated Files")

    print("-------------------------------------------")

    print("correlation_matrix.csv")

    print("highly_correlated_features.csv")

    print("top20_correlations.csv")

    print("correlation_summary.json")

    print("correlation_heatmap.png")

    print("correlation_distribution.png")

    print("-------------------------------------------")

    print("\nCorrelation Analysis Completed Successfully.")

    return correlated_df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from loader import load_dataset
    from config import PROCESSED_DATASET

    dataframe = load_dataset(PROCESSED_DATASET)

    correlation_analysis(dataframe)