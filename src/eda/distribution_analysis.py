import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import FIGURE_PATH, TABLE_PATH


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

def create_directories():

    os.makedirs(TABLE_PATH, exist_ok=True)

    os.makedirs(FIGURE_PATH, exist_ok=True)

    os.makedirs(
        os.path.join(FIGURE_PATH, "histograms"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(FIGURE_PATH, "density"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(FIGURE_PATH, "boxplots"),
        exist_ok=True
    )


# ==========================================================
# DISTRIBUTION ANALYSIS
# ==========================================================

def distribution_analysis(
        df: pd.DataFrame,
        sample_size=10000):
    """
    Generate distribution plots for numerical features.

    Parameters
    ----------
    df : DataFrame

    sample_size : int
        Number of rows used for plotting.
    """

    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    create_directories()

    # ------------------------------------------------------
    # Numerical Features
    # ------------------------------------------------------

    numerical_df = df.select_dtypes(include=np.number)

    if "Label" in numerical_df.columns:

        numerical_df = numerical_df.drop(columns=["Label"])

    # ------------------------------------------------------
    # Sampling
    # ------------------------------------------------------

    if len(numerical_df) > sample_size:

        sampled_df = numerical_df.sample(
            sample_size,
            random_state=42
        )

    else:

        sampled_df = numerical_df.copy()

    print(f"\nSample Used : {len(sampled_df):,}")

    summary = []

    # ------------------------------------------------------
    # Loop through Features
    # ------------------------------------------------------

    for feature in sampled_df.columns:

        data = (
            sampled_df[feature]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(data) < 3:
            continue

        if data.nunique() <= 1:
            print(f"Skipping constant feature: {feature}")
            continue

        if data.std() < 1e-10:
            print(f"Skipping near-constant feature: {feature}")
            continue

        # ==============================================
        # Histogram
        # ==============================================

        plt.figure(figsize=(8,5))

        plt.hist(
            data,
            bins=min(40, max(5, data.nunique()))
        )

        plt.title(f"Histogram - {feature}")

        plt.xlabel(feature)

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                FIGURE_PATH,
                "histograms",
                f"{feature}.png"

            ),

            dpi=300

        )

        plt.close()

        # ==============================================
        # Density Plot
        # ==============================================

        if (
            len(data) > 2
            and data.nunique() > 1
            and data.std() > 1e-10
        ):
            try:
                plt.figure(figsize=(8,5))
                data.plot(kind="density")
                plt.title(f"Density Plot - {feature}")
                plt.xlabel(feature)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        FIGURE_PATH,
                        "density",
                        f"{feature}.png"
                    ),
                    dpi=300
                )
                plt.close()
            except Exception as e:
                print(f"Skipping Density Plot for {feature}: {e}")
                plt.close()
        else:
            print(f"Skipping Density Plot for {feature} (constant feature)")

        # ==============================================
        # Box Plot
        # ==============================================

        plt.figure(figsize=(8,2))

        plt.boxplot(
            data,
            vert=False
        )

        plt.title(f"Box Plot - {feature}")

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

        # ==============================================
        # Statistics
        # ==============================================

        summary.append({

            "Feature": feature,

            "Mean": data.mean(),

            "Median": data.median(),

            "Std": data.std(),

            "Minimum": data.min(),

            "Maximum": data.max(),

            "Skewness": data.skew(),

            "Kurtosis": data.kurt()

        })

    # ------------------------------------------------------
    # Save Summary
    # ------------------------------------------------------

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(

        os.path.join(

            TABLE_PATH,
            "distribution_summary.csv"

        ),

        index=False

    )

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(f"\nFeatures Processed : {len(summary_df)}")

    print("\nGenerated Files")

    print("-------------------------------------------")

    print("distribution_summary.csv")

    print("Histograms")

    print("Density Plots")

    print("Box Plots")

    print("-------------------------------------------")

    print("\nDistribution Analysis Completed Successfully.")

    return summary_df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from loader import load_dataset
    from config import PROCESSED_DATASET

    df = load_dataset(PROCESSED_DATASET)

    distribution_analysis(df)