"""
pipeline.py

Complete Exploratory Data Analysis Pipeline

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import time

from .config import PROCESSED_DATASET

from .loader import load_dataset
from .dataset_info import dataset_information
from .missing_analysis import missing_value_analysis
from .label_analysis import label_analysis
from .statistical_summary import statistical_summary
from .distribution_analysis import distribution_analysis
from .correlation_analysis import correlation_analysis
from .outlier_analysis import outlier_analysis
from .feature_importance import feature_importance_analysis
from .pca_analysis import pca_analysis
from .tsne_analysis import tsne_analysis
from .umap_analysis import umap_analysis
from .report_generator import generate_report


# ==========================================================
# RUN SINGLE STEP
# ==========================================================

def run_step(step_name, function, *args, **kwargs):

    print("\n" + "=" * 80)
    print(f"STARTING : {step_name}")
    print("=" * 80)

    start = time.time()

    try:

        result = function(*args, **kwargs)

        elapsed = time.time() - start

        print(f"\n{step_name} Completed Successfully.")
        print(f"Execution Time : {elapsed:.2f} seconds")

        return result

    except Exception as e:

        print(f"\nERROR IN {step_name}")
        print(e)

        raise


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():

    total_start = time.time()

    print("\n" + "=" * 80)
    print("NETWORK INTRUSION DETECTION SYSTEM")
    print("EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 80)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    df = run_step(

        "Load Dataset",

        load_dataset,

        PROCESSED_DATASET

    )

    # ------------------------------------------------------
    # Dataset Information
    # ------------------------------------------------------

    run_step(

        "Dataset Information",

        dataset_information,

        df

    )

    # ------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------

    run_step(

        "Missing Value Analysis",

        missing_value_analysis,

        df

    )

    # ------------------------------------------------------
    # Label Analysis
    # ------------------------------------------------------

    run_step(

        "Label Analysis",

        label_analysis,

        df

    )

    # ------------------------------------------------------
    # Statistical Summary
    # ------------------------------------------------------

    run_step(

        "Statistical Summary",

        statistical_summary,

        df

    )

    # ------------------------------------------------------
    # Distribution Analysis
    # ------------------------------------------------------

    run_step(

        "Distribution Analysis",

        distribution_analysis,

        df

    )

    # ------------------------------------------------------
    # Correlation Analysis
    # ------------------------------------------------------

    run_step(

        "Correlation Analysis",

        correlation_analysis,

        df

    )

    # ------------------------------------------------------
    # Outlier Analysis
    # ------------------------------------------------------

    run_step(

        "Outlier Analysis",

        outlier_analysis,

        df

    )

    # ------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------

    run_step(

        "Feature Importance Analysis",

        feature_importance_analysis,

        df

    )

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    run_step(

        "PCA Analysis",

        pca_analysis,

        df

    )

    # ------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------

    run_step(

        "t-SNE Analysis",

        tsne_analysis,

        df

    )

    # ------------------------------------------------------
    # UMAP
    # ------------------------------------------------------

    run_step(

        "UMAP Analysis",

        umap_analysis,

        df

    )

    # ------------------------------------------------------
    # Final Report
    # ------------------------------------------------------

    run_step(

        "EDA Report Generation",

        generate_report

    )

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("EDA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(f"\nTotal Execution Time : {total_time:.2f} seconds")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()