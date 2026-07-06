"""
report_generator.py

Generate Final EDA Report

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import json
import pandas as pd

from .config import TABLE_PATH, REPORT_PATH


# ==========================================================
# CREATE DIRECTORY
# ==========================================================

def create_directory():

    os.makedirs(REPORT_PATH, exist_ok=True)


# ==========================================================
# SAFE LOADER
# ==========================================================

def load_json(filename):

    path = os.path.join(TABLE_PATH, filename)

    if os.path.exists(path):

        with open(path, "r") as f:

            return json.load(f)

    return None


def load_csv(filename):

    path = os.path.join(TABLE_PATH, filename)

    if os.path.exists(path):

        return pd.read_csv(path)

    return None


# ==========================================================
# GENERATE REPORT
# ==========================================================

def generate_report():

    print("\n" + "=" * 70)
    print("GENERATING FINAL EDA REPORT")
    print("=" * 70)

    create_directory()

    dataset = load_json("dataset_summary.json")
    missing = load_json("missing_value_summary.json")
    labels = load_json("label_summary.json")
    correlation = load_json("correlation_summary.json")
    feature = load_json("feature_importance_summary.json")
    pca = load_json("pca_summary.json")
    tsne = load_json("tsne_summary.json")
    umap = load_json("umap_summary.json")
    outlier = load_json("outlier_summary.json")

    report = {

        "Dataset": dataset,

        "Missing Value Analysis": missing,

        "Label Analysis": labels,

        "Correlation Analysis": correlation,

        "Outlier Analysis": outlier,

        "Feature Importance": feature,

        "PCA": pca,

        "t-SNE": tsne,

        "UMAP": umap

    }

    # ------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------

    json_path = os.path.join(

        REPORT_PATH,

        "EDA_Report.json"

    )

    with open(json_path, "w") as f:

        json.dump(report, f, indent=4)

    # ------------------------------------------------------
    # Markdown Report
    # ------------------------------------------------------

    md_path = os.path.join(

        REPORT_PATH,

        "EDA_Report.md"

    )

    with open(md_path, "w", encoding="utf-8") as f:

        f.write("# Exploratory Data Analysis Report\n\n")

        f.write("## Dataset Summary\n\n")

        if dataset:

            for k, v in dataset.items():
                f.write(f"- **{k}** : {v}\n")

        f.write("\n---\n")

        f.write("## Missing Value Analysis\n\n")

        if missing:

            for k, v in missing.items():
                f.write(f"- **{k}** : {v}\n")

        f.write("\n---\n")

        f.write("## Label Analysis\n\n")

        if labels:

            for k, v in labels.items():
                f.write(f"- **{k}** : {v}\n")

        f.write("\n---\n")

        f.write("## Correlation Analysis\n\n")

        if correlation:

            for k, v in correlation.items():
                f.write(f"- **{k}** : {v}\n")

        f.write("\n---\n")

        f.write("## Outlier Analysis\n\n")

        if outlier:

            for k, v in outlier.items():
                f.write(f"- **{k}** : {v}\n")

        f.write("\n---\n")

        f.write("## Feature Importance\n\n")

        if feature:

            for k, v in feature.items():
                f.write(f"- **{k}** : {v}\n")