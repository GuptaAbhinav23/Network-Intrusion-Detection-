"""
label_analysis.py

Analyze attack labels for Network Intrusion Detection Dataset.

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import json
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
# LABEL ANALYSIS
# ==========================================================

def label_analysis(df: pd.DataFrame):
    """
    Perform attack label analysis.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    print("\n" + "=" * 70)
    print("LABEL ANALYSIS")
    print("=" * 70)

    create_directories()

    # ------------------------------------------------------
    # Check Label Column
    # ------------------------------------------------------

    if "Label" not in df.columns:
        raise Exception("Label column not found.")

    # ------------------------------------------------------
    # Attack Distribution
    # ------------------------------------------------------

    label_count = (
        df["Label"]
        .value_counts()
        .sort_index()
    )

    label_percentage = (
        label_count / len(df)
    ) * 100

    report = pd.DataFrame({

        "Label": label_count.index,
        "Count": label_count.values,
        "Percentage": label_percentage.values

    })

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    csv_path = os.path.join(
        TABLE_PATH,
        "label_distribution.csv"
    )

    report.to_csv(
        csv_path,
        index=False
    )

    # ------------------------------------------------------
    # JSON Summary
    # ------------------------------------------------------

    summary = {

        "Total Samples":
            int(len(df)),

        "Number of Classes":
            int(report.shape[0]),

        "Largest Class":
            str(
                report.loc[
                    report["Count"].idxmax(),
                    "Label"
                ]
            ),

        "Largest Class Samples":
            int(report["Count"].max()),

        "Smallest Class":
            str(
                report.loc[
                    report["Count"].idxmin(),
                    "Label"
                ]
            ),

        "Smallest Class Samples":
            int(report["Count"].min())

    }

    json_path = os.path.join(
        TABLE_PATH,
        "label_summary.json"
    )

    with open(json_path, "w") as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ------------------------------------------------------
    # Bar Chart
    # ------------------------------------------------------

    plt.figure(figsize=(12,6))

    plt.bar(
        report["Label"].astype(str),
        report["Count"]
    )

    plt.xticks(rotation=45, ha="right")

    plt.xlabel("Attack Class")

    plt.ylabel("Number of Samples")

    plt.title("Attack Class Distribution")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            FIGURE_PATH,
            "attack_distribution_bar.png"
        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Pie Chart
    # ------------------------------------------------------

    plt.figure(figsize=(9,9))

    plt.pie(

        report["Count"],

        labels=report["Label"].astype(str),

        autopct="%1.1f%%",

        startangle=90

    )

    plt.title("Attack Class Distribution")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            FIGURE_PATH,
            "attack_distribution_pie.png"
        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Benign vs Attack
    # ------------------------------------------------------

    benign_count = 0

    if "BENIGN" in df["Label"].astype(str).values:

        benign_count = (
            df["Label"]
            .astype(str)
            .eq("BENIGN")
            .sum()
        )

    attack_count = len(df) - benign_count

    binary_report = pd.DataFrame({

        "Class":[
            "BENIGN",
            "ATTACK"
        ],

        "Count":[
            benign_count,
            attack_count
        ]

    })

    binary_report.to_csv(

        os.path.join(
            TABLE_PATH,
            "binary_distribution.csv"
        ),

        index=False

    )

    plt.figure(figsize=(6,5))

    plt.bar(

        binary_report["Class"],

        binary_report["Count"]

    )

    plt.title("Benign vs Attack")

    plt.ylabel("Samples")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            FIGURE_PATH,
            "benign_vs_attack.png"
        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(f"\nTotal Samples : {len(df):,}")

    print(f"Total Classes : {summary['Number of Classes']}")

    print(f"Largest Class : {summary['Largest Class']}")

    print(f"Smallest Class : {summary['Smallest Class']}")

    print("\nAttack Distribution\n")

    print(report)

    print("\nFiles Generated")

    print("------------------------------------------")
    print("label_distribution.csv")
    print("label_summary.json")
    print("binary_distribution.csv")
    print("attack_distribution_bar.png")
    print("attack_distribution_pie.png")
    print("benign_vs_attack.png")
    print("------------------------------------------")

    print("\nLabel Analysis Completed Successfully.")

    return report