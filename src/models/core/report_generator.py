import os
from .config import REPORT_DIR
import pandas as pd


def generate_report(
        evaluation,
        model_name,
        report_directory = REPORT_DIR
):

    model_directory = os.path.join(report_directory, model_name)

    comparison_directory = os.path.join(report_directory, "comparison")

    os.makedirs(
        model_directory,
        exist_ok=True
    )

    os.makedirs(
        comparison_directory,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    metrics = evaluation["metrics"].copy()

    metrics.pop("Classification Report", None)
    metrics.pop("Confusion Matrix", None)

    metrics_df = pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Value"]
    )

    metrics_df.to_csv(
        os.path.join(model_directory,"metrics.csv"),
        index=False
    )

    # ------------------------------------------------------
    # Classification Report
    # ------------------------------------------------------

    classification_df = pd.DataFrame(
        evaluation["classification_report"]
    ).transpose()

    classification_df.to_csv(
        os.path.join(model_directory, "classification_report.csv")
    )

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    confusion_df = pd.DataFrame(

        evaluation["confusion_matrix"]

    )

    confusion_df.to_csv(
        os.path.join(model_directory, "confusion_matrix.csv"),
        index=False
    )

    # ------------------------------------------------------
    # Model Comparison
    # ------------------------------------------------------

    comparison_file = os.path.join(comparison_directory, "model_comparison.csv")

    comparison_row = {
        "Model":
        metrics["Model"],

        "Accuracy":
        metrics["Accuracy"],

        "Balanced Accuracy":
        metrics["Balanced Accuracy"],

        "Precision":
        metrics["Precision"],

        "Recall":
        metrics["Recall"],

        "F1 Score":
        metrics["F1 Score"],

        "ROC AUC":
        metrics["ROC AUC"],

        "MCC":
        metrics["Matthews Correlation Coefficient"],

        "Cohen Kappa":
        metrics["Cohen Kappa"],

        "Training Time":
        metrics["Training Time (s)"],

        "Prediction Time":
        metrics["Prediction Time (s)"]
    }

    comparison_df = pd.DataFrame(
        [comparison_row]
    )

    if os.path.exists(comparison_file):
        existing = pd.read_csv(
            comparison_file
        )

        existing = existing[
            existing["Model"] != model_name
        ]

        comparison_df = pd.concat(
            [existing, comparison_df],
            ignore_index=True
        )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    print("\nReports Saved Successfully.")