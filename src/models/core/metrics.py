#hello
import time
import numpy as np


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(
        model,
        X_test,
        y_test,
        training_time,
        memory_usage,
        model_name="Model"
):
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model : sklearn estimator

    X_test : ndarray

    y_test : ndarray

    training_time : float

    model_name : str

    Returns
    -------
    dict
    """

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction_start = time.time()

    y_pred = model.predict(X_test)

    prediction_time = time.time() - prediction_start

    # ------------------------------------------------------
    # Throughput
    # ------------------------------------------------------

    throughput = len(X_test) / prediction_time if prediction_time > 0 else 0

    # ------------------------------------------------------
    # Latency
    # ------------------------------------------------------

    latency = (prediction_time * 1000) / len(X_test) if len(X_test) > 0 else 0

    # ------------------------------------------------------
    # Basic Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    kappa = cohen_kappa_score(
        y_test,
        y_pred
    )

    # ------------------------------------------------------
    # ROC AUC
    # ------------------------------------------------------

    roc_auc = None

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(X_test)

            roc_auc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="weighted"
            )

        except Exception:

            roc_auc = None
    
    elif hasattr(model, "decision_function"):

        try:
            scores = model.decision_function(X_test)

            roc_auc = roc_auc_score(
                y_test,
                scores,
                multi_class="ovr",
                average="weighted"
            )
        except Exception:

            roc_auc=None


    # ------------------------------------------------------
    # Reports
    # ------------------------------------------------------

    class_report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_test,
        y_pred
    )

    # ------------------------------------------------------
    # Results Dictionary
    # ------------------------------------------------------

    results = {

        "Model": model_name,

        "Accuracy": accuracy,

        "Balanced Accuracy": balanced_accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc_auc,

        "Matthews Correlation Coefficient": mcc,

        "Cohen Kappa": kappa,

        "Training Time (s)": training_time,

        "Prediction Time (s)": prediction_time,

        "Throughput (samples/sec)": throughput,

        "Latency (ms/sample)": latency,

        "Memory Usage (MB)": memory_usage,

        "Classification Report": class_report,

        "Confusion Matrix": confusion

    }

    # ------------------------------------------------------
    # Print Summary
    # ------------------------------------------------------

    print(f"Accuracy               : {accuracy:.4f}")
    print(f"Balanced Accuracy      : {balanced_accuracy:.4f}")
    print(f"Precision              : {precision:.4f}")
    print(f"Recall                 : {recall:.4f}")
    print(f"F1 Score               : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC AUC                : {roc_auc:.4f}")

    print(f"MCC                    : {mcc:.4f}")
    print(f"Cohen Kappa            : {kappa:.4f}")
    print(f"Training Time (s)      : {training_time:.2f}")
    print(f"Prediction Time (s)    : {prediction_time:.2f}")
    print(f"Throughput             : {throughput:,.2f} samples/sec")
    print(f"Latency                : {latency:.8f} ms/sample")
    print(f"Memory Usage (MB)      : {memory_usage:.2f}")

    print("\nEvaluation Completed Successfully.")

    return {
        "metrics": results,
        "confusion_matrix": confusion,
        "classification_report": class_report,
    }