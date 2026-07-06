import os
import glob
import joblib
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from .config import (
    RAW_DATASET_PATH,
    PROCESSED_PATH,
    SCALER_PATH,
    ENCODER_PATH
)

# =====================================================
# SCALE FEATURES
# =====================================================

def scale_features(
        X_train,
        X_test,
        scaler_name="RobustScaler"):
    """
    Scale train and test features.

    Parameters
    ----------
    X_train : DataFrame

    X_test : DataFrame

    scaler_name : str

    Returns
    -------
    X_train_scaled
    X_test_scaled
    """

    print("\n" + "=" * 70)
    print("SCALING FEATURES")
    print("=" * 70)

    print(f"\nScaler Used : {scaler_name}")

    # ---------------------------------------------
    # Select Scaler
    # ---------------------------------------------

    if scaler_name == "StandardScaler":

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

    elif scaler_name == "MinMaxScaler":

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()

    elif scaler_name == "RobustScaler":

        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()

    else:

        raise ValueError("Invalid Scaler")

    # ---------------------------------------------
    # Fit ONLY on training data
    # ---------------------------------------------

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------
    # Convert back to DataFrame
    # ---------------------------------------------

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    # ---------------------------------------------
    # Save Scaler
    # ---------------------------------------------

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print("\nScaler Saved Successfully")

    print(f"\nTraining Shape : {X_train_scaled.shape}")

    print(f"Testing Shape  : {X_test_scaled.shape}")

    print("\nFeature Scaling Completed Successfully.")

    return X_train_scaled, X_test_scaled
