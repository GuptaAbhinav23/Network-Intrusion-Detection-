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
    MERGED_DATASET,
    SCALER_PATH,
    ENCODER_PATH
)

# =====================================================
# ENCODE LABELS
# =====================================================

def encode_labels(df):
    """
    Encode attack labels into integer values.

    Saves:
        - label_encoder.pkl
        - label_mapping.json
    """

    print("\n" + "=" * 70)
    print("ENCODING LABELS")
    print("=" * 70)

    # ---------------------------------------------
    # Check Label Column
    # ---------------------------------------------

    if "Label" not in df.columns:
        raise Exception("Label column not found.")

    # Remove leading/trailing spaces
    # Remove invalid unicode replacement characters
    df["Label"] = (
        df["Label"]
        .str.replace("\ufffd", "", regex=False)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
    )

    # ---------------------------------------------
    # Display Classes
    # ---------------------------------------------

    unique_labels = sorted(df["Label"].unique())

    print("\nAttack Classes Found\n")

    for label in unique_labels:
        print(repr(label))

    print("\nTotal Classes :", len(unique_labels))

    # ---------------------------------------------
    # Encode Labels
    # ---------------------------------------------

    encoder = LabelEncoder()

    df["Label"] = encoder.fit_transform(df["Label"])

    # ---------------------------------------------
    # Mapping Dictionary
    # ---------------------------------------------

    mapping = {}

    for original, encoded in zip(
            encoder.classes_,
            encoder.transform(encoder.classes_)):

        mapping[str(original)] = int(encoded)

    print("\nEncoded Labels\n")

    for key, value in mapping.items():

        print(f"{key:25s} --> {value}")

    # ---------------------------------------------
    # Save Encoder
    # ---------------------------------------------

    joblib.dump(
        encoder,
        ENCODER_PATH
    )

    # ---------------------------------------------
    # Save Mapping
    # ---------------------------------------------

    mapping_path = os.path.join(
        PROCESSED_PATH,
        "label_mapping.json"
    )

    with open(mapping_path, "w") as file:

        json.dump(
            mapping,
            file,
            indent=4
        )

    print("\nLabel Encoder Saved")

    print("Label Mapping Saved")

    print("\nEncoding Completed Successfully.")

    return df