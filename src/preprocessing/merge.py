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
# MERGE CSV FILES
# =====================================================

def merge_csv_files():

    csv_files = glob.glob(os.path.join(RAW_DATASET_PATH, "*.csv"))

    if len(csv_files) == 0:
        raise Exception("No CSV files found inside dataset/raw")

    print("=" * 60)
    print("Found", len(csv_files), "CSV files")
    print("=" * 60)

    dataframe_list = []

    for file in csv_files:

        print("Reading :", os.path.basename(file))

        try:
            df = pd.read_csv(
                file,
                encoding="utf-8",
                low_memory=False
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                file,
                encoding="latin1",
                low_memory=False
            )

        dataframe_list.append(df)

    merged_df = pd.concat(dataframe_list, ignore_index=True)

    print("\nMerged Shape :", merged_df.shape)

    merged_df.to_csv(MERGED_DATASET, index=False)

    print("[OK] Merged dataset saved.")

    return merged_df
