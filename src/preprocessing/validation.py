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
# VALIDATE SCHEMA CONSISTENCY
# =====================================================

def validate_schema(csv_files):
    """
    Validate that all CSV files have the same schema.
    """

    print("\n" + "=" * 70)
    print("VALIDATING SCHEMA CONSISTENCY")
    print("=" * 70)

    if len(csv_files) == 0:
        raise Exception("No CSV files found.")

    # Read first CSV as reference
    reference_df = pd.read_csv(csv_files[0], nrows=5)

    reference_df.columns = reference_df.columns.str.strip()

    reference_columns = list(reference_df.columns)

    print(f"\nReference File : {os.path.basename(csv_files[0])}")
    print(f"Number of Features : {len(reference_columns)}")

    validation_passed = True

    for file in csv_files[1:]:

        current_df = pd.read_csv(file, nrows=5)

        current_df.columns = current_df.columns.str.strip()

        current_columns = list(current_df.columns)

        # Compare number of columns
        if len(current_columns) != len(reference_columns):

            print(f"\n❌ Column Count Mismatch")
            print(os.path.basename(file))
            print("Expected :", len(reference_columns))
            print("Found    :", len(current_columns))

            validation_passed = False

        # Compare column names
        if reference_columns != current_columns:

            print(f"\n❌ Column Name Mismatch")
            print(os.path.basename(file))

            missing = set(reference_columns) - set(current_columns)
            extra = set(current_columns) - set(reference_columns)

            if len(missing) > 0:
                print("Missing Columns")
                print(sorted(missing))

            if len(extra) > 0:
                print("Extra Columns")
                print(sorted(extra))

            validation_passed = False

    if validation_passed:

        print("\n✅ Schema Validation Passed")
        print("All CSV files have identical schema.")

    else:

        raise Exception(
            "\nSchema Validation Failed.\n"
            "Please fix the inconsistent CSV files before merging."
        )
