import numpy as np
#hello
import os
from .config import DATASET_DIR

def load_train_test_data():

    X_train = np.load(os.path.join(DATASET_DIR,"X_train.npy"))
    X_test = np.load(os.path.join(DATASET_DIR,"X_test.npy"))

    y_train = np.load(os.path.join(DATASET_DIR,"y_train.npy"))
    y_test = np.load(os.path.join(DATASET_DIR,"y_test.npy"))

    print("="*70)
    print("TRAIN / TEST DATA LOADED")
    print("=" * 70)

    print(f"Training Samples : {X_train.shape[0]:,}")
    print(f"Testing Samples  : {X_test.shape[0]:,}")
    print(f"Features         : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def load_feature_names():

    import pandas as pd

    df = pd.read_csv(DATASET_DIR, "processed_dataset.csv")

    feature_names = df.drop(columns=["Label"]).columns.tolist()

    return feature_names