import os

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATASET_PATH = r"S:\Network Intrusion Detection\dataset\raw"

PROCESSED_PATH = r"S:\Network Intrusion Detection\dataset\processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

MERGED_DATASET = os.path.join(PROCESSED_PATH, "merged_dataset.csv")

PROCESSED_DATASET = os.path.join(PROCESSED_PATH, "processed_dataset.csv")

SCALER_PATH = os.path.join(PROCESSED_PATH, "scaler.pkl")

ENCODER_PATH = os.path.join(PROCESSED_PATH, "label_encoder.pkl")

os.makedirs(PROCESSED_PATH, exist_ok=True)
