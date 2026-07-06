import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# RAW_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "raw")

# BASE_DIR = Path(__file__).resolve().parent[3]

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "processed")

MODEL_DIR = os.path.join(BASE_DIR, "models")

REPORT_DIR = os.path.join(BASE_DIR, "reports", "models")

PLOT_DIR = os.path.join(REPORT_DIR, "plots")

TABLE_DIR = os.path.join(REPORT_DIR, "tables")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
