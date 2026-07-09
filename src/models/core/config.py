import os
#hello

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "processed")

MODEL_DIR = os.path.join(BASE_DIR, "models")

MACHINE_LEARNING_DIR = os.path.join(MODEL_DIR, "machine_learning")

REPORT_DIR = os.path.join(MACHINE_LEARNING_DIR, "reports")

PLOT_DIR = os.path.join(REPORT_DIR, "plots")

TABLE_DIR = os.path.join(REPORT_DIR, "tables")

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(MACHINE_LEARNING_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
# os.makedirs(PLOT_DIR, exist_ok=True)
# os.makedirs(TABLE_DIR, exist_ok=True)
