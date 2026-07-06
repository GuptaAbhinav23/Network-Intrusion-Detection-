"""
config.py

Configuration file for EDA module.
"""

import os

# ==========================================================
# PROJECT ROOT
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# ==========================================================
# DATASET PATHS
# ==========================================================

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

PROCESSED_DATASET = os.path.join(
    DATASET_DIR,
    "processed",
    "processed_dataset.csv"
)

# ==========================================================
# REPORT PATHS
# ==========================================================

REPORT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "eda"
)

TABLE_PATH = os.path.join(
    REPORT_PATH,
    "tables"
)

FIGURE_PATH = os.path.join(
    REPORT_PATH,
    "figures"
)

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

os.makedirs(REPORT_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)
