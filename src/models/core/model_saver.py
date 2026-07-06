import joblib

from pathlib import Path

from .config import MODEL_DIR


def save_model(model,name):

    path = MODEL_DIR / f"{name}.pkl"

    joblib.dump(model,path)

    print(f"Model Saved : {path}")

if __name__ == "__main__":
    model = ...
    save_model(model,"RandomForest")