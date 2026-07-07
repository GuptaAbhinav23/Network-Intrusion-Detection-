import joblib
import os
from .config import MODEL_DIR


def save_model(model,name):

    path = os.path.join(MODEL_DIR, f"{name}.pkl")

    joblib.dump(model, path)

    size = os.path.getsize(path) / (1024 * 1024)

    print(f"Model Saved : {path}")

    print(f"Model Size  : {size:.2f} MB")

    return size

if __name__ == "__main__":
    model = ...
    save_model(model,"RandomForest")