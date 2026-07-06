import joblib
import os
from .config import MODEL_DIR


def save_model(model,name):

    path = os.path.join(MODEL_DIR, f"{name}.pkl")

    joblib.dump(model,path)

    print(f"Model Saved : {path}")

if __name__ == "__main__":
    model = ...
    save_model(model,"RandomForest")