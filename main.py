from src.preprocessing.pipeline import main as preprocessing
from src.eda.pipeline import main as eda_pipeline
from src.models.pipeline import main as model_pipeline

def main():

    preprocessing()
    eda_pipeline()
    model_pipeline()

if __name__ == "__main__":

    main()