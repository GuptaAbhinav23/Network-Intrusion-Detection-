from src.preprocessing.pipeline import main as preprocessing
from src.eda.pipeline import main as eda_pipeline
def main():

    preprocessing()
    eda_pipeline()

if __name__ == "__main__":

    main()