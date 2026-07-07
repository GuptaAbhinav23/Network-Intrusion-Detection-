from sklearn.neighbors import KNeighborsClassifier

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer,
)


def knn(n_neighbors, weights, algorithm, leaf_size, metric, p, n_jobs):

    Logger.title("K-NEAREST NEIGHBORS")

    X_train, X_test, y_train, y_test = load_train_test_data()

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        n_jobs=n_jobs
    )

    Logger.info("Training Model...")
    timer = Timer()
    timer.tic()
    model.fit(X_train,y_train)
    training_time = timer.toc()

    Logger.success("Model Training")

    evaluation = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        training_time=training_time,
        model_name="KNN"
    )

    model_size = save_model(   
        model,
        "KNN"
    )

    evaluation["metrics"]["Model Size (MB)"] = model_size

    generate_report(
        evaluation,
        model_name="KNN"
    )

    Logger.success("KNN Completed")

    return evaluation

def train(n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        n_jobs=-1):
    knn(n_neighbors, weights, algorithm, leaf_size, metric, p, n_jobs)


if __name__ == "__main__":

    train()