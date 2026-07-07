from sklearn.linear_model import LogisticRegression

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)

def logistic_regression_model(MAX_ITER, RANDOM_STATE, N_JOBS):

    Logger.title("LOGISTIC REGRESSION")

    X_train, X_test, y_train, y_test = load_train_test_data()

    model = LogisticRegression(

        max_iter=MAX_ITER,

        random_state=RANDOM_STATE,

        n_jobs=N_JOBS
    )

    Logger.info("Training Model...")

    timer = Timer()

    timer.tic()

    model.fit(X_train, y_train)

    training_time, memory_usage = timer.toc()

    Logger.success("Model Training")

    Logger.info("Evaluating Model...")

    evaluation = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        training_time=training_time,
        memory_usage=memory_usage,
        model_name="LogisticRegression"
    )

    model_size = save_model(
        model,
        "LogisticRegression"
    )

    evaluation["metrics"]["Model Size (MB)"] = model_size

    generate_report(
        evaluation,
        model_name="LogisticRegression"
    )
    Logger.success("Logistic Regression")

    return evaluation

def train(MAX_ITER=1000, RANDOM_STATE=42, N_JOBS=-1):

    logistic_regression_model(MAX_ITER,RANDOM_STATE,N_JOBS)

if __name__ == "__main__":
    train()