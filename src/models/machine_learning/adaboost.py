from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)


def adaboost(max_depth,n_estimators,learning_rate,random_state):
    Logger.title("ADABOOST")

    X_train, X_test, y_train, y_test = load_train_test_data()

    base_estimator = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )

    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )

    Logger.info("Training Model...")
    timer=Timer()
    timer.tic()
    model.fit(X_train, y_train)
    training_time, memory_usage = timer.toc()

    Logger.success("Model Training")

    evaluation = evaluate_model(
        model = model,
        X_test= X_test,
        y_test= y_test,
        training_time= training_time,
        memory_usage=memory_usage,
        model_name= "AdaBoost"
    )

    model_size = save_model(
        model,
        "AdaBoost"
    )

    evaluation["metrics"]["Model Size (MB)"] = model_size

    generate_report(
        evaluation,
        model_name= "AdaBoost"
    )

    Logger.success("AdaBoost")

    return evaluation


def train(
        max_depth=1,
        n_estimators=200,
        learning_rate=1.0,
        random_state=42):
    adaboost(max_depth,n_estimators,learning_rate,random_state)


if __name__ == "__main__":
    train()