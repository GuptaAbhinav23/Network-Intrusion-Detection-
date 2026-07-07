from lightgbm import LGBMClassifier

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)


def lightgbm(objective, n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, subsample, 
             colsample_bytree, reg_alpha, reg_lambda, random_state, n_jobs, verbosity):
    Logger.title("LIGHTGBM")

    X_train, X_test, y_train, y_test = load_train_test_data()

    model = LGBMClassifier(
        objective=objective,
        num_class=len(set(y_train)),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=verbosity
    )

    Logger.info("Training Model...")
    timer = Timer()
    timer.tic()
    model.fit(X_train, y_train)
    training_time, memory_usage = timer.toc()

    Logger.success("Model Training")

    evaluation = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        training_time=training_time,
        memory_usage=memory_usage,
        model_name="LightGBM"
    )

    model_size = save_model(
        model,
        "LightGBM"
    )

    evaluation["metrics"]["Model Size (MB)"] = model_size

    generate_report(
        evaluation,
        model_name="LightGBM"
    )

    Logger.success("LightGBM")

    return evaluation

def train(objective="multiclass", n_estimators=300, learning_rate=0.1, max_depth=-1, num_leaves=31, 
          min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0, 
          random_state=42, n_jobs=-1, verbosity=-1):
    lightgbm(objective, n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, subsample, 
             colsample_bytree, reg_alpha, reg_lambda, random_state, n_jobs, verbosity)

if __name__ == "__main__":
    train()

