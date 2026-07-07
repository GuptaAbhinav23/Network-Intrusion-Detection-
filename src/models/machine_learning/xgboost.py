from xgboost import XGBClassifier

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)


def xgboost(objective, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight,
            gamma, reg_alpha, reg_lambda, tree_method, eval_metric, random_state, n_jobs):
    Logger.title("XGBOOST")

    X_train, X_test, y_train, y_test = load_train_test_data()

    model = XGBClassifier(
        objective=objective,
        num_class=len(set(y_train)),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method=tree_method,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=n_jobs
    )

    Logger.info("Training Model...")
    timer = Timer()
    timer.tic()
    model.fit(X_train, y_train)
    training_time, memory_usage = timer.toc()

    Logger.success("Model Training")

    evaluation = evaluate_model(
        model = model,
        X_test= X_test,
        y_test=y_test,
        training_time=training_time,
        memory_usage=memory_usage,
        model_name="XGBoost"
    )

    model_size = save_model(
        evaluation,
        "XGBoost"
    )

    evaluation["metrics"]["Model Size (MB)"] = model_size

    generate_report(
        evaluation,
        model_name="XGBoost"
    )

    Logger.success("XGBoost")

    return evaluation

def train(objective="multi:softprob", n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, 
          colsample_bytree=0.8, min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1, tree_method="hist", 
          eval_metric="mlogloss", random_state=42, n_jobs=-1):
    xgboost(objective, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight,
            gamma, reg_alpha, reg_lambda, tree_method, eval_metric, random_state, n_jobs)


if __name__ == "__main__":
    train()