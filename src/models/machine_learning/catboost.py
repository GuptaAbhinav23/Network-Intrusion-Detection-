from catboost import CatBoostClassifier

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)

def catboost(loss_function, iterations, learning_rate, depth, l2_leaf_reg, random_seed, thread_count, verbose):
    Logger.title("CATBOOST")

    X_train, X_test, y_train, y_test = load_train_test_data()

    model = CatBoostClassifier(
        loss_function=loss_function,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        thread_count=thread_count,
        verbose=verbose
    )

    Logger.info("Training Model...")
    timer=Timer()
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
        model_name="CatBoost"
    )

    model_size = save_model(
        model,
        "CatBoost"
    )

    evaluation["metrics"]["Model Size (MB)"] = model_size

    generate_report(
        evaluation,
        model_name="CatBoost"
    )

    Logger.success("CatBoost")

    return evaluation


def train(loss_function="MultiClass", iterations=300, learning_rate=0.1, depth=8, l2_leaf_reg=3, 
          random_seed=42, thread_count=-1, verbose=False):
    catboost(loss_function, iterations, learning_rate, depth, l2_leaf_reg, random_seed, thread_count, verbose)

if __name__ == "__main__":
    train()