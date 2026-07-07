from catboost import CatBoostClassifier

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)

def catboost(loss_function, iterations, learning_rate, depth, l2_leaf_reg, random_seed, thread_count, verbose, bootstrap_type, subsample, allow_writing_files):
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
        verbose=verbose,
        bootstrap_type=bootstrap_type,
        subsample=subsample,
        allow_writing_files=allow_writing_files
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


def train(loss_function="MultiClass", iterations=150, learning_rate=0.1, depth=5, l2_leaf_reg=3, 
          random_seed=42, thread_count=-1, verbose=False, bootstrap_type="Bernoulli", subsample=0.2, allow_writing_files=False):
    catboost(loss_function, iterations, learning_rate, depth, l2_leaf_reg, random_seed, thread_count, verbose, bootstrap_type, subsample, allow_writing_files)

if __name__ == "__main__":
    train()