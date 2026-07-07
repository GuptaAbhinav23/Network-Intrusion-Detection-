from sklearn.ensemble import ExtraTreesClassifier

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)

def extra_trees(n_estimators,criterion,max_depth,min_samples_split,min_samples_leaf,max_features,bootstrap,n_jobs,random_state):
    Logger.title("EXTRA TREES")

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = load_train_test_data()

    # --------------------------------------------------
    # Initialize Model
    # --------------------------------------------------

    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    Logger.info("Training Model...")

    timer = Timer()

    timer.tic()

    model.fit(X_train, y_train)

    training_time, memory_usage = timer.toc()

    Logger.success("Model Training")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    evaluation = evaluate_model(

        model=model,

        X_test=X_test,

        y_test=y_test,

        training_time=training_time,

        memory_usage = memory_usage,

        model_name="ExtraTrees"

    )

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    model_size = save_model(

        model,

        "RandomForest"

    )

    evaluation["metrics"]["Model Size (MB)"] = model_size

    # --------------------------------------------------
    # Save Reports
    # --------------------------------------------------

    generate_report(

        evaluation,

        model_name="ExtraTrees"

    )

    Logger.success("Extra Trees")

    return evaluation

def train(n_estimators=300, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, 
          max_features="sqrt", bootstrap=False, n_jobs=-1, random_state=42
):
    extra_trees(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, 
                max_features, bootstrap, n_jobs, random_state)

if __name__ == "__main__":
    train()

