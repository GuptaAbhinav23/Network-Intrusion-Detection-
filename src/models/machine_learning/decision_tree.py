from sklearn.tree import DecisionTreeClassifier

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)

def decision_tree(criterion, max_depth, min_samples_split, min_samples_leaf, random_state):
    Logger.title("DECISION TREE")

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = load_train_test_data()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = DecisionTreeClassifier(

        criterion=criterion,

        max_depth=max_depth,

        min_samples_split=min_samples_split,

        min_samples_leaf=min_samples_leaf,

        random_state=random_state

    )

    # --------------------------------------------------
    # Train
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

        memory_usage=memory_usage,

        model_name="DecisionTree"

    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    model_size = save_model(

        model,

        "DecisionTree"

    )

    evaluation["metrics"]["Model Size (MB)"] = model_size

    generate_report(

        evaluation,

        model_name="DecisionTree"
    )

    Logger.success("Decision Tree")

    return evaluation


def train(criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ):
    decision_tree(criterion,max_depth,min_samples_split,min_samples_leaf,random_state)


if __name__ == "__main__":
    train()