# Your dataset has ~2.5 million training samples.

# A standard SVC from scikit-learn has quadratic to cubic training complexity and is not practical for 
# datasets of this size. Training could take days or exhaust memory.

# For this project, I recommend using LinearSVC (fast linear SVM) or SGDClassifier(loss="hinge") as the scalable 
# baseline. If your goal is to include the kernel SVM in the comparison, train it on a representative stratified 
# subset (for example, 50k–100k samples) and clearly document that choice in your report. This is a common and 
# acceptable practice in research on very large datasets.


from sklearn.svm import LinearSVC

from ..core import (
    load_train_test_data,
    evaluate_model,
    save_model,
    generate_report,
    Logger,
    Timer
)

def svm(C,loss,penalty,dual,max_iter,random_state):
    Logger.title("LINEAR SVM")
    # Suitable for large dataset i.e 2.5 Million

    X_train, X_test, y_train, y_test = load_train_test_data()

    model = LinearSVC(
        C=C,
        loss=loss,
        penalty=penalty,
        dual=dual,
        max_iter=max_iter,
        random_state=random_state
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
        model_name="Linear SVM"
    )

    model_size = save_model(
        model,
        "Linear SVM"
    )

    evaluation["metric"]["Model Size (MB)"] = model_size

    generate_report(
        evaluation,
        model_name="Linear SVM"
    )

    return evaluation


def train(C=1.0, loss="squared_hinge", penalty="l2", dual=False, max_iter=5000, random_state=42):
    svm(C,loss,penalty,dual,max_iter,random_state)

if __name__ == "__main__":
    train()

