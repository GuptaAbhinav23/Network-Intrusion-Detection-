from .data_loader import load_train_test_data

from .metrics import evaluate_model

from .model_saver import save_model

# from .plots import *

from .logger import Logger

from .timer import Timer


# after utils every model is 
# Logger.title("Random Forest")

# X_train,X_test,y_train,y_test = load_train_test_data()

# timer = Timer()

# timer.tic()

# model = RandomForestClassifier(...)

# model.fit(X_train,y_train)

# training_time = timer.toc()

# metrics = evaluate_model(...)

# save_model(model,"RandomForest")