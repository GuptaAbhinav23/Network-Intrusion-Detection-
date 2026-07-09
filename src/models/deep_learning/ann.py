"""
Advanced Artificial Neural Network (ANN) classifier.

Follows the exact same conventions used across `models.machine_learning.*`:
it loads data via the shared `core.data_loader`, trains / evaluates / saves /
reports through the shared `core` utilities, and exposes a module-level
`train()` function so it can be dropped into `models.pipeline` exactly like
every other model in this project.

Architecture
------------
A fully-connected feed-forward network with 3 hidden blocks, each:

    Dense (L2-regularized) -> BatchNormalization -> Activation -> Dropout

followed by a softmax output layer sized to the number of classes.

Labels are assumed to already be integer-encoded (as produced by the
existing preprocessing pipeline / `core.data_loader`), so the model is
trained with sparse categorical cross-entropy and no one-hot encoding step
is required or duplicated here.

Given the strong class imbalance typically present in this project's
dataset (see `machine_learning/reports/*/classification_report.csv`),
balanced class weights are computed and applied by default.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from ..core import load_train_test_data, Logger, DeepLearningException, load_keras_model
from .base import BaseNeuralNetwork, KerasClassifierWrapper


class ANNModel(BaseNeuralNetwork):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        **hyperparameters: Any,
    ) -> None:

        super().__init__(
            model_name="ANN",
            input_dim=input_dim,
            num_classes=num_classes,
            random_state=random_state,
            **hyperparameters,
        )

        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate

    def build_model(self) -> Any:

        try:
            from tensorflow.keras import Input, Model
            from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
            from tensorflow.keras.regularizers import l2
            from tensorflow.keras.optimizers import Adam

        except ImportError as exc:
            raise DeepLearningException("TensorFlow Is Required To Build The ANN Model.", exc)

        inputs = Input(shape=(self.input_dim,), name="input_features")

        x = inputs

        for index, units in enumerate(self.hidden_layers, start=1):

            x = Dense(
                units,
                kernel_regularizer=l2(self.l2_reg),
                name=f"dense_{index}",
            )(x)

            x = BatchNormalization(name=f"batch_norm_{index}")(x)

            x = Activation(self.activation, name=f"activation_{index}")(x)

            x = Dropout(self.dropout_rate, name=f"dropout_{index}")(x)

        outputs = Dense(self.num_classes, activation="softmax", name="output")(x)

        model = Model(inputs=inputs, outputs=outputs, name="ANN")

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.summary(print_fn=Logger.info)

        return model


def ann(
    hidden_layers: Optional[List[int]],
    activation: str,
    dropout_rate: float,
    l2_reg: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
    use_class_weights: bool,
    random_state: int,
) -> Dict[str, Any]:

    try:

        X_train, X_test, y_train, y_test = load_train_test_data()

        input_dim = X_train.shape[1]
        num_classes = int(len(np.unique(y_train)))

        model = ANNModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            random_state=random_state,
        )

        class_weight = None

        if use_class_weights:

            classes = np.unique(y_train)

            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train,
            )

            class_weight = dict(zip(classes.tolist(), weights.tolist()))

            Logger.info(f"Class Weights Enabled : {len(class_weight)} Classes")

        evaluation = model.run(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight,
        )

        return evaluation

    except DeepLearningException:
        raise

    except Exception as exc:
        raise DeepLearningException("ANN Pipeline Failed.", exc)


def train(
    hidden_layers: Optional[List[int]] = None,
    activation: str = "relu",
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 256,
    validation_split: float = 0.1,
    use_class_weights: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:

    return ann(
        hidden_layers,
        activation,
        dropout_rate,
        l2_reg,
        learning_rate,
        epochs,
        batch_size,
        validation_split,
        use_class_weights,
        random_state,
    )


def predict(X: np.ndarray) -> np.ndarray:

    try:
        keras_model = load_keras_model("ANN")
        return KerasClassifierWrapper(keras_model).predict(X)

    except Exception as exc:
        raise DeepLearningException("ANN Prediction Failed.", exc)


def predict_proba(X: np.ndarray) -> np.ndarray:

    try:
        keras_model = load_keras_model("ANN")
        return KerasClassifierWrapper(keras_model).predict_proba(X)

    except Exception as exc:
        raise DeepLearningException("ANN Prediction Failed.", exc)


if __name__ == "__main__":
    train()