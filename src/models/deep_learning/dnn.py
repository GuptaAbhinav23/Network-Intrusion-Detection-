"""
Advanced Deep Neural Network (DNN) classifier.

Follows the exact same conventions used across `models.machine_learning.*`
and mirrors `models.deep_learning.ann`: it loads data via the shared
`core.data_loader`, trains / evaluates / saves / reports through the shared
`core` utilities, and exposes a module-level `train()` function so it can be
dropped into `models.pipeline` exactly like every other model in this
project.

Architecture
------------
Where the ANN is a plain 3-block feed-forward network, the DNN is a
deeper, residual feed-forward network intended for larger hidden-unit
budgets without suffering from vanishing gradients / optimization
degradation as depth increases:

    GaussianNoise (optional, input regularization)
    -> N x [ Dense (L1/L2-regularized) -> BatchNormalization -> Activation
             -> Dropout -> Residual/Skip Add ]
    -> softmax output layer sized to the number of classes

Each block adds a skip connection from its input to its output (a
1x1 `Dense` projection is inserted automatically when the block changes
width), which is what distinguishes this from the plain ANN and is what
makes the deeper default stack ([512, 256, 256, 128, 128, 64]) trainable
in practice.

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


class DNNModel(BaseNeuralNetwork):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.35,
        l1_reg: float = 1e-6,
        l2_reg: float = 1e-4,
        learning_rate: float = 1e-3,
        use_residual: bool = True,
        noise_stddev: float = 0.05,
        random_state: int = 42,
        **hyperparameters: Any,
    ) -> None:

        super().__init__(
            model_name="DNN",
            input_dim=input_dim,
            num_classes=num_classes,
            random_state=random_state,
            **hyperparameters,
        )

        self.hidden_layers = hidden_layers or [512, 256, 256, 128, 128, 64]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.use_residual = use_residual
        self.noise_stddev = noise_stddev

    def build_model(self) -> Any:

        try:
            from tensorflow.keras import Input, Model
            from tensorflow.keras.layers import (
                Dense,
                BatchNormalization,
                Activation,
                Dropout,
                GaussianNoise,
                Add,
            )
            from tensorflow.keras.regularizers import l1_l2
            from tensorflow.keras.optimizers import Adam

        except ImportError as exc:
            raise DeepLearningException("TensorFlow Is Required To Build The DNN Model.", exc)

        inputs = Input(shape=(self.input_dim,), name="input_features")

        x = inputs

        if self.noise_stddev > 0:
            x = GaussianNoise(self.noise_stddev, name="input_noise")(x)

        for index, units in enumerate(self.hidden_layers, start=1):

            shortcut = x

            x = Dense(
                units,
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f"dense_{index}",
            )(x)

            x = BatchNormalization(name=f"batch_norm_{index}")(x)

            x = Activation(self.activation, name=f"activation_{index}")(x)

            x = Dropout(self.dropout_rate, name=f"dropout_{index}")(x)

            if self.use_residual:

                if shortcut.shape[-1] != units:

                    shortcut = Dense(
                        units,
                        kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                        name=f"shortcut_projection_{index}",
                    )(shortcut)

                x = Add(name=f"residual_add_{index}")([x, shortcut])

        outputs = Dense(self.num_classes, activation="softmax", name="output")(x)

        model = Model(inputs=inputs, outputs=outputs, name="DNN")

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.summary(print_fn=Logger.info)

        return model


def dnn(
    hidden_layers: Optional[List[int]],
    activation: str,
    dropout_rate: float,
    l1_reg: float,
    l2_reg: float,
    learning_rate: float,
    use_residual: bool,
    noise_stddev: float,
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

        model = DNNModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            use_residual=use_residual,
            noise_stddev=noise_stddev,
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
        raise DeepLearningException("DNN Pipeline Failed.", exc)


def train(
    hidden_layers: Optional[List[int]] = None,
    activation: str = "relu",
    dropout_rate: float = 0.35,
    l1_reg: float = 1e-6,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
    use_residual: bool = True,
    noise_stddev: float = 0.05,
    epochs: int = 150,
    batch_size: int = 256,
    validation_split: float = 0.1,
    use_class_weights: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:

    return dnn(
        hidden_layers,
        activation,
        dropout_rate,
        l1_reg,
        l2_reg,
        learning_rate,
        use_residual,
        noise_stddev,
        epochs,
        batch_size,
        validation_split,
        use_class_weights,
        random_state,
    )


def predict(X: np.ndarray) -> np.ndarray:

    try:
        keras_model = load_keras_model("DNN")
        return KerasClassifierWrapper(keras_model).predict(X)

    except Exception as exc:
        raise DeepLearningException("DNN Prediction Failed.", exc)


def predict_proba(X: np.ndarray) -> np.ndarray:

    try:
        keras_model = load_keras_model("DNN")
        return KerasClassifierWrapper(keras_model).predict_proba(X)

    except Exception as exc:
        raise DeepLearningException("DNN Prediction Failed.", exc)


if __name__ == "__main__":
    train()
    