"""
Advanced 1D Convolutional Neural Network (CNN) classifier.

Follows the exact same conventions used across `models.machine_learning.*`
and mirrors `models.deep_learning.dnn`: it loads data via the shared
`core.data_loader`, trains / evaluates / saves / reports through the shared
`core` utilities, and exposes a module-level `train()` function so it can be
dropped into `models.pipeline` exactly like every other model in this
project.

Why a *1D* CNN
--------------
The project's dataset is tabular (one feature vector per row), not
image/grid data, so a standard 2D CNN does not apply. Each row is instead
treated as a length-`input_dim` sequence and reshaped to
`(input_dim, 1)`, letting `Conv1D` filters slide across neighbouring
feature positions to learn local feature interactions that plain Dense
layers (as in the ANN/DNN) can miss.

Architecture
------------
    Reshape(input_dim, 1)
    -> GaussianNoise (optional, input regularization)
    -> N x [ Conv1D (L1/L2-regularized) -> BatchNormalization -> Activation
             -> SpatialDropout1D -> Residual/Skip Add -> MaxPooling1D ]
    -> GlobalAveragePooling1D
    -> Dense head (BatchNormalization -> Activation -> Dropout)
    -> softmax output layer sized to the number of classes

As in the DNN, a residual (skip) connection is added around every conv
block; a `1x1 Conv1D` projection is inserted automatically whenever a
block changes the channel count, so the shortcut always matches shape.

Labels are assumed to already be integer-encoded (as produced by the
existing preprocessing pipeline / `core.data_loader`), so the model is
trained with sparse categorical cross-entropy and no one-hot encoding step
is required or duplicated here.

Given the strong class imbalance typically present in this project's
dataset (see `machine_learning/reports/*/classification_report.csv`),
balanced class weights are computed and applied by default - identical to
the ANN and DNN.

A note on accuracy
-------------------
This module is built to the same production standard as the ANN and DNN,
and reuses the same residual/regularization techniques that helped the
DNN. No architecture can *guarantee* a specific accuracy figure (e.g.
0.9993) ahead of time - that depends on the dataset itself (feature
separability, label noise, and the class imbalance already visible in
this project's reports, where some classes have only a handful of
samples). Treat any target accuracy as an aspiration to evaluate against,
not a property the code can promise on its own.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from ..core import load_train_test_data, Logger, DeepLearningException, load_keras_model
from .base import BaseNeuralNetwork, KerasClassifierWrapper


class CNNModel(BaseNeuralNetwork):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_blocks: Optional[List[Tuple[int, int]]] = None,
        dense_units: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.35,
        spatial_dropout_rate: float = 0.2,
        l1_reg: float = 1e-6,
        l2_reg: float = 1e-4,
        learning_rate: float = 1e-3,
        use_residual: bool = True,
        noise_stddev: float = 0.05,
        pool_size: int = 2,
        random_state: int = 42,
        **hyperparameters: Any,
    ) -> None:

        super().__init__(
            model_name="CNN",
            input_dim=input_dim,
            num_classes=num_classes,
            random_state=random_state,
            **hyperparameters,
        )

        # (filters, kernel_size) per convolutional block
        self.conv_blocks = conv_blocks or [(48, 3), (96, 5), (128, 3)]
        self.dense_units = dense_units or [128, 64]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.use_residual = use_residual
        self.noise_stddev = noise_stddev
        self.pool_size = pool_size

    def build_model(self) -> Any:

        try:
            from tensorflow.keras import Input, Model
            from tensorflow.keras.layers import (
                Reshape,
                SeparableConv1D,
                Conv1D,
                BatchNormalization,
                Activation,
                SpatialDropout1D,
                Dropout,
                GaussianNoise,
                MaxPooling1D,
                GlobalAveragePooling1D,
                GlobalMaxPooling1D,
                Dense,
                Add,
                Concatenate,
            )
            from tensorflow.keras.regularizers import l1_l2
            from tensorflow.keras.optimizers import Adam

        except ImportError as exc:
            raise DeepLearningException("TensorFlow Is Required To Build The CNN Model.", exc)

        inputs = Input(shape=(self.input_dim,), name="input_features")

        x = Reshape((self.input_dim, 1), name="reshape_to_sequence")(inputs)

        if self.noise_stddev > 0:
            x = GaussianNoise(self.noise_stddev, name="input_noise")(x)

        for index, (filters, kernel_size) in enumerate(self.conv_blocks, start=1):

            shortcut = x

            x = SeparableConv1D(
                filters,
                kernel_size=kernel_size,
                padding="same",
                depthwise_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                pointwise_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f"separable_conv_{index}",
            )(x)

            x = BatchNormalization(name=f"batch_norm_{index}")(x)

            x = Activation(self.activation, name=f"activation_{index}")(x)

            x = SpatialDropout1D(self.spatial_dropout_rate, name=f"spatial_dropout_{index}")(x)

            if self.use_residual:

                if shortcut.shape[-1] != filters:

                    shortcut = Conv1D(
                        filters,
                        kernel_size=1,
                        padding="same",
                        kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                        name=f"shortcut_projection_{index}",
                    )(shortcut)

                x = Add(name=f"residual_add_{index}")([x, shortcut])

            # Only downsample while the sequence length can still support it
            if x.shape[1] is not None and x.shape[1] >= self.pool_size:
                x = MaxPooling1D(pool_size=self.pool_size, padding="same", name=f"max_pool_{index}")(x)

        average_pool = GlobalAveragePooling1D(name="global_average_pool")(x)
        max_pool = GlobalMaxPooling1D(name="global_max_pool")(x)
        x = Concatenate(name="global_pool_concat")([average_pool, max_pool])

        x = BatchNormalization(name="global_pool_batch_norm")(x)

        for index, units in enumerate(self.dense_units, start=1):

            x = Dense(
                units,
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f"dense_head_{index}",
            )(x)

            x = BatchNormalization(name=f"dense_batch_norm_{index}")(x)

            x = Activation(self.activation, name=f"dense_activation_{index}")(x)

            x = Dropout(self.dropout_rate, name=f"dense_dropout_{index}")(x)

        outputs = Dense(self.num_classes, activation="softmax", name="output")(x)

        model = Model(inputs=inputs, outputs=outputs, name="CNN")

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.summary(print_fn=Logger.info)

        return model


def cnn(
    conv_blocks: Optional[List[Tuple[int, int]]],
    dense_units: Optional[List[int]],
    activation: str,
    dropout_rate: float,
    spatial_dropout_rate: float,
    l1_reg: float,
    l2_reg: float,
    learning_rate: float,
    use_residual: bool,
    noise_stddev: float,
    pool_size: int,
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

        model = CNNModel(
            input_dim=input_dim,
            num_classes=num_classes,
            conv_blocks=conv_blocks,
            dense_units=dense_units,
            activation=activation,
            dropout_rate=dropout_rate,
            spatial_dropout_rate=spatial_dropout_rate,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            use_residual=use_residual,
            noise_stddev=noise_stddev,
            pool_size=pool_size,
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
        raise DeepLearningException("CNN Pipeline Failed.", exc)


def train(
    conv_blocks: Optional[List[Tuple[int, int]]] = None,
    dense_units: Optional[List[int]] = None,
    activation: str = "relu",
    dropout_rate: float = 0.35,
    spatial_dropout_rate: float = 0.2,
    l1_reg: float = 1e-6,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
    use_residual: bool = True,
    noise_stddev: float = 0.05,
    pool_size: int = 2,
    epochs: int = 120,
    batch_size: int = 256,
    validation_split: float = 0.1,
    use_class_weights: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:

    return cnn(
        conv_blocks,
        dense_units,
        activation,
        dropout_rate,
        spatial_dropout_rate,
        l1_reg,
        l2_reg,
        learning_rate,
        use_residual,
        noise_stddev,
        pool_size,
        epochs,
        batch_size,
        validation_split,
        use_class_weights,
        random_state,
    )


def predict(X: np.ndarray) -> np.ndarray:

    try:
        keras_model = load_keras_model("CNN")
        return KerasClassifierWrapper(keras_model).predict(X)

    except Exception as exc:
        raise DeepLearningException("CNN Prediction Failed.", exc)


def predict_proba(X: np.ndarray) -> np.ndarray:

    try:
        keras_model = load_keras_model("CNN")
        return KerasClassifierWrapper(keras_model).predict_proba(X)

    except Exception as exc:
        raise DeepLearningException("CNN Prediction Failed.", exc)


if __name__ == "__main__":
    train()
