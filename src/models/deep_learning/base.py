"""
Shared infrastructure for every Deep Learning model in this project
(ANN, DNN, and any future neural network added under `models.deep_learning`).

This module deliberately contains everything that would otherwise be
duplicated across ANN, DNN, etc.:

    - `set_seed`              : reproducibility across python / numpy / tensorflow
    - `configure_gpu`         : automatic GPU detection with safe CPU fallback
    - `KerasClassifierWrapper`: makes a trained Keras model look like a
                                 scikit-learn classifier (`.predict`, `.predict_proba`)
                                 so the project's existing `core.metrics.evaluate_model`
                                 and `core.report_generator.generate_report`
                                 utilities can be reused completely unchanged.
    - `BaseNeuralNetwork`     : abstract base class implementing the full
                                 train -> evaluate -> save -> report workflow,
                                 with EarlyStopping, ModelCheckpoint,
                                 ReduceLROnPlateau and TensorBoard wired in.
                                 Subclasses (ANNModel, DNNModel, ...)
                                 only need to implement `build_model`.
"""

import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core import (
    Logger,
    Timer,
    evaluate_model,
    generate_report,
    plot_training_history,
    plot_confusion_matrix,
    save_keras_model,
    load_keras_model,
    DeepLearningException,
)
from ..core.config import (
    DEEP_LEARNING_REPORT_DIR,
    DEEP_LEARNING_ARTIFACTS_DIR,
    DEEP_LEARNING_LOG_DIR,
)


def set_seed(seed: int = 42) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)

    except ImportError as exc:
        raise DeepLearningException("TensorFlow Is Required For Deep Learning Models.", exc)


def configure_gpu() -> str:

    try:
        import tensorflow as tf

    except ImportError as exc:
        raise DeepLearningException("TensorFlow Is Required For Deep Learning Models.", exc)

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            Logger.info(f"GPU(s) Detected : {len(gpus)} -> Memory Growth Enabled.")

            return "GPU"

        except RuntimeError as exc:
            Logger.error(f"GPU Configuration Failed : {exc}")
            return "CPU"

    Logger.info("No GPU Detected. Falling Back To CPU.")

    return "CPU"


class KerasClassifierWrapper:

    def __init__(self, keras_model: Any) -> None:
        self.keras_model = keras_model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.keras_model.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class BaseNeuralNetwork(ABC):

    def __init__(
        self,
        model_name: str,
        input_dim: int,
        num_classes: int,
        random_state: int = 42,
        **hyperparameters: Any,
    ) -> None:

        self.model_name = model_name
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.random_state = random_state
        self.hyperparameters = hyperparameters

        self.model: Optional[Any] = None
        self.history: Optional[Any] = None

        set_seed(self.random_state)
        self.device = configure_gpu()

    @abstractmethod
    def build_model(self) -> Any:
        raise NotImplementedError

    def get_callbacks(self) -> List[Any]:

        try:
            from tensorflow.keras.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
                ReduceLROnPlateau,
                TensorBoard,
            )

        except ImportError as exc:
            raise DeepLearningException("TensorFlow Is Required To Build Callbacks.", exc)

        checkpoint_dir = os.path.join(DEEP_LEARNING_ARTIFACTS_DIR, self.model_name, "checkpoints")
        log_dir = os.path.join(DEEP_LEARNING_LOG_DIR, self.model_name)

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [

            EarlyStopping(
                monitor="val_loss",
                patience=self.hyperparameters.get("early_stopping_patience", 10),
                restore_best_weights=True,
                verbose=1,
            ),

            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "best_model.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.hyperparameters.get("reduce_lr_factor", 0.5),
                patience=self.hyperparameters.get("reduce_lr_patience", 5),
                min_lr=self.hyperparameters.get("min_lr", 1e-6),
                verbose=1,
            ),

            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
            ),

        ]

        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
        class_weight: Optional[Dict[int, float]] = None,
        verbose: int = 1,
    ) -> Tuple[Any, float, float]:

        try:

            Logger.info("Building Model...")

            self.model = self.build_model()

            Logger.info(f"Training On Device : {self.device}")

            Logger.info("Training Model...")

            timer = Timer()

            timer.tic()

            fit_kwargs: Dict[str, Any] = dict(
                x=X_train,
                y=y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.get_callbacks(),
                class_weight=class_weight,
                verbose=verbose,
            )

            if X_val is not None and y_val is not None:
                fit_kwargs["validation_data"] = (X_val, y_val)
            else:
                fit_kwargs["validation_split"] = validation_split

            self.history = self.model.fit(**fit_kwargs)

            training_time, memory_usage = timer.toc()

            Logger.success("Model Training")

            return self.history, training_time, memory_usage

        except Exception as exc:
            raise DeepLearningException(f"Training Failed For Model '{self.model_name}'.", exc)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        training_time: float,
        memory_usage: float,
    ) -> Dict[str, Any]:

        try:

            if self.model is None:
                raise DeepLearningException(
                    f"Model '{self.model_name}' Has Not Been Trained Or Loaded Yet."
                )

            Logger.info("Evaluating Model...")

            wrapped_model = KerasClassifierWrapper(self.model)

            evaluation = evaluate_model(
                model=wrapped_model,
                X_test=X_test,
                y_test=y_test,
                training_time=training_time,
                memory_usage=memory_usage,
                model_name=self.model_name,
            )

            return evaluation

        except DeepLearningException:
            raise

        except Exception as exc:
            raise DeepLearningException(f"Evaluation Failed For Model '{self.model_name}'.", exc)

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.model is None:
            raise DeepLearningException(
                f"Model '{self.model_name}' Has Not Been Trained Or Loaded Yet."
            )

        return KerasClassifierWrapper(self.model).predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if self.model is None:
            raise DeepLearningException(
                f"Model '{self.model_name}' Has Not Been Trained Or Loaded Yet."
            )

        return KerasClassifierWrapper(self.model).predict_proba(X)

    def save(self) -> float:

        if self.model is None:
            raise DeepLearningException(
                f"Model '{self.model_name}' Has Not Been Trained Yet. Nothing To Save."
            )

        return save_keras_model(self.model, self.model_name)

    def load(self) -> Any:

        self.model = load_keras_model(self.model_name)

        return self.model

    def report(self, evaluation: Dict[str, Any]) -> None:

        generate_report(
            evaluation,
            model_name=self.model_name,
            report_directory=DEEP_LEARNING_REPORT_DIR,
        )

        plot_directory = os.path.join(DEEP_LEARNING_REPORT_DIR, self.model_name, "plots")

        if self.history is not None:
            plot_training_history(self.history, self.model_name, plot_directory)

        plot_confusion_matrix(
            evaluation["confusion_matrix"],
            self.model_name,
            plot_directory,
        )

    def run(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:

        Logger.title(self.model_name)

        _, training_time, memory_usage = self.train(
            X_train=X_train,
            y_train=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight,
        )

        evaluation = self.evaluate(
            X_test=X_test,
            y_test=y_test,
            training_time=training_time,
            memory_usage=memory_usage,
        )

        model_size = self.save()

        evaluation["metrics"]["Model Size (MB)"] = model_size

        self.report(evaluation)

        return evaluation