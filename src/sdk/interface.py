import logging
from typing import Any

import numpy as np

import src.shared.config as config
from src.data.generator import SineWaveDatasetGenerator
from src.data.processor import to_one_hot
from src.sdk.data_prep import split_dataset
from src.sdk.evaluation import create_model, evaluate_model, format_metrics_table
from src.sdk.gatekeeper import Gatekeeper
from src.sdk.sensitivity import run_sensitivity_analysis as run_sdk_sensitivity_analysis
from src.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)
class SignalDenoiserSDK:
    """
    Public Entry Point for the Signal De-noising System.
    Orchestrates data generation, training, and evaluation strictly through this interface.
    """

    def __init__(self, config_path: str = "config.py"):
        """
        Initialize the SDK.
        Args:
            config_path: Path to the centralized configuration file.
        """
        self.config_path = config_path
        self.dataset_splits: dict[str, dict[str, np.ndarray]] = {}
        self.trained_models: dict[str, Any] = {}
        self.training_runs: dict[str, dict[str, Any]] = {}
        # TODO: Load config dynamically in Phase 2/3

    def _build_training_example(
        self, vectors: dict[str, np.ndarray], target_index: int, start_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        noisy_window = self._slice_signal_window(vectors["sum_noise"], start_index)
        pure_window = self._slice_signal_window(vectors[f"pure_{target_index + 1}"], start_index)

        Gatekeeper.validate_window_dimensions(noisy_window)
        Gatekeeper.validate_window_dimensions(pure_window)

        x_input = self._compose_input_vector(target_index, noisy_window)
        y_true = pure_window.astype(np.float32)

        if y_true.shape != (config.OUTPUT_SIZE,):
            raise ValueError(f"Expected Y_true shape {(config.OUTPUT_SIZE,)}, got {y_true.shape}")

        return x_input, y_true

    def _slice_signal_window(self, signal: np.ndarray, start_index: int) -> np.ndarray:
        """Extract a fixed-width window from the requested start index."""
        return signal[start_index : start_index + config.WINDOW_SIZE]

    def _compose_input_vector(self, target_index: int, noisy_window: np.ndarray) -> np.ndarray:
        """Compose the model input as OHE class bits on the left and samples on the right."""
        class_vector = to_one_hot(target_index, config.NUM_FREQUENCIES)
        return np.concatenate([class_vector, noisy_window], dtype=np.float32)

    def prepare_data(
        self, dataset_size: int = config.DATASET_SIZE
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Generates signals and partitions the dataset (70/15/15).
        Implemented in Phase 3 & 5.
        """
        vectors = SineWaveDatasetGenerator().generate_all_vectors()
        max_start_index = config.TOTAL_SAMPLES - config.WINDOW_SIZE
        inputs = np.zeros((dataset_size, config.INPUT_SIZE), dtype=np.float32)
        targets = np.zeros((dataset_size, config.OUTPUT_SIZE), dtype=np.float32)

        for example_index in range(dataset_size):
            target_index = int(np.random.randint(0, config.NUM_FREQUENCIES))
            start_index = int(np.random.randint(0, max_start_index + 1))
            try:
                x_input, y_true = self._build_training_example(vectors, target_index, start_index)
                Gatekeeper.validate_input_vector(x_input)
            except ValueError as error:
                logger.error(
                    "Failed to build training example %s with target=%s start=%s: %s",
                    example_index,
                    target_index,
                    start_index,
                    error,
                )
                raise
            inputs[example_index] = x_input
            targets[example_index] = y_true

        Gatekeeper.validate_input_batch(inputs)
        self.dataset_splits = split_dataset(inputs, targets, dataset_size)
        return self.dataset_splits

    def run_training(
        self,
        model_type: str,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
    ) -> dict[str, Any]:
        """
        Executes the training pipeline for a specific architecture.
        Args:
            model_type: One of ['FC', 'RNN', 'LSTM'].
        Returns:
            Metrics and resource tracking results.
        """
        normalized_model_type = model_type.upper()
        model = create_model(normalized_model_type)
        if not self.dataset_splits:
            self.prepare_data()
        trainer = ModelTrainer(model)
        training_result = trainer.fit(self.dataset_splits, epochs=epochs, batch_size=batch_size)
        self.trained_models[normalized_model_type] = trainer.model
        self.training_runs[normalized_model_type] = training_result
        return {"model_type": normalized_model_type, **training_result}

    def evaluate_on_test_set(
        self,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
    ) -> dict[str, Any]:
        """Train missing models, score the unseen test set, and return a summary table."""
        if not self.dataset_splits:
            self.prepare_data()

        test_split = self.dataset_splits["test"]
        metrics_by_model = {}
        for model_type in ("FC", "RNN", "LSTM"):
            if model_type not in self.trained_models:
                self.run_training(model_type, epochs=epochs, batch_size=batch_size)
            metrics_by_model[model_type] = evaluate_model(
                self.trained_models[model_type],
                test_split["inputs"],
                test_split["targets"],
            )

        return {
            "metrics": metrics_by_model,
            "summary_table": format_metrics_table(metrics_by_model),
        }
    def run_sensitivity_analysis(self, noise_levels: list[float] | None = None) -> dict[str, Any]:
        """Sweep noise levels, evaluate all models, and export research figures."""
        return run_sdk_sensitivity_analysis(self, noise_levels=noise_levels)
    def generate_report(self) -> str:
        """
        Analyzes results and generates a performance report.
        Returns:
            Path to the generated report.
        """
        return ""
