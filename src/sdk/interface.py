import logging
from typing import Any

import numpy as np

import src.shared.config as config
from src.data.generator import SineWaveDatasetGenerator
from src.sdk.data_prep import split_dataset
from src.sdk.evaluation import create_model
from src.sdk.gatekeeper import Gatekeeper
from src.sdk.reporting import evaluate_sdk_on_test_set, generate_sdk_report
from src.sdk.sample_builder import (
    build_training_example,
    compose_input_vector,
    resolve_noise_level,
    slice_signal_window,
)
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
        return build_training_example(vectors, target_index, start_index)

    def _slice_signal_window(self, signal: np.ndarray, start_index: int) -> np.ndarray:
        """Extract a fixed-width window from the requested start index."""
        return slice_signal_window(signal, start_index)

    def _resolve_noise_level(self) -> float:
        """Return the single sigma value expected by the homework contract."""
        return resolve_noise_level()

    def _compose_input_vector(
        self,
        target_index: int,
        noise_level: float,
        noisy_window: np.ndarray,
    ) -> np.ndarray:
        """Compose the input as OHE class bits, sigma, then the noisy signal window."""
        return compose_input_vector(target_index, noise_level, noisy_window)

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
        export_artifacts: bool = True,
    ) -> dict[str, Any]:
        """Train missing models, score the unseen test set, and return a summary table."""
        return evaluate_sdk_on_test_set(
            self,
            epochs=epochs,
            batch_size=batch_size,
            export_artifacts=export_artifacts,
        )

    def run_sensitivity_analysis(self, noise_levels: list[float] | None = None) -> dict[str, Any]:
        """Sweep noise levels, evaluate all models, and export research figures."""
        return run_sdk_sensitivity_analysis(self, noise_levels=noise_levels)

    def generate_report(self) -> str:
        """
        Generate a concise markdown report from the current test-set evaluation.
        Returns:
            Markdown report content.
        """
        return generate_sdk_report(self)
