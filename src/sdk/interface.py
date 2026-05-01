from typing import Any

import numpy as np

import src.shared.config as config
from src.data.generator import SineWaveDatasetGenerator
from src.data.processor import to_one_hot


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
        # TODO: Load config dynamically in Phase 2/3

    def _build_training_example(
        self, vectors: dict[str, np.ndarray], target_index: int, start_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        noisy_window = vectors["sum_noise"][start_index : start_index + config.WINDOW_SIZE]
        pure_window = vectors[f"pure_{target_index + 1}"][
            start_index : start_index + config.WINDOW_SIZE
        ]
        x_input = np.concatenate(
            [to_one_hot(target_index, config.NUM_FREQUENCIES), noisy_window],
            dtype=np.float32,
        )
        y_true = pure_window.astype(np.float32)

        if x_input.shape != (config.INPUT_SIZE,):
            raise ValueError(f"Expected X_input shape {(config.INPUT_SIZE,)}, got {x_input.shape}")
        if y_true.shape != (config.OUTPUT_SIZE,):
            raise ValueError(f"Expected Y_true shape {(config.OUTPUT_SIZE,)}, got {y_true.shape}")

        return x_input, y_true

    def _split_dataset(
        self, inputs: np.ndarray, targets: np.ndarray
    ) -> dict[str, dict[str, np.ndarray]]:
        permutation = np.random.permutation(config.DATASET_SIZE)
        shuffled_inputs = inputs[permutation]
        shuffled_targets = targets[permutation]

        train_end = int(config.DATASET_SIZE * config.TRAIN_SPLIT)
        validation_end = train_end + int(config.DATASET_SIZE * config.VAL_SPLIT)

        return {
            "train": {
                "inputs": shuffled_inputs[:train_end],
                "targets": shuffled_targets[:train_end],
            },
            "validation": {
                "inputs": shuffled_inputs[train_end:validation_end],
                "targets": shuffled_targets[train_end:validation_end],
            },
            "test": {
                "inputs": shuffled_inputs[validation_end:],
                "targets": shuffled_targets[validation_end:],
            },
        }

    def prepare_data(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Generates signals and partitions the dataset (70/15/15).
        Implemented in Phase 3 & 5.
        """
        vectors = SineWaveDatasetGenerator().generate_all_vectors()
        max_start_index = config.TOTAL_SAMPLES - config.WINDOW_SIZE
        inputs = np.zeros((config.DATASET_SIZE, config.INPUT_SIZE), dtype=np.float32)
        targets = np.zeros((config.DATASET_SIZE, config.OUTPUT_SIZE), dtype=np.float32)

        for example_index in range(config.DATASET_SIZE):
            target_index = int(np.random.randint(0, config.NUM_FREQUENCIES))
            start_index = int(np.random.randint(0, max_start_index + 1))
            x_input, y_true = self._build_training_example(vectors, target_index, start_index)
            inputs[example_index] = x_input
            targets[example_index] = y_true

        self.dataset_splits = self._split_dataset(inputs, targets)
        return self.dataset_splits

    def run_training(self, model_type: str) -> dict[str, Any]:
        """
        Executes the training pipeline for a specific architecture.
        Args:
            model_type: One of ['FC', 'RNN', 'LSTM'].
        Returns:
            Metrics and resource tracking results.
        """
        return {}

    def generate_report(self) -> str:
        """
        Analyzes results and generates a performance report.
        Returns:
            Path to the generated report.
        """
        return ""
