import numpy as np

import src.shared.config as config


def split_dataset(
    inputs: np.ndarray, targets: np.ndarray, dataset_size: int
) -> dict[str, dict[str, np.ndarray]]:
    """Shuffle the dataset once and partition it into train, validation, and test splits."""
    permutation = np.random.permutation(dataset_size)
    shuffled_inputs = inputs[permutation]
    shuffled_targets = targets[permutation]

    train_end = int(dataset_size * config.TRAIN_SPLIT)
    validation_end = train_end + int(dataset_size * config.VAL_SPLIT)

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
