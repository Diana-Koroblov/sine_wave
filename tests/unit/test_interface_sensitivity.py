import numpy as np

import src.shared.config as config
from src.sdk.interface import SignalDenoiserSDK


def test_prepare_data_accepts_runtime_dataset_size(monkeypatch):
    """Verify prepare_data can build smaller runtime-scoped dataset splits."""
    sdk = SignalDenoiserSDK()
    monkeypatch.setattr(config, "DATASET_SIZE", 60000)
    monkeypatch.setattr(config, "TOTAL_SAMPLES", 100)
    monkeypatch.setattr(config, "WINDOW_SIZE", 10)
    monkeypatch.setattr(config, "INPUT_SIZE", 14)
    monkeypatch.setattr(config, "OUTPUT_SIZE", 10)
    monkeypatch.setattr(config, "NUM_FREQUENCIES", 4)

    vectors = {
        "sum_noise": np.arange(100, dtype=np.float32),
        "sum_pure": np.zeros(100, dtype=np.float32),
    }
    for index in range(config.NUM_FREQUENCIES):
        vectors[f"pure_{index + 1}"] = np.full(100, index + 1, dtype=np.float32)
        vectors[f"noisy_{index + 1}"] = np.full(100, index + 11, dtype=np.float32)

    class FakeGenerator:
        def generate_all_vectors(self):
            return vectors

    monkeypatch.setattr("src.sdk.interface.SineWaveDatasetGenerator", FakeGenerator)
    monkeypatch.setattr("src.sdk.interface.np.random.randint", lambda low, high=None, size=None: 0)
    monkeypatch.setattr("src.sdk.interface.np.random.permutation", lambda size: np.arange(size))

    dataset_splits = sdk.prepare_data(dataset_size=20)

    assert dataset_splits["train"]["inputs"].shape == (14, config.INPUT_SIZE)
    assert dataset_splits["validation"]["inputs"].shape == (3, config.INPUT_SIZE)
    assert dataset_splits["test"]["inputs"].shape == (3, config.INPUT_SIZE)
