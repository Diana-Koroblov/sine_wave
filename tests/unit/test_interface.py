import numpy as np
import pytest

import src.sdk.interface as interface_module
import src.shared.config as config
from src.sdk.interface import SignalDenoiserSDK


def test_sdk_initialization():
    """Verify SDK correctly stores config path."""
    sdk = SignalDenoiserSDK(config_path="custom_config.py")
    assert sdk.config_path == "custom_config.py"


def test_prepare_data_builds_strict_dataset_splits(monkeypatch):
    """Verify prepare_data creates 60k examples and strict 70/15/15 partitions."""

    vectors = {
        "sum_noise": np.arange(config.TOTAL_SAMPLES, dtype=np.float32),
        "sum_pure": np.zeros(config.TOTAL_SAMPLES, dtype=np.float32),
    }
    for index in range(config.NUM_FREQUENCIES):
        vectors[f"pure_{index + 1}"] = np.full(config.TOTAL_SAMPLES, index + 1, dtype=np.float32)
        vectors[f"noisy_{index + 1}"] = np.full(config.TOTAL_SAMPLES, index + 11, dtype=np.float32)

    permutation_calls = []

    class FakeGenerator:
        def generate_all_vectors(self):
            return vectors

    def fake_randint(low, high=None, size=None):
        if high == config.NUM_FREQUENCIES:
            return 1
        return 7

    def fake_permutation(size):
        permutation_calls.append(size)
        return np.arange(size - 1, -1, -1)

    monkeypatch.setattr(interface_module, "SineWaveDatasetGenerator", FakeGenerator)
    monkeypatch.setattr(interface_module.np.random, "randint", fake_randint)
    monkeypatch.setattr(interface_module.np.random, "permutation", fake_permutation)

    sdk = SignalDenoiserSDK()
    dataset_splits = sdk.prepare_data()

    expected_input = np.concatenate(
        [np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), vectors["sum_noise"][7:17]]
    )
    expected_target = np.full(config.WINDOW_SIZE, 2.0, dtype=np.float32)

    assert permutation_calls == [config.DATASET_SIZE]
    assert dataset_splits is sdk.dataset_splits
    assert dataset_splits["train"]["inputs"].shape == (42000, config.INPUT_SIZE)
    assert dataset_splits["train"]["targets"].shape == (42000, config.OUTPUT_SIZE)
    assert dataset_splits["validation"]["inputs"].shape == (9000, config.INPUT_SIZE)
    assert dataset_splits["validation"]["targets"].shape == (9000, config.OUTPUT_SIZE)
    assert dataset_splits["test"]["inputs"].shape == (9000, config.INPUT_SIZE)
    assert dataset_splits["test"]["targets"].shape == (9000, config.OUTPUT_SIZE)
    np.testing.assert_allclose(dataset_splits["train"]["inputs"][0], expected_input)
    np.testing.assert_allclose(dataset_splits["train"]["targets"][0], expected_target)


def test_prepare_data_raises_for_invalid_window_shapes(monkeypatch):
    """Verify prepare_data rejects malformed sampled windows."""

    class FakeGenerator:
        def generate_all_vectors(self):
            short_wave = np.arange(config.WINDOW_SIZE - 1, dtype=np.float32)
            vectors = {
                "sum_noise": short_wave,
                "sum_pure": short_wave,
            }
            for index in range(config.NUM_FREQUENCIES):
                vectors[f"pure_{index + 1}"] = short_wave
                vectors[f"noisy_{index + 1}"] = short_wave
            return vectors

    def fake_randint(low, high=None, size=None):
        return 0

    monkeypatch.setattr(interface_module, "SineWaveDatasetGenerator", FakeGenerator)
    monkeypatch.setattr(interface_module.np.random, "randint", fake_randint)

    with pytest.raises(ValueError, match="Expected window size"):
        SignalDenoiserSDK().prepare_data()


def test_prepare_data_raises_for_invalid_target_shapes(monkeypatch):
    """Verify prepare_data rejects malformed target windows."""

    class FakeGenerator:
        def generate_all_vectors(self):
            valid_noise = np.arange(config.TOTAL_SAMPLES, dtype=np.float32)
            vectors = {
                "sum_noise": valid_noise,
                "sum_pure": valid_noise,
            }
            vectors["pure_1"] = np.arange(config.WINDOW_SIZE - 1, dtype=np.float32)
            for index in range(1, config.NUM_FREQUENCIES):
                vectors[f"pure_{index + 1}"] = valid_noise
            for index in range(config.NUM_FREQUENCIES):
                vectors[f"noisy_{index + 1}"] = valid_noise
            return vectors

    def fake_randint(low, high=None, size=None):
        if high == config.NUM_FREQUENCIES:
            return 0
        return 0

    monkeypatch.setattr(interface_module, "SineWaveDatasetGenerator", FakeGenerator)
    monkeypatch.setattr(interface_module.np.random, "randint", fake_randint)

    with pytest.raises(ValueError, match="Expected window size"):
        SignalDenoiserSDK().prepare_data()


def test_sdk_other_public_methods_keep_expected_types():
    """Verify the remaining public interface methods keep their expected types."""

    sdk = SignalDenoiserSDK()
    assert isinstance(sdk.run_training("FC"), dict)
    assert isinstance(sdk.generate_report(), str)
