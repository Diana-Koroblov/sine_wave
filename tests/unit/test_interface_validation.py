import logging

import numpy as np
import pytest

import src.sdk.interface as interface_module
import src.shared.config as config
from src.sdk.interface import SignalDenoiserSDK


def test_prepare_data_validates_each_generated_input(monkeypatch):
    """Verify Gatekeeper validation runs inside the dataset creation loop."""

    vectors = {
        "sum_noise": np.arange(config.TOTAL_SAMPLES, dtype=np.float32),
        "sum_pure": np.zeros(config.TOTAL_SAMPLES, dtype=np.float32),
    }
    for index in range(config.NUM_FREQUENCIES):
        vectors[f"pure_{index + 1}"] = np.full(config.TOTAL_SAMPLES, index + 1, dtype=np.float32)
        vectors[f"noisy_{index + 1}"] = np.full(config.TOTAL_SAMPLES, index + 11, dtype=np.float32)

    calls = []

    class FakeGenerator:
        def generate_all_vectors(self):
            return vectors

    def fake_randint(low, high=None, size=None):
        if high == config.NUM_FREQUENCIES:
            return 0
        return 3

    def fake_validate_input_vector(x_input):
        calls.append(x_input.copy())

    monkeypatch.setattr(interface_module, "SineWaveDatasetGenerator", FakeGenerator)
    monkeypatch.setattr(interface_module.np.random, "randint", fake_randint)
    monkeypatch.setattr(interface_module.config, "DATASET_SIZE", 3)
    monkeypatch.setattr(
        interface_module.Gatekeeper, "validate_input_vector", fake_validate_input_vector
    )
    monkeypatch.setattr(interface_module.Gatekeeper, "validate_input_batch", lambda batch: None)

    SignalDenoiserSDK().prepare_data(dataset_size=3)

    assert len(calls) == 3
    for recorded_input in calls:
        assert recorded_input.shape == (config.INPUT_SIZE,)


def test_build_training_example_places_ohe_on_left_and_window_on_right():
    """Verify inputs use OHE, sigma, then the selected noisy window in that order."""

    sdk = SignalDenoiserSDK()
    vectors = {
        "sum_noise": np.arange(config.TOTAL_SAMPLES, dtype=np.float32),
        "sum_pure": np.zeros(config.TOTAL_SAMPLES, dtype=np.float32),
    }
    for index in range(config.NUM_FREQUENCIES):
        vectors[f"pure_{index + 1}"] = np.arange(
            config.TOTAL_SAMPLES,
            dtype=np.float32,
        ) + (index + 1) * 1000
        vectors[f"noisy_{index + 1}"] = np.arange(
            config.TOTAL_SAMPLES,
            dtype=np.float32,
        ) + (index + 1) * 100

    x_input, y_true = sdk._build_training_example(vectors, target_index=2, start_index=5)

    np.testing.assert_array_equal(x_input[: config.NUM_FREQUENCIES], np.array([0, 0, 1, 0]))
    assert x_input[config.SIGMA_INDEX] == pytest.approx(config.NOISE_ALPHA)
    np.testing.assert_allclose(
        x_input[config.SIGNAL_START_INDEX :],
        vectors["noisy_3"][5:15],
    )
    np.testing.assert_allclose(y_true, vectors["pure_3"][5:15])


def test_slice_signal_window_uses_t_to_t_plus_window_size():
    """Verify the sampling helper extracts exactly 10 values from the requested start index."""

    sdk = SignalDenoiserSDK()
    signal = np.arange(20, dtype=np.float32)

    window = sdk._slice_signal_window(signal, start_index=4)

    np.testing.assert_allclose(window, np.arange(4, 14, dtype=np.float32))


def test_prepare_data_logs_generation_errors(monkeypatch, caplog):
    """Verify prepare_data logs clear errors before re-raising generation failures."""

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

    monkeypatch.setattr(interface_module, "SineWaveDatasetGenerator", FakeGenerator)
    monkeypatch.setattr(interface_module.np.random, "randint", lambda low, high=None, size=None: 0)

    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="Expected window size"):
        SignalDenoiserSDK().prepare_data()

    assert "Failed to build training example 0" in caplog.text
