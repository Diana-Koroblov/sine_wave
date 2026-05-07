from unittest.mock import patch

import numpy as np
import pytest

import src.shared.config as config
from src.data.generator import SineWaveDatasetGenerator


def test_generate_gaussian_noise():
    """Verify noise generator produces correct shapes and statistical properties."""
    gen = SineWaveDatasetGenerator()
    size = 1000
    noise = gen._generate_gaussian_noise(size)
    assert noise.shape == (size,)
    # For standard Gaussian, mean is approx 0 and std is approx 1
    assert np.isclose(np.mean(noise), 0.0, atol=0.2)
    assert np.isclose(np.std(noise), 1.0, atol=0.2)


def test_create_noisy_waves():
    """Verify noisy wave generation and formula impact."""
    gen = SineWaveDatasetGenerator()
    pure_waves = gen._create_pure_waves()
    noisy_waves = gen._create_noisy_waves()

    assert len(noisy_waves) == 4
    for i in range(4):
        assert noisy_waves[i].shape == (10000,)
        # Noisy wave must be mathematically different from pure wave
        assert not np.array_equal(noisy_waves[i], pure_waves[i])
        # Check for NaN/Inf
        assert not np.any(np.isnan(noisy_waves[i]))
        assert not np.any(np.isinf(noisy_waves[i]))


def test_generate_all_vectors():
    """Verify the primary output dictionary structure and shapes."""
    gen = SineWaveDatasetGenerator()
    vectors = gen.generate_all_vectors()

    assert isinstance(vectors, dict)
    assert len(vectors) == 10

    expected_keys = [
        "sum_pure",
        "sum_noise",
        "pure_1",
        "pure_2",
        "pure_3",
        "pure_4",
        "noisy_1",
        "noisy_2",
        "noisy_3",
        "noisy_4",
    ]
    for key in expected_keys:
        assert key in vectors
        assert vectors[key].shape == (10000,)
        assert vectors[key].dtype == np.float32


def test_sum_noise_vs_sum_pure():
    """Verify that combined noisy signal differs from combined pure signal."""
    gen = SineWaveDatasetGenerator()
    vectors = gen.generate_all_vectors()

    assert not np.array_equal(vectors["sum_noise"], vectors["sum_pure"])
    # Noise should add variance
    assert np.var(vectors["sum_noise"]) != np.var(vectors["sum_pure"])


def test_noise_independence():
    """Verify that noise added to different waves is independent (uncorrelated)."""
    gen = SineWaveDatasetGenerator()
    # To get reliable correlation, we use the raw noise generator
    n1 = gen._generate_gaussian_noise(10000)
    n2 = gen._generate_gaussian_noise(10000)

    correlation = np.corrcoef(n1, n2)[0, 1]
    # Correlation between independent Gaussian variables should be near 0
    assert abs(correlation) < 0.1


def test_generator_error_handling():
    """Verify that the generator raises ValueError if sample count is corrupted."""
    gen = SineWaveDatasetGenerator()
    # Mocking total_samples to trigger ValueError in generate_all_vectors logic
    gen.total_samples = 9999
    with pytest.raises(ValueError, match="has incorrect shape"):
        gen.generate_all_vectors()


def test_noise_application_integrity():
    """
    Verify noisy signal accounts for both amplitude and phase noise components.
    Formula: S'_i(t) = (A_i * (1 + alpha * noise)) * sin(2*pi * f_i * t + theta_i + beta * noise)
    """
    gen = SineWaveDatasetGenerator()
    # Mock noise to return constant 1.0
    with patch.object(
        SineWaveDatasetGenerator, "_generate_gaussian_noise", return_value=np.ones(10000)
    ):
        noisy_waves = gen._create_noisy_waves()
        t = gen._generate_time_axis()

        for i in range(4):
            # Manual calculation
            expected_amp = config.AMPLITUDES[i] * (1.0 + config.NOISE_ALPHA)
            expected_phase = config.PHASES[i] + config.NOISE_BETA
            expected_wave = expected_amp * np.sin(
                2 * np.pi * config.FREQUENCIES[i] * t + expected_phase
            )

            np.testing.assert_allclose(noisy_waves[i], expected_wave, atol=1e-5)


def test_summation_math():
    """Verify that sum_pure/sum_noise match the manual sum of individual waves."""
    gen = SineWaveDatasetGenerator()
    vectors = gen.generate_all_vectors()

    # Manual sum of pure waves
    manual_sum_pure = np.zeros(10000, dtype=np.float32)
    manual_sum_noise = np.zeros(10000, dtype=np.float32)
    for i in range(1, 5):
        manual_sum_pure += vectors[f"pure_{i}"]
        manual_sum_noise += vectors[f"noisy_{i}"]

    np.testing.assert_allclose(vectors["sum_pure"], manual_sum_pure, atol=1e-5)
    np.testing.assert_allclose(vectors["sum_noise"], manual_sum_noise, atol=1e-5)


def test_pure_waves_match_configured_frequencies():
    """Verify each pure wave's dominant frequency matches the configured value."""
    gen = SineWaveDatasetGenerator()
    pure_waves = gen._create_pure_waves()
    frequency_bins = np.fft.rfftfreq(config.TOTAL_SAMPLES, d=1 / config.SAMPLING_RATE)

    for wave, expected_frequency in zip(pure_waves, config.FREQUENCIES, strict=True):
        magnitude = np.abs(np.fft.rfft(wave))
        dominant_frequency = frequency_bins[1 + np.argmax(magnitude[1:])]
        assert np.isclose(dominant_frequency, expected_frequency, atol=0.1)
