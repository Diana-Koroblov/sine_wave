import numpy as np

import src.shared.config as config
from src.data.generator import SineWaveDatasetGenerator


def test_generator_initialization():
    """Verify the generator correctly loads parameters from config.py."""
    gen = SineWaveDatasetGenerator()
    assert gen.sampling_rate == config.SAMPLING_RATE
    assert gen.duration == config.DURATION
    assert gen.total_samples == config.TOTAL_SAMPLES
    assert gen.frequencies == config.FREQUENCIES
    assert gen.amplitudes == config.AMPLITUDES
    assert gen.phases == config.PHASES


def test_generate_time_axis():
    """Verify the time axis has the correct shape and range."""
    gen = SineWaveDatasetGenerator()
    t = gen._generate_time_axis()
    assert t.shape == (10000,)
    assert t[0] == 0.0
    # Last sample should be just before DURATION (10s)
    assert t[-1] < config.DURATION


def test_create_pure_waves():
    """Verify that 4 pure sine waves are generated with correct shapes."""
    gen = SineWaveDatasetGenerator()
    pure_waves = gen._create_pure_waves()

    assert isinstance(pure_waves, list)
    assert len(pure_waves) == 4

    for i, wave in enumerate(pure_waves):
        assert wave.shape == (10000,)
        assert wave.dtype == np.float32
        # Amplitude check: should not exceed configured amplitude (+ small epsilon for float)
        assert np.max(np.abs(wave)) <= config.AMPLITUDES[i] + 1e-5


def test_pure_waves_frequency():
    """Basic check to ensure waves are not just zeros and vary over time."""
    gen = SineWaveDatasetGenerator()
    pure_waves = gen._create_pure_waves()

    for wave in pure_waves:
        assert not np.all(wave == 0)
        # Periodic check: if it's a sine wave, mean should be near zero
        assert np.isclose(np.mean(wave), 0.0, atol=1e-2)
