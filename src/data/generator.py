import numpy as np

import src.shared.config as config


class SineWaveDatasetGenerator:
    """
    Engine responsible for the mathematical synthesis of pure and noisy sine waves.
    Follows the formula: S_i(t) = A_i * sin(2*pi * f_i * t + theta_i)
    """

    def __init__(self):
        """Initialize generator with configuration parameters."""
        self.sampling_rate = config.SAMPLING_RATE
        self.duration = config.DURATION
        self.total_samples = config.TOTAL_SAMPLES
        self.frequencies = config.FREQUENCIES
        self.amplitudes = config.AMPLITUDES
        self.phases = config.PHASES

    def _generate_time_axis(self) -> np.ndarray:
        """
        Creates a 1D array of time steps.
        Returns:
            np.ndarray: Time axis from 0 to DURATION with TOTAL_SAMPLES points.
        """
        return np.linspace(0, self.duration, self.total_samples, endpoint=False)

    def _generate_gaussian_noise(self, size: int) -> np.ndarray:
        """
        Generates standard Gaussian noise N(0, 1).
        Args:
            size: Number of samples to generate.
        Returns:
            np.ndarray: Array of random samples.
        """
        return np.random.normal(0, 1, size)

    def _create_pure_waves(self) -> list[np.ndarray]:
        """
        Generates the 4 pure sine waves (S_i) as defined in the project specs.
        Returns:
            list[np.ndarray]: List of 4 arrays, each containing 10,000 samples.
        """
        t = self._generate_time_axis()
        pure_waves = []

        for i in range(len(self.frequencies)):
            amp = self.amplitudes[i]
            freq = self.frequencies[i]
            phase = self.phases[i]

            # S_i(t) = A_i * sin(2 * pi * f_i * t + theta_i)
            wave = amp * np.sin(2 * np.pi * freq * t + phase)
            pure_waves.append(wave.astype(np.float32))

        return pure_waves

    def _create_noisy_waves(self) -> list[np.ndarray]:
        """
        Generates the 4 noisy sine waves (S'_i) using the Gaussian formula.
        Formula: S'_i(t) = (A_i + alpha * noise) * sin(2*pi * f_i * t + (theta_i + beta * noise))
        Returns:
            list[np.ndarray]: List of 4 noisy arrays, each containing 10,000 samples.
        """
        t = self._generate_time_axis()
        noisy_waves = []

        # Load alpha and beta from config
        alpha = config.NOISE_ALPHA
        beta = config.NOISE_BETA

        for i in range(len(self.frequencies)):
            amp = self.amplitudes[i]
            freq = self.frequencies[i]
            phase = self.phases[i]

            # Generate independent noise for amplitude and phase per wave
            noise_amp = self._generate_gaussian_noise(self.total_samples)
            noise_phase = self._generate_gaussian_noise(self.total_samples)

            # Apply formula: (A_i + alpha * noise) * sin(2*pi*f_i*t + theta_i + beta * noise)
            noisy_amp = amp + (alpha * noise_amp)
            noisy_phase = phase + (beta * noise_phase)

            wave = noisy_amp * np.sin(2 * np.pi * freq * t + noisy_phase)
            noisy_waves.append(wave.astype(np.float32))

        return noisy_waves

    def generate_all_vectors(self) -> dict[str, np.ndarray]:
        """
        Orchestrates the generation of all 10 required vectors.
        Returns:
            dict: {
                'sum_noise': Σ_noise,
                'sum_pure': Σ_pure,
                'pure_1-4': individual S_i,
                'noisy_1-4': individual S'_i
            }
        """
        pure_waves = self._create_pure_waves()
        noisy_waves = self._create_noisy_waves()

        # Compute combined signals
        sum_pure = np.sum(pure_waves, axis=0)
        sum_noise = np.sum(noisy_waves, axis=0)

        # Structure the export dictionary
        vectors = {
            "sum_pure": sum_pure.astype(np.float32),
            "sum_noise": sum_noise.astype(np.float32),
        }

        for i in range(len(pure_waves)):
            vectors[f"pure_{i + 1}"] = pure_waves[i]
            vectors[f"noisy_{i + 1}"] = noisy_waves[i]

        # Verification of Constraints: Assert the configured sample count per wave
        for key, vec in vectors.items():
            if vec.shape[0] != config.TOTAL_SAMPLES:
                raise ValueError(f"Vector {key} has incorrect shape: {vec.shape}")

        return vectors
