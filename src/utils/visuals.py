from pathlib import Path

import matplotlib
import numpy as np

import src.shared.config as config

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class Visualizer:
    """Export research figures to the configured assets directory."""

    def __init__(self, assets_dir: str | Path = config.ASSETS_PATH):
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self.assets_dir / filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def plot_reconstruction(
        self,
        noisy_input: np.ndarray,
        pure_signal: np.ndarray,
        reconstructed_signal: np.ndarray,
        model_name: str,
        filename: str = "reconstruction.png",
    ) -> Path:
        """Plot signal reconstruction comparison with explicit metadata."""
        # Extract OHE from the first 4 elements of the input vector
        ohe = noisy_input[:4]
        freq_class = int(np.argmax(ohe))
        noisy_signal = noisy_input[4:]

        figure, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Title formatting: Extract noise from filename or assume default
        noise_val = "Unknown"
        if "noise" in filename.lower():
            # e.g., reconstruction_lstm_freq2_noise0_9.png -> 0_9 -> 0.9
            noise_part = filename.split("noise")[-1].replace(".png", "")
            noise_val = noise_part.replace("_", ".")

        title_meta = f"Model: {model_name} | Noise: {noise_val} | Freq Class: {freq_class}"

        axes[0].plot(pure_signal, label="Pure", linewidth=2)
        axes[0].plot(noisy_signal, label="Noisy", alpha=0.8)
        axes[0].set_title(f"Target vs Noisy\n({title_meta})")

        axes[1].plot(pure_signal, label="Pure", linewidth=2)
        axes[1].plot(reconstructed_signal, label="Reconstructed", alpha=0.8)
        axes[1].set_title(f"Target vs Reconstructed\n({title_meta})")
        for axis in axes:
            axis.set_xlabel("Sample")
            axis.set_ylabel("Amplitude")
            axis.legend()
        return self._save(figure, filename)

    def plot_loss_curves(
        self,
        histories: dict[str, dict[str, list[float]]],
        filename: str = "loss_curves.png",
    ) -> Path:
        figure, axes = plt.subplots(len(histories), 1, figsize=(10, 4 * max(len(histories), 1)))
        axes_array = np.atleast_1d(axes)
        for axis, (model_name, history) in zip(axes_array, histories.items(), strict=False):
            axis.plot(history.get("train_losses", []), label="Train")
            axis.plot(history.get("validation_losses", []), label="Validation")
            axis.set_title(f"{model_name} Loss")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("MSE")
            axis.legend()
        return self._save(figure, filename)

    def plot_sensitivity_curve(
        self,
        noise_levels: list[float],
        metrics_by_model: dict[str, list[dict[str, float]]],
        metric: str = "mse",
        filename: str = "sensitivity_mse.png",
    ) -> Path:
        figure, axis = plt.subplots(figsize=(10, 5))
        for model_name, metric_points in metrics_by_model.items():
            axis.plot(
                noise_levels,
                [point[metric] for point in metric_points],
                marker="o",
                label=model_name,
            )
        axis.set_title(f"Reconstruction Quality vs Noise ({metric.upper()})")
        axis.set_xlabel("Noise Intensity (alpha = beta)")
        axis.set_ylabel(metric.upper())
        axis.legend()
        return self._save(figure, filename)
