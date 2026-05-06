import numpy as np
import torch

from src.models.fc import FCModel
from src.models.lstm import LSTMModel
from src.models.rnn import RNNModel

MODEL_BUILDERS = {
    "FC": FCModel,
    "RNN": RNNModel,
    "LSTM": LSTMModel,
}


def create_model(model_type: str):
    """Build one of the supported denoising architectures."""
    try:
        return MODEL_BUILDERS[model_type]()
    except KeyError as error:
        supported = ", ".join(MODEL_BUILDERS)
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Expected one of: {supported}"
        ) from error


def _pearson_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Pearson correlation on flattened arrays with constant-vector guards."""
    prediction_values = predictions.reshape(-1)
    target_values = targets.reshape(-1)
    if prediction_values.std() == 0.0 or target_values.std() == 0.0:
        return 1.0 if np.allclose(prediction_values, target_values) else 0.0
    correlation_matrix = np.corrcoef(prediction_values, target_values)
    return float(correlation_matrix[0, 1])


@torch.no_grad()
def evaluate_model(model, inputs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Run inference on the test split and compute core regression metrics."""
    model.eval()
    prediction_tensor = model(torch.as_tensor(inputs, dtype=torch.float32))
    predictions = prediction_tensor.detach().cpu().numpy()
    error = predictions - targets
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "pearson_correlation": _pearson_correlation(predictions, targets),
    }


def format_metrics_table(metrics_by_model: dict[str, dict[str, float]]) -> str:
    """Format evaluation results as a markdown table for notebooks and README content."""
    lines = [
        "| Model | MSE | MAE | Pearson Correlation |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model_type in ("FC", "RNN", "LSTM"):
        metrics = metrics_by_model[model_type]
        lines.append(
            "| "
            f"{model_type} | {metrics['mse']:.6f} | {metrics['mae']:.6f} | "
            f"{metrics['pearson_correlation']:.6f} |"
        )
    return "\n".join(lines)
