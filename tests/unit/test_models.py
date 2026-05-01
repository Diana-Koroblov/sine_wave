import pytest
import torch
import torch.nn as nn

import src.shared.config as config
from src.models.fc import FCModel
from src.models.lstm import LSTMModel
from src.models.rnn import RNNModel


def make_valid_input() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0] + [0.25] * 10], dtype=torch.float32)


@pytest.mark.parametrize("model_cls", [FCModel, RNNModel, LSTMModel])
def test_model_forward_output_shape(model_cls) -> None:
    model = model_cls()

    output = model(make_valid_input())

    assert output.shape == (1, config.OUTPUT_SIZE)


@pytest.mark.parametrize(
    ("model_cls", "reference_builder", "layer_getter"),
    [
        (
            FCModel,
            lambda: nn.Linear(config.INPUT_SIZE, config.HIDDEN_SIZE),
            lambda model: next(
                layer for layer in model.network if isinstance(layer, nn.Linear)
            ),
        ),
        (
            RNNModel,
            lambda: nn.RNN(
                input_size=5,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                batch_first=True,
                nonlinearity="tanh",
            ),
            lambda model: model.rnn,
        ),
        (
            LSTMModel,
            lambda: nn.LSTM(
                input_size=5,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                batch_first=True,
            ),
            lambda model: model.lstm,
        ),
    ],
)
def test_init_weights_changes_default_parameters(
    model_cls, reference_builder, layer_getter
) -> None:
    torch.manual_seed(7)
    reference_layer = reference_builder()
    reference_parameters = {
        name: parameter.detach().clone()
        for name, parameter in reference_layer.named_parameters()
    }

    torch.manual_seed(7)
    model = model_cls()
    initialized_layer = layer_getter(model)

    assert any(
        not torch.allclose(parameter.detach(), reference_parameters[name])
        for name, parameter in initialized_layer.named_parameters()
    )
