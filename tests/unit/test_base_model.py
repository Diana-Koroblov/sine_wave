import pytest
import torch

from src.models.base import BaseModel


class MockModel(BaseModel):
    """Concrete implementation of BaseModel for testing purposes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.output_size))


def test_base_model_initialization():
    """Verify correct initialization of BaseModel dimensions."""
    model = MockModel(input_size=14, output_size=10)
    assert model.input_size == 14
    assert model.output_size == 10


def test_base_model_invalid_dimensions():
    """Verify that BaseModel raises ValueError for incorrect dimensions."""
    with pytest.raises(ValueError, match="Expected input_size 14"):
        MockModel(input_size=13, output_size=10)

    with pytest.raises(ValueError, match="Expected output_size 10"):
        MockModel(input_size=14, output_size=9)


def test_base_model_forward_coverage():
    """Verify that calling forward on a mock works (covering the pass line)."""

    class CoverageMock(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(x)

    model = CoverageMock(input_size=14, output_size=10)
    # This will return None because of 'pass' in parent
    assert model.forward(torch.zeros((1, 14))) is None


def test_base_model_weight_init():
    """Verify that weight initialization logic executes without error."""

    class LinearMock(MockModel):
        def __init__(self, input_size, output_size):
            super().__init__(input_size, output_size)
            self.fc = torch.nn.Linear(input_size, output_size)

    model = LinearMock(input_size=14, output_size=10)
    model.init_weights()
    assert isinstance(model.fc.weight, torch.Tensor)
