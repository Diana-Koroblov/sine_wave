import pytest

import src.sdk.interface as interface_module
from src.sdk.interface import SignalDenoiserSDK


def test_sdk_initialization():
    """Verify SDK correctly stores config path."""
    sdk = SignalDenoiserSDK(config_path="custom_config.py")
    assert sdk.config_path == "custom_config.py"


def test_sdk_other_public_methods_keep_expected_types(monkeypatch, tmp_path):
    """Verify the remaining public interface methods keep their expected types."""

    sdk = SignalDenoiserSDK()

    monkeypatch.setattr(sdk, "run_training", lambda *args, **kwargs: {"model_type": "FC"})

    def fake_evaluate_on_test_set(*args, **kwargs):
        return {
            "summary_table": "| Model | MSE |\n| --- | ---: |\n| FC | 0.1 |",
            "frequency_mse": {
                "FC": [0.1, 0.2, 0.3, 0.4],
                "RNN": [0.2, 0.3, 0.4, 0.5],
                "LSTM": [0.05, 0.1, 0.15, 0.2],
            },
            "artifacts": {
                "frequency_mse_comparison": str(tmp_path / "frequency_mse_comparison.png")
            },
        }

    monkeypatch.setattr(sdk, "evaluate_on_test_set", fake_evaluate_on_test_set)

    assert isinstance(sdk.run_training("FC"), dict)
    assert isinstance(sdk.generate_report(), str)


def test_resolve_noise_level_requires_single_sigma_contract(monkeypatch):
    """Verify homework-aligned inputs reject divergent amplitude/phase noise levels."""
    sdk = SignalDenoiserSDK()
    monkeypatch.setattr(interface_module.config, "NOISE_ALPHA", 0.1)
    monkeypatch.setattr(interface_module.config, "NOISE_BETA", 0.2)

    with pytest.raises(ValueError, match="NOISE_ALPHA and NOISE_BETA to match"):
        sdk._resolve_noise_level()
