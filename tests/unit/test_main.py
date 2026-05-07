import main as main_module


class FakeSDK:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.calls: list[object] = []

    def prepare_data(self):
        self.calls.append("prepare_data")
        return {}

    def run_training(self, model_type: str):
        self.calls.append(("run_training", model_type))
        return {"model_type": model_type}

    def evaluate_on_test_set(self):
        self.calls.append("evaluate_on_test_set")
        return {
            "summary_table": "| Model | MSE |\n| --- | ---: |\n| FC | 0.1 |",
            "artifacts": {
                "frequency_mse_comparison": "assets/v2_high_freq/frequency_mse_comparison.png"
            },
        }

    def run_sensitivity_analysis(self):
        self.calls.append("run_sensitivity_analysis")
        return {"artifacts": {"sensitivity_mse": "assets/v2_high_freq/sensitivity_mse.png"}}

    def generate_report(self):
        self.calls.append("generate_report")
        return "# Signal Denoising Report"


def test_main_runs_sdk_workflow_via_public_interface(monkeypatch, capsys):
    """Verify the CLI entrypoint orchestrates the workflow exclusively through the SDK."""

    captured_instances: list[FakeSDK] = []

    def fake_sdk_factory(config_path: str):
        instance = FakeSDK(config_path)
        captured_instances.append(instance)
        return instance

    monkeypatch.setattr(main_module, "SignalDenoiserSDK", fake_sdk_factory)

    exit_code = main_module.main(["--models", "FC", "LSTM"])

    assert exit_code == 0
    assert len(captured_instances) == 1
    assert captured_instances[0].config_path == "src/shared/config.py"
    assert captured_instances[0].calls == [
        "prepare_data",
        ("run_training", "FC"),
        ("run_training", "LSTM"),
        "evaluate_on_test_set",
        "run_sensitivity_analysis",
        "generate_report",
    ]

    output = capsys.readouterr().out
    assert "| Model | MSE |" in output
    assert "frequency_mse_comparison.png" in output
    assert "sensitivity_mse" in output
    assert "# Signal Denoising Report" in output
