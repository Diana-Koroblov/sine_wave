import argparse

from src.sdk.interface import SignalDenoiserSDK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the SDK-driven experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run the sine-wave denoising workflow through the public SDK interface."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["FC", "RNN", "LSTM"],
        type=str.upper,
        choices=("FC", "RNN", "LSTM"),
        help="Model families to train before evaluation.",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip the sensitivity sweep and its exported plots.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip markdown report generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sdk = SignalDenoiserSDK(config_path="src/shared/config.py")

    sdk.prepare_data()
    for model_type in dict.fromkeys(args.models):
        sdk.run_training(model_type=model_type)

    evaluation = sdk.evaluate_on_test_set()
    print(evaluation["summary_table"])
    if evaluation["artifacts"]:
        print(f"Frequency comparison: {evaluation['artifacts']['frequency_mse_comparison']}")

    if not args.skip_sensitivity:
        sensitivity = sdk.run_sensitivity_analysis()
        print(f"Sensitivity artifacts: {sensitivity['artifacts']}")

    if not args.skip_report:
        print(sdk.generate_report())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
