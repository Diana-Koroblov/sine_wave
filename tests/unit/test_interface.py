from src.sdk.interface import SignalDenoiserSDK


def test_sdk_initialization():
    """Verify SDK correctly stores config path."""
    sdk = SignalDenoiserSDK(config_path="custom_config.py")
    assert sdk.config_path == "custom_config.py"


def test_sdk_stubs():
    """Verify SDK public interface stubs exist and return expected types."""
    sdk = SignalDenoiserSDK()

    # These are currently stubs, verify they don't crash
    assert sdk.prepare_data() is None
    assert isinstance(sdk.run_training("FC"), dict)
    assert isinstance(sdk.generate_report(), str)
