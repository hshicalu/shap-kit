import pytest

from configs import TrainingConfig
from main import get_config


def test_get_config_returns_registered_config() -> None:
    config = get_config("Baseline")

    assert isinstance(config, TrainingConfig)
    assert config.name == "baseline"


def test_get_config_rejects_unknown_config() -> None:
    with pytest.raises(ValueError, match="Unknown config"):
        get_config("unknown")
