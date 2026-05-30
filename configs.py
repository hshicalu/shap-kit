from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    name: str = "baseline"
    seed: int = 2021
    train_path: Path = Path("input/train.csv")
    test_path: Path = Path("input/test.csv")
    output_dir: Path = Path("output")
    test_size: float = 0.2
    num_boost_round: int = 1000
    early_stopping_rounds: int = 10
    lgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "boosting_type": "gbdt",
            "max_depth": -1,
            "objective": "binary",
            "num_leaves": 64,
            "learning_rate": 0.05,
            "max_bin": 512,
            "subsample_for_bin": 200,
            "subsample": 1,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 5,
            "reg_lambda": 10,
            "min_split_gain": 0.5,
            "min_child_weight": 1,
            "min_child_samples": 5,
            "scale_pos_weight": 1,
            "metric": "binary_error",
            "verbosity": -1,
        }
    )

    @property
    def artifact_dir(self) -> Path:
        return self.output_dir / self.name


CONFIGS: dict[str, TrainingConfig] = {
    "Baseline": TrainingConfig(),
    "baseline": TrainingConfig(),
}
