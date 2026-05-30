import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from configs import CONFIGS, TrainingConfig
from preprocess import preprocess

ID_COLUMN = "PassengerId"
TARGET_COLUMN = "Survived"


@dataclass(frozen=True)
class PreparedData:
    x: pd.DataFrame
    y: pd.Series
    x_test: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM Titanic model and generate a SHAP summary plot."
    )
    parser.add_argument(
        "--config",
        default="Baseline",
        choices=sorted(CONFIGS),
        help="Training configuration name.",
    )
    return parser.parse_args()


def get_config(config_name: str) -> TrainingConfig:
    try:
        return CONFIGS[config_name]
    except KeyError as exc:
        available = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available}") from exc


def load_dataset(config: TrainingConfig) -> PreparedData:
    from sklearn.preprocessing import RobustScaler

    train = pd.read_csv(config.train_path)
    test = pd.read_csv(config.test_path)

    processed_train = preprocess(train, train)
    processed_test = preprocess(test, train)

    x = processed_train.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = processed_train[TARGET_COLUMN]
    x_test = processed_test.drop(columns=[ID_COLUMN])

    scaler = RobustScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index,
    )
    return PreparedData(x=x_scaled, y=y, x_test=x_test_scaled)


def train_model(config: TrainingConfig, x: np.ndarray, y: np.ndarray):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )
    lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train, free_raw_data=False)

    return lgb.train(
        config.lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "eval"],
        num_boost_round=config.num_boost_round,
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.early_stopping(config.early_stopping_rounds),
        ],
    )


def save_shap_summary(model, x_test: pd.DataFrame, output_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(model=model, data=x_test)
    shap_values = explainer.shap_values(x_test, check_additivity=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, features=x_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def run(config: TrainingConfig) -> Path:
    data = load_dataset(config)
    model = train_model(config, data.x, data.y)
    output_path = config.artifact_dir / "shap_summary.png"
    return save_shap_summary(model, data.x_test, output_path)


def main() -> None:
    args = parse_args()
    config = get_config(args.config)
    artifact_path = run(config)
    print(f"Saved SHAP summary plot to {artifact_path}")


if __name__ == "__main__":
    main()
