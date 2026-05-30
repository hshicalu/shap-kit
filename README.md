# shap-kit

A compact example of training a LightGBM model on the Titanic dataset and generating a
SHAP summary plot.

## Setup

This project uses `uv` for dependency management.

```bash
uv sync --dev
```

Place Kaggle Titanic CSV files under `input/`:

```text
input/train.csv
input/test.csv
```

## Run

```bash
uv run python main.py --config Baseline
```

The SHAP bar summary is written to:

```text
output/baseline/shap_summary.png
```

## Test

```bash
uv run pytest
uv run ruff check .
```
