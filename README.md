# ScoutRadar - Data Science Project (Phase 2)

ScoutRadar is an offline-runnable transfer-success classification project for football recruitment. The Phase 2 pipeline builds a transfer-level modeling table, labels historical transfers as success/failure, trains five classifiers, logs experiments in MLflow, and produces EDA/model-comparison artifacts for the final report.

## Setup Instructions

This project uses `poetry` for dependency management.

1. Install Poetry if it is not already available.
2. Run `poetry install` in the project root.
3. Activate the environment with `poetry shell`, or prefix commands with `poetry run`.

## Phase 2 Commands

- `make transfer-dataset`
  Builds the transfer-level modeling dataset and saves:
  `data/processed/transfer_modeling_dataset.csv`
  `data/processed/transfer_labeled_cohort.csv`
  `data/processed/transfer_excluded_audit.csv`
  `data/processed/transfer_label_failures.csv`
- `make train-phase2`
  Trains and compares the five required classifiers:
  `DummyClassifier`
  `LogisticRegression`
  `RandomForestClassifier`
  `GradientBoostingClassifier`
  `SVC`
  Outputs are written under `models/generated/`.
- `make smoke`
  Runs the synthetic end-to-end pipeline test without requiring the local raw dataset or live API access.
- `make test`
  Runs the full pytest suite with coverage.

## Modeling Summary

- Cohort: incoming transfers into Big Five clubs between `2018-07-01` and `2022-06-30`
- Target: `transfer_success`
- Success rule: at least `1,800` minutes for the destination club in the first `24` months after the transfer, plus end-of-window market value greater than or equal to pre-transfer market value
- Splitting: chronological `70/15/15` train/validation/test
- Tracking: MLflow file-based tracking under `models/generated/mlruns` by default

## Notebook

The offline EDA and report dashboard notebook lives at `notebooks/phase2_dashboard.ipynb`. It loads the generated transfer dataset and model-comparison outputs, then renders the visualizations and comparison tables needed for the report.

## Folder Structure

- `data/` : raw inputs, cached API samples, and processed modeling datasets
- `docs/` : project documentation and PDFs
- `notebooks/` : exploratory analysis and report-facing dashboards
- `src/` : pipelines for validation, feature engineering, and model training
- `tests/` : unit and integration tests, including the synthetic smoke pipeline
- `models/` : generated model artifacts and MLflow runs
