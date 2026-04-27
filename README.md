# ScoutRadar - Data Science Project (Phase 2)

ScoutRadar is an offline-runnable transfer-success data pipeline for football recruitment. This trimmed workspace keeps Kirollos's data and automation scope: validation, preprocessing, feature transformation, transfer-success target building, tests, Makefile workflows, and CI.

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
- `make test`
  Runs the full pytest suite with coverage.

## Modeling Summary

- Cohort: incoming transfers into Big Five clubs between `2018-07-01` and `2022-06-30`
- Target: `transfer_success`
- Success rule: at least `1,800` minutes for the destination club in the first `24` months after the transfer, plus end-of-window market value greater than or equal to pre-transfer market value
- Splitting: chronological `70/15/15` train/validation/test
- Feature processing: chronological `70/15/15` split, leakage-safe target exclusion, numeric scaling, and categorical one-hot encoding

## Folder Structure

- `data/` : raw inputs, cached API samples, and processed modeling datasets
- `docs/` : project documentation and PDFs
- `notebooks/` : Phase 1 validation notebook
- `src/` : pipelines for validation, transfer dataset construction, and feature engineering
- `tests/` : unit and integration tests for the retained data pipeline
