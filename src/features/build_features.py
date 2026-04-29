from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_TARGET_COLUMN = "transfer_success"
DEFAULT_SORT_COLUMN = "transfer_date"

NON_FEATURE_COLUMNS = {
    "transfer_key",
    "player_id",
    "player_full_name",
    "player_name",
    "source_club_id",
    "destination_club_id",
    "source_club_name",
    "destination_club_name",
    "source_club_dataset_name",
    "destination_club_dataset_name",
    "follow_up_window_end",
    "pre_transfer_market_value_date",
    "market_value_180d_prior_date",
    "market_value_365d_prior_date",
    "target_end_market_value_date",
    "target_failure_reason",
    "target_is_eligible",
    "target_minutes_window_start",
    "target_minutes_window_end",
}


def chronological_split(
    df: pd.DataFrame,
    *,
    sort_column: str = DEFAULT_SORT_COLUMN,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, pd.DataFrame]:
    if df.empty:
        raise ValueError("The modeling dataset is empty; nothing can be split.")
    if len(df) < 3:
        raise ValueError("At least three rows are required for a train/validation/test split.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave room for a test split.")

    ordered = df.sort_values(sort_column, kind="stable").reset_index(drop=True)
    total_rows = len(ordered)
    train_end = max(1, int(total_rows * train_ratio))
    val_end = max(train_end + 1, int(total_rows * (train_ratio + val_ratio)))
    val_end = min(val_end, total_rows - 1)

    if train_end >= val_end:
        val_end = train_end + 1
    if val_end >= total_rows:
        raise ValueError("The requested split ratios leave no rows for the test split.")

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    val_df = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test_df = ordered.iloc[val_end:].reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "split_summary": pd.DataFrame(
            [
                {
                    "split": "train",
                    "rows": len(train_df),
                    "start_date": train_df[sort_column].min(),
                    "end_date": train_df[sort_column].max(),
                },
                {
                    "split": "val",
                    "rows": len(val_df),
                    "start_date": val_df[sort_column].min(),
                    "end_date": val_df[sort_column].max(),
                },
                {
                    "split": "test",
                    "rows": len(test_df),
                    "start_date": test_df[sort_column].min(),
                    "end_date": test_df[sort_column].max(),
                },
            ]
        ),
    }


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET_COLUMN,
    extra_excluded: set[str] | None = None,
) -> list[str]:
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.add(target_col)
    if extra_excluded:
        excluded.update(extra_excluded)

    feature_columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        if column.startswith("target_"):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        if column.endswith("_date"):
            continue
        feature_columns.append(column)
    return feature_columns


def _build_preprocessor(train_df: pd.DataFrame, feature_columns: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(train_df[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    logger.info(
        "Building feature preprocessor with %s numeric and %s categorical columns.",
        len(numeric_columns),
        len(categorical_columns),
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns


def _transform_frame(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    clean_frame = frame[feature_columns].copy().replace({pd.NA: np.nan})
    transformed = preprocessor.transform(clean_frame)
    feature_names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(transformed, index=frame.index, columns=feature_names)


def transform_with_preprocessor_artifact(artifact: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Transform new rows with a fitted preprocessing artifact."""
    return _transform_frame(
        artifact["preprocessor"],
        frame,
        artifact["feature_columns"],
    )


def save_preprocessor_artifact(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Save the fitted preprocessing schema for reuse."""
    artifact = {
        "preprocessor": payload["preprocessor"],
        "feature_columns": payload["feature_columns"],
        "feature_names": payload["feature_names"],
        "numeric_columns": payload["numeric_columns"],
        "categorical_columns": payload["categorical_columns"],
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(artifact, file)
    return path


def load_preprocessor_artifact(input_path: str | Path) -> dict[str, Any]:
    """Load a fitted preprocessing schema saved by save_preprocessor_artifact."""
    with Path(input_path).open("rb") as file:
        return pickle.load(file)


def build_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    *,
    target_col: str = DEFAULT_TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    extra_excluded: set[str] | None = None,
) -> dict[str, Any]:
    if train_df.empty:
        raise ValueError("The training dataframe is empty.")

    selected_columns = feature_columns or infer_feature_columns(train_df, target_col=target_col, extra_excluded=extra_excluded)
    selected_columns = [column for column in selected_columns if not train_df[column].isna().all()]
    if not selected_columns:
        raise ValueError("No usable feature columns were found in the provided dataframe.")

    preprocessor, numeric_columns, categorical_columns = _build_preprocessor(train_df, selected_columns)
    preprocessor.fit(train_df[selected_columns].copy().replace({pd.NA: np.nan}))

    transformed = {
        "X_train": _transform_frame(preprocessor, train_df, selected_columns),
        "y_train": train_df[target_col].astype(int).reset_index(drop=True),
        "feature_columns": selected_columns,
        "feature_names": list(preprocessor.get_feature_names_out()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "preprocessor": preprocessor,
    }

    if val_df is not None:
        transformed["X_val"] = _transform_frame(preprocessor, val_df, selected_columns)
        transformed["y_val"] = val_df[target_col].astype(int).reset_index(drop=True)
    if test_df is not None:
        transformed["X_test"] = _transform_frame(preprocessor, test_df, selected_columns)
        transformed["y_test"] = test_df[target_col].astype(int).reset_index(drop=True)

    return transformed


def build_feature_splits(
    df: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET_COLUMN,
    sort_column: str = DEFAULT_SORT_COLUMN,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    feature_columns: list[str] | None = None,
    extra_excluded: set[str] | None = None,
) -> dict[str, Any]:
    splits = chronological_split(df, sort_column=sort_column, train_ratio=train_ratio, val_ratio=val_ratio)
    payload = build_features(
        splits["train"],
        splits["val"],
        splits["test"],
        target_col=target_col,
        feature_columns=feature_columns,
        extra_excluded=extra_excluded,
    )
    payload["train_df"] = splits["train"]
    payload["val_df"] = splits["val"]
    payload["test_df"] = splits["test"]
    payload["split_summary"] = splits["split_summary"]
    return payload
