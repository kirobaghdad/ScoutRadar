from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/processed/transfer_modeling_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_MISSINGNESS_THRESHOLD = 0.95
DEFAULT_OUTLIER_MULTIPLIER = 3

VALID_POSITIONS = {"Attack", "Midfield", "Defender", "Goalkeeper"}
PROTECTED_COLUMNS = {
    "transfer_key",
    "player_id",
    "transfer_date",
    "transfer_success",
}
NON_OUTLIER_COLUMNS = {
    "player_id",
    "source_club_id",
    "destination_club_id",
    "season_start_year",
    "transfer_success",
    "target_destination_matches_24m",
    "target_destination_minutes_24m",
    "target_destination_goals_24m",
    "target_destination_assists_24m",
}
FINANCIAL_COLUMNS = {
    "transfer_fee",
    "transfer_fee_for_model",
    "transfer_fee_log1p",
    "market_value_in_eur",
    "market_value_in_eur_log1p",
    "pre_transfer_market_value",
    "pre_transfer_market_value_log1p",
    "market_value_180d_prior",
    "market_value_365d_prior",
    "market_value_change_180d",
    "market_value_change_365d",
    "transfer_fee_to_market_value_ratio",
    "transfer_fee_minus_market_value",
    "target_end_market_value",
    "target_end_market_value_log1p",
    "target_market_value_delta_24m",
    "highest_market_value_in_eur",
    "highest_market_value_in_eur_log1p",
    "player_current_market_value",
    "player_current_market_value_log1p",
}
RATE_ZERO_FILL_MAP = {
    "player_goals_per90_180d_pre": "player_minutes_180d_pre",
    "player_assists_per90_180d_pre": "player_minutes_180d_pre",
    "player_goals_per90_365d_pre": "player_minutes_365d_pre",
    "player_assists_per90_365d_pre": "player_minutes_365d_pre",
}


def _append_reason(existing: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    updated = existing.fillna("").astype(str)
    updated.loc[mask] = updated.loc[mask].map(lambda value: reason if not value else f"{value};{reason}")
    return updated.replace("", pd.NA)


def _log_step(records: list[dict[str, Any]], *, step: str, action: str, affected_rows: int, details: str) -> None:
    records.append(
        {
            "step": step,
            "action": action,
            "affected_rows": int(affected_rows),
            "details": details,
        }
    )


def _standardize_types(df: pd.DataFrame, log_records: list[dict[str, Any]]) -> pd.DataFrame:
    cleaned = df.copy()
    date_columns = [
        column
        for column in cleaned.columns
        if "date" in column.lower() or column in {"follow_up_window_end"}
    ]
    for column in date_columns:
        before_nulls = int(cleaned[column].isna().sum())
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
        after_nulls = int(cleaned[column].isna().sum())
        _log_step(
            log_records,
            step="type_standardization",
            action=f"parsed_datetime:{column}",
            affected_rows=max(0, after_nulls - before_nulls),
            details="Parsed date-like column with pd.to_datetime(errors='coerce').",
        )
    return cleaned


def _strip_strings(df: pd.DataFrame, log_records: list[dict[str, Any]]) -> pd.DataFrame:
    cleaned = df.copy()
    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    changed_cells = 0
    for column in object_columns:
        original = cleaned[column].copy()
        cleaned[column] = cleaned[column].astype("string").str.strip()
        changed_cells += int((original.astype("string") != cleaned[column]).fillna(False).sum())
    _log_step(
        log_records,
        step="consistency",
        action="trim_whitespace",
        affected_rows=changed_cells,
        details="Trimmed leading and trailing whitespace from string columns.",
    )
    return cleaned


def _quarantine_invalid_rows(df: pd.DataFrame, log_records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["cleaning_rejection_reason"] = pd.Series(pd.NA, index=working.index, dtype="object")

    invalid_position = working["position"].notna() & ~working["position"].isin(VALID_POSITIONS)
    invalid_target = working["transfer_success"].isna() | ~working["transfer_success"].isin([0, 1])
    invalid_age = working["age_at_transfer"].notna() & ~working["age_at_transfer"].between(15, 45)
    missing_key = working["transfer_key"].isna() | working["transfer_key"].astype("string").str.strip().eq("")
    missing_date = working["transfer_date"].isna()
    duplicate_key = working["transfer_key"].duplicated(keep="first")

    non_negative_columns = [
        column
        for column in [
            "transfer_fee",
            "pre_transfer_market_value",
            "market_value_in_eur",
            "player_minutes_180d_pre",
            "player_minutes_365d_pre",
            "target_end_market_value",
            "source_squad_size",
            "destination_squad_size",
            "source_stadium_seats",
            "destination_stadium_seats",
        ]
        if column in working.columns
    ]

    negative_masks: list[pd.Series] = []
    for column in non_negative_columns:
        mask = working[column].notna() & working[column].lt(0)
        negative_masks.append(mask)
        working["cleaning_rejection_reason"] = _append_reason(
            working["cleaning_rejection_reason"],
            mask,
            f"negative_{column}",
        )

    working["cleaning_rejection_reason"] = _append_reason(working["cleaning_rejection_reason"], invalid_position, "invalid_position")
    working["cleaning_rejection_reason"] = _append_reason(working["cleaning_rejection_reason"], invalid_target, "invalid_target")
    working["cleaning_rejection_reason"] = _append_reason(working["cleaning_rejection_reason"], invalid_age, "invalid_age")
    working["cleaning_rejection_reason"] = _append_reason(working["cleaning_rejection_reason"], missing_key, "missing_transfer_key")
    working["cleaning_rejection_reason"] = _append_reason(working["cleaning_rejection_reason"], missing_date, "missing_transfer_date")
    working["cleaning_rejection_reason"] = _append_reason(
        working["cleaning_rejection_reason"],
        duplicate_key,
        "duplicate_transfer_key",
    )

    quarantine = working[working["cleaning_rejection_reason"].notna()].copy().reset_index(drop=True)
    cleaned = working[working["cleaning_rejection_reason"].isna()].drop(columns=["cleaning_rejection_reason"]).reset_index(drop=True)
    _log_step(
        log_records,
        step="accuracy",
        action="quarantine_invalid_rows",
        affected_rows=len(quarantine),
        details="Quarantined rows that violate critical business rules instead of forcing unsafe corrections.",
    )
    return cleaned, quarantine


def _apply_rule_based_corrections(df: pd.DataFrame, log_records: list[dict[str, Any]]) -> pd.DataFrame:
    cleaned = df.copy()
    corrected_rows = 0
    for rate_column, minutes_column in RATE_ZERO_FILL_MAP.items():
        if rate_column not in cleaned.columns or minutes_column not in cleaned.columns:
            continue
        mask = cleaned[rate_column].isna() & cleaned[minutes_column].fillna(0).eq(0)
        corrected_rows += int(mask.sum())
        cleaned.loc[mask, rate_column] = 0.0

    if "transfer_fee" in cleaned.columns:
        missing_mask = cleaned["transfer_fee"].isna()
        zero_mask = cleaned["transfer_fee"].fillna(0).eq(0)
        cleaned["transfer_fee_missing_flag"] = missing_mask.astype(int)
        cleaned["is_free_transfer"] = (~missing_mask & zero_mask).astype(int)
        cleaned["transfer_fee_for_model"] = cleaned["transfer_fee"].fillna(0.0)
        cleaned["transfer_fee_log1p"] = np.log1p(cleaned["transfer_fee_for_model"])
        corrected_rows += int(missing_mask.sum())
        corrected_rows += int((~missing_mask & zero_mask).sum())

    for column in [
        "pre_transfer_market_value",
        "market_value_in_eur",
        "target_end_market_value",
        "highest_market_value_in_eur",
        "player_current_market_value",
    ]:
        if column in cleaned.columns:
            cleaned[f"{column}_log1p"] = np.log1p(cleaned[column].clip(lower=0).fillna(0.0))

    _log_step(
        log_records,
        step="accuracy",
        action="rule_based_corrections",
        affected_rows=corrected_rows,
        details="Filled per-90 stats with 0 when prior minutes are 0 and created domain-aware transfer fee and market-value helper features.",
    )
    return cleaned


def _drop_sparse_columns(
    df: pd.DataFrame,
    log_records: list[dict[str, Any]],
    *,
    missingness_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    missing_pct = df.isna().mean()
    columns_to_drop = [
        column
        for column, ratio in missing_pct.items()
        if ratio > missingness_threshold and column not in PROTECTED_COLUMNS
    ]
    cleaned = df.drop(columns=columns_to_drop).copy()
    _log_step(
        log_records,
        step="completeness",
        action="drop_sparse_columns",
        affected_rows=len(columns_to_drop),
        details=f"Dropped columns with missingness greater than {missingness_threshold:.0%}: {columns_to_drop}",
    )
    return cleaned, columns_to_drop


def _impute_remaining_missing(df: pd.DataFrame, log_records: list[dict[str, Any]]) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    bool_cols = cleaned.select_dtypes(include=["bool"]).columns

    numeric_imputed = 0
    for column in numeric_cols:
        if column == "transfer_fee":
            continue
        if cleaned[column].isna().any():
            numeric_imputed += int(cleaned[column].isna().sum())
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    object_imputed = 0
    for column in object_cols:
        if cleaned[column].isna().any():
            object_imputed += int(cleaned[column].isna().sum())
            mode = cleaned[column].mode(dropna=True)
            cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    bool_imputed = 0
    for column in bool_cols:
        if cleaned[column].isna().any():
            bool_imputed += int(cleaned[column].isna().sum())
            mode = cleaned[column].mode(dropna=True)
            cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else False)

    _log_step(
        log_records,
        step="completeness",
        action="impute_missing_values",
        affected_rows=numeric_imputed + object_imputed + bool_imputed,
        details="Imputed remaining numeric values with medians and categorical values with mode.",
    )
    return cleaned


def _cap_outliers_iqr(
    df: pd.DataFrame,
    log_records: list[dict[str, Any]],
    *,
    multiplier: float,
) -> pd.DataFrame:
    cleaned = df.copy()
    capped_cells = 0
    for column in cleaned.select_dtypes(include=["number"]).columns:
        if (
            column in NON_OUTLIER_COLUMNS
            or column in FINANCIAL_COLUMNS
            or column.endswith("_matched")
            or "id" in column.lower()
        ):
            continue
        if cleaned[column].nunique(dropna=True) < 5:
            continue

        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = cleaned[column].lt(lower) | cleaned[column].gt(upper)
        capped_cells += int(mask.sum())
        cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)

    _log_step(
        log_records,
        step="outliers",
        action="cap_iqr_outliers",
        affected_rows=capped_cells,
        details=f"Capped continuous numeric outliers using IQR bounds with multiplier={multiplier}.",
    )
    return cleaned


def clean_transfer_modeling_dataframe(
    df: pd.DataFrame,
    *,
    missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
    outlier_multiplier: float = DEFAULT_OUTLIER_MULTIPLIER,
) -> dict[str, pd.DataFrame | list[str]]:
    log_records: list[dict[str, Any]] = []

    cleaned = df.copy()
    cleaned = _standardize_types(cleaned, log_records)
    cleaned = _strip_strings(cleaned, log_records)
    cleaned, quarantine = _quarantine_invalid_rows(cleaned, log_records)
    cleaned = _apply_rule_based_corrections(cleaned, log_records)
    cleaned, dropped_columns = _drop_sparse_columns(
        cleaned,
        log_records,
        missingness_threshold=missingness_threshold,
    )
    cleaned = _impute_remaining_missing(cleaned, log_records)
    cleaned = _cap_outliers_iqr(cleaned, log_records, multiplier=outlier_multiplier)

    cleaning_log = pd.DataFrame(log_records)
    return {
        "cleaned_dataset": cleaned.reset_index(drop=True),
        "quarantine_dataset": quarantine.reset_index(drop=True),
        "cleaning_log": cleaning_log,
        "dropped_columns": pd.DataFrame({"column": dropped_columns}),
    }


def clean_transfer_modeling_dataset(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
    outlier_multiplier: float = DEFAULT_OUTLIER_MULTIPLIER,
) -> dict[str, str]:
    path = Path(input_path)
    df = pd.read_csv(path, low_memory=False)
    outputs = clean_transfer_modeling_dataframe(
        df,
        missingness_threshold=missingness_threshold,
        outlier_multiplier=outlier_multiplier,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    file_map = {
        "cleaned_dataset": output_root / "transfer_modeling_dataset_clean.csv",
        "quarantine_dataset": output_root / "transfer_modeling_dataset_quarantine.csv",
        "cleaning_log": output_root / "transfer_modeling_dataset_cleaning_log.csv",
        "dropped_columns": output_root / "transfer_modeling_dataset_dropped_columns.csv",
    }

    saved: dict[str, str] = {}
    for key, out_path in file_map.items():
        frame = outputs[key]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(out_path, index=False)
        saved[key] = str(out_path)

    logger.info("Saved transfer modeling cleaning outputs: %s", json.dumps(saved, indent=2))
    return saved


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean the transfer-level modeling dataset.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--missingness-threshold", type=float, default=DEFAULT_MISSINGNESS_THRESHOLD)
    parser.add_argument("--outlier-multiplier", type=float, default=DEFAULT_OUTLIER_MULTIPLIER)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_transfer_modeling_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        missingness_threshold=args.missingness_threshold,
        outlier_multiplier=args.outlier_multiplier,
    )


if __name__ == "__main__":
    main()
