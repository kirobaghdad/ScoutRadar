from __future__ import annotations

import contextlib
import io
import json
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import great_expectations as gx
except ModuleNotFoundError:  # pragma: no cover
    gx = None



from .config import *

def load_primary_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_path = Path(raw_dir).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")

    tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(raw_path.glob("*.csv")):
        tables[csv_path.stem] = pd.read_csv(csv_path, low_memory=False)

    if not tables:
        raise FileNotFoundError(f"No CSV files found in raw data directory: {raw_path}")

    return tables


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _stringify(series: pd.Series) -> pd.Series:
    return series.map(lambda value: None if pd.isna(value) else str(value))


def _stringify_with_one_decimal(series: pd.Series) -> pd.Series:
    return series.map(lambda value: None if pd.isna(value) else f"{float(value):.1f}")


def _short_year_to_full(value: Any) -> int | None:
    if pd.isna(value):
        return None
    year = int(value)
    return 1900 + year if year >= 90 else 2000 + year


def _require_great_expectations() -> Any:
    if gx is None:
        raise ModuleNotFoundError(
            "great_expectations is required for the validation workflow. "
            "Install it in the project .conda environment before running these checks."
        )
    return gx


def _gx_validator(frame: pd.DataFrame, suite_name: str) -> Any:
    _require_great_expectations()
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name=f"{suite_name}_datasource")
    asset = datasource.add_dataframe_asset(name=f"{suite_name}_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe(name="batch")
    batch_request = batch_definition.build_batch_request(batch_parameters={"dataframe": frame})
    return context.get_validator(batch_request=batch_request, create_expectation_suite_with_name=f"{suite_name}_suite")


def _run_expectation(validator: Any, expectation_type: str, **kwargs: Any) -> Any:
    expectation = getattr(validator, expectation_type)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="great_expectations")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return expectation(result_format="SUMMARY", **kwargs)


def _check_row(table_name: str, dimension: str, check_name: str, description: str, expectation_type: str, result: Any) -> dict[str, Any]:
    payload = result.result or {}
    unexpected = payload.get("partial_unexpected_list") or []
    applicable_rows = int(payload.get("element_count", 0) or 0)
    failed_rows = int(payload.get("unexpected_count", 0) or 0)
    return {
        "dimension": dimension,
        "table_name": table_name,
        "check_name": check_name,
        "description": description,
        "expectation_type": expectation_type,
        "status": "passed" if result.success else "failed",
        "applicable_rows": applicable_rows,
        "failed_rows": failed_rows,
        "failed_pct": round((failed_rows / max(applicable_rows, 1)) * 100, 4),
        "missing_columns": "",
        "sample_unexpected_values": json.dumps(unexpected[:5], ensure_ascii=False, default=str),
    }


def _manual_check_row(
    table_name: str,
    dimension: str,
    check_name: str,
    description: str,
    applicable_rows: int,
    failed_rows: int,
    sample_unexpected_values: list[Any] | None = None,
    expectation_type: str = "pandas_rule",
) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "table_name": table_name,
        "check_name": check_name,
        "description": description,
        "expectation_type": expectation_type,
        "status": "passed" if failed_rows == 0 else "failed",
        "applicable_rows": int(applicable_rows),
        "failed_rows": int(failed_rows),
        "failed_pct": round((int(failed_rows) / max(int(applicable_rows), 1)) * 100, 4),
        "missing_columns": "",
        "sample_unexpected_values": json.dumps((sample_unexpected_values or [])[:5], ensure_ascii=False, default=str),
    }


def _skipped_row(
    table_name: str,
    dimension: str,
    check_name: str,
    description: str,
    expectation_type: str,
    frame: pd.DataFrame,
    missing_columns: list[str],
) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "table_name": table_name,
        "check_name": check_name,
        "description": description,
        "expectation_type": expectation_type,
        "status": "skipped",
        "applicable_rows": int(len(frame)),
        "failed_rows": None,
        "failed_pct": None,
        "missing_columns": ", ".join(missing_columns),
        "sample_unexpected_values": "[]",
    }


def _run_table_checks(table_name: str, frame: pd.DataFrame, dimension: str, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    validator = _gx_validator(frame, f"{dimension}_{table_name}")
    rows: list[dict[str, Any]] = []
    for check in checks:
        missing_columns = [column for column in check["columns"] if column not in frame.columns]
        if missing_columns:
            rows.append(
                _skipped_row(
                    table_name,
                    dimension,
                    check["check_name"],
                    check["description"],
                    check["expectation_type"],
                    frame,
                    missing_columns,
                )
            )
            continue

        result = _run_expectation(validator, check["expectation_type"], **check["kwargs"])
        rows.append(_check_row(table_name, dimension, check["check_name"], check["description"], check["expectation_type"], result))

    return rows


def _finalize_dimension_checks(rows: list[dict[str, Any]], dimension: str) -> pd.DataFrame:
    if not rows:
        rows = [
            {
                "dimension": dimension,
                "table_name": "n/a",
                "check_name": "no_checks_ran",
                "description": "No matching tables were found.",
                "expectation_type": "n/a",
                "status": "skipped",
                "applicable_rows": 0,
                "failed_rows": None,
                "failed_pct": None,
                "missing_columns": "",
                "sample_unexpected_values": "[]",
            }
        ]

    return pd.DataFrame(rows).sort_values(
        ["status", "failed_pct", "table_name", "check_name"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

__all__ = [k for k in globals().keys() if not k.startswith('__') and k not in ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']]
