from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "run_accuracy_checks": ".primary",
    "run_consistency_checks": ".primary",
    "run_completeness_checks": ".primary",
    "run_uniqueness_checks": ".primary",
    "run_outlier_checks": ".primary",
    "run_timeliness_checks": ".primary",
    "run_distribution_checks": ".primary",
    "run_relationship_checks": ".primary",
    "summarize_dimension_checks": ".primary",
    "fetch_api_football_fixtures": ".api",
    "fetch_api_football_big_five_fixtures": ".api",
    "run_api_fixture_validation": ".api",
    "build_api_report_snippets": ".api",
    "build_merged_fixture_dataset": ".merged",
    "run_merged_fixture_validation": ".merged",
    "build_merged_report_snippets": ".merged",
    "load_primary_tables": ".utils",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
