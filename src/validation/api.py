from __future__ import annotations

import contextlib
import io
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import warnings
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import great_expectations as gx
except ModuleNotFoundError:  # pragma: no cover
    gx = None



from .config import *
from .utils import *
from .primary import *

def _normalize_api_errors(errors: Any) -> list[str]:
    if errors in (None, "", [], {}):
        return []
    if isinstance(errors, dict):
        normalized = []
        for key, value in errors.items():
            if value in (None, "", [], {}):
                continue
            normalized.append(f"{key}: {value}")
        return normalized
    if isinstance(errors, list):
        normalized = []
        for value in errors:
            text = str(value).strip()
            if text:
                normalized.append(text)
        return normalized
    text = str(errors).strip()
    return [text] if text else []


def _normalize_api_parameters(parameters: Any) -> dict[str, str]:
    if parameters is None:
        return {}
    if isinstance(parameters, dict):
        return {str(key): str(value) for key, value in parameters.items()}
    if isinstance(parameters, list):
        normalized: dict[str, str] = {}
        for item in parameters:
            if isinstance(item, dict):
                for key, value in item.items():
                    normalized[str(key)] = str(value)
        return normalized
    return {}


def _missing_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        mask = mask | series.fillna("").astype(str).str.strip().eq("")
    return mask


def _header_lookup(headers: dict[str, Any]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _sample_records(frame: pd.DataFrame, mask: pd.Series, columns: list[str], limit: int = 5) -> list[dict[str, Any]]:
    if frame.empty or not mask.any():
        return []
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return []
    return frame.loc[mask, available_columns].head(limit).to_dict("records")


def load_api_key(key_path: str | Path = "key_api.txt") -> str:
    key_file = Path(key_path).expanduser().resolve()
    if not key_file.exists():
        raise FileNotFoundError(f"API key file was not found: {key_file}")

    raw_text = key_file.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"API key file is empty: {key_file}")

    candidates = re.findall(r"[A-Za-z0-9_-]{16,}", raw_text)
    if not candidates:
        raise ValueError(
            f"Could not parse an API key from {key_file}. "
            "Store the plain key value or include it in a simple key=value line."
        )
    return candidates[-1]


def _build_fetch_result(
    request_params: dict[str, Any],
    request_url: str,
    status_code: int,
    elapsed_seconds: float,
    headers: dict[str, Any],
    payload: dict[str, Any],
    source: str,
    cache_path: str | None = None,
) -> dict[str, Any]:
    return {
        "url": API_FOOTBALL_FIXTURES_URL,
        "request_url": request_url,
        "request_params": request_params,
        "status_code": int(status_code),
        "elapsed_seconds": round(float(elapsed_seconds), 4),
        "headers": headers,
        "payload": payload,
        "get": payload.get("get"),
        "parameters": payload.get("parameters"),
        "errors": payload.get("errors"),
        "results": payload.get("results"),
        "paging": payload.get("paging"),
        "response": payload.get("response"),
        "source": source,
        "cache_path": cache_path,
    }


def _save_api_fixture_cache(fetch_result: dict[str, Any], cache_path: str | Path) -> None:
    cache_file = Path(cache_path).expanduser().resolve()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(fetch_result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_api_fixture_cache(
    cache_path: str | Path,
    request_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache_file = Path(cache_path).expanduser().resolve()
    if not cache_file.exists():
        raise FileNotFoundError(f"API fixture cache file was not found: {cache_file}")

    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    if isinstance(cached, dict) and "payload" in cached:
        payload = cached.get("payload") or {}
        return _build_fetch_result(
            request_params=cached.get("request_params") or request_params or dict(DEFAULT_API_FOOTBALL_FIXTURE_PARAMS),
            request_url=cached.get("request_url") or API_FOOTBALL_FIXTURES_URL,
            status_code=int(cached.get("status_code", 200)),
            elapsed_seconds=float(cached.get("elapsed_seconds", 0.0)),
            headers=cached.get("headers") or {},
            payload=payload,
            source="cache_file",
            cache_path=str(cache_file),
        )

    if isinstance(cached, dict):
        payload = cached
        normalized_params = request_params or dict(DEFAULT_API_FOOTBALL_FIXTURE_PARAMS)
        query_string = urllib.parse.urlencode({key: value for key, value in normalized_params.items() if value is not None})
        request_url = f"{API_FOOTBALL_FIXTURES_URL}?{query_string}"
        return _build_fetch_result(
            request_params=normalized_params,
            request_url=request_url,
            status_code=200,
            elapsed_seconds=0.0,
            headers={},
            payload=payload,
            source="cache_file",
            cache_path=str(cache_file),
        )

    raise ValueError(
        f"Cache file {cache_file} must contain either the saved fetch result dictionary or the raw API JSON payload."
    )


def fetch_api_football_fixtures(
    params: dict[str, Any] | None = None,
    key_path: str | Path = "key_api.txt",
    timeout: int = 30,
    cache_path: str | Path | None = None,
    use_cache_on_failure: bool = True,
    save_cache: bool = True,
) -> dict[str, Any]:
    request_params = {**DEFAULT_API_FOOTBALL_FIXTURE_PARAMS, **(params or {})}
    api_key = load_api_key(key_path)
    query_string = urllib.parse.urlencode({key: value for key, value in request_params.items() if value is not None})
    request_url = f"{API_FOOTBALL_FIXTURES_URL}?{query_string}"
    request = urllib.request.Request(
        request_url,
        headers={
            "x-apisports-key": api_key,
            "Accept": "application/json",
        },
        method="GET",
    )

    started_at = time.perf_counter()
    status_code: int | None = None
    response_headers: dict[str, Any] = {}
    response_body = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(response.getcode())
            response_headers = dict(response.info().items())
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        response_headers = dict(exc.headers.items()) if exc.headers is not None else {}
        response_body = exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        if cache_path is not None and use_cache_on_failure and Path(cache_path).expanduser().resolve().exists():
            return _load_api_fixture_cache(cache_path, request_params=request_params)
        reason = exc.reason if getattr(exc, "reason", None) is not None else exc
        raise ConnectionError(
            "Could not reach API-Football. "
            f"Network error: {reason}. "
            f"If this environment blocks outbound requests, save one `/fixtures` JSON response to `{cache_path or DEFAULT_API_FOOTBALL_CACHE_PATH}` and rerun."
        ) from exc

    elapsed_seconds = round(time.perf_counter() - started_at, 4)

    try:
        payload = json.loads(response_body) if response_body else {}
    except json.JSONDecodeError as exc:
        raise ValueError("API-Football returned a non-JSON response.") from exc

    fetch_result = _build_fetch_result(
        request_params=request_params,
        request_url=request_url,
        status_code=int(status_code or 0),
        elapsed_seconds=elapsed_seconds,
        headers=response_headers,
        payload=payload,
        source="live_api",
        cache_path=str(Path(cache_path).expanduser().resolve()) if cache_path is not None else None,
    )
    if cache_path is not None and save_cache and int(fetch_result["status_code"]) == 200:
        _save_api_fixture_cache(fetch_result, cache_path)
    return fetch_result


def flatten_fixture_response(payload: dict[str, Any]) -> pd.DataFrame:
    response_rows = payload.get("response")
    if not isinstance(response_rows, list):
        response_rows = []

    records: list[dict[str, Any]] = []
    for item in response_rows:
        fixture = item.get("fixture") or {}
        fixture_status = fixture.get("status") or {}
        league = item.get("league") or {}
        teams = item.get("teams") or {}
        home_team = teams.get("home") or {}
        away_team = teams.get("away") or {}
        goals = item.get("goals") or {}

        records.append(
            {
                "fixture_id": fixture.get("id"),
                "fixture_date": fixture.get("date"),
                "fixture_timestamp": fixture.get("timestamp"),
                "status_short": fixture_status.get("short"),
                "status_long": fixture_status.get("long"),
                "status_elapsed": fixture_status.get("elapsed"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "league_country": league.get("country"),
                "season": league.get("season"),
                "home_team_id": home_team.get("id"),
                "home_team_name": home_team.get("name"),
                "home_winner": home_team.get("winner"),
                "away_team_id": away_team.get("id"),
                "away_team_name": away_team.get("name"),
                "away_winner": away_team.get("winner"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
            }
        )

    columns = [
        "fixture_id",
        "fixture_date",
        "fixture_timestamp",
        "status_short",
        "status_long",
        "status_elapsed",
        "league_id",
        "league_name",
        "league_country",
        "season",
        "home_team_id",
        "home_team_name",
        "home_winner",
        "away_team_id",
        "away_team_name",
        "away_winner",
        "home_goals",
        "away_goals",
    ]
    return pd.DataFrame(records, columns=columns)


def _api_request_metadata(fetch_result: dict[str, Any], fixtures: pd.DataFrame) -> pd.DataFrame:
    payload = fetch_result["payload"]
    normalized_headers = _header_lookup(fetch_result.get("headers", {}))
    paging = payload.get("paging") if isinstance(payload.get("paging"), dict) else {}
    response_rows = payload.get("response") if isinstance(payload.get("response"), list) else []
    return pd.DataFrame(
        [
            {
                "endpoint": fetch_result["url"],
                "source": fetch_result.get("source", "live_api"),
                "cache_path": fetch_result.get("cache_path"),
                "status_code": fetch_result["status_code"],
                "elapsed_seconds": fetch_result["elapsed_seconds"],
                "request_url": fetch_result["request_url"],
                "request_params": json.dumps(fetch_result["request_params"], ensure_ascii=False, default=str),
                "echoed_parameters": json.dumps(_normalize_api_parameters(payload.get("parameters")), ensure_ascii=False, default=str),
                "results": payload.get("results"),
                "response_count": int(len(response_rows)),
                "paging_current": paging.get("current"),
                "paging_total": paging.get("total"),
                "daily_requests_remaining": normalized_headers.get("x-ratelimit-requests-remaining"),
                "minute_requests_remaining": normalized_headers.get("x-ratelimit-remaining"),
                "fixture_rows": int(len(fixtures)),
            }
        ]
    )


def _run_api_accuracy_checks(fetch_result: dict[str, Any], fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    payload = fetch_result["payload"]
    echoed_parameters = _normalize_api_parameters(payload.get("parameters"))
    response_rows = payload.get("response") if isinstance(payload.get("response"), list) else []
    errors = _normalize_api_errors(payload.get("errors"))
    request_season = str(fetch_result["request_params"].get("season"))
    rows: list[dict[str, Any]] = []

    rows.append(
        _manual_check_row(
            "api_wrapper",
            "accuracy",
            "http_status_is_200",
            "The fixtures endpoint should return HTTP 200.",
            1,
            0 if int(fetch_result["status_code"]) == 200 else 1,
            [{"status_code": fetch_result["status_code"]}],
        )
    )
    rows.append(
        _manual_check_row(
            "api_wrapper",
            "accuracy",
            "errors_array_is_empty",
            "The API wrapper should not report request errors.",
            1,
            0 if not errors else 1,
            errors,
        )
    )

    get_value = str(payload.get("get", "")).strip().strip("/")
    rows.append(
        _manual_check_row(
            "api_wrapper",
            "accuracy",
            "get_field_is_fixtures",
            "The wrapper get field should identify the fixtures endpoint.",
            1,
            0 if get_value == "fixtures" else 1,
            [{"get": payload.get("get")}],
        )
    )

    results_value = pd.to_numeric(pd.Series([payload.get("results")]), errors="coerce").iloc[0]
    rows.append(
        _manual_check_row(
            "api_wrapper",
            "accuracy",
            "results_matches_response_count",
            "The wrapper results count should match the number of returned records.",
            1,
            0 if pd.notna(results_value) and int(results_value) == len(response_rows) else 1,
            [{"results": payload.get("results"), "response_count": len(response_rows)}],
        )
    )

    rows.append(
        _manual_check_row(
            "api_wrapper",
            "accuracy",
            "season_parameter_echoes_request",
            "The echoed season parameter should match the requested season.",
            1,
            0 if echoed_parameters.get("season") == request_season else 1,
            [{"request_season": request_season, "echoed_season": echoed_parameters.get("season")}],
        )
    )

    season_values = pd.to_numeric(fixtures["season"], errors="coerce")
    season_mismatch_mask = season_values.ne(pd.to_numeric(pd.Series([request_season]), errors="coerce").iloc[0]) | season_values.isna()
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "accuracy",
            "fixture_season_matches_request",
            "Each returned fixture should belong to the requested season.",
            len(fixtures),
            int(season_mismatch_mask.sum()),
            _sample_records(fixtures, season_mismatch_mask, ["fixture_id", "season", "league_name"]),
        )
    )

    fixture_id_mask = pd.to_numeric(fixtures["fixture_id"], errors="coerce").le(0) | pd.to_numeric(fixtures["fixture_id"], errors="coerce").isna()
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "accuracy",
            "fixture_ids_are_positive",
            "Fixture IDs should be positive integers.",
            len(fixtures),
            int(fixture_id_mask.sum()),
            _sample_records(fixtures, fixture_id_mask, ["fixture_id", "league_name", "home_team_name", "away_team_name"]),
        )
    )

    league_id_values = pd.to_numeric(fixtures["league_id"], errors="coerce")
    league_id_mask = league_id_values.le(0) | league_id_values.isna()
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "accuracy",
            "league_ids_are_positive",
            "League IDs should be positive integers.",
            len(fixtures),
            int(league_id_mask.sum()),
            _sample_records(fixtures, league_id_mask, ["fixture_id", "league_id", "league_name"]),
        )
    )

    home_team_ids = pd.to_numeric(fixtures["home_team_id"], errors="coerce")
    away_team_ids = pd.to_numeric(fixtures["away_team_id"], errors="coerce")
    team_id_mask = home_team_ids.le(0) | away_team_ids.le(0) | home_team_ids.isna() | away_team_ids.isna()
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "accuracy",
            "team_ids_are_positive",
            "Home and away team IDs should be positive integers.",
            len(fixtures),
            int(team_id_mask.sum()),
            _sample_records(fixtures, team_id_mask, ["fixture_id", "home_team_id", "away_team_id", "home_team_name", "away_team_name"]),
        )
    )

    home_goals = pd.to_numeric(fixtures["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(fixtures["away_goals"], errors="coerce")
    negative_goals_mask = home_goals.lt(0) | away_goals.lt(0)
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "accuracy",
            "goals_are_non_negative",
            "Reported goals should be non-negative.",
            len(fixtures),
            int(negative_goals_mask.sum()),
            _sample_records(fixtures, negative_goals_mask, ["fixture_id", "home_goals", "away_goals", "home_team_name", "away_team_name"]),
        )
    )

    checks = _finalize_dimension_checks(rows, "accuracy")
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
    }


def _run_api_consistency_checks(fetch_result: dict[str, Any], fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    payload = fetch_result["payload"]
    echoed_parameters = _normalize_api_parameters(payload.get("parameters"))
    request_params = {str(key): str(value) for key, value in fetch_result["request_params"].items()}
    rows: list[dict[str, Any]] = []

    mismatched_parameters = [
        {
            "parameter": key,
            "requested": expected_value,
            "echoed": echoed_parameters.get(key),
        }
        for key, expected_value in request_params.items()
        if echoed_parameters.get(key) != expected_value
    ]
    rows.append(
        _manual_check_row(
            "api_wrapper",
            "consistency",
            "echoed_parameters_match_request",
            "The wrapper should echo the submitted parameters consistently.",
            len(request_params),
            len(mismatched_parameters),
            mismatched_parameters,
        )
    )

    parsed_dates = pd.to_datetime(fixtures["fixture_date"], errors="coerce", utc=True)
    invalid_date_mask = parsed_dates.isna()
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "consistency",
            "fixture_date_is_parseable",
            "Each fixture date should parse cleanly into a timestamp.",
            len(fixtures),
            int(invalid_date_mask.sum()),
            _sample_records(fixtures, invalid_date_mask, ["fixture_id", "fixture_date", "status_short"]),
        )
    )

    parsed_timestamps = pd.to_datetime(pd.to_numeric(fixtures["fixture_timestamp"], errors="coerce"), unit="s", errors="coerce", utc=True)
    comparable_mask = parsed_dates.notna() & parsed_timestamps.notna()
    timestamp_mismatch_mask = comparable_mask & (parsed_dates.sub(parsed_timestamps).abs().dt.total_seconds().gt(60))
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "consistency",
            "fixture_timestamp_matches_fixture_date",
            "The ISO fixture date and Unix fixture timestamp should describe the same moment.",
            int(comparable_mask.sum()),
            int(timestamp_mismatch_mask.sum()),
            _sample_records(fixtures, timestamp_mismatch_mask, ["fixture_id", "fixture_date", "fixture_timestamp"]),
        )
    )

    status_short_missing = _missing_mask(fixtures["status_short"])
    status_long_missing = _missing_mask(fixtures["status_long"])
    inconsistent_status_mask = status_short_missing ^ status_long_missing
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "consistency",
            "status_fields_are_populated_together",
            "status_short and status_long should be populated together.",
            len(fixtures),
            int(inconsistent_status_mask.sum()),
            _sample_records(fixtures, inconsistent_status_mask, ["fixture_id", "status_short", "status_long"]),
        )
    )

    team_structure_mask = (
        _missing_mask(fixtures["home_team_id"])
        | _missing_mask(fixtures["home_team_name"])
        | _missing_mask(fixtures["away_team_id"])
        | _missing_mask(fixtures["away_team_name"])
    )
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "consistency",
            "home_and_away_team_structure_is_complete",
            "Each fixture should consistently include home and away team identifiers and names.",
            len(fixtures),
            int(team_structure_mask.sum()),
            _sample_records(fixtures, team_structure_mask, ["fixture_id", "home_team_id", "home_team_name", "away_team_id", "away_team_name"]),
        )
    )

    checks = _finalize_dimension_checks(rows, "consistency")
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
    }


def _run_api_completeness_checks(payload: dict[str, Any], fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    for field_name in API_REQUIRED_WRAPPER_FIELDS:
        rows.append(
            _manual_check_row(
                "api_wrapper",
                "completeness",
                f"wrapper_field_{field_name}_present",
                f"The wrapper should include the {field_name} field.",
                1,
                0 if field_name in payload else 1,
                [] if field_name in payload else [{"missing_field": field_name}],
            )
        )

    response_rows = payload.get("response") if isinstance(payload.get("response"), list) else []
    rows.append(
        _manual_check_row(
            "api_wrapper",
            "completeness",
            "response_contains_rows",
            "The response array should not be empty for the validation sample.",
            1,
            0 if len(response_rows) > 0 else 1,
            [{"response_count": len(response_rows)}],
        )
    )

    required_fields = set(API_REQUIRED_FIXTURE_FIELDS)
    for column in fixtures.columns:
        missing_count = int(_missing_mask(fixtures[column]).sum())
        field_rows.append(
            {
                "table_name": "api_fixtures",
                "column_name": column,
                "field_role": "required" if column in required_fields else "optional",
                "missing_count": missing_count,
                "missing_pct": round((missing_count / max(len(fixtures), 1)) * 100, 4),
            }
        )

    for column in API_REQUIRED_FIXTURE_FIELDS:
        if column not in fixtures.columns:
            rows.append(
                _manual_check_row(
                    "api_fixtures",
                    "completeness",
                    f"{column}_not_missing",
                    f"{column} should be present and populated.",
                    len(fixtures),
                    len(fixtures),
                    [{"missing_column": column}],
                )
            )
            continue
        missing_mask = _missing_mask(fixtures[column])
        rows.append(
            _manual_check_row(
                "api_fixtures",
                "completeness",
                f"{column}_not_missing",
                f"{column} should be present and populated.",
                len(fixtures),
                int(missing_mask.sum()),
                _sample_records(fixtures, missing_mask, ["fixture_id", column, "home_team_name", "away_team_name"]),
            )
        )

    checks = _finalize_dimension_checks(rows, "completeness")
    field_missingness = (
        pd.DataFrame(field_rows)
        .sort_values(["field_role", "missing_pct", "column_name"], ascending=[True, False, True])
        .reset_index(drop=True)
        if field_rows
        else _empty_dataframe(["table_name", "column_name", "field_role", "missing_count", "missing_pct"])
    )
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
        "field_missingness": field_missingness,
    }


def _run_api_uniqueness_checks(fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    exact_duplicate_mask = fixtures.duplicated(keep=False)
    fixture_id_duplicate_mask = fixtures["fixture_id"].duplicated(keep=False) & fixtures["fixture_id"].notna()
    business_key_duplicate_mask = fixture_id_duplicate_mask.copy()

    checks = _finalize_dimension_checks(
        [
            _manual_check_row(
                "api_fixtures",
                "uniqueness",
                "no_exact_duplicate_rows",
                "The flattened fixture sample should not contain exact duplicate rows.",
                len(fixtures),
                int(fixtures.duplicated().sum()),
                _sample_records(fixtures, exact_duplicate_mask, ["fixture_id", "fixture_date", "home_team_name", "away_team_name"]),
            ),
            _manual_check_row(
                "api_fixtures",
                "uniqueness",
                "fixture_id_is_unique",
                "Each fixture_id should appear only once.",
                len(fixtures),
                int(fixture_id_duplicate_mask.sum()),
                _sample_records(fixtures, fixture_id_duplicate_mask, ["fixture_id", "fixture_date", "home_team_name", "away_team_name"]),
            ),
            _manual_check_row(
                "api_fixtures",
                "uniqueness",
                "fixture_business_key_is_unique",
                "The fixture_id business key should not repeat.",
                len(fixtures),
                int(business_key_duplicate_mask.sum()),
                _sample_records(fixtures, business_key_duplicate_mask, ["fixture_id", "league_name", "home_team_name", "away_team_name"]),
            ),
        ],
        "uniqueness",
    )

    summary = pd.DataFrame(
        [
            {
                "table_name": "api_fixtures",
                "rows": int(len(fixtures)),
                "exact_duplicate_rows": int(fixtures.duplicated().sum()),
                "duplicate_fixture_ids": int(fixture_id_duplicate_mask.sum()),
                "duplicate_business_keys": int(business_key_duplicate_mask.sum()),
            }
        ]
    )
    duplicate_rows = (
        fixtures.loc[exact_duplicate_mask | fixture_id_duplicate_mask]
        .sort_values(["fixture_id", "fixture_date"], kind="stable")
        .reset_index(drop=True)
        if (exact_duplicate_mask | fixture_id_duplicate_mask).any()
        else _empty_dataframe(list(fixtures.columns))
    )
    return {
        "checks": checks,
        "summary": summary,
        "duplicate_rows": duplicate_rows,
    }


def _run_api_outlier_checks(fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    numeric_columns = ["home_goals", "away_goals"]
    if "status_elapsed" in fixtures.columns and pd.to_numeric(fixtures["status_elapsed"], errors="coerce").notna().any():
        numeric_columns.append("status_elapsed")

    labels = fixtures["home_team_name"].fillna("").astype(str) + " vs " + fixtures["away_team_name"].fillna("").astype(str)

    for column in numeric_columns:
        values = pd.to_numeric(fixtures[column], errors="coerce")
        clean_values = values.dropna()
        if len(clean_values) < 4:
            continue

        q1 = clean_values.quantile(0.25)
        q3 = clean_values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = values.lt(lower_bound) | values.gt(upper_bound)
        outlier_count = int(outlier_mask.sum())

        summary_rows.append(
            {
                "table_name": "api_fixtures",
                "column_name": column,
                "non_null_count": int(clean_values.shape[0]),
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4),
                "outlier_count": outlier_count,
                "outlier_pct": round((outlier_count / max(len(clean_values), 1)) * 100, 4),
            }
        )

        if outlier_count == 0:
            continue

        outlier_rows = fixtures.loc[outlier_mask, ["fixture_id", column]].copy()
        outlier_rows["table_name"] = "api_fixtures"
        outlier_rows["column_name"] = column
        outlier_rows["value"] = pd.to_numeric(outlier_rows[column], errors="coerce")
        outlier_rows["label"] = labels.loc[outlier_mask].values
        example_rows.extend(
            outlier_rows.sort_values("value", ascending=False)[["table_name", "column_name", "fixture_id", "label", "value"]]
            .head(10)
            .to_dict("records")
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["outlier_count", "column_name"], ascending=[False, True])
        .reset_index(drop=True)
        if summary_rows
        else _empty_dataframe(
            ["table_name", "column_name", "non_null_count", "q1", "q3", "iqr", "lower_bound", "upper_bound", "outlier_count", "outlier_pct"]
        )
    )
    examples = (
        pd.DataFrame(example_rows)
        .sort_values(["column_name", "value"], ascending=[True, False])
        .reset_index(drop=True)
        if example_rows
        else _empty_dataframe(["table_name", "column_name", "fixture_id", "label", "value"])
    )
    return {
        "summary": summary,
        "examples": examples,
    }


def _run_api_timeliness_checks(fetch_result: dict[str, Any], fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    request_params = fetch_result["request_params"]
    requested_season = int(request_params["season"])
    completed_statuses = {status for status in str(request_params.get("status", "")).split("-") if status} or API_COMPLETED_STATUSES
    rows: list[dict[str, Any]] = []

    parsed_dates = pd.to_datetime(fixtures["fixture_date"], errors="coerce", utc=True)
    valid_dates = parsed_dates.dropna()
    invalid_date_mask = parsed_dates.isna()
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "timeliness",
            "fixture_dates_are_parseable",
            "Fixture dates should be parseable timestamps.",
            len(fixtures),
            int(invalid_date_mask.sum()),
            _sample_records(fixtures, invalid_date_mask, ["fixture_id", "fixture_date", "status_short"]),
        )
    )

    season_window_mask = valid_dates.dt.year.isin([requested_season, requested_season + 1])
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "timeliness",
            "fixture_dates_fall_in_requested_season_window",
            "Fixture dates should fall in the requested season year or the following calendar year.",
            int(len(valid_dates)),
            int((~season_window_mask).sum()),
            fixtures.loc[valid_dates.index[~season_window_mask], ["fixture_id", "fixture_date", "season"]].head(5).to_dict("records")
            if len(valid_dates) > 0
            else [],
        )
    )

    completed_mask = fixtures["status_short"].fillna("").astype(str).isin(completed_statuses) & parsed_dates.notna()
    current_time = pd.Timestamp.now(tz="UTC")
    future_completed_mask = completed_mask & parsed_dates.gt(current_time)
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "timeliness",
            "completed_fixtures_are_not_future_dated",
            "Completed fixtures should not have timestamps in the future.",
            int(completed_mask.sum()),
            int(future_completed_mask.sum()),
            _sample_records(fixtures, future_completed_mask, ["fixture_id", "fixture_date", "status_short"]),
        )
    )

    unique_days = int(valid_dates.dt.normalize().nunique()) if not valid_dates.empty else 0
    rows.append(
        _manual_check_row(
            "api_fixtures",
            "timeliness",
            "sample_spans_multiple_match_days",
            "The season sample should span multiple match days.",
            1,
            0 if unique_days >= 2 else 1,
            [{"unique_days": unique_days}],
        )
    )

    checks = _finalize_dimension_checks(rows, "timeliness")
    coverage = pd.DataFrame(
        [
            {
                "table_name": "api_fixtures",
                "rows": int(len(fixtures)),
                "valid_dates": int(valid_dates.shape[0]),
                "invalid_dates": int(invalid_date_mask.sum()),
                "min_date": valid_dates.min(),
                "max_date": valid_dates.max(),
                "unique_days": unique_days,
                "span_days": int((valid_dates.max() - valid_dates.min()).days) if valid_dates.shape[0] >= 2 else 0,
            }
        ]
    )
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
        "coverage": coverage,
    }


def _run_api_distribution_checks(fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []

    numeric_columns = ["home_goals", "away_goals"]
    if "status_elapsed" in fixtures.columns and pd.to_numeric(fixtures["status_elapsed"], errors="coerce").notna().any():
        numeric_columns.append("status_elapsed")

    for column in numeric_columns:
        numeric = pd.to_numeric(fixtures[column], errors="coerce")
        clean = numeric.dropna()
        if clean.empty:
            continue

        distribution_type, histogram_payload = _distribution_payload_from_numeric(clean)
        skewness = clean.skew()
        kurtosis = clean.kurt()
        numeric_rows.append(
            {
                "table_name": "api_fixtures",
                "column_name": column,
                "non_null_count": int(clean.shape[0]),
                "missing_count": int(numeric.isna().sum()),
                "missing_pct": round((int(numeric.isna().sum()) / max(len(fixtures), 1)) * 100, 4),
                "distinct_count": int(clean.nunique()),
                "min_value": round(float(clean.min()), 4),
                "q1": round(float(clean.quantile(0.25)), 4),
                "median": round(float(clean.median()), 4),
                "mean": round(float(clean.mean()), 4),
                "q3": round(float(clean.quantile(0.75)), 4),
                "max_value": round(float(clean.max()), 4),
                "std_dev": round(float(clean.std()), 4) if clean.shape[0] > 1 else 0.0,
                "skewness": round(float(skewness), 4) if pd.notna(skewness) else None,
                "abs_skewness": round(float(abs(skewness)), 4) if pd.notna(skewness) else None,
                "kurtosis": round(float(kurtosis), 4) if pd.notna(kurtosis) else None,
                "distribution_type": distribution_type,
                "histogram_payload": histogram_payload,
            }
        )

    for column in ["status_short", "status_long", "league_name", "league_country"]:
        if column not in fixtures.columns:
            continue

        clean = fixtures[column].dropna().astype(str)
        if clean.empty:
            continue

        top_values = clean.value_counts()
        top_value = top_values.index[0]
        top_count = int(top_values.iloc[0])
        categorical_rows.append(
            {
                "table_name": "api_fixtures",
                "column_name": column,
                "non_null_count": int(clean.shape[0]),
                "missing_count": int(fixtures[column].isna().sum()),
                "missing_pct": round((int(fixtures[column].isna().sum()) / max(len(fixtures), 1)) * 100, 4),
                "distinct_count": int(clean.nunique()),
                "most_frequent_value": top_value,
                "most_frequent_count": top_count,
                "most_frequent_pct": round((top_count / max(len(clean), 1)) * 100, 4),
                "top_values_payload": _top_frequency_payload(clean),
            }
        )

    summary = pd.DataFrame(
        [
            {
                "table_name": "api_fixtures",
                "numeric_fields_profiled": len(numeric_rows),
                "categorical_fields_profiled": len(categorical_rows),
            }
        ]
    )
    return {
        "summary": summary,
        "numeric_profiles": pd.DataFrame(numeric_rows).sort_values("column_name").reset_index(drop=True)
        if numeric_rows
        else _empty_dataframe(
            [
                "table_name",
                "column_name",
                "non_null_count",
                "missing_count",
                "missing_pct",
                "distinct_count",
                "min_value",
                "q1",
                "median",
                "mean",
                "q3",
                "max_value",
                "std_dev",
                "skewness",
                "abs_skewness",
                "kurtosis",
                "distribution_type",
                "histogram_payload",
            ]
        ),
        "categorical_profiles": pd.DataFrame(categorical_rows).sort_values("column_name").reset_index(drop=True)
        if categorical_rows
        else _empty_dataframe(
            [
                "table_name",
                "column_name",
                "non_null_count",
                "missing_count",
                "missing_pct",
                "distinct_count",
                "most_frequent_value",
                "most_frequent_count",
                "most_frequent_pct",
                "top_values_payload",
            ]
        ),
    }


def _run_api_relationship_checks(fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    home_goals = pd.to_numeric(fixtures["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(fixtures["away_goals"], errors="coerce")
    score_available_mask = home_goals.notna() & away_goals.notna()
    decisive_mask = score_available_mask & home_goals.ne(away_goals)
    draw_mask = score_available_mask & home_goals.eq(away_goals)

    home_mismatch_mask = decisive_mask & home_goals.gt(away_goals) & ~fixtures["home_winner"].eq(True)
    away_mismatch_mask = decisive_mask & away_goals.gt(home_goals) & ~fixtures["away_winner"].eq(True)
    draw_winner_mask = draw_mask & fixtures["home_winner"].eq(True) & fixtures["away_winner"].eq(True)

    checks = _finalize_dimension_checks(
        [
            _manual_check_row(
                "api_fixtures",
                "relationships",
                "home_winner_matches_score_difference",
                "If home_goals exceed away_goals, home_winner should be True.",
                int(decisive_mask.sum()),
                int(home_mismatch_mask.sum()),
                _sample_records(fixtures, home_mismatch_mask, ["fixture_id", "home_goals", "away_goals", "home_winner", "home_team_name", "away_team_name"]),
            ),
            _manual_check_row(
                "api_fixtures",
                "relationships",
                "away_winner_matches_score_difference",
                "If away_goals exceed home_goals, away_winner should be True.",
                int(decisive_mask.sum()),
                int(away_mismatch_mask.sum()),
                _sample_records(fixtures, away_mismatch_mask, ["fixture_id", "home_goals", "away_goals", "away_winner", "home_team_name", "away_team_name"]),
            ),
            _manual_check_row(
                "api_fixtures",
                "relationships",
                "draws_do_not_mark_both_teams_as_winners",
                "Drawn matches should not mark both teams as winners.",
                int(draw_mask.sum()),
                int(draw_winner_mask.sum()),
                _sample_records(fixtures, draw_winner_mask, ["fixture_id", "home_goals", "away_goals", "home_winner", "away_winner"]),
            ),
        ],
        "relationships",
    )
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
    }


def _build_api_fixture_validation_results(fetch_result: dict[str, Any]) -> dict[str, Any]:
    payload = fetch_result["payload"]
    fixtures = flatten_fixture_response(payload)
    errors = _normalize_api_errors(payload.get("errors"))
    response_rows = payload.get("response") if isinstance(payload.get("response"), list) else []

    if int(fetch_result["status_code"]) != 200:
        raise RuntimeError(
            f"API-Football returned HTTP {fetch_result['status_code']}. "
            "Retry later or confirm the endpoint and credentials."
        )
    if errors:
        raise ValueError(
            "API-Football returned request-level errors: "
            + "; ".join(errors[:3])
            + ". Update the sample parameters or retry later."
        )
    if not response_rows or fixtures.empty:
        raise ValueError(
            "API-Football returned zero fixtures for the validation sample. "
            "Update the sample parameters or retry later."
        )

    results = {
        "request_metadata": _api_request_metadata(fetch_result, fixtures),
        "fixtures": fixtures,
        "accuracy": _run_api_accuracy_checks(fetch_result, fixtures),
        "consistency": _run_api_consistency_checks(fetch_result, fixtures),
        "completeness": _run_api_completeness_checks(payload, fixtures),
        "uniqueness": _run_api_uniqueness_checks(fixtures),
        "outliers": _run_api_outlier_checks(fixtures),
        "timeliness": _run_api_timeliness_checks(fetch_result, fixtures),
        "distribution": _run_api_distribution_checks(fixtures),
        "relationships": _run_api_relationship_checks(fixtures),
    }
    results["report_snippets"] = build_api_report_snippets(results)
    return results


def run_api_fixture_validation(
    params: dict[str, Any] | None = None,
    key_path: str | Path = "key_api.txt",
    cache_path: str | Path | None = DEFAULT_API_FOOTBALL_CACHE_PATH,
    use_cache_on_failure: bool = True,
) -> dict[str, Any]:
    fetch_result = fetch_api_football_fixtures(
        params=params,
        key_path=key_path,
        cache_path=cache_path,
        use_cache_on_failure=use_cache_on_failure,
        save_cache=cache_path is not None,
    )
    return _build_api_fixture_validation_results(fetch_result)


def _failed_api_check_names(checks: pd.DataFrame, limit: int = 3) -> str:
    if checks.empty or "status" not in checks.columns:
        return "no major issues"
    failed = checks.loc[checks["status"] == "failed", "check_name"].astype(str).head(limit).tolist()
    return ", ".join(failed) if failed else "no major issues"


def build_api_report_snippets(api_results: dict[str, Any]) -> pd.DataFrame:
    metadata = api_results["request_metadata"].iloc[0]
    sample_size = int(metadata["fixture_rows"])
    accuracy_failed_checks = int(api_results["accuracy"]["summary"]["failed_checks"].sum())
    consistency_failed_checks = int(api_results["consistency"]["summary"]["failed_checks"].sum())
    completeness_failed_checks = int(api_results["completeness"]["summary"]["failed_checks"].sum())
    uniqueness_failed_checks = int(api_results["uniqueness"]["checks"]["status"].eq("failed").sum())
    timeliness_failed_checks = int(api_results["timeliness"]["summary"]["failed_checks"].sum())
    relationship_failed_checks = int(api_results["relationships"]["summary"]["failed_checks"].sum())

    required_missingness = api_results["completeness"]["field_missingness"]
    required_missingness = required_missingness[required_missingness["field_role"] == "required"]
    max_required_missing_pct = round(float(required_missingness["missing_pct"].max()), 4) if not required_missingness.empty else 0.0

    outlier_count = int(api_results["outliers"]["summary"]["outlier_count"].sum()) if not api_results["outliers"]["summary"].empty else 0
    distribution_profiles = api_results["distribution"]["categorical_profiles"]
    status_row = (
        distribution_profiles[distribution_profiles["column_name"] == "status_short"].head(1)
        if not distribution_profiles.empty
        else pd.DataFrame()
    )
    dominant_status = status_row["most_frequent_value"].iloc[0] if not status_row.empty else "completed statuses"
    accuracy_issue_names = _failed_api_check_names(api_results["accuracy"]["checks"])
    consistency_issue_names = _failed_api_check_names(api_results["consistency"]["checks"])
    completeness_issue_names = _failed_api_check_names(api_results["completeness"]["checks"])
    uniqueness_issue_names = _failed_api_check_names(api_results["uniqueness"]["checks"])
    timeliness_issue_names = _failed_api_check_names(api_results["timeliness"]["checks"])
    relationship_issue_names = _failed_api_check_names(api_results["relationships"]["checks"])

    rows = [
        {
            "dimension": "Accuracy",
            "status": "passed" if accuracy_failed_checks == 0 else "failed",
            "report_text": (
                "Accuracy was evaluated by checking the API wrapper, row-count agreement, positive identifiers, non-negative goals, and season alignment. "
                f"All applied accuracy checks passed for the sampled `{sample_size}` fixtures, so the `/fixtures` response appears logically valid for downstream use."
                if accuracy_failed_checks == 0
                else "Accuracy was evaluated by checking the API wrapper, row-count agreement, positive identifiers, non-negative goals, and season alignment. "
                f"Most checks passed, but {accuracy_failed_checks} accuracy issue(s) were flagged, mainly in {accuracy_issue_names}, so these fields should be reviewed before use."
            ),
        },
        {
            "dimension": "Consistency",
            "status": "passed" if consistency_failed_checks == 0 else "failed",
            "report_text": (
                "Consistency was evaluated by comparing echoed parameters, parsing fixture dates, aligning timestamps, and checking the home/away team schema. "
                f"All consistency checks passed across the sampled `{sample_size}` fixtures, indicating that the API uses one stable representation for the core fields."
                if consistency_failed_checks == 0
                else "Consistency was evaluated by comparing echoed parameters, parsing fixture dates, aligning timestamps, and checking the home/away team schema. "
                f"Most checks passed, but {consistency_failed_checks} consistency issue(s) were detected, mainly in {consistency_issue_names}."
            ),
        },
        {
            "dimension": "Completeness",
            "status": "passed" if completeness_failed_checks == 0 else "failed",
            "report_text": (
                "Completeness was evaluated by checking required wrapper fields, confirming that the response was non-empty, and profiling missingness in the flattened fixture fields. "
                "The sampled API response was complete for the core required attributes, while any remaining missingness was concentrated in optional fields."
                if completeness_failed_checks == 0
                else "Completeness was evaluated by checking required wrapper fields, confirming that the response was non-empty, and profiling missingness in the flattened fixture fields. "
                f"The sample showed completeness gaps, with required fields reaching up to {max_required_missing_pct}% missingness and the main issues appearing in {completeness_issue_names}."
            ),
        },
        {
            "dimension": "Uniqueness",
            "status": "passed" if uniqueness_failed_checks == 0 else "failed",
            "report_text": (
                "Uniqueness was evaluated using exact duplicate rows and repeated fixture identifiers in the flattened response. "
                "No duplicate rows or repeated fixture IDs were detected, so the API sample preserved record uniqueness."
                if uniqueness_failed_checks == 0
                else "Uniqueness was evaluated using exact duplicate rows and repeated fixture identifiers in the flattened response. "
                f"The sample showed {uniqueness_failed_checks} uniqueness issue(s), mainly in {uniqueness_issue_names}."
            ),
        },
        {
            "dimension": "Outliers",
            "status": "passed",
            "report_text": (
                "Outliers were evaluated with the IQR method on home goals, away goals, and elapsed time when available. "
                + (
                    f"The sampled API response produced {outlier_count} outlier value(s); these should be treated as extreme football observations rather than automatic data errors."
                    if outlier_count > 0
                    else "No IQR outliers were flagged in the sampled numeric API fields."
                )
            ),
        },
        {
            "dimension": "Timeliness",
            "status": "passed" if timeliness_failed_checks == 0 else "failed",
            "report_text": (
                "Timeliness was evaluated by parsing fixture timestamps, checking the requested season window, and ensuring that completed matches were not future-dated. "
                "All timeliness checks passed, indicating that the sampled API records were logically dated and aligned with the requested season."
                if timeliness_failed_checks == 0
                else "Timeliness was evaluated by parsing fixture timestamps, checking the requested season window, and ensuring that completed matches were not future-dated. "
                f"Most checks passed, but {timeliness_failed_checks} timeliness issue(s) were detected, mainly in {timeliness_issue_names}."
            ),
        },
        {
            "dimension": "Distribution",
            "status": "passed",
            "report_text": (
                "Distribution was profiled on the main numeric and categorical API fields, including goals and match-status labels. "
                f"The main pattern was that goal counts were concentrated in low-scoring values, while `{dominant_status}` was the dominant status category in the sampled response."
            ),
        },
        {
            "dimension": "Relationships",
            "status": "passed" if relationship_failed_checks == 0 else "failed",
            "report_text": (
                "Relationships were evaluated using direct dependency rules between score differences and winner flags. "
                "All applied checks passed, indicating that the API preserved the expected match-outcome logic."
                if relationship_failed_checks == 0
                else "Relationships were evaluated using direct dependency rules between score differences and winner flags. "
                f"The sample showed {relationship_failed_checks} relationship issue(s), mainly in {relationship_issue_names}."
            ),
        },
    ]
    return pd.DataFrame(rows)

__all__ = [k for k in globals().keys() if not k.startswith('__') and k not in ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']]
