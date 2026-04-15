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
from .api import *

MERGED_TEAM_NAME_STOPWORDS = {
    "football",
    "club",
    "fc",
    "afc",
    "association",
    "sporting",
    "hotspur",
    "town",
    "city",
    "united",
    "wanderers",
    "albion",
    "hove",
    "and",
    "s",
    "a",
    "d",
}

MERGED_TEAM_NAME_ALIASES = {
    "wolves": "wolverhampton",
}

MERGED_REQUIRED_FIELDS = [
    "game_id",
    "competition_id",
    "season",
    "date",
    "home_club_id",
    "away_club_id",
    "home_club_name",
    "away_club_name",
    "home_club_goals",
    "away_club_goals",
    "api_fixture_id",
    "api_fixture_date",
    "api_league_id",
    "api_league_name",
    "api_home_team_id",
    "api_away_team_id",
    "api_home_team_name",
    "api_away_team_name",
    "api_home_goals",
    "api_away_goals",
    "api_status_short",
    "api_status_long",
]

MERGED_OUTLIER_FIELDS = ["home_club_goals", "away_club_goals", "attendance"]

MERGED_DISTRIBUTION_NUMERIC_FIELDS = ["home_club_goals", "away_club_goals", "attendance"]

MERGED_DISTRIBUTION_CATEGORICAL_FIELDS = ["api_status_short", "competition_type", "home_club_formation", "away_club_formation"]


def load_api_fixture_sample(cache_path: str | Path = DEFAULT_API_FOOTBALL_CACHE_PATH) -> dict[str, Any]:
    return _load_api_fixture_cache(cache_path, request_params=dict(DEFAULT_API_FOOTBALL_FIXTURE_PARAMS))


def _normalize_team_name_for_merge(value: Any) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""

    tokens = [token for token in normalized.split() if token not in MERGED_TEAM_NAME_STOPWORDS]
    team_key = " ".join(tokens)
    return MERGED_TEAM_NAME_ALIASES.get(team_key, team_key)


def _map_api_league_to_competition(fixtures: pd.DataFrame, competitions: pd.DataFrame) -> dict[str, Any]:
    if fixtures.empty:
        raise ValueError("API fixtures are required to map the API league to the Kaggle competition.")
    if competitions.empty:
        raise ValueError("competitions.csv is required to map the API league to the Kaggle competition.")

    league_names = fixtures["league_name"].dropna().astype(str)
    league_countries = fixtures["league_country"].dropna().astype(str)
    league_name = league_names.mode().iloc[0] if not league_names.empty else ""
    league_country = league_countries.mode().iloc[0] if not league_countries.empty else ""
    league_name_norm = _normalize_text(league_name)
    league_country_norm = _normalize_text(league_country)

    candidates = competitions.copy()
    candidates["name_norm"] = candidates["name"].map(_normalize_text)
    candidates["code_norm"] = candidates["competition_code"].map(_normalize_text)
    candidates["country_norm"] = candidates["country_name"].map(_normalize_text)

    country_candidates = candidates[candidates["country_norm"] == league_country_norm].copy() if league_country_norm else candidates.copy()
    exact_candidates = country_candidates[
        country_candidates["name_norm"].eq(league_name_norm) | country_candidates["code_norm"].eq(league_name_norm)
    ].copy()
    if not exact_candidates.empty:
        return exact_candidates.iloc[0].to_dict()

    scored = country_candidates.copy()
    scored["name_similarity"] = scored["name_norm"].map(lambda value: SequenceMatcher(a=league_name_norm, b=value).ratio())
    scored["code_similarity"] = scored["code_norm"].map(lambda value: SequenceMatcher(a=league_name_norm, b=value).ratio())
    scored["best_similarity"] = scored[["name_similarity", "code_similarity"]].max(axis=1)
    scored = scored.sort_values(["best_similarity", "competition_id"], ascending=[False, True]).reset_index(drop=True)

    if scored.empty or float(scored.iloc[0]["best_similarity"]) < 0.55:
        raise ValueError(
            "Could not map the API league "
            f"'{league_name}' ({league_country}) to one Kaggle competition."
        )

    return scored.iloc[0].to_dict()


def build_merged_fixture_dataset(
    raw_dir: str | Path = "data/player_scores_data",
    api_cache_path: str | Path = DEFAULT_API_FOOTBALL_CACHE_PATH,
    api_fetch_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tables = load_primary_tables(raw_dir)
    games = tables.get("games")
    competitions = tables.get("competitions")
    if games is None or games.empty:
        raise ValueError("games.csv is required to build the merged fixture dataset.")
    if competitions is None or competitions.empty:
        raise ValueError("competitions.csv is required to build the merged fixture dataset.")

    fetch_result = api_fetch_result or load_api_fixture_sample(api_cache_path)
    fixtures = flatten_fixture_response(fetch_result["payload"])
    if fixtures.empty:
        raise ValueError("The cached API fixture sample is empty, so the merged dataset could not be built.")

    competition_row = _map_api_league_to_competition(fixtures, competitions)
    season_values = sorted(pd.to_numeric(fixtures["season"], errors="coerce").dropna().astype(int).unique().tolist())
    if not season_values:
        raise ValueError("The API fixture sample does not contain a valid season value.")

    api = fixtures.copy()
    api["api_season"] = pd.to_numeric(api["season"], errors="coerce")
    api["match_date"] = pd.to_datetime(api["fixture_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    api["home_team_key"] = api["home_team_name"].map(_normalize_team_name_for_merge)
    api["away_team_key"] = api["away_team_name"].map(_normalize_team_name_for_merge)
    api = api.rename(
        columns={
            "fixture_id": "api_fixture_id",
            "fixture_date": "api_fixture_date",
            "fixture_timestamp": "api_fixture_timestamp",
            "status_short": "api_status_short",
            "status_long": "api_status_long",
            "status_elapsed": "api_status_elapsed",
            "league_id": "api_league_id",
            "league_name": "api_league_name",
            "league_country": "api_league_country",
            "home_team_id": "api_home_team_id",
            "home_team_name": "api_home_team_name",
            "home_winner": "api_home_winner",
            "away_team_id": "api_away_team_id",
            "away_team_name": "api_away_team_name",
            "away_winner": "api_away_winner",
            "home_goals": "api_home_goals",
            "away_goals": "api_away_goals",
        }
    )

    kaggle = games.copy()
    kaggle = kaggle[
        kaggle["competition_id"].astype(str).eq(str(competition_row["competition_id"]))
        & pd.to_numeric(kaggle["season"], errors="coerce").isin(season_values)
    ].copy()
    kaggle["match_date"] = pd.to_datetime(kaggle["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    kaggle["home_team_key"] = kaggle["home_club_name"].map(_normalize_team_name_for_merge)
    kaggle["away_team_key"] = kaggle["away_club_name"].map(_normalize_team_name_for_merge)

    merge_key = ["match_date", "season", "home_team_key", "away_team_key"]
    outer = api.merge(kaggle, on=merge_key, how="outer", indicator=True)
    merged = outer[outer["_merge"] == "both"].copy().drop(columns=["_merge"])
    unmatched_api = (
        outer.loc[outer["_merge"] == "left_only", ["api_fixture_id", "api_fixture_date", "api_home_team_name", "api_away_team_name", "season", "match_date"]]
        .sort_values(["match_date", "api_fixture_id"], kind="stable")
        .reset_index(drop=True)
    )
    unmatched_kaggle = (
        outer.loc[outer["_merge"] == "right_only", ["game_id", "date", "home_club_name", "away_club_name", "competition_id", "season", "match_date"]]
        .sort_values(["match_date", "game_id"], kind="stable")
        .reset_index(drop=True)
    )

    merged["mapped_competition_name"] = competition_row["name"]
    merged["mapped_competition_country"] = competition_row["country_name"]
    merged["merge_strategy"] = "feature_enrichment_same_match"

    api_rows = int(len(api))
    kaggle_rows = int(len(kaggle))
    matched_rows = int(len(merged))
    added_api_features = int(len([column for column in merged.columns if column.startswith("api_")]))

    merge_summary = pd.DataFrame(
        [
            {
                "league_name": api["api_league_name"].dropna().astype(str).mode().iloc[0] if api["api_league_name"].notna().any() else "",
                "league_country": api["api_league_country"].dropna().astype(str).mode().iloc[0] if api["api_league_country"].notna().any() else "",
                "competition_id": competition_row["competition_id"],
                "competition_name": competition_row["name"],
                "season_values": ", ".join(str(value) for value in season_values),
                "api_rows": api_rows,
                "kaggle_rows_considered": kaggle_rows,
                "matched_rows": matched_rows,
                "api_unmatched_rows": int(len(unmatched_api)),
                "kaggle_unmatched_rows": int(len(unmatched_kaggle)),
                "api_match_rate_pct": round((matched_rows / max(api_rows, 1)) * 100, 4),
                "kaggle_match_rate_pct": round((matched_rows / max(kaggle_rows, 1)) * 100, 4),
                "added_api_features": added_api_features,
                "merge_key": "season + match_date + canonicalized home team + canonicalized away team",
                "name_aliases_used": ", ".join(f"{left}->{right}" for left, right in MERGED_TEAM_NAME_ALIASES.items()) or "none",
            }
        ]
    )

    return {
        "fetch_result": fetch_result,
        "api_fixtures": api,
        "kaggle_games": kaggle,
        "merged_fixtures": merged,
        "unmatched_api": unmatched_api,
        "unmatched_kaggle": unmatched_kaggle,
        "merge_summary": merge_summary,
        "competition_mapping": pd.DataFrame([competition_row]),
    }


def _run_merged_accuracy_checks(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    api_dates = pd.to_datetime(merged["api_fixture_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    kaggle_dates = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    api_home_goals = pd.to_numeric(merged["api_home_goals"], errors="coerce")
    kaggle_home_goals = pd.to_numeric(merged["home_club_goals"], errors="coerce")
    api_away_goals = pd.to_numeric(merged["api_away_goals"], errors="coerce")
    kaggle_away_goals = pd.to_numeric(merged["away_club_goals"], errors="coerce")
    api_season = pd.to_numeric(merged["api_season"], errors="coerce")
    kaggle_season = pd.to_numeric(merged["season"], errors="coerce")

    league_name_match = merged["api_league_name"].map(_normalize_text).eq(merged["mapped_competition_name"].map(_normalize_text))
    country_match = merged["api_league_country"].map(_normalize_text).eq(merged["mapped_competition_country"].map(_normalize_text))

    checks = _finalize_dimension_checks(
        [
            _manual_check_row(
                "merged_fixtures",
                "accuracy",
                "api_home_goals_match_kaggle_home_goals",
                "The API and Kaggle home-goal values should agree for the same match.",
                len(merged),
                int(api_home_goals.ne(kaggle_home_goals).sum()),
                _sample_records(
                    merged,
                    api_home_goals.ne(kaggle_home_goals),
                    ["game_id", "api_fixture_id", "home_club_name", "away_club_name", "home_club_goals", "api_home_goals"],
                ),
            ),
            _manual_check_row(
                "merged_fixtures",
                "accuracy",
                "api_away_goals_match_kaggle_away_goals",
                "The API and Kaggle away-goal values should agree for the same match.",
                len(merged),
                int(api_away_goals.ne(kaggle_away_goals).sum()),
                _sample_records(
                    merged,
                    api_away_goals.ne(kaggle_away_goals),
                    ["game_id", "api_fixture_id", "home_club_name", "away_club_name", "away_club_goals", "api_away_goals"],
                ),
            ),
            _manual_check_row(
                "merged_fixtures",
                "accuracy",
                "api_fixture_date_matches_kaggle_game_date",
                "The API fixture date should match the Kaggle game date after removing the time component.",
                len(merged),
                int(api_dates.ne(kaggle_dates).sum()),
                _sample_records(merged, api_dates.ne(kaggle_dates), ["game_id", "api_fixture_id", "date", "api_fixture_date"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "accuracy",
                "api_season_matches_kaggle_season",
                "The API season should match the Kaggle season for the merged match.",
                len(merged),
                int(api_season.ne(kaggle_season).sum()),
                _sample_records(merged, api_season.ne(kaggle_season), ["game_id", "api_fixture_id", "season", "api_season"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "accuracy",
                "api_league_name_matches_kaggle_competition_mapping",
                "The API league name should match the mapped Kaggle competition.",
                len(merged),
                int((~league_name_match).sum()),
                _sample_records(merged, ~league_name_match, ["game_id", "api_fixture_id", "api_league_name", "mapped_competition_name"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "accuracy",
                "api_league_country_matches_kaggle_competition_country",
                "The API league country should match the Kaggle competition country.",
                len(merged),
                int((~country_match).sum()),
                _sample_records(merged, ~country_match, ["game_id", "api_fixture_id", "api_league_country", "mapped_competition_country"]),
            ),
        ],
        "accuracy",
    )
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
    }


def _run_merged_consistency_checks(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    status_short_missing = _missing_mask(merged["api_status_short"])
    status_long_missing = _missing_mask(merged["api_status_long"])
    inconsistent_status_mask = status_short_missing ^ status_long_missing

    home_team_structure_mask = _missing_mask(merged["api_home_team_id"]) ^ _missing_mask(merged["api_home_team_name"])
    away_team_structure_mask = _missing_mask(merged["api_away_team_id"]) ^ _missing_mask(merged["api_away_team_name"])

    competition_count = int(merged["competition_id"].nunique(dropna=True))
    league_count = int(merged["api_league_name"].nunique(dropna=True))

    checks = _finalize_dimension_checks(
        [
            _manual_check_row(
                "merged_fixtures",
                "consistency",
                "api_status_fields_are_populated_together",
                "api_status_short and api_status_long should be populated together.",
                len(merged),
                int(inconsistent_status_mask.sum()),
                _sample_records(merged, inconsistent_status_mask, ["game_id", "api_fixture_id", "api_status_short", "api_status_long"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "consistency",
                "api_home_team_id_and_name_are_paired",
                "Each merged row should include both api_home_team_id and api_home_team_name together.",
                len(merged),
                int(home_team_structure_mask.sum()),
                _sample_records(merged, home_team_structure_mask, ["game_id", "api_fixture_id", "api_home_team_id", "api_home_team_name"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "consistency",
                "api_away_team_id_and_name_are_paired",
                "Each merged row should include both api_away_team_id and api_away_team_name together.",
                len(merged),
                int(away_team_structure_mask.sum()),
                _sample_records(merged, away_team_structure_mask, ["game_id", "api_fixture_id", "api_away_team_id", "api_away_team_name"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "consistency",
                "merged_slice_contains_one_competition",
                "The merged validation slice should represent one Kaggle competition.",
                1,
                0 if competition_count == 1 else 1,
                [{"competition_count": competition_count}],
            ),
            _manual_check_row(
                "merged_fixtures",
                "consistency",
                "merged_slice_contains_one_api_league",
                "The merged validation slice should represent one API league.",
                1,
                0 if league_count == 1 else 1,
                [{"league_count": league_count}],
            ),
        ],
        "consistency",
    )
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
    }


def _run_merged_completeness_checks(
    merged: pd.DataFrame,
    unmatched_api: pd.DataFrame,
    unmatched_kaggle: pd.DataFrame,
    merge_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    for column in merged.columns:
        missing_count = int(_missing_mask(merged[column]).sum())
        field_rows.append(
            {
                "table_name": "merged_fixtures",
                "column_name": column,
                "field_role": "required" if column in MERGED_REQUIRED_FIELDS else "optional",
                "missing_count": missing_count,
                "missing_pct": round((missing_count / max(len(merged), 1)) * 100, 4),
            }
        )

    for column in MERGED_REQUIRED_FIELDS:
        if column not in merged.columns:
            rows.append(
                _manual_check_row(
                    "merged_fixtures",
                    "completeness",
                    f"{column}_present_and_populated",
                    f"{column} should be present and populated in the merged dataset.",
                    len(merged),
                    len(merged),
                    [{"missing_column": column}],
                )
            )
            continue

        missing_mask = _missing_mask(merged[column])
        rows.append(
            _manual_check_row(
                "merged_fixtures",
                "completeness",
                f"{column}_present_and_populated",
                f"{column} should be present and populated in the merged dataset.",
                len(merged),
                int(missing_mask.sum()),
                _sample_records(merged, missing_mask, ["game_id", "api_fixture_id", column]),
            )
        )

    metadata = merge_summary.iloc[0]
    rows.extend(
        [
            _manual_check_row(
                "merged_fixtures",
                "completeness",
                "all_api_fixtures_were_matched",
                "All API fixtures in the selected league-season slice should find a Kaggle match.",
                int(metadata["api_rows"]),
                int(metadata["api_unmatched_rows"]),
                unmatched_api.head(5).to_dict("records"),
            ),
            _manual_check_row(
                "merged_fixtures",
                "completeness",
                "all_kaggle_games_were_matched",
                "All Kaggle games in the mapped league-season slice should find an API fixture.",
                int(metadata["kaggle_rows_considered"]),
                int(metadata["kaggle_unmatched_rows"]),
                unmatched_kaggle.head(5).to_dict("records"),
            ),
        ]
    )

    checks = _finalize_dimension_checks(rows, "completeness")
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
        "field_missingness": pd.DataFrame(field_rows).sort_values(["field_role", "missing_pct", "column_name"], ascending=[True, False, True]).reset_index(drop=True),
    }


def _run_merged_uniqueness_checks(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    exact_duplicate_mask = merged.duplicated(keep=False)
    game_id_duplicate_mask = merged["game_id"].duplicated(keep=False) & merged["game_id"].notna()
    fixture_id_duplicate_mask = merged["api_fixture_id"].duplicated(keep=False) & merged["api_fixture_id"].notna()
    natural_key_duplicate_mask = merged.duplicated(["season", "match_date", "home_team_key", "away_team_key"], keep=False)

    checks = _finalize_dimension_checks(
        [
            _manual_check_row(
                "merged_fixtures",
                "uniqueness",
                "merged_rows_are_not_exact_duplicates",
                "The merged table should not contain exact duplicate rows.",
                len(merged),
                int(merged.duplicated().sum()),
                _sample_records(merged, exact_duplicate_mask, ["game_id", "api_fixture_id", "match_date", "home_club_name", "away_club_name"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "uniqueness",
                "game_id_is_unique",
                "Each Kaggle game_id should appear only once in the merged table.",
                len(merged),
                int(game_id_duplicate_mask.sum()),
                _sample_records(merged, game_id_duplicate_mask, ["game_id", "api_fixture_id", "match_date", "home_club_name", "away_club_name"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "uniqueness",
                "api_fixture_id_is_unique",
                "Each API fixture_id should appear only once in the merged table.",
                len(merged),
                int(fixture_id_duplicate_mask.sum()),
                _sample_records(merged, fixture_id_duplicate_mask, ["game_id", "api_fixture_id", "match_date", "home_club_name", "away_club_name"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "uniqueness",
                "match_business_key_is_unique",
                "The merged natural match key should identify one row.",
                len(merged),
                int(natural_key_duplicate_mask.sum()),
                _sample_records(merged, natural_key_duplicate_mask, ["game_id", "api_fixture_id", "match_date", "home_club_name", "away_club_name"]),
            ),
        ],
        "uniqueness",
    )
    duplicate_rows = (
        merged.loc[exact_duplicate_mask | game_id_duplicate_mask | fixture_id_duplicate_mask | natural_key_duplicate_mask]
        .sort_values(["match_date", "home_club_name", "away_club_name"], kind="stable")
        .reset_index(drop=True)
        if (exact_duplicate_mask | game_id_duplicate_mask | fixture_id_duplicate_mask | natural_key_duplicate_mask).any()
        else _empty_dataframe(list(merged.columns))
    )
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
        "duplicate_rows": duplicate_rows,
    }


def _run_merged_outlier_checks(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    labels = merged["home_club_name"].fillna("").astype(str) + " vs " + merged["away_club_name"].fillna("").astype(str)

    for column in MERGED_OUTLIER_FIELDS:
        if column not in merged.columns:
            continue

        values = pd.to_numeric(merged[column], errors="coerce")
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
                "table_name": "merged_fixtures",
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

        outlier_rows = merged.loc[outlier_mask, ["game_id", "api_fixture_id", column]].copy()
        outlier_rows["table_name"] = "merged_fixtures"
        outlier_rows["column_name"] = column
        outlier_rows["value"] = pd.to_numeric(outlier_rows[column], errors="coerce")
        outlier_rows["label"] = labels.loc[outlier_mask].values
        example_rows.extend(
            outlier_rows.sort_values("value", ascending=False)[["table_name", "column_name", "game_id", "api_fixture_id", "label", "value"]]
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
        else _empty_dataframe(["table_name", "column_name", "game_id", "api_fixture_id", "label", "value"])
    )
    return {
        "summary": summary,
        "examples": examples,
    }


def _run_merged_timeliness_checks(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    kaggle_dates = pd.to_datetime(merged["date"], errors="coerce")
    api_dates = pd.to_datetime(merged["api_fixture_date"], utc=True, errors="coerce")
    api_date_only = api_dates.dt.strftime("%Y-%m-%d")
    kaggle_date_only = kaggle_dates.dt.strftime("%Y-%m-%d")
    season_values = pd.to_numeric(merged["season"], errors="coerce")
    current_time = pd.Timestamp.now(tz="UTC")

    invalid_kaggle_date_mask = kaggle_dates.isna()
    invalid_api_date_mask = api_dates.isna()
    aligned_date_mask = api_date_only.eq(kaggle_date_only)
    season_window_mask = api_dates.dt.year.isin(season_values.astype("Int64")) | api_dates.dt.year.eq(season_values.astype("Int64") + 1)
    completed_mask = merged["api_status_short"].fillna("").astype(str).isin(API_COMPLETED_STATUSES) & api_dates.notna()
    future_completed_mask = completed_mask & api_dates.gt(current_time)
    valid_api_dates = api_dates.dropna()
    unique_days = int(valid_api_dates.dt.normalize().nunique()) if not valid_api_dates.empty else 0

    checks = _finalize_dimension_checks(
        [
            _manual_check_row(
                "merged_fixtures",
                "timeliness",
                "kaggle_game_dates_are_parseable",
                "Kaggle game dates should parse cleanly.",
                len(merged),
                int(invalid_kaggle_date_mask.sum()),
                _sample_records(merged, invalid_kaggle_date_mask, ["game_id", "date", "home_club_name", "away_club_name"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "timeliness",
                "api_fixture_dates_are_parseable",
                "API fixture dates should parse cleanly.",
                len(merged),
                int(invalid_api_date_mask.sum()),
                _sample_records(merged, invalid_api_date_mask, ["api_fixture_id", "api_fixture_date", "api_home_team_name", "api_away_team_name"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "timeliness",
                "api_and_kaggle_dates_are_aligned",
                "The API and Kaggle date fields should point to the same match day.",
                len(merged),
                int((~aligned_date_mask).sum()),
                _sample_records(merged, ~aligned_date_mask, ["game_id", "api_fixture_id", "date", "api_fixture_date"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "timeliness",
                "api_fixture_dates_fall_in_season_window",
                "Merged fixture dates should fall in the season year or the following calendar year.",
                len(merged),
                int((~season_window_mask).sum()),
                _sample_records(merged, ~season_window_mask, ["game_id", "api_fixture_id", "season", "api_fixture_date"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "timeliness",
                "completed_fixtures_are_not_future_dated",
                "Completed fixtures should not be dated in the future.",
                int(completed_mask.sum()),
                int(future_completed_mask.sum()),
                _sample_records(merged, future_completed_mask, ["game_id", "api_fixture_id", "api_fixture_date", "api_status_short"]),
            ),
            _manual_check_row(
                "merged_fixtures",
                "timeliness",
                "merged_sample_spans_multiple_match_days",
                "The merged season slice should span multiple match days.",
                1,
                0 if unique_days >= 2 else 1,
                [{"unique_days": unique_days}],
            ),
        ],
        "timeliness",
    )
    coverage = pd.DataFrame(
        [
            {
                "table_name": "merged_fixtures",
                "rows": int(len(merged)),
                "valid_api_dates": int(valid_api_dates.shape[0]),
                "invalid_api_dates": int(invalid_api_date_mask.sum()),
                "min_date": valid_api_dates.min(),
                "max_date": valid_api_dates.max(),
                "unique_days": unique_days,
                "span_days": int((valid_api_dates.max() - valid_api_dates.min()).days) if valid_api_dates.shape[0] >= 2 else 0,
            }
        ]
    )
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
        "coverage": coverage,
    }


def _run_merged_distribution_checks(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []

    for column in MERGED_DISTRIBUTION_NUMERIC_FIELDS:
        if column not in merged.columns:
            continue

        numeric = pd.to_numeric(merged[column], errors="coerce")
        clean = numeric.dropna()
        if clean.empty:
            continue

        distribution_type, histogram_payload = _distribution_payload_from_numeric(clean)
        skewness = clean.skew()
        kurtosis = clean.kurt()
        numeric_rows.append(
            {
                "table_name": "merged_fixtures",
                "column_name": column,
                "non_null_count": int(clean.shape[0]),
                "missing_count": int(numeric.isna().sum()),
                "missing_pct": round((int(numeric.isna().sum()) / max(len(merged), 1)) * 100, 4),
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

    for column in MERGED_DISTRIBUTION_CATEGORICAL_FIELDS:
        if column not in merged.columns:
            continue

        clean = merged[column].dropna().astype(str)
        if clean.empty:
            continue

        top_values = clean.value_counts()
        top_value = top_values.index[0]
        top_count = int(top_values.iloc[0])
        categorical_rows.append(
            {
                "table_name": "merged_fixtures",
                "column_name": column,
                "non_null_count": int(clean.shape[0]),
                "missing_count": int(merged[column].isna().sum()),
                "missing_pct": round((int(merged[column].isna().sum()) / max(len(merged), 1)) * 100, 4),
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
                "table_name": "merged_fixtures",
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


def _run_merged_relationship_checks(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    home_goals = pd.to_numeric(merged["home_club_goals"], errors="coerce")
    away_goals = pd.to_numeric(merged["away_club_goals"], errors="coerce")
    score_available_mask = home_goals.notna() & away_goals.notna()
    decisive_mask = score_available_mask & home_goals.ne(away_goals)
    draw_mask = score_available_mask & home_goals.eq(away_goals)

    home_mismatch_mask = decisive_mask & home_goals.gt(away_goals) & ~merged["api_home_winner"].eq(True)
    away_mismatch_mask = decisive_mask & away_goals.gt(home_goals) & ~merged["api_away_winner"].eq(True)
    draw_winner_mask = draw_mask & merged["api_home_winner"].eq(True) & merged["api_away_winner"].eq(True)

    checks = _finalize_dimension_checks(
        [
            _manual_check_row(
                "merged_fixtures",
                "relationships",
                "api_home_winner_matches_score_difference",
                "If home goals exceed away goals, api_home_winner should be True.",
                int(decisive_mask.sum()),
                int(home_mismatch_mask.sum()),
                _sample_records(
                    merged,
                    home_mismatch_mask,
                    ["game_id", "api_fixture_id", "home_club_goals", "away_club_goals", "api_home_winner", "home_club_name", "away_club_name"],
                ),
            ),
            _manual_check_row(
                "merged_fixtures",
                "relationships",
                "api_away_winner_matches_score_difference",
                "If away goals exceed home goals, api_away_winner should be True.",
                int(decisive_mask.sum()),
                int(away_mismatch_mask.sum()),
                _sample_records(
                    merged,
                    away_mismatch_mask,
                    ["game_id", "api_fixture_id", "home_club_goals", "away_club_goals", "api_away_winner", "home_club_name", "away_club_name"],
                ),
            ),
            _manual_check_row(
                "merged_fixtures",
                "relationships",
                "draws_do_not_mark_both_teams_as_winners",
                "Drawn matches should not mark both teams as winners in the API flags.",
                int(draw_mask.sum()),
                int(draw_winner_mask.sum()),
                _sample_records(merged, draw_winner_mask, ["game_id", "api_fixture_id", "home_club_goals", "away_club_goals", "api_home_winner", "api_away_winner"]),
            ),
        ],
        "relationships",
    )
    return {
        "checks": checks,
        "summary": summarize_dimension_checks(checks),
    }


def run_merged_fixture_validation(
    raw_dir: str | Path = "data/player_scores_data",
    api_cache_path: str | Path = DEFAULT_API_FOOTBALL_CACHE_PATH,
    api_fetch_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_payload = build_merged_fixture_dataset(
        raw_dir=raw_dir,
        api_cache_path=api_cache_path,
        api_fetch_result=api_fetch_result,
    )
    merged = merged_payload["merged_fixtures"]
    unmatched_api = merged_payload["unmatched_api"]
    unmatched_kaggle = merged_payload["unmatched_kaggle"]
    merge_summary = merged_payload["merge_summary"]

    results = {
        **merged_payload,
        "accuracy": _run_merged_accuracy_checks(merged),
        "consistency": _run_merged_consistency_checks(merged),
        "completeness": _run_merged_completeness_checks(merged, unmatched_api, unmatched_kaggle, merge_summary),
        "uniqueness": _run_merged_uniqueness_checks(merged),
        "outliers": _run_merged_outlier_checks(merged),
        "timeliness": _run_merged_timeliness_checks(merged),
        "distribution": _run_merged_distribution_checks(merged),
        "relationships": _run_merged_relationship_checks(merged),
    }
    results["report_snippets"] = build_merged_report_snippets(results)
    return results


def _failed_merged_check_names(checks: pd.DataFrame, limit: int = 3) -> str:
    if checks.empty or "status" not in checks.columns:
        return "no major issues"
    failed = checks.loc[checks["status"] == "failed", "check_name"].astype(str).head(limit).tolist()
    return ", ".join(failed) if failed else "no major issues"


def build_merged_report_snippets(merged_results: dict[str, Any]) -> pd.DataFrame:
    metadata = merged_results["merge_summary"].iloc[0]
    matched_rows = int(metadata["matched_rows"])
    api_rows = int(metadata["api_rows"])
    kaggle_rows = int(metadata["kaggle_rows_considered"])
    api_unmatched_rows = int(metadata["api_unmatched_rows"])
    kaggle_unmatched_rows = int(metadata["kaggle_unmatched_rows"])

    accuracy_failed_checks = int(merged_results["accuracy"]["summary"]["failed_checks"].sum())
    consistency_failed_checks = int(merged_results["consistency"]["summary"]["failed_checks"].sum())
    completeness_failed_checks = int(merged_results["completeness"]["summary"]["failed_checks"].sum())
    uniqueness_failed_checks = int(merged_results["uniqueness"]["summary"]["failed_checks"].sum())
    timeliness_failed_checks = int(merged_results["timeliness"]["summary"]["failed_checks"].sum())
    relationship_failed_checks = int(merged_results["relationships"]["summary"]["failed_checks"].sum())

    required_missingness = merged_results["completeness"]["field_missingness"]
    required_missingness = required_missingness[required_missingness["field_role"] == "required"]
    max_required_missing_pct = round(float(required_missingness["missing_pct"].max()), 4) if not required_missingness.empty else 0.0

    outlier_count = int(merged_results["outliers"]["summary"]["outlier_count"].sum()) if not merged_results["outliers"]["summary"].empty else 0
    numeric_profiles = merged_results["distribution"]["numeric_profiles"]
    home_goals_row = numeric_profiles[numeric_profiles["column_name"] == "home_club_goals"].head(1)
    away_goals_row = numeric_profiles[numeric_profiles["column_name"] == "away_club_goals"].head(1)
    home_goal_median = float(home_goals_row["median"].iloc[0]) if not home_goals_row.empty else 0.0
    away_goal_median = float(away_goals_row["median"].iloc[0]) if not away_goals_row.empty else 0.0
    status_profiles = merged_results["distribution"]["categorical_profiles"]
    status_row = status_profiles[status_profiles["column_name"] == "api_status_short"].head(1)
    dominant_status = status_row["most_frequent_value"].iloc[0] if not status_row.empty else "FT"

    accuracy_issue_names = _failed_merged_check_names(merged_results["accuracy"]["checks"])
    consistency_issue_names = _failed_merged_check_names(merged_results["consistency"]["checks"])
    completeness_issue_names = _failed_merged_check_names(merged_results["completeness"]["checks"])
    uniqueness_issue_names = _failed_merged_check_names(merged_results["uniqueness"]["checks"])
    timeliness_issue_names = _failed_merged_check_names(merged_results["timeliness"]["checks"])
    relationship_issue_names = _failed_merged_check_names(merged_results["relationships"]["checks"])

    rows = [
        {
            "dimension": "Accuracy",
            "status": "passed" if accuracy_failed_checks == 0 else "failed",
            "report_text": (
                "Accuracy was evaluated through cross-source agreement on scores, dates, season values, and league mapping. "
                f"All applied checks passed across the `{matched_rows}` merged matches, so the Kaggle and API records aligned cleanly on the shared match entity."
                if accuracy_failed_checks == 0
                else "Accuracy was evaluated through cross-source agreement on scores, dates, season values, and league mapping. "
                f"The merged table showed {accuracy_failed_checks} accuracy issue(s), mainly in {accuracy_issue_names}."
            ),
        },
        {
            "dimension": "Consistency",
            "status": "passed" if consistency_failed_checks == 0 else "failed",
            "report_text": (
                "Consistency was evaluated by checking stable competition coverage plus paired API status and team-identity fields. "
                "All consistency checks passed, which indicates that the merged slice used one coherent schema for the integrated match records."
                if consistency_failed_checks == 0
                else "Consistency was evaluated by checking stable competition coverage plus paired API status and team-identity fields. "
                f"The merged slice showed {consistency_failed_checks} consistency issue(s), mainly in {consistency_issue_names}."
            ),
        },
        {
            "dimension": "Completeness",
            "status": "passed" if completeness_failed_checks == 0 else "failed",
            "report_text": (
                "Completeness was evaluated on required merged fields and merge coverage across both sources. "
                f"All `{api_rows}` API fixtures matched `{kaggle_rows}` Kaggle Premier League rows, and the required merged fields were fully populated."
                if completeness_failed_checks == 0
                else "Completeness was evaluated on required merged fields and merge coverage across both sources. "
                f"The merged table showed {completeness_failed_checks} completeness issue(s), with up to {max_required_missing_pct}% missingness in required fields and unmatched rows of {api_unmatched_rows} API records plus {kaggle_unmatched_rows} Kaggle records; main issues were {completeness_issue_names}."
            ),
        },
        {
            "dimension": "Uniqueness",
            "status": "passed" if uniqueness_failed_checks == 0 else "failed",
            "report_text": (
                "Uniqueness was evaluated on exact duplicates, `game_id`, `api_fixture_id`, and the merged natural match key. "
                "No duplicate merged rows were detected, so both technical identifiers and the business key remained unique."
                if uniqueness_failed_checks == 0
                else "Uniqueness was evaluated on exact duplicates, `game_id`, `api_fixture_id`, and the merged natural match key. "
                f"The merged table showed {uniqueness_failed_checks} uniqueness issue(s), mainly in {uniqueness_issue_names}."
            ),
        },
        {
            "dimension": "Outliers",
            "status": "passed",
            "report_text": (
                "Outliers were evaluated with the IQR method on home goals, away goals, and attendance. "
                + (
                    f"The merged table produced `{outlier_count}` outlier value(s), which should be interpreted as plausible extreme football outcomes or attendance peaks rather than automatic errors."
                    if outlier_count > 0
                    else "No IQR outliers were flagged in the selected merged numeric fields."
                )
            ),
        },
        {
            "dimension": "Timeliness",
            "status": "passed" if timeliness_failed_checks == 0 else "failed",
            "report_text": (
                "Timeliness was evaluated by parsing both source dates, checking cross-source date alignment, validating the 2024/2025 season window, and ensuring completed fixtures were not future-dated. "
                "All timeliness checks passed, so the merged records were temporally aligned and suitable for downstream analysis."
                if timeliness_failed_checks == 0
                else "Timeliness was evaluated by parsing both source dates, checking cross-source date alignment, validating the 2024/2025 season window, and ensuring completed fixtures were not future-dated. "
                f"The merged table showed {timeliness_failed_checks} timeliness issue(s), mainly in {timeliness_issue_names}."
            ),
        },
        {
            "dimension": "Distribution",
            "status": "passed",
            "report_text": (
                "Distribution was profiled on goals, attendance, match formations, and API match status. "
                f"The integrated match data remained low-scoring overall, with median home goals of `{home_goal_median:.1f}`, median away goals of `{away_goal_median:.1f}`, and `{dominant_status}` as the dominant API status."
            ),
        },
        {
            "dimension": "Relationships",
            "status": "passed" if relationship_failed_checks == 0 else "failed",
            "report_text": (
                "Relationships were evaluated by checking whether API winner flags matched the merged score differences. "
                "All dependency checks passed, indicating that the integrated rows preserved the expected link between goals and match outcomes."
                if relationship_failed_checks == 0
                else "Relationships were evaluated by checking whether API winner flags matched the merged score differences. "
                f"The merged table showed {relationship_failed_checks} relationship issue(s), mainly in {relationship_issue_names}."
            ),
        },
    ]
    return pd.DataFrame(rows)

__all__ = [k for k in globals().keys() if not k.startswith('__') and k not in ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']]
