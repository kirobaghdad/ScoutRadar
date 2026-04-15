from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

try:
    import great_expectations as gx
except ModuleNotFoundError:  # pragma: no cover
    gx = None



from .config import *
from .utils import *

def run_accuracy_checks(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    players = tables.get("players")
    if players is not None and not players.empty:
        frame = players.copy()
        frame["date_of_birth"] = _parse_datetime(frame["date_of_birth"])
        frame["age_at_last_season"] = frame["last_season"] - frame["date_of_birth"].dt.year
        rows.extend(
            _run_table_checks(
                "players",
                frame,
                "accuracy",
                [
                    {
                        "check_name": "age_at_last_season_between_15_and_50",
                        "description": "Derived player age at the recorded last season should be between 15 and 50.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["age_at_last_season"],
                        "kwargs": {"column": "age_at_last_season", "min_value": 15, "max_value": 50},
                    },
                    {
                        "check_name": "height_in_cm_between_140_and_230",
                        "description": "Player height should be between 140 and 230 cm.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["height_in_cm"],
                        "kwargs": {"column": "height_in_cm", "min_value": 140, "max_value": 230},
                    },
                    {
                        "check_name": "market_value_in_eur_non_negative",
                        "description": "Market value should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["market_value_in_eur"],
                        "kwargs": {"column": "market_value_in_eur", "min_value": 0},
                    },
                    {
                        "check_name": "highest_market_value_in_eur_non_negative",
                        "description": "Highest market value should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["highest_market_value_in_eur"],
                        "kwargs": {"column": "highest_market_value_in_eur", "min_value": 0},
                    },
                ],
            )
        )

    clubs = tables.get("clubs")
    if clubs is not None and not clubs.empty:
        rows.extend(
            _run_table_checks(
                "clubs",
                clubs,
                "accuracy",
                [
                    {
                        "check_name": "squad_size_greater_than_zero",
                        "description": "Squad size should be greater than 0.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["squad_size"],
                        "kwargs": {"column": "squad_size", "min_value": 1},
                    },
                    {
                        "check_name": "average_age_between_15_and_45",
                        "description": "Average squad age should be between 15 and 45.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["average_age"],
                        "kwargs": {"column": "average_age", "min_value": 15, "max_value": 45},
                    },
                    {
                        "check_name": "foreigners_percentage_between_0_and_100",
                        "description": "Foreign players percentage should be between 0 and 100.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["foreigners_percentage"],
                        "kwargs": {"column": "foreigners_percentage", "min_value": 0, "max_value": 100},
                    },
                    {
                        "check_name": "stadium_seats_greater_than_zero",
                        "description": "Stadium seats should be greater than 0.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["stadium_seats"],
                        "kwargs": {"column": "stadium_seats", "min_value": 1},
                    },
                ],
            )
        )

    games = tables.get("games")
    if games is not None and not games.empty:
        rows.extend(
            _run_table_checks(
                "games",
                games,
                "accuracy",
                [
                    {
                        "check_name": "home_goals_non_negative",
                        "description": "Home goals should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["home_club_goals"],
                        "kwargs": {"column": "home_club_goals", "min_value": 0},
                    },
                    {
                        "check_name": "away_goals_non_negative",
                        "description": "Away goals should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["away_club_goals"],
                        "kwargs": {"column": "away_club_goals", "min_value": 0},
                    },
                    {
                        "check_name": "attendance_non_negative",
                        "description": "Attendance should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["attendance"],
                        "kwargs": {"column": "attendance", "min_value": 0},
                    },
                ],
            )
        )

    appearances = tables.get("appearances")
    if appearances is not None and not appearances.empty:
        rows.extend(
            _run_table_checks(
                "appearances",
                appearances,
                "accuracy",
                [
                    {
                        "check_name": "minutes_played_between_0_and_130",
                        "description": "Minutes played should be between 0 and 130.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["minutes_played"],
                        "kwargs": {"column": "minutes_played", "min_value": 0, "max_value": 130},
                    },
                    {
                        "check_name": "goals_non_negative",
                        "description": "Goals should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["goals"],
                        "kwargs": {"column": "goals", "min_value": 0},
                    },
                    {
                        "check_name": "assists_non_negative",
                        "description": "Assists should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["assists"],
                        "kwargs": {"column": "assists", "min_value": 0},
                    },
                    {
                        "check_name": "yellow_cards_between_0_and_2",
                        "description": "Yellow cards should be between 0 and 2 per appearance.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["yellow_cards"],
                        "kwargs": {"column": "yellow_cards", "min_value": 0, "max_value": 2},
                    },
                    {
                        "check_name": "red_cards_between_0_and_1",
                        "description": "Red cards should be between 0 and 1 per appearance.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["red_cards"],
                        "kwargs": {"column": "red_cards", "min_value": 0, "max_value": 1},
                    },
                ],
            )
        )

    club_games = tables.get("club_games")
    if club_games is not None and not club_games.empty:
        rows.extend(
            _run_table_checks(
                "club_games",
                club_games,
                "accuracy",
                [
                    {
                        "check_name": "own_goals_non_negative",
                        "description": "Own goals should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["own_goals"],
                        "kwargs": {"column": "own_goals", "min_value": 0},
                    },
                    {
                        "check_name": "opponent_goals_non_negative",
                        "description": "Opponent goals should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["opponent_goals"],
                        "kwargs": {"column": "opponent_goals", "min_value": 0},
                    },
                    {
                        "check_name": "hosting_is_home_or_away",
                        "description": "Hosting should be either Home or Away.",
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "columns": ["hosting"],
                        "kwargs": {"column": "hosting", "value_set": ["Home", "Away"]},
                    },
                    {
                        "check_name": "is_win_is_binary",
                        "description": "is_win should be either 0 or 1.",
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "columns": ["is_win"],
                        "kwargs": {"column": "is_win", "value_set": [0, 1]},
                    },
                ],
            )
        )

    player_valuations = tables.get("player_valuations")
    if player_valuations is not None and not player_valuations.empty:
        rows.extend(
            _run_table_checks(
                "player_valuations",
                player_valuations,
                "accuracy",
                [
                    {
                        "check_name": "market_value_non_negative",
                        "description": "Player valuation should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["market_value_in_eur"],
                        "kwargs": {"column": "market_value_in_eur", "min_value": 0},
                    }
                ],
            )
        )

    transfers = tables.get("transfers")
    if transfers is not None and not transfers.empty:
        rows.extend(
            _run_table_checks(
                "transfers",
                transfers,
                "accuracy",
                [
                    {
                        "check_name": "transfer_fee_non_negative",
                        "description": "Transfer fee should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["transfer_fee"],
                        "kwargs": {"column": "transfer_fee", "min_value": 0},
                    },
                    {
                        "check_name": "market_value_non_negative",
                        "description": "Player market value at transfer time should be non-negative.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["market_value_in_eur"],
                        "kwargs": {"column": "market_value_in_eur", "min_value": 0},
                    },
                ],
            )
        )

    game_events = tables.get("game_events")
    if game_events is not None and not game_events.empty:
        rows.extend(
            _run_table_checks(
                "game_events",
                game_events,
                "accuracy",
                [
                    {
                        "check_name": "minute_between_0_and_130",
                        "description": "Event minute should be between 0 and 130.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["minute"],
                        "kwargs": {"column": "minute", "min_value": 0, "max_value": 130},
                    }
                ],
            )
        )

    game_lineups = tables.get("game_lineups")
    if game_lineups is not None and not game_lineups.empty:
        frame = game_lineups.copy()
        frame["shirt_number_numeric"] = pd.to_numeric(frame["number"], errors="coerce")
        rows.extend(
            _run_table_checks(
                "game_lineups",
                frame,
                "accuracy",
                [
                    {
                        "check_name": "team_captain_is_binary",
                        "description": "team_captain should be either 0 or 1.",
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "columns": ["team_captain"],
                        "kwargs": {"column": "team_captain", "value_set": [0, 1]},
                    },
                    {
                        "check_name": "shirt_number_between_1_and_99",
                        "description": "Numeric shirt numbers should be between 1 and 99.",
                        "expectation_type": "expect_column_values_to_be_between",
                        "columns": ["shirt_number_numeric"],
                        "kwargs": {"column": "shirt_number_numeric", "min_value": 1, "max_value": 99},
                    },
                ],
            )
        )

    return _finalize_dimension_checks(rows, "accuracy")


def run_consistency_checks(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    players = tables.get("players")
    if players is not None and not players.empty:
        frame = players.copy()
        frame["last_season_text"] = _stringify(frame["last_season"])
        rows.extend(
            _run_table_checks(
                "players",
                frame,
                "consistency",
                [
                    {
                        "check_name": "date_of_birth_uses_datetime_format",
                        "description": "date_of_birth should consistently use YYYY-MM-DD HH:MM:SS.",
                        "expectation_type": "expect_column_values_to_match_strftime_format",
                        "columns": ["date_of_birth"],
                        "kwargs": {"column": "date_of_birth", "strftime_format": "%Y-%m-%d %H:%M:%S"},
                    },
                    {
                        "check_name": "contract_expiration_date_uses_datetime_format",
                        "description": "contract_expiration_date should consistently use YYYY-MM-DD HH:MM:SS.",
                        "expectation_type": "expect_column_values_to_match_strftime_format",
                        "columns": ["contract_expiration_date"],
                        "kwargs": {"column": "contract_expiration_date", "strftime_format": "%Y-%m-%d %H:%M:%S"},
                    },
                    {
                        "check_name": "last_season_uses_four_digit_year",
                        "description": "last_season should consistently use a four-digit year.",
                        "expectation_type": "expect_column_values_to_match_regex",
                        "columns": ["last_season_text"],
                        "kwargs": {"column": "last_season_text", "regex": r"^\d{4}$"},
                    },
                ],
            )
        )

    clubs = tables.get("clubs")
    if clubs is not None and not clubs.empty:
        frame = clubs.copy()
        frame["last_season_text"] = _stringify(frame["last_season"])
        frame["average_age_text"] = _stringify_with_one_decimal(frame["average_age"])
        rows.extend(
            _run_table_checks(
                "clubs",
                frame,
                "consistency",
                [
                    {
                        "check_name": "last_season_uses_four_digit_year",
                        "description": "last_season should consistently use a four-digit year.",
                        "expectation_type": "expect_column_values_to_match_regex",
                        "columns": ["last_season_text"],
                        "kwargs": {"column": "last_season_text", "regex": r"^\d{4}$"},
                    },
                    {
                        "check_name": "average_age_uses_one_decimal_place",
                        "description": "average_age should consistently use one decimal place.",
                        "expectation_type": "expect_column_values_to_match_regex",
                        "columns": ["average_age_text"],
                        "kwargs": {"column": "average_age_text", "regex": r"^\d{1,2}\.\d$"},
                    },
                ],
            )
        )

    games = tables.get("games")
    if games is not None and not games.empty:
        frame = games.copy()
        frame["season_text"] = _stringify(frame["season"])
        rows.extend(
            _run_table_checks(
                "games",
                frame,
                "consistency",
                [
                    {
                        "check_name": "date_uses_iso_date_format",
                        "description": "date should consistently use YYYY-MM-DD.",
                        "expectation_type": "expect_column_values_to_match_strftime_format",
                        "columns": ["date"],
                        "kwargs": {"column": "date", "strftime_format": "%Y-%m-%d"},
                    },
                    {
                        "check_name": "season_uses_four_digit_year",
                        "description": "season should consistently use a four-digit year.",
                        "expectation_type": "expect_column_values_to_match_regex",
                        "columns": ["season_text"],
                        "kwargs": {"column": "season_text", "regex": r"^\d{4}$"},
                    },
                ],
            )
        )

    for table_name in ["appearances", "game_events", "game_lineups", "player_valuations"]:
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue
        rows.extend(
            _run_table_checks(
                table_name,
                frame,
                "consistency",
                [
                    {
                        "check_name": "date_uses_iso_date_format",
                        "description": "date should consistently use YYYY-MM-DD.",
                        "expectation_type": "expect_column_values_to_match_strftime_format",
                        "columns": ["date"],
                        "kwargs": {"column": "date", "strftime_format": "%Y-%m-%d"},
                    }
                ],
            )
        )

    transfers = tables.get("transfers")
    if transfers is not None and not transfers.empty:
        rows.extend(
            _run_table_checks(
                "transfers",
                transfers,
                "consistency",
                [
                    {
                        "check_name": "transfer_date_uses_iso_date_format",
                        "description": "transfer_date should consistently use YYYY-MM-DD.",
                        "expectation_type": "expect_column_values_to_match_strftime_format",
                        "columns": ["transfer_date"],
                        "kwargs": {"column": "transfer_date", "strftime_format": "%Y-%m-%d"},
                    },
                    {
                        "check_name": "transfer_season_uses_nn_nn_format",
                        "description": "transfer_season should consistently use the NN/NN format.",
                        "expectation_type": "expect_column_values_to_match_regex",
                        "columns": ["transfer_season"],
                        "kwargs": {"column": "transfer_season", "regex": r"^\d{2}/\d{2}$"},
                    },
                ],
            )
        )

    return _finalize_dimension_checks(rows, "consistency")


def run_completeness_checks(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    table_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    required_rows: list[dict[str, Any]] = []

    for table_name, frame in sorted(tables.items()):
        required_fields = REQUIRED_FIELDS.get(table_name, [])
        total_cells = max(len(frame) * max(len(frame.columns), 1), 1)
        missing_cells = int(frame.isna().sum().sum())
        rows_with_missing = int(frame.isna().any(axis=1).sum()) if not frame.empty else 0

        table_rows.append(
            {
                "table_name": table_name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "missing_cells": missing_cells,
                "missing_cells_pct": round((missing_cells / total_cells) * 100, 4),
                "rows_with_missing": rows_with_missing,
                "rows_with_missing_pct": round((rows_with_missing / max(len(frame), 1)) * 100, 4),
                "expected_min_rows": 1,
                "size_status": "passed" if len(frame) > 0 else "failed",
            }
        )

        for column in frame.columns:
            missing_count = int(frame[column].isna().sum())
            field_rows.append(
                {
                    "table_name": table_name,
                    "column_name": column,
                    "field_role": "required" if column in required_fields else "optional",
                    "missing_count": missing_count,
                    "missing_pct": round((missing_count / max(len(frame), 1)) * 100, 4),
                }
            )

        if frame.empty:
            continue

        validator = _gx_validator(frame, f"completeness_{table_name}")
        for column in required_fields:
            if column not in frame.columns:
                required_rows.append(
                    _skipped_row(
                        table_name,
                        "completeness",
                        f"{column}_required",
                        f"{column} is required and should not be missing.",
                        "expect_column_values_to_not_be_null",
                        frame,
                        [column],
                    )
                )
                continue

            result = _run_expectation(validator, "expect_column_values_to_not_be_null", column=column)
            required_rows.append(
                _check_row(
                    table_name=table_name,
                    dimension="completeness",
                    check_name=f"{column}_required",
                    description=f"{column} is required and should not be missing.",
                    expectation_type="expect_column_values_to_not_be_null",
                    result=result,
                )
            )

    table_summary = pd.DataFrame(table_rows).sort_values(
        ["size_status", "missing_cells_pct", "table_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    field_completeness = pd.DataFrame(field_rows).sort_values(
        ["missing_pct", "table_name", "column_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    required_field_checks = _finalize_dimension_checks(required_rows, "completeness")
    required_field_issues = required_field_checks[required_field_checks["status"] != "passed"].reset_index(drop=True)
    missing_patterns = (
        field_completeness[field_completeness["missing_count"] > 0]
        .assign(pattern=lambda df: df["missing_pct"].map(lambda value: "high_missingness" if value >= 20 else "some_missingness"))
        .reset_index(drop=True)
    )

    return {
        "table_summary": table_summary,
        "field_completeness": field_completeness,
        "required_field_checks": required_field_checks,
        "required_field_issues": required_field_issues,
        "missing_patterns": missing_patterns,
    }


def run_uniqueness_checks(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    exact_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    fuzzy_rows: list[dict[str, Any]] = []

    for table_name, frame in sorted(tables.items()):
        duplicate_rows = int(frame.duplicated().sum())
        exact_rows.append(
            {
                "table_name": table_name,
                "rows": int(len(frame)),
                "exact_duplicate_rows": duplicate_rows,
                "exact_duplicate_pct": round((duplicate_rows / max(len(frame), 1)) * 100, 4),
            }
        )

        for column in STRICT_UNIQUE_FIELDS.get(table_name, []):
            if column not in frame.columns:
                strict_rows.append(
                    {
                        "table_name": table_name,
                        "column_name": column,
                        "status": "skipped",
                        "duplicate_count": None,
                        "duplicate_pct": None,
                        "note": "Column not found.",
                    }
                )
                continue

            series = frame[column].dropna()
            duplicate_count = int(series.duplicated().sum())
            strict_rows.append(
                {
                    "table_name": table_name,
                    "column_name": column,
                    "status": "passed" if duplicate_count == 0 else "failed",
                    "duplicate_count": duplicate_count,
                    "duplicate_pct": round((duplicate_count / max(len(frame), 1)) * 100, 4),
                    "note": "Strict unique field.",
                }
            )

        for column, note in CANDIDATE_UNIQUE_FIELDS.get(table_name, {}).items():
            if column not in frame.columns:
                continue

            series = frame[column].dropna()
            duplicate_count = int(series.duplicated().sum())
            candidate_rows.append(
                {
                    "table_name": table_name,
                    "column_name": column,
                    "duplicate_count": duplicate_count,
                    "duplicate_pct": round((duplicate_count / max(len(frame), 1)) * 100, 4),
                    "note": note,
                }
            )

        keys = SUBSET_UNIQUE_KEYS.get(table_name)
        if not keys:
            continue

        available_keys = [column for column in keys if column in frame.columns]
        if len(available_keys) != len(keys):
            subset_rows.append(
                {
                    "table_name": table_name,
                    "key_fields": ", ".join(keys),
                    "status": "skipped",
                    "duplicate_count": None,
                    "duplicate_groups": None,
                    "duplicate_pct": None,
                    "note": "One or more key fields are missing.",
                }
            )
            continue

        working = frame[available_keys].dropna()
        duplicate_count = int(working.duplicated(subset=available_keys).sum())
        duplicate_groups = int(
            working[working.duplicated(subset=available_keys, keep=False)]
            .drop_duplicates(subset=available_keys)
            .shape[0]
        )
        subset_rows.append(
            {
                "table_name": table_name,
                "key_fields": ", ".join(available_keys),
                "status": "passed" if duplicate_count == 0 else "failed",
                "duplicate_count": duplicate_count,
                "duplicate_groups": duplicate_groups,
                "duplicate_pct": round((duplicate_count / max(len(frame), 1)) * 100, 4),
            }
        )

    players = tables.get("players")
    if players is not None and {"player_id", "name", "date_of_birth"}.issubset(players.columns):
        working = players[["player_id", "name", "date_of_birth"]].dropna().copy()
        working["normalized_name"] = working["name"].map(_normalize_text)

        for date_of_birth, group in working.groupby("date_of_birth"):
            values = group[["player_id", "name", "normalized_name"]].drop_duplicates().to_dict("records")
            if len(values) < 2 or len(values) > 20:
                continue

            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    left = values[i]
                    right = values[j]
                    if left["normalized_name"] == right["normalized_name"]:
                        continue
                    similarity = SequenceMatcher(a=left["normalized_name"], b=right["normalized_name"]).ratio()
                    if similarity >= 0.88:
                        fuzzy_rows.append(
                            {
                                "table_name": "players",
                                "left_player_id": left["player_id"],
                                "left_name": left["name"],
                                "right_player_id": right["player_id"],
                                "right_name": right["name"],
                                "date_of_birth": date_of_birth,
                                "similarity": round(similarity, 4),
                            }
                        )

    exact_duplicates = pd.DataFrame(exact_rows).sort_values(
        ["exact_duplicate_rows", "table_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    strict_unique_checks = pd.DataFrame(strict_rows).sort_values(
        ["status", "duplicate_count", "table_name", "column_name"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    candidate_field_reviews = pd.DataFrame(candidate_rows).sort_values(
        ["duplicate_count", "table_name", "column_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    subset_duplicate_checks = pd.DataFrame(subset_rows).sort_values(
        ["status", "duplicate_count", "table_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    fuzzy_duplicate_candidates = (
        pd.DataFrame(fuzzy_rows).sort_values(["similarity", "left_name"], ascending=[False, True]).reset_index(drop=True)
        if fuzzy_rows
        else pd.DataFrame(columns=["table_name", "left_player_id", "left_name", "right_player_id", "right_name", "date_of_birth", "similarity"])
    )

    summary = pd.DataFrame(
        [
            {
                "check_type": "exact_duplicates",
                "issue_count": int((exact_duplicates["exact_duplicate_rows"] > 0).sum()),
                "affected_rows": int(exact_duplicates["exact_duplicate_rows"].sum()),
            },
            {
                "check_type": "strict_unique_fields",
                "issue_count": int((strict_unique_checks["status"] == "failed").sum()),
                "affected_rows": int(pd.to_numeric(strict_unique_checks["duplicate_count"], errors="coerce").fillna(0).sum()),
            },
            {
                "check_type": "subset_duplicates",
                "issue_count": int((subset_duplicate_checks["status"] == "failed").sum()),
                "affected_rows": int(pd.to_numeric(subset_duplicate_checks["duplicate_count"], errors="coerce").fillna(0).sum()),
            },
            {
                "check_type": "fuzzy_duplicate_candidates",
                "issue_count": int(len(fuzzy_duplicate_candidates)),
                "affected_rows": int(len(fuzzy_duplicate_candidates)),
            },
        ]
    )

    return {
        "summary": summary,
        "exact_duplicates": exact_duplicates,
        "strict_unique_checks": strict_unique_checks,
        "candidate_field_reviews": candidate_field_reviews,
        "subset_duplicate_checks": subset_duplicate_checks,
        "fuzzy_duplicate_candidates": fuzzy_duplicate_candidates,
    }


def run_outlier_checks(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    for table_name, columns in OUTLIER_FIELDS.items():
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue

        working = frame.copy()
        if table_name == "players" and {"last_season", "date_of_birth"}.issubset(working.columns):
            working["date_of_birth"] = _parse_datetime(working["date_of_birth"])
            working["age_at_last_season"] = working["last_season"] - working["date_of_birth"].dt.year

        id_column = next((column for column in ["player_id", "club_id", "game_id"] if column in working.columns), None)
        label_column = next((column for column in ["name", "player_name", "home_club_name"] if column in working.columns), None)

        for column in columns:
            if column not in working.columns:
                continue

            values = pd.to_numeric(working[column], errors="coerce")
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
                    "table_name": table_name,
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

            outlier_rows = working.loc[outlier_mask, [column]].copy()
            outlier_rows["table_name"] = table_name
            outlier_rows["column_name"] = column
            outlier_rows["value"] = pd.to_numeric(outlier_rows[column], errors="coerce")
            outlier_rows["row_id"] = working.loc[outlier_mask, id_column].astype(str).values if id_column else outlier_rows.index.astype(str)
            outlier_rows["label"] = working.loc[outlier_mask, label_column].astype(str).values if label_column else ""

            example_rows.extend(
                outlier_rows.sort_values("value", ascending=False)[["table_name", "column_name", "row_id", "label", "value"]]
                .head(5)
                .to_dict("records")
            )

    outlier_summary = pd.DataFrame(summary_rows).sort_values(
        ["outlier_count", "table_name", "column_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    outlier_examples = pd.DataFrame(example_rows).reset_index(drop=True)

    return {
        "summary": outlier_summary,
        "examples": outlier_examples,
    }


def _empty_dataframe(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _build_monthly_timeliness_profile(table_name: str, valid_dates: pd.Series) -> pd.DataFrame:
    columns = [
        "table_name",
        "year_month",
        "calendar_month",
        "row_count",
        "baseline_median",
        "pct_diff_from_baseline",
        "volume_status",
        "gap_status",
        "is_edge_month",
    ]
    if valid_dates.empty:
        return _empty_dataframe(columns)

    monthly_counts = valid_dates.dt.to_period("M").value_counts().sort_index()
    full_months = pd.period_range(monthly_counts.index.min(), monthly_counts.index.max(), freq="M")
    profile = pd.DataFrame(
        {
            "table_name": table_name,
            "year_month": [str(period) for period in full_months],
            "calendar_month": [period.month for period in full_months],
            "row_count": monthly_counts.reindex(full_months, fill_value=0).to_numpy(),
        }
    )
    profile["is_edge_month"] = False
    profile.loc[[0, len(profile) - 1], "is_edge_month"] = True

    baseline_source = profile.loc[~profile["is_edge_month"], ["calendar_month", "row_count"]]
    month_baselines = baseline_source.groupby("calendar_month")["row_count"].median() if not baseline_source.empty else pd.Series(dtype=float)
    profile["baseline_median"] = pd.to_numeric(profile["calendar_month"].map(month_baselines), errors="coerce")
    profile["pct_diff_from_baseline"] = pd.NA

    valid_baseline = profile["baseline_median"].notna() & profile["baseline_median"].gt(0)
    profile.loc[valid_baseline, "pct_diff_from_baseline"] = (
        (profile.loc[valid_baseline, "row_count"] - profile.loc[valid_baseline, "baseline_median"])
        / profile.loc[valid_baseline, "baseline_median"]
        * 100
    ).round(4)

    analyzable_mask = (
        ~profile["is_edge_month"]
        & valid_baseline
        & profile["baseline_median"].ge(TIMELINESS_MIN_BASELINE_ROWS)
    )
    profile["volume_status"] = "skipped"
    profile.loc[analyzable_mask, "volume_status"] = profile.loc[analyzable_mask, "pct_diff_from_baseline"].abs().le(
        TIMELINESS_MONTHLY_DEVIATION_PCT
    ).map({True: "passed", False: "failed"})

    profile["gap_status"] = "skipped"
    profile.loc[analyzable_mask & profile["row_count"].gt(0), "gap_status"] = "passed"
    profile.loc[analyzable_mask & profile["row_count"].eq(0), "gap_status"] = "failed"

    return profile[columns]


def _distribution_payload_from_numeric(clean: pd.Series, max_bins: int = 8) -> tuple[str, str]:
    unique_count = int(clean.nunique())
    if unique_count <= 10:
        frequency = clean.value_counts().sort_index()
        payload = [
            {"value": round(float(value), 4), "count": int(count)}
            for value, count in frequency.items()
        ]
        return "exact_frequency", json.dumps(payload, ensure_ascii=False, default=str)

    bin_count = min(max_bins, unique_count)
    binned = pd.cut(clean, bins=bin_count, duplicates="drop")
    frequency = binned.value_counts(sort=False)
    payload = [
        {
            "bin_start": round(float(interval.left), 4),
            "bin_end": round(float(interval.right), 4),
            "count": int(count),
        }
        for interval, count in frequency.items()
    ]
    return "equal_width_bins", json.dumps(payload, ensure_ascii=False, default=str)


def _top_frequency_payload(series: pd.Series, max_items: int = 5) -> str:
    clean = series.dropna().astype(str)
    if clean.empty:
        return "[]"

    frequency = clean.value_counts().head(max_items)
    payload = [
        {
            "value": value,
            "count": int(count),
            "pct": round((int(count) / max(len(clean), 1)) * 100, 4),
        }
        for value, count in frequency.items()
    ]
    return json.dumps(payload, ensure_ascii=False, default=str)


def _correlation_strength(value: float) -> str:
    absolute = abs(value)
    if absolute >= 0.9:
        return "very_high"
    if absolute >= 0.7:
        return "high"
    if absolute >= 0.5:
        return "moderate"
    if absolute >= 0.3:
        return "weak"
    return "very_weak"


def run_timeliness_checks(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    check_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    seasonal_pattern_rows: list[dict[str, Any]] = []
    unexpected_gap_rows: list[dict[str, Any]] = []
    current_date = pd.Timestamp.today().normalize()
    coverage_columns = [
        "table_name",
        "date_column",
        "rows",
        "valid_dates",
        "invalid_dates",
        "min_date",
        "max_date",
        "unique_days",
        "unique_months",
        "full_months_in_range",
        "unexpected_gap_months",
        "median_monthly_rows",
        "max_monthly_rows",
    ]
    seasonal_pattern_columns = [
        "table_name",
        "year_month",
        "calendar_month",
        "row_count",
        "baseline_median",
        "pct_diff_from_baseline",
        "volume_status",
        "gap_status",
    ]
    gap_columns = [
        "table_name",
        "year_month",
        "row_count",
        "baseline_median",
        "pct_diff_from_baseline",
        "gap_status",
    ]
    season_volume_columns = ["season", "game_count", "pct_vs_median", "volume_status"]

    for table_name, date_column in TIME_COLUMNS.items():
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue
        if date_column not in frame.columns:
            check_rows.append(
                _skipped_row(
                    table_name,
                    "timeliness",
                    "time_column_exists",
                    f"{date_column} is required for timeliness checks.",
                    "pandas_rule",
                    frame,
                    [date_column],
                )
            )
            continue

        parsed = _parse_datetime(frame[date_column])
        valid_dates = parsed.dropna()
        monthly_counts = valid_dates.dt.to_period("M").value_counts().sort_index()
        profile = (
            _build_monthly_timeliness_profile(table_name, valid_dates)
            if table_name in SEASONAL_TIMELINESS_TABLES
            else _empty_dataframe(seasonal_pattern_columns + ["is_edge_month"])
        )
        analyzable_months = profile[profile["volume_status"] != "skipped"] if not profile.empty else profile
        gap_issues = profile[profile["gap_status"] == "failed"] if not profile.empty else profile
        seasonal_issues = (
            profile[(profile["volume_status"] == "failed") & profile["row_count"].gt(0)]
            if not profile.empty
            else profile
        )

        if not profile.empty:
            seasonal_pattern_rows.extend(profile[seasonal_pattern_columns].to_dict("records"))
            unexpected_gap_rows.extend(gap_issues[gap_columns].to_dict("records"))

        coverage_rows.append(
            {
                "table_name": table_name,
                "date_column": date_column,
                "rows": int(len(frame)),
                "valid_dates": int(valid_dates.shape[0]),
                "invalid_dates": int(parsed.isna().sum()),
                "min_date": valid_dates.min(),
                "max_date": valid_dates.max(),
                "unique_days": int(valid_dates.dt.normalize().nunique()) if not valid_dates.empty else 0,
                "unique_months": int(monthly_counts.shape[0]),
                "full_months_in_range": int(profile.shape[0]) if not profile.empty else int(monthly_counts.shape[0]),
                "unexpected_gap_months": int(len(gap_issues)),
                "median_monthly_rows": int(monthly_counts.median()) if not monthly_counts.empty else 0,
                "max_monthly_rows": int(monthly_counts.max()) if not monthly_counts.empty else 0,
            }
        )

        invalid_mask = parsed.isna()
        check_rows.append(
            _manual_check_row(
                table_name,
                "timeliness",
                "date_values_are_parseable",
                f"{date_column} should contain valid dates.",
                len(frame),
                int(invalid_mask.sum()),
                frame.loc[invalid_mask, date_column].astype(str).head(5).tolist(),
            )
        )

        future_mask = parsed > current_date
        check_rows.append(
            _manual_check_row(
                table_name,
                "timeliness",
                "no_future_dates",
                f"{date_column} should not be later than {current_date.date()}.",
                int(parsed.notna().sum()),
                int(future_mask.sum()),
                parsed.loc[future_mask].dt.strftime("%Y-%m-%d").head(5).tolist(),
            )
        )

        if table_name in SEASONAL_TIMELINESS_TABLES and not profile.empty:
            check_rows.append(
                _manual_check_row(
                    table_name,
                    "timeliness",
                    "monthly_volume_within_expected_range",
                    f"Active months should stay within +/-{int(TIMELINESS_MONTHLY_DEVIATION_PCT)}% of the historical median for the same calendar month.",
                    int(len(analyzable_months)),
                    int((analyzable_months["volume_status"] == "failed").sum()),
                    analyzable_months.loc[
                        analyzable_months["volume_status"] == "failed",
                        ["year_month", "row_count", "baseline_median", "pct_diff_from_baseline"],
                    ]
                    .head(5)
                    .to_dict("records"),
                )
            )
            check_rows.append(
                _manual_check_row(
                    table_name,
                    "timeliness",
                    "seasonal_pattern_consistent",
                    "Monthly activity should follow the usual seasonal pattern for that table.",
                    int(len(analyzable_months[analyzable_months["row_count"] > 0])),
                    int(len(seasonal_issues)),
                    seasonal_issues[["year_month", "row_count", "baseline_median", "pct_diff_from_baseline"]]
                    .head(5)
                    .to_dict("records"),
                )
            )
            check_rows.append(
                _manual_check_row(
                    table_name,
                    "timeliness",
                    "no_unexpected_gap_months",
                    "Months that are normally active should not be completely missing.",
                    int(len(profile[profile["gap_status"] != "skipped"])),
                    int(len(gap_issues)),
                    gap_issues[["year_month", "baseline_median"]].head(5).to_dict("records"),
                )
            )

    games = tables.get("games")
    if games is not None and not games.empty and {"game_id", "date"}.issubset(games.columns):
        game_dates = games[["game_id", "date"]].copy()
        game_dates["game_date"] = _parse_datetime(game_dates["date"])
        game_dates = game_dates[["game_id", "game_date"]]

        for table_name in ["appearances", "game_events", "game_lineups"]:
            frame = tables.get(table_name)
            if frame is None or frame.empty:
                continue
            if not {"game_id", "date"}.issubset(frame.columns):
                check_rows.append(
                    _skipped_row(
                        table_name,
                        "timeliness",
                        "date_matches_games_date",
                        f"{table_name}.date should match games.date for the same game_id.",
                        "pandas_rule",
                        frame,
                        [column for column in ["game_id", "date"] if column not in frame.columns],
                    )
                )
                continue

            merged = frame[["game_id", "date"]].copy()
            merged["child_date"] = _parse_datetime(merged["date"])
            merged = merged.merge(game_dates, on="game_id", how="left")

            applicable_mask = merged["child_date"].notna() & merged["game_date"].notna()
            mismatch_mask = applicable_mask & (merged["child_date"] != merged["game_date"])
            sample = merged.loc[mismatch_mask, ["game_id", "child_date", "game_date"]].head(5).to_dict("records")
            check_rows.append(
                _manual_check_row(
                    table_name,
                    "timeliness",
                    "date_matches_games_date",
                    f"{table_name}.date should match games.date for the same game_id.",
                    int(applicable_mask.sum()),
                    int(mismatch_mask.sum()),
                    sample,
                )
            )

        if {"season", "date"}.issubset(games.columns):
            parsed_games = games[["game_id", "season", "date"]].copy()
            parsed_games["season"] = pd.to_numeric(parsed_games["season"], errors="coerce")
            parsed_games["game_date"] = _parse_datetime(parsed_games["date"])
            parsed_games["date_year"] = parsed_games["game_date"].dt.year
            applicable_mask = parsed_games["season"].notna() & parsed_games["date_year"].notna()
            mismatch_mask = applicable_mask & ~(
                (parsed_games["date_year"] == parsed_games["season"])
                | (parsed_games["date_year"] == parsed_games["season"] + 1)
            )
            sample = parsed_games.loc[mismatch_mask, ["game_id", "season", "date"]].head(5).to_dict("records")
            check_rows.append(
                _manual_check_row(
                    "games",
                    "timeliness",
                    "season_matches_date_year",
                    "Game dates should fall in the listed season year or the following calendar year.",
                    int(applicable_mask.sum()),
                    int(mismatch_mask.sum()),
                    sample,
                )
            )

            season_volume = (
                parsed_games.dropna(subset=["season"])
                .groupby("season")
                .size()
                .rename("game_count")
                .sort_index()
                .reset_index()
            )
            season_volume["season"] = season_volume["season"].astype(int)
            if not season_volume.empty:
                latest_season = int(season_volume["season"].max())
                baseline = season_volume.loc[season_volume["season"] < latest_season, "game_count"]
                if baseline.empty:
                    baseline = season_volume["game_count"]
                median_volume = float(baseline.median())
                lower_bound = median_volume * 0.8
                upper_bound = median_volume * 1.2
                season_volume["pct_vs_median"] = (
                    (season_volume["game_count"] - median_volume) / max(median_volume, 1) * 100
                ).round(4)
                season_volume["volume_status"] = season_volume["game_count"].map(
                    lambda value: "passed" if lower_bound <= value <= upper_bound else "failed"
                )

                completed_outliers = season_volume[
                    (season_volume["season"] < latest_season) & (season_volume["volume_status"] == "failed")
                ]
                check_rows.append(
                    _manual_check_row(
                        "games",
                        "timeliness",
                        "season_volume_within_expected_range",
                        "Completed seasons should have game counts within +/-20% of the typical season volume.",
                        int((season_volume["season"] < latest_season).sum()),
                        int(len(completed_outliers)),
                        completed_outliers[["season", "game_count"]].head(5).to_dict("records"),
                    )
                )
            else:
                season_volume = _empty_dataframe(season_volume_columns)
        else:
            season_volume = _empty_dataframe(season_volume_columns)
    else:
        season_volume = _empty_dataframe(season_volume_columns)

    transfers = tables.get("transfers")
    if transfers is not None and not transfers.empty and {"transfer_date", "transfer_season"}.issubset(transfers.columns):
        working = transfers[["transfer_date", "transfer_season", "player_name"]].copy()
        working["transfer_date"] = _parse_datetime(working["transfer_date"])
        season_parts = working["transfer_season"].astype(str).str.extract(r"^(?P<start>\d{2})/(?P<end>\d{2})$")
        working["start_year"] = season_parts["start"].map(_short_year_to_full)
        working["end_year"] = season_parts["end"].map(_short_year_to_full)
        working["transfer_year"] = working["transfer_date"].dt.year

        applicable_mask = working["transfer_year"].notna() & working["start_year"].notna() & working["end_year"].notna()
        mismatch_mask = applicable_mask & ~(
            (working["transfer_year"] == working["start_year"])
            | (working["transfer_year"] == working["end_year"])
        )
        sample = working.loc[mismatch_mask, ["transfer_date", "transfer_season", "player_name"]].head(5).to_dict("records")
        check_rows.append(
            _manual_check_row(
                "transfers",
                "timeliness",
                "transfer_season_matches_transfer_date",
                "transfer_date should align with the listed transfer_season.",
                int(applicable_mask.sum()),
                int(mismatch_mask.sum()),
                sample,
            )
        )

    for table_name, keys in TIMELINESS_DUPLICATE_KEYS.items():
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue

        missing_columns = [column for column in keys if column not in frame.columns]
        if missing_columns:
            check_rows.append(
                _skipped_row(
                    table_name,
                    "timeliness",
                    "no_duplicate_time_keys",
                    f"{', '.join(keys)} should uniquely identify time-based records.",
                    "pandas_rule",
                    frame,
                    missing_columns,
                )
            )
            continue

        working = frame[keys].dropna()
        duplicate_mask = working.duplicated(subset=keys, keep=False)
        duplicate_count = int(working.duplicated(subset=keys).sum())
        sample = working.loc[duplicate_mask].head(5).to_dict("records")
        check_rows.append(
            _manual_check_row(
                table_name,
                "timeliness",
                "no_duplicate_time_keys",
                f"{', '.join(keys)} should uniquely identify time-based records.",
                int(len(working)),
                duplicate_count,
                sample,
            )
        )

    checks = _finalize_dimension_checks(check_rows, "timeliness")
    coverage = (
        pd.DataFrame(coverage_rows, columns=coverage_columns).sort_values(["table_name"], ascending=[True]).reset_index(drop=True)
        if coverage_rows
        else _empty_dataframe(coverage_columns)
    )
    seasonal_patterns = (
        pd.DataFrame(seasonal_pattern_rows, columns=seasonal_pattern_columns)
        .sort_values(["table_name", "year_month"], ascending=[True, True])
        .reset_index(drop=True)
        if seasonal_pattern_rows
        else _empty_dataframe(seasonal_pattern_columns)
    )
    monthly_gaps = (
        pd.DataFrame(unexpected_gap_rows, columns=gap_columns)
        .sort_values(["table_name", "year_month"], ascending=[True, True])
        .reset_index(drop=True)
        if unexpected_gap_rows
        else _empty_dataframe(gap_columns)
    )

    return {
        "checks": checks,
        "coverage": coverage,
        "seasonal_patterns": seasonal_patterns,
        "monthly_gaps": monthly_gaps,
        "season_volume": season_volume.reset_index(drop=True),
    }


def run_distribution_checks(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []

    for table_name, columns in DISTRIBUTION_NUMERIC_FIELDS.items():
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue

        for column in columns:
            if column not in frame.columns:
                continue

            numeric = pd.to_numeric(frame[column], errors="coerce")
            clean = numeric.dropna()
            if clean.empty:
                continue

            distribution_type, histogram_payload = _distribution_payload_from_numeric(clean)
            skewness = clean.skew()
            kurtosis = clean.kurt()
            numeric_rows.append(
                {
                    "table_name": table_name,
                    "column_name": column,
                    "non_null_count": int(clean.shape[0]),
                    "missing_count": int(numeric.isna().sum()),
                    "missing_pct": round((int(numeric.isna().sum()) / max(len(frame), 1)) * 100, 4),
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

    for table_name, columns in DISTRIBUTION_CATEGORICAL_FIELDS.items():
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue

        for column in columns:
            if column not in frame.columns:
                continue

            clean = frame[column].dropna().astype(str)
            if clean.empty:
                continue

            frequency = clean.value_counts()
            top_value = frequency.index[0]
            top_count = int(frequency.iloc[0])
            categorical_rows.append(
                {
                    "table_name": table_name,
                    "column_name": column,
                    "non_null_count": int(clean.shape[0]),
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_pct": round((int(frame[column].isna().sum()) / max(len(frame), 1)) * 100, 4),
                    "distinct_count": int(clean.nunique()),
                    "mode_value": top_value,
                    "mode_count": top_count,
                    "mode_pct": round((top_count / max(len(clean), 1)) * 100, 4),
                    "top_frequencies": _top_frequency_payload(clean),
                }
            )

    numeric_profiles = (
        pd.DataFrame(numeric_rows)
        .sort_values(["table_name", "abs_skewness", "column_name"], ascending=[True, False, True])
        .reset_index(drop=True)
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
        )
    )
    categorical_profiles = (
        pd.DataFrame(categorical_rows)
        .sort_values(["table_name", "distinct_count", "column_name"], ascending=[True, False, True])
        .reset_index(drop=True)
        if categorical_rows
        else _empty_dataframe(
            [
                "table_name",
                "column_name",
                "non_null_count",
                "missing_count",
                "missing_pct",
                "distinct_count",
                "mode_value",
                "mode_count",
                "mode_pct",
                "top_frequencies",
            ]
        )
    )

    profiled_tables = sorted(set(numeric_profiles.get("table_name", pd.Series(dtype=str))) | set(categorical_profiles.get("table_name", pd.Series(dtype=str))))
    summary_rows: list[dict[str, Any]] = []
    for table_name in profiled_tables:
        numeric_subset = numeric_profiles[numeric_profiles["table_name"] == table_name]
        categorical_subset = categorical_profiles[categorical_profiles["table_name"] == table_name]
        summary_rows.append(
            {
                "table_name": table_name,
                "numeric_fields_profiled": int(len(numeric_subset)),
                "categorical_fields_profiled": int(len(categorical_subset)),
                "high_skew_numeric_fields": int(numeric_subset["abs_skewness"].fillna(0).ge(1).sum()) if not numeric_subset.empty else 0,
                "high_cardinality_categorical_fields": int(categorical_subset["distinct_count"].ge(20).sum()) if not categorical_subset.empty else 0,
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["numeric_fields_profiled", "categorical_fields_profiled", "table_name"], ascending=[False, False, True])
        .reset_index(drop=True)
        if summary_rows
        else _empty_dataframe(
            [
                "table_name",
                "numeric_fields_profiled",
                "categorical_fields_profiled",
                "high_skew_numeric_fields",
                "high_cardinality_categorical_fields",
            ]
        )
    )

    return {
        "summary": summary,
        "numeric_profiles": numeric_profiles,
        "categorical_profiles": categorical_profiles,
    }


def run_relationship_checks(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    correlation_rows: list[dict[str, Any]] = []
    dependency_rows: list[dict[str, Any]] = []

    for table_name, pair_configs in RELATIONSHIP_CORRELATION_PAIRS.items():
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue

        for config in pair_configs:
            left = config["left"]
            right = config["right"]
            if left not in frame.columns or right not in frame.columns:
                continue

            pair_frame = frame[[left, right]].copy()
            pair_frame[left] = pd.to_numeric(pair_frame[left], errors="coerce")
            pair_frame[right] = pd.to_numeric(pair_frame[right], errors="coerce")
            pair_frame = pair_frame.dropna()
            if len(pair_frame) < RELATIONSHIP_MIN_SHARED_ROWS:
                continue

            pearson_corr = pair_frame[left].corr(pair_frame[right], method="pearson")
            spearman_corr = pair_frame[left].corr(pair_frame[right], method="spearman")
            max_abs_corr = max(abs(float(pearson_corr)), abs(float(spearman_corr)))
            stronger_method = "pearson" if abs(float(pearson_corr)) >= abs(float(spearman_corr)) else "spearman"
            stronger_value = float(pearson_corr) if stronger_method == "pearson" else float(spearman_corr)

            correlation_rows.append(
                {
                    "table_name": table_name,
                    "left_column": left,
                    "right_column": right,
                    "description": config["description"],
                    "shared_rows": int(len(pair_frame)),
                    "pearson_corr": round(float(pearson_corr), 4),
                    "spearman_corr": round(float(spearman_corr), 4),
                    "max_abs_corr": round(max_abs_corr, 4),
                    "stronger_method": stronger_method,
                    "stronger_corr": round(stronger_value, 4),
                    "direction": "positive" if stronger_value >= 0 else "negative",
                    "strength": _correlation_strength(stronger_value),
                }
            )

    clubs = tables.get("clubs")
    if clubs is not None and not clubs.empty and {"club_id", "squad_size", "foreigners_number", "foreigners_percentage"}.issubset(clubs.columns):
        working = clubs[["club_id", "squad_size", "foreigners_number", "foreigners_percentage"]].copy()
        for column in ["squad_size", "foreigners_number", "foreigners_percentage"]:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.dropna(subset=["squad_size", "foreigners_number", "foreigners_percentage"])
        working = working[working["squad_size"] > 0]
        working["expected_foreigners_pct"] = (working["foreigners_number"] / working["squad_size"]) * 100
        working["abs_pct_diff"] = (working["foreigners_percentage"] - working["expected_foreigners_pct"]).abs()
        mismatch_mask = working["abs_pct_diff"] > RELATIONSHIP_FOREIGNERS_PCT_TOLERANCE
        dependency_rows.append(
            _manual_check_row(
                "clubs",
                "relationships",
                "foreigners_percentage_matches_counts",
                "foreigners_percentage should match foreigners_number divided by squad_size.",
                int(len(working)),
                int(mismatch_mask.sum()),
                working.loc[mismatch_mask, ["club_id", "foreigners_percentage", "expected_foreigners_pct", "abs_pct_diff"]]
                .head(5)
                .to_dict("records"),
            )
        )

    club_games = tables.get("club_games")
    if club_games is not None and not club_games.empty and {"game_id", "club_id", "own_goals", "opponent_goals", "is_win"}.issubset(club_games.columns):
        working = club_games[["game_id", "club_id", "own_goals", "opponent_goals", "is_win"]].copy()
        for column in ["own_goals", "opponent_goals", "is_win"]:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.dropna(subset=["own_goals", "opponent_goals", "is_win"])
        working["expected_is_win"] = (working["own_goals"] > working["opponent_goals"]).astype(int)
        mismatch_mask = working["is_win"] != working["expected_is_win"]
        dependency_rows.append(
            _manual_check_row(
                "club_games",
                "relationships",
                "is_win_matches_scoreline",
                "is_win should equal 1 when own_goals exceeds opponent_goals, otherwise 0.",
                int(len(working)),
                int(mismatch_mask.sum()),
                working.loc[mismatch_mask, ["game_id", "club_id", "own_goals", "opponent_goals", "is_win", "expected_is_win"]]
                .head(5)
                .to_dict("records"),
            )
        )

    players = tables.get("players")
    if players is not None and not players.empty and {"player_id", "market_value_in_eur", "highest_market_value_in_eur"}.issubset(players.columns):
        working = players[["player_id", "market_value_in_eur", "highest_market_value_in_eur"]].copy()
        for column in ["market_value_in_eur", "highest_market_value_in_eur"]:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.dropna(subset=["market_value_in_eur", "highest_market_value_in_eur"])
        mismatch_mask = working["market_value_in_eur"] > working["highest_market_value_in_eur"]
        dependency_rows.append(
            _manual_check_row(
                "players",
                "relationships",
                "current_value_not_above_highest_value",
                "Current market value should not exceed highest recorded market value.",
                int(len(working)),
                int(mismatch_mask.sum()),
                working.loc[mismatch_mask, ["player_id", "market_value_in_eur", "highest_market_value_in_eur"]]
                .head(5)
                .to_dict("records"),
            )
        )

    correlations = (
        pd.DataFrame(correlation_rows)
        .sort_values(["max_abs_corr", "table_name", "left_column"], ascending=[False, True, True])
        .reset_index(drop=True)
        if correlation_rows
        else _empty_dataframe(
            [
                "table_name",
                "left_column",
                "right_column",
                "description",
                "shared_rows",
                "pearson_corr",
                "spearman_corr",
                "max_abs_corr",
                "stronger_method",
                "stronger_corr",
                "direction",
                "strength",
            ]
        )
    )
    strong_correlations = (
        correlations[correlations["max_abs_corr"] >= RELATIONSHIP_STRONG_THRESHOLD].reset_index(drop=True)
        if not correlations.empty
        else correlations
    )
    dependency_checks = _finalize_dimension_checks(dependency_rows, "relationships")

    summary_rows: list[dict[str, Any]] = []
    all_tables = sorted(set(correlations.get("table_name", pd.Series(dtype=str))) | set(dependency_checks.get("table_name", pd.Series(dtype=str))))
    for table_name in all_tables:
        table_corr = correlations[correlations["table_name"] == table_name]
        table_checks = dependency_checks[dependency_checks["table_name"] == table_name]
        summary_rows.append(
            {
                "table_name": table_name,
                "correlation_pairs_profiled": int(len(table_corr)),
                "strong_pairs": int((table_corr["max_abs_corr"] >= RELATIONSHIP_STRONG_THRESHOLD).sum()) if not table_corr.empty else 0,
                "dependency_checks_run": int(len(table_checks)),
                "failed_dependency_checks": int((table_checks["status"] == "failed").sum()) if not table_checks.empty else 0,
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["strong_pairs", "correlation_pairs_profiled", "table_name"], ascending=[False, False, True])
        .reset_index(drop=True)
        if summary_rows
        else _empty_dataframe(
            [
                "table_name",
                "correlation_pairs_profiled",
                "strong_pairs",
                "dependency_checks_run",
                "failed_dependency_checks",
            ]
        )
    )

    return {
        "summary": summary,
        "correlations": correlations,
        "strong_correlations": strong_correlations,
        "dependency_checks": dependency_checks,
    }


def summarize_dimension_checks(checks: pd.DataFrame) -> pd.DataFrame:
    if checks.empty:
        return pd.DataFrame()

    working = checks.copy()
    working["failed_rows"] = pd.to_numeric(working["failed_rows"], errors="coerce").fillna(0).astype(int)
    summary = (
        working.groupby(["dimension", "table_name"])
        .agg(
            checks_run=("check_name", "count"),
            failed_checks=("status", lambda s: int((s == "failed").sum())),
            skipped_checks=("status", lambda s: int((s == "skipped").sum())),
            total_failed_rows=("failed_rows", "sum"),
        )
        .reset_index()
        .sort_values(
            ["dimension", "failed_checks", "total_failed_rows", "table_name"],
            ascending=[True, False, False, True],
        )
    )
    return summary

__all__ = [k for k in globals().keys() if not k.startswith('__') and k not in ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']]
