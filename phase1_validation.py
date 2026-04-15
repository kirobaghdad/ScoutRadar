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


REQUIRED_FIELDS = {
    "competitions": ["competition_id", "name", "type"],
    "clubs": ["club_id", "name", "domestic_competition_id"],
    "games": ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"],
    "players": ["player_id", "name", "last_season", "current_club_id", "date_of_birth", "position"],
    "appearances": ["appearance_id", "game_id", "player_id", "date", "minutes_played"],
    "club_games": ["game_id", "club_id", "opponent_id", "hosting", "is_win"],
    "game_events": ["game_event_id", "game_id", "date", "minute", "type"],
    "game_lineups": ["game_lineups_id", "game_id", "player_id", "date", "club_id", "player_name"],
    "player_valuations": ["player_id", "date", "market_value_in_eur"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
}

STRICT_UNIQUE_FIELDS = {
    "players": ["player_id", "url"],
    "clubs": ["club_id", "club_code", "url"],
    "competitions": ["competition_id", "url"],
    "games": ["game_id", "url"],
    "appearances": ["appearance_id"],
    "game_events": ["game_event_id"],
    "game_lineups": ["game_lineups_id"],
}

CANDIDATE_UNIQUE_FIELDS = {
    "players": {
        "name": "Player names can repeat across different people.",
        "player_code": "player_code behaves like a slug and may repeat for similar names.",
    },
    "clubs": {
        "name": "Club names are business-key candidates, not strict technical keys.",
    },
    "competitions": {
        "name": "Competition names may repeat across countries.",
        "competition_code": "Competition codes may repeat across countries.",
    },
}

SUBSET_UNIQUE_KEYS = {
    "player_valuations": ["player_id", "date"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
    "club_games": ["game_id", "club_id"],
    "appearances": ["game_id", "player_id"],
    "game_lineups": ["game_id", "player_id", "club_id"],
    "games": ["competition_id", "season", "date", "home_club_id", "away_club_id"],
}

# Only use IQR on continuous business fields where extreme values are meaningful.
OUTLIER_FIELDS = {
    "players": ["age_at_last_season", "height_in_cm", "market_value_in_eur", "highest_market_value_in_eur"],
    "clubs": ["squad_size", "average_age", "national_team_players", "stadium_seats"],
    "games": ["attendance"],
    "player_valuations": ["market_value_in_eur"],
    "transfers": ["market_value_in_eur"],
}

TIME_COLUMNS = {
    "games": "date",
    "appearances": "date",
    "game_events": "date",
    "game_lineups": "date",
    "player_valuations": "date",
    "transfers": "transfer_date",
}

TIMELINESS_DUPLICATE_KEYS = {
    "games": ["competition_id", "season", "date", "home_club_id", "away_club_id"],
    "appearances": ["game_id", "player_id", "date"],
    "game_lineups": ["game_id", "player_id", "club_id", "date"],
    "player_valuations": ["player_id", "date"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
}

SEASONAL_TIMELINESS_TABLES = {"games", "appearances", "game_events", "game_lineups"}
TIMELINESS_MIN_BASELINE_ROWS = 100
TIMELINESS_MONTHLY_DEVIATION_PCT = 40.0

DISTRIBUTION_NUMERIC_FIELDS = {
    "players": ["height_in_cm", "market_value_in_eur", "highest_market_value_in_eur"],
    "clubs": ["squad_size", "average_age", "foreigners_percentage", "national_team_players", "stadium_seats"],
    "games": ["home_club_goals", "away_club_goals", "home_club_position", "away_club_position", "attendance"],
    "appearances": ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"],
    "club_games": ["own_goals", "opponent_goals", "own_position", "opponent_position"],
    "game_events": ["minute"],
    "player_valuations": ["market_value_in_eur"],
    "transfers": ["transfer_fee", "market_value_in_eur"],
}

DISTRIBUTION_CATEGORICAL_FIELDS = {
    "competitions": ["type", "sub_type", "confederation", "is_major_national_league"],
    "players": ["position", "sub_position", "foot"],
    "clubs": ["domestic_competition_id"],
    "games": ["competition_type", "home_club_formation", "away_club_formation"],
    "appearances": ["competition_id"],
    "club_games": ["hosting", "is_win"],
    "game_events": ["type"],
    "game_lineups": ["type", "position"],
    "player_valuations": ["player_club_domestic_competition_id"],
    "transfers": ["transfer_season"],
}

RELATIONSHIP_CORRELATION_PAIRS = {
    "players": [
        {
            "left": "market_value_in_eur",
            "right": "highest_market_value_in_eur",
            "description": "Current and peak market value should move together.",
        }
    ],
    "clubs": [
        {
            "left": "foreigners_number",
            "right": "foreigners_percentage",
            "description": "Foreign player count should strongly align with foreigner percentage.",
        },
        {
            "left": "squad_size",
            "right": "national_team_players",
            "description": "Larger squads often contain more national-team players.",
        },
    ],
    "games": [
        {
            "left": "home_club_goals",
            "right": "home_club_position",
            "description": "Better-ranked home teams tend to score more.",
        },
        {
            "left": "away_club_goals",
            "right": "away_club_position",
            "description": "Better-ranked away teams tend to score more.",
        },
        {
            "left": "attendance",
            "right": "home_club_position",
            "description": "Higher-ranked home teams often attract larger attendances.",
        },
    ],
    "appearances": [
        {
            "left": "minutes_played",
            "right": "goals",
            "description": "Longer playing time may weakly increase goal totals.",
        },
        {
            "left": "minutes_played",
            "right": "assists",
            "description": "Longer playing time may weakly increase assist totals.",
        },
        {
            "left": "minutes_played",
            "right": "yellow_cards",
            "description": "Longer playing time may slightly increase booking risk.",
        },
    ],
    "club_games": [
        {
            "left": "own_goals",
            "right": "is_win",
            "description": "Scoring more should strongly relate to winning.",
        },
        {
            "left": "opponent_goals",
            "right": "is_win",
            "description": "Conceding more should relate negatively to winning.",
        },
        {
            "left": "own_position",
            "right": "is_win",
            "description": "Better-ranked teams should generally win more often.",
        },
    ],
    "transfers": [
        {
            "left": "transfer_fee",
            "right": "market_value_in_eur",
            "description": "Transfer fee should positively relate to market value.",
        }
    ],
}

RELATIONSHIP_MIN_SHARED_ROWS = 100
RELATIONSHIP_STRONG_THRESHOLD = 0.6
RELATIONSHIP_FOREIGNERS_PCT_TOLERANCE = 1.0

API_FOOTBALL_FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"
DEFAULT_API_FOOTBALL_FIXTURE_PARAMS = {
    "league": 39,
    "season": 2024,
    "status": "FT-AET-PEN",
    "timezone": "Africa/Cairo",
}
API_REQUIRED_WRAPPER_FIELDS = ["get", "parameters", "errors", "results", "paging", "response"]
API_REQUIRED_FIXTURE_FIELDS = [
    "fixture_id",
    "fixture_date",
    "fixture_timestamp",
    "status_short",
    "status_long",
    "league_id",
    "league_name",
    "season",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_goals",
    "away_goals",
]
API_COMPLETED_STATUSES = {"FT", "AET", "PEN"}
DEFAULT_API_FOOTBALL_CACHE_PATH = "data/api_football_fixtures_sample.json"


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
