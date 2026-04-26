from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.validation.api import flatten_fixture_response
from src.validation.utils import _normalize_text, load_primary_tables

logger = logging.getLogger(__name__)

BIG_FIVE_LEAGUES = ("GB1", "ES1", "IT1", "L1", "FR1")
DEFAULT_TRANSFER_START_DATE = "2018-07-01"
DEFAULT_TRANSFER_END_DATE = "2022-06-30"
DEFAULT_FOLLOW_UP_MONTHS = 24
DEFAULT_MINUTES_THRESHOLD = 1800

PLAYER_LOOKBACK_WINDOWS_DAYS = (180, 365)
CLUB_LOOKBACK_WINDOW_DAYS = 365


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float)
    result = numerator.astype(float).div(denominator.where(denominator.ne(0)))
    return result.replace([float("inf"), float("-inf")], pd.NA)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _append_reason(existing: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    updated = existing.fillna("").astype(str)
    updated.loc[mask] = updated.loc[mask].map(lambda value: reason if not value else f"{value};{reason}")
    return updated.replace("", pd.NA)


def _season_start_year(season_value: Any, fallback_date: Any) -> int | None:
    if pd.notna(season_value):
        text = str(season_value).strip()
        if "/" in text:
            try:
                short_year = int(text.split("/")[0])
                return 1900 + short_year if short_year >= 90 else 2000 + short_year
            except ValueError:
                return None
    if pd.isna(fallback_date):
        return None
    fallback_ts = pd.Timestamp(fallback_date)
    return int(fallback_ts.year if fallback_ts.month >= 7 else fallback_ts.year - 1)


def _merge_asof_snapshot(
    anchors: pd.DataFrame,
    history: pd.DataFrame,
    *,
    by_columns: list[str],
    anchor_date_col: str,
    history_date_col: str,
    value_columns: Iterable[str],
    prefix: str,
    allow_exact_matches: bool = True,
) -> pd.DataFrame:
    value_columns = list(value_columns)
    if anchors.empty:
        return anchors.copy()

    result = anchors.reset_index(drop=True).copy()
    result["_row_id"] = range(len(result))

    right_columns = list(dict.fromkeys(by_columns + [history_date_col] + value_columns))
    right = history[right_columns].copy().sort_values([history_date_col] + by_columns, kind="stable")
    left = result[by_columns + [anchor_date_col, "_row_id"]].copy().sort_values([anchor_date_col] + by_columns, kind="stable")

    merged = pd.merge_asof(
        left,
        right,
        left_on=anchor_date_col,
        right_on=history_date_col,
        by=by_columns,
        direction="backward",
        allow_exact_matches=allow_exact_matches,
    ).sort_values("_row_id", kind="stable")

    renamed = {column: f"{prefix}{column}" for column in value_columns}
    if history_date_col in merged.columns:
        renamed[history_date_col] = f"{prefix}{history_date_col}"
    merge_columns = ["_row_id"] + [column for column in renamed if column in merged.columns]
    snapshot = merged[merge_columns].rename(columns=renamed)

    result = result.merge(snapshot, on="_row_id", how="left", sort=False)
    return result.drop(columns=["_row_id"])


def _prepare_cumulative_history(
    frame: pd.DataFrame,
    *,
    by_columns: list[str],
    date_col: str,
    metric_columns: list[str],
) -> pd.DataFrame:
    grouped = frame.groupby(by_columns + [date_col], as_index=False)[metric_columns].sum()
    grouped = grouped.sort_values(by_columns + [date_col], kind="stable").copy()
    for column in metric_columns:
        grouped[f"cum_{column}"] = grouped.groupby(by_columns, dropna=False)[column].cumsum()
    return grouped[by_columns + [date_col] + [f"cum_{column}" for column in metric_columns]]


def _lookup_cumulative_values(
    anchors: pd.DataFrame,
    history: pd.DataFrame,
    *,
    by_columns: list[str],
    anchor_date_col: str,
    history_date_col: str,
    cumulative_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    reference = anchors.reset_index(drop=True).copy()
    reference["_row_id"] = range(len(reference))

    right_columns = by_columns + [history_date_col] + cumulative_columns
    right = history[right_columns].copy().sort_values([history_date_col] + by_columns, kind="stable")
    left = reference[by_columns + [anchor_date_col, "_row_id"]].copy().sort_values([anchor_date_col] + by_columns, kind="stable")

    merged = pd.merge_asof(
        left,
        right,
        left_on=anchor_date_col,
        right_on=history_date_col,
        by=by_columns,
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_id", kind="stable")

    out = merged[["_row_id"]].copy()
    for column in cumulative_columns:
        out[f"{prefix}{column}"] = merged[column].fillna(0.0)
    return out


def _attach_window_totals(
    anchors: pd.DataFrame,
    history: pd.DataFrame,
    *,
    by_columns: list[str],
    history_date_col: str,
    window_start_col: str,
    window_end_col: str,
    metric_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    cumulative_columns = [f"cum_{column}" for column in metric_columns]
    end_lookup = _lookup_cumulative_values(
        anchors,
        history,
        by_columns=by_columns,
        anchor_date_col=window_end_col,
        history_date_col=history_date_col,
        cumulative_columns=cumulative_columns,
        prefix="end_",
    )
    start_lookup = _lookup_cumulative_values(
        anchors,
        history,
        by_columns=by_columns,
        anchor_date_col=window_start_col,
        history_date_col=history_date_col,
        cumulative_columns=cumulative_columns,
        prefix="start_",
    )

    result = anchors.reset_index(drop=True).copy()
    for column in metric_columns:
        end_col = f"end_cum_{column}"
        start_col = f"start_cum_{column}"
        result[f"{prefix}{column}"] = end_lookup[end_col] - start_lookup[start_col]
    return result


def _build_transfer_audit(
    transfers: pd.DataFrame,
    clubs: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    big_five_leagues: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audited = transfers.copy()
    audited["transfer_date"] = _ensure_datetime(audited["transfer_date"])
    audited = audited.rename(
        columns={
            "from_club_id": "source_club_id",
            "to_club_id": "destination_club_id",
            "from_club_name": "source_club_name",
            "to_club_name": "destination_club_name",
        }
    )

    destination_lookup = clubs[["club_id", "domestic_competition_id", "name"]].rename(
        columns={
            "club_id": "destination_club_id",
            "domestic_competition_id": "destination_competition_id",
            "name": "destination_club_dataset_name",
        }
    )
    audited = audited.merge(destination_lookup, on="destination_club_id", how="left")

    audited["transfer_key"] = (
        audited["player_id"].astype(str)
        + "_"
        + audited["transfer_date"].dt.strftime("%Y-%m-%d").fillna("missing-date")
        + "_"
        + audited["source_club_id"].astype("Int64").astype(str)
        + "_"
        + audited["destination_club_id"].astype("Int64").astype(str)
    )
    audited["exclusion_reason"] = pd.Series(pd.NA, index=audited.index, dtype="object")

    audited["exclusion_reason"] = _append_reason(audited["exclusion_reason"], audited["transfer_date"].isna(), "missing_transfer_date")
    audited["exclusion_reason"] = _append_reason(
        audited["exclusion_reason"],
        audited["transfer_date"].notna() & audited["transfer_date"].lt(pd.Timestamp(start_date)),
        "before_modeling_window",
    )
    audited["exclusion_reason"] = _append_reason(
        audited["exclusion_reason"],
        audited["transfer_date"].notna() & audited["transfer_date"].gt(pd.Timestamp(end_date)),
        "after_modeling_window",
    )
    audited["exclusion_reason"] = _append_reason(
        audited["exclusion_reason"],
        audited["destination_competition_id"].isna(),
        "missing_destination_competition",
    )
    audited["exclusion_reason"] = _append_reason(
        audited["exclusion_reason"],
        audited["destination_competition_id"].notna() & ~audited["destination_competition_id"].isin(big_five_leagues),
        "destination_not_big_five",
    )

    duplicate_mask = audited.duplicated(["player_id", "transfer_date", "source_club_id", "destination_club_id"], keep="first")
    audited["exclusion_reason"] = _append_reason(audited["exclusion_reason"], duplicate_mask, "duplicate_transfer_business_key")

    excluded = audited[audited["exclusion_reason"].notna()].copy()
    included = audited[audited["exclusion_reason"].isna()].copy()
    return included.reset_index(drop=True), excluded.reset_index(drop=True)


def _prepare_base_transfer_cohort(
    tables: dict[str, pd.DataFrame],
    *,
    start_date: str,
    end_date: str,
    big_five_leagues: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfers = tables["transfers"]
    players = tables["players"].copy()
    clubs = tables["clubs"].copy()
    competitions = tables["competitions"].copy()

    included, excluded = _build_transfer_audit(
        transfers,
        clubs,
        start_date=start_date,
        end_date=end_date,
        big_five_leagues=big_five_leagues,
    )

    players["date_of_birth"] = _ensure_datetime(players["date_of_birth"])
    players = players.rename(
        columns={
            "name": "player_full_name",
            "market_value_in_eur": "player_current_market_value",
        }
    )
    included = included.merge(
        players[
            [
                "player_id",
                "player_full_name",
                "date_of_birth",
                "position",
                "sub_position",
                "foot",
                "height_in_cm",
                "country_of_citizenship",
                "highest_market_value_in_eur",
                "player_current_market_value",
            ]
        ],
        on="player_id",
        how="left",
    )

    club_columns = [
        "club_id",
        "name",
        "domestic_competition_id",
        "squad_size",
        "average_age",
        "foreigners_percentage",
        "national_team_players",
        "stadium_seats",
    ]
    source_clubs = clubs[club_columns].rename(
        columns={
            "club_id": "source_club_id",
            "name": "source_club_dataset_name",
            "domestic_competition_id": "source_competition_id",
            "squad_size": "source_squad_size",
            "average_age": "source_average_age",
            "foreigners_percentage": "source_foreigners_percentage",
            "national_team_players": "source_national_team_players",
            "stadium_seats": "source_stadium_seats",
        }
    )
    destination_clubs = clubs[
        [
            "club_id",
            "squad_size",
            "average_age",
            "foreigners_percentage",
            "national_team_players",
            "stadium_seats",
        ]
    ].rename(
        columns={
            "club_id": "destination_club_id",
            "squad_size": "destination_squad_size",
            "average_age": "destination_average_age",
            "foreigners_percentage": "destination_foreigners_percentage",
            "national_team_players": "destination_national_team_players",
            "stadium_seats": "destination_stadium_seats",
        }
    )

    competitions_lookup = competitions[["competition_id", "name", "country_name"]]
    source_competitions = competitions_lookup.rename(
        columns={
            "competition_id": "source_competition_id",
            "name": "source_competition_name",
            "country_name": "source_country_name",
        }
    )
    destination_competitions = competitions_lookup.rename(
        columns={
            "competition_id": "destination_competition_id",
            "name": "destination_competition_name",
            "country_name": "destination_country_name",
        }
    )

    included = included.merge(source_clubs, on="source_club_id", how="left")
    included = included.merge(destination_clubs, on="destination_club_id", how="left")
    included = included.merge(source_competitions, on="source_competition_id", how="left")
    included = included.merge(destination_competitions, on="destination_competition_id", how="left")

    included["age_at_transfer"] = (
        included["transfer_date"].dt.year
        - included["date_of_birth"].dt.year
        - (
            (included["transfer_date"].dt.month < included["date_of_birth"].dt.month)
            | (
                (included["transfer_date"].dt.month == included["date_of_birth"].dt.month)
                & (included["transfer_date"].dt.day < included["date_of_birth"].dt.day)
            )
        ).astype("Int64")
    )
    included["season_start_year"] = [
        _season_start_year(season_value, transfer_date)
        for season_value, transfer_date in zip(included["transfer_season"], included["transfer_date"], strict=False)
    ]
    included["is_same_league_move"] = included["source_competition_id"].eq(included["destination_competition_id"]).astype(int)
    included["is_same_country_move"] = included["source_country_name"].eq(included["destination_country_name"]).astype(int)
    return included.sort_values("transfer_date", kind="stable").reset_index(drop=True), excluded


def _attach_pre_transfer_valuation_features(cohort: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    if valuations.empty:
        result = cohort.copy()
        for column in [
            "pre_transfer_market_value",
            "pre_transfer_market_value_date",
            "pre_transfer_market_value_source",
            "market_value_180d_prior",
            "market_value_365d_prior",
            "market_value_change_180d",
            "market_value_change_365d",
        ]:
            result[column] = pd.NA
        return result

    valuations = valuations.copy()
    valuations["date"] = _ensure_datetime(valuations["date"])
    valuations = valuations.dropna(subset=["player_id", "date"]).sort_values(["player_id", "date"], kind="stable")

    base = _merge_asof_snapshot(
        cohort,
        valuations,
        by_columns=["player_id"],
        anchor_date_col="transfer_date",
        history_date_col="date",
        value_columns=["market_value_in_eur", "current_club_id", "player_club_domestic_competition_id"],
        prefix="pre_transfer_",
        allow_exact_matches=True,
    )
    base = base.rename(
        columns={
            "pre_transfer_market_value_in_eur": "pre_transfer_market_value",
            "pre_transfer_date": "pre_transfer_market_value_date",
            "pre_transfer_current_club_id": "pre_transfer_market_value_club_id",
            "pre_transfer_player_club_domestic_competition_id": "pre_transfer_market_value_league_id",
        }
    )

    base["pre_transfer_market_value_source"] = "valuation_history"
    fallback_mask = base["pre_transfer_market_value"].isna() & base["market_value_in_eur"].notna()
    base.loc[fallback_mask, "pre_transfer_market_value"] = base.loc[fallback_mask, "market_value_in_eur"]
    base.loc[fallback_mask, "pre_transfer_market_value_source"] = "transfer_row_market_value"

    for window in PLAYER_LOOKBACK_WINDOWS_DAYS:
        cutoff_col = f"valuation_cutoff_{window}d"
        base[cutoff_col] = base["transfer_date"] - pd.Timedelta(days=window)
        base = _merge_asof_snapshot(
            base,
            valuations,
            by_columns=["player_id"],
            anchor_date_col=cutoff_col,
            history_date_col="date",
            value_columns=["market_value_in_eur"],
            prefix=f"value_{window}d_prior_",
            allow_exact_matches=True,
        )
        base = base.rename(
            columns={
                f"value_{window}d_prior_market_value_in_eur": f"market_value_{window}d_prior",
                f"value_{window}d_prior_date": f"market_value_{window}d_prior_date",
            }
        )
        base[f"market_value_change_{window}d"] = base["pre_transfer_market_value"] - base[f"market_value_{window}d_prior"]

    base["transfer_fee_to_market_value_ratio"] = _safe_divide(base["transfer_fee"], base["pre_transfer_market_value"])
    base["transfer_fee_minus_market_value"] = base["transfer_fee"] - base["pre_transfer_market_value"]
    return base


def _attach_player_performance_features(cohort: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    if appearances.empty:
        result = cohort.copy()
        for window in PLAYER_LOOKBACK_WINDOWS_DAYS:
            for metric in ["matches", "minutes", "goals", "assists", "goals_per90", "assists_per90"]:
                result[f"player_{metric}_{window}d_pre"] = 0.0 if metric not in {"goals_per90", "assists_per90"} else pd.NA
        return result

    appearances = appearances.copy()
    appearances["date"] = _ensure_datetime(appearances["date"])
    appearances = appearances.dropna(subset=["player_id", "date"])
    appearances["matches"] = 1.0
    history = _prepare_cumulative_history(
        appearances,
        by_columns=["player_id"],
        date_col="date",
        metric_columns=["matches", "minutes_played", "goals", "assists"],
    )

    result = cohort.copy()
    for window in PLAYER_LOOKBACK_WINDOWS_DAYS:
        start_col = f"player_window_start_{window}d"
        end_col = f"player_window_end_{window}d"
        result[start_col] = result["transfer_date"] - pd.Timedelta(days=window)
        result[end_col] = result["transfer_date"] - pd.Timedelta(days=1)
        result = _attach_window_totals(
            result,
            history,
            by_columns=["player_id"],
            history_date_col="date",
            window_start_col=start_col,
            window_end_col=end_col,
            metric_columns=["matches", "minutes_played", "goals", "assists"],
            prefix=f"player_{window}d_pre_",
        )
        result = result.rename(
            columns={
                f"player_{window}d_pre_matches": f"player_matches_{window}d_pre",
                f"player_{window}d_pre_minutes_played": f"player_minutes_{window}d_pre",
                f"player_{window}d_pre_goals": f"player_goals_{window}d_pre",
                f"player_{window}d_pre_assists": f"player_assists_{window}d_pre",
            }
        )
        result[f"player_goals_per90_{window}d_pre"] = _safe_divide(
            result[f"player_goals_{window}d_pre"] * 90,
            result[f"player_minutes_{window}d_pre"],
        )
        result[f"player_assists_per90_{window}d_pre"] = _safe_divide(
            result[f"player_assists_{window}d_pre"] * 90,
            result[f"player_minutes_{window}d_pre"],
        )
    return result


def _prepare_club_form_history(club_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if club_games.empty or games.empty:
        return pd.DataFrame()

    dates = games[["game_id", "date"]].copy()
    dates["date"] = _ensure_datetime(dates["date"])
    club_form = club_games.merge(dates, on="game_id", how="left")
    club_form = club_form.dropna(subset=["club_id", "date"]).copy()
    club_form["matches"] = 1.0
    club_form["wins"] = pd.to_numeric(club_form["is_win"], errors="coerce").fillna(0.0)
    club_form["draws"] = ((club_form["wins"].eq(0)) & club_form["own_goals"].eq(club_form["opponent_goals"])).astype(float)
    club_form["points"] = (club_form["wins"] * 3) + club_form["draws"]
    club_form["goal_diff"] = pd.to_numeric(club_form["own_goals"], errors="coerce").fillna(0.0) - pd.to_numeric(
        club_form["opponent_goals"], errors="coerce"
    ).fillna(0.0)
    return _prepare_cumulative_history(
        club_form,
        by_columns=["club_id"],
        date_col="date",
        metric_columns=["matches", "wins", "draws", "points", "goal_diff"],
    )


def _attach_club_form_features(cohort: pd.DataFrame, club_history: pd.DataFrame) -> pd.DataFrame:
    if club_history.empty:
        result = cohort.copy()
        for prefix in ["source", "destination"]:
            for metric in ["matches_365d_pre", "win_rate_365d_pre", "points_per_match_365d_pre", "goal_diff_per_match_365d_pre"]:
                result[f"{prefix}_{metric}"] = pd.NA
        return result

    result = cohort.copy()
    for prefix, club_column in [("source", "source_club_id"), ("destination", "destination_club_id")]:
        working = result.rename(columns={club_column: "club_id"}).copy()
        start_col = f"{prefix}_club_window_start"
        end_col = f"{prefix}_club_window_end"
        working[start_col] = working["transfer_date"] - pd.Timedelta(days=CLUB_LOOKBACK_WINDOW_DAYS)
        working[end_col] = working["transfer_date"] - pd.Timedelta(days=1)

        working = _attach_window_totals(
            working,
            club_history,
            by_columns=["club_id"],
            history_date_col="date",
            window_start_col=start_col,
            window_end_col=end_col,
            metric_columns=["matches", "wins", "points", "goal_diff"],
            prefix=f"{prefix}_club_",
        )
        result[f"{prefix}_matches_365d_pre"] = working[f"{prefix}_club_matches"]
        result[f"{prefix}_win_rate_365d_pre"] = _safe_divide(working[f"{prefix}_club_wins"], working[f"{prefix}_club_matches"])
        result[f"{prefix}_points_per_match_365d_pre"] = _safe_divide(
            working[f"{prefix}_club_points"],
            working[f"{prefix}_club_matches"],
        )
        result[f"{prefix}_goal_diff_per_match_365d_pre"] = _safe_divide(
            working[f"{prefix}_club_goal_diff"],
            working[f"{prefix}_club_matches"],
        )
    return result


def _load_cached_api_team_context(cache_dir: str | Path) -> pd.DataFrame:
    cache_root = Path(cache_dir).expanduser().resolve()
    frames: list[pd.DataFrame] = []
    for cache_path in sorted(cache_root.glob("api_football*.json")):
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "payload" in payload:
                payload = payload["payload"]
            if not isinstance(payload, dict):
                continue

            fixtures = flatten_fixture_response(payload)
            if fixtures.empty:
                continue

            home = fixtures[["season", "league_name", "home_team_id", "home_team_name", "home_goals", "away_goals", "home_winner"]].rename(
                columns={
                    "home_team_id": "team_id",
                    "home_team_name": "team_name",
                    "home_goals": "goals_for",
                    "away_goals": "goals_against",
                    "home_winner": "is_win",
                }
            )
            away = fixtures[["season", "league_name", "away_team_id", "away_team_name", "away_goals", "home_goals", "away_winner"]].rename(
                columns={
                    "away_team_id": "team_id",
                    "away_team_name": "team_name",
                    "away_goals": "goals_for",
                    "home_goals": "goals_against",
                    "away_winner": "is_win",
                }
            )
            combined = pd.concat([home, away], ignore_index=True)
            combined["team_key"] = combined["team_name"].map(_normalize_text)
            combined["matches"] = 1.0
            combined["wins"] = pd.to_numeric(combined["is_win"], errors="coerce").fillna(0).astype(int)
            aggregated = combined.groupby(["season", "team_key"], as_index=False).agg(
                api_cached_matches=("matches", "sum"),
                api_cached_wins=("wins", "sum"),
                api_cached_goals_for=("goals_for", "sum"),
                api_cached_goals_against=("goals_against", "sum"),
            )
            aggregated["api_cached_win_rate"] = _safe_divide(aggregated["api_cached_wins"], aggregated["api_cached_matches"])
            aggregated["api_cached_goal_diff_per_match"] = _safe_divide(
                aggregated["api_cached_goals_for"] - aggregated["api_cached_goals_against"],
                aggregated["api_cached_matches"],
            )
            aggregated["api_cache_file"] = cache_path.name
            frames.append(aggregated)
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping API cache %s because it could not be parsed: %s", cache_path, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["season", "team_key"], keep="last").reset_index(drop=True)


def _attach_optional_api_features(cohort: pd.DataFrame, cache_dir: str | Path) -> pd.DataFrame:
    api_context = _load_cached_api_team_context(cache_dir)
    result = cohort.copy()
    for prefix, team_col in [("source", "source_club_name"), ("destination", "destination_club_name")]:
        result[f"{prefix}_api_cached_matches"] = pd.NA
        result[f"{prefix}_api_cached_win_rate"] = pd.NA
        result[f"{prefix}_api_cached_goal_diff_per_match"] = pd.NA
        result[f"{prefix}_api_cache_file"] = pd.NA
        result[f"{prefix}_team_key"] = result[team_col].map(_normalize_text)

    if api_context.empty:
        return result.drop(columns=["source_team_key", "destination_team_key"])

    for prefix, team_key_col in [("source", "source_team_key"), ("destination", "destination_team_key")]:
        merged = result[[team_key_col, "season_start_year"]].merge(
            api_context,
            left_on=[team_key_col, "season_start_year"],
            right_on=["team_key", "season"],
            how="left",
        )
        result[f"{prefix}_api_cached_matches"] = merged["api_cached_matches"]
        result[f"{prefix}_api_cached_win_rate"] = merged["api_cached_win_rate"]
        result[f"{prefix}_api_cached_goal_diff_per_match"] = merged["api_cached_goal_diff_per_match"]
        result[f"{prefix}_api_cache_file"] = merged["api_cache_file"]

    return result.drop(columns=["source_team_key", "destination_team_key"])


def create_transfer_success_labels(
    cohort: pd.DataFrame,
    appearances: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    follow_up_months: int = DEFAULT_FOLLOW_UP_MONTHS,
    minutes_threshold: int = DEFAULT_MINUTES_THRESHOLD,
) -> pd.DataFrame:
    result = cohort.copy()
    result["follow_up_window_end"] = result["transfer_date"] + pd.DateOffset(months=follow_up_months)

    valuation_history = valuations.copy()
    valuation_history["date"] = _ensure_datetime(valuation_history["date"])
    valuation_history = valuation_history.dropna(subset=["player_id", "date"]).sort_values(["player_id", "date"], kind="stable")

    result = _merge_asof_snapshot(
        result,
        valuation_history,
        by_columns=["player_id"],
        anchor_date_col="follow_up_window_end",
        history_date_col="date",
        value_columns=["market_value_in_eur"],
        prefix="target_end_window_",
        allow_exact_matches=True,
    ).rename(
        columns={
            "target_end_window_market_value_in_eur": "target_end_market_value",
            "target_end_window_date": "target_end_market_value_date",
        }
    )

    result["target_market_value_delta_24m"] = result["target_end_market_value"] - result["pre_transfer_market_value"]

    appearance_history = appearances.copy()
    appearance_history["date"] = _ensure_datetime(appearance_history["date"])
    appearance_history = appearance_history.dropna(subset=["player_id", "player_club_id", "date"])
    appearance_history["matches"] = 1.0
    destination_history = _prepare_cumulative_history(
        appearance_history,
        by_columns=["player_id", "player_club_id"],
        date_col="date",
        metric_columns=["matches", "minutes_played", "goals", "assists"],
    )

    lookup = result.rename(columns={"destination_club_id": "player_club_id"}).copy()
    lookup["target_minutes_window_start"] = lookup["transfer_date"] - pd.Timedelta(days=1)
    lookup["target_minutes_window_end"] = lookup["follow_up_window_end"]
    lookup = _attach_window_totals(
        lookup,
        destination_history,
        by_columns=["player_id", "player_club_id"],
        history_date_col="date",
        window_start_col="target_minutes_window_start",
        window_end_col="target_minutes_window_end",
        metric_columns=["matches", "minutes_played", "goals", "assists"],
        prefix="target_destination_",
    )

    result["target_destination_matches_24m"] = lookup["target_destination_matches"]
    result["target_destination_minutes_24m"] = lookup["target_destination_minutes_played"]
    result["target_destination_goals_24m"] = lookup["target_destination_goals"]
    result["target_destination_assists_24m"] = lookup["target_destination_assists"]

    valuation_max_date = valuation_history["date"].max() if not valuation_history.empty else pd.NaT
    data_horizon = valuation_max_date
    result["target_is_eligible"] = True
    result["target_failure_reason"] = pd.Series(pd.NA, index=result.index, dtype="object")

    result["target_is_eligible"] &= result["pre_transfer_market_value"].notna()
    result["target_failure_reason"] = _append_reason(
        result["target_failure_reason"],
        result["pre_transfer_market_value"].isna(),
        "missing_pre_transfer_market_value",
    )

    result["target_is_eligible"] &= result["target_end_market_value"].notna()
    result["target_failure_reason"] = _append_reason(
        result["target_failure_reason"],
        result["target_end_market_value"].isna(),
        "missing_follow_up_market_value",
    )

    result["target_is_eligible"] &= result["target_end_market_value_date"].gt(result["transfer_date"])
    result["target_failure_reason"] = _append_reason(
        result["target_failure_reason"],
        result["target_end_market_value_date"].le(result["transfer_date"]) | result["target_end_market_value_date"].isna(),
        "no_market_value_inside_follow_up_window",
    )

    if pd.notna(data_horizon):
        result["target_is_eligible"] &= result["follow_up_window_end"].le(data_horizon)
        result["target_failure_reason"] = _append_reason(
            result["target_failure_reason"],
            result["follow_up_window_end"].gt(data_horizon),
            "incomplete_follow_up_window",
        )

    success_mask = (
        result["target_destination_minutes_24m"].fillna(0).ge(minutes_threshold)
        & result["target_end_market_value"].ge(result["pre_transfer_market_value"])
    )
    result["transfer_success"] = success_mask.astype(int)
    result.loc[~result["target_is_eligible"], "transfer_success"] = pd.NA
    return result


def _finalize_modeling_dataset(labeled: pd.DataFrame) -> pd.DataFrame:
    modeling_dataset = labeled[labeled["target_is_eligible"]].copy()
    modeling_dataset["transfer_success"] = modeling_dataset["transfer_success"].astype(int)
    modeling_dataset["source_api_cache_available"] = modeling_dataset["source_api_cached_matches"].notna().astype(int)
    modeling_dataset["destination_api_cache_available"] = modeling_dataset["destination_api_cached_matches"].notna().astype(int)

    ordered_columns = [
        "transfer_key",
        "player_id",
        "player_full_name",
        "transfer_date",
        "transfer_season",
        "source_club_id",
        "source_club_name",
        "source_competition_id",
        "source_competition_name",
        "destination_club_id",
        "destination_club_name",
        "destination_competition_id",
        "destination_competition_name",
        "age_at_transfer",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
        "country_of_citizenship",
        "transfer_fee",
        "market_value_in_eur",
        "pre_transfer_market_value",
        "pre_transfer_market_value_source",
        "market_value_180d_prior",
        "market_value_365d_prior",
        "market_value_change_180d",
        "market_value_change_365d",
        "transfer_fee_to_market_value_ratio",
        "transfer_fee_minus_market_value",
        "player_matches_180d_pre",
        "player_minutes_180d_pre",
        "player_goals_180d_pre",
        "player_assists_180d_pre",
        "player_goals_per90_180d_pre",
        "player_assists_per90_180d_pre",
        "player_matches_365d_pre",
        "player_minutes_365d_pre",
        "player_goals_365d_pre",
        "player_assists_365d_pre",
        "player_goals_per90_365d_pre",
        "player_assists_per90_365d_pre",
        "source_squad_size",
        "source_average_age",
        "source_foreigners_percentage",
        "source_national_team_players",
        "source_stadium_seats",
        "destination_squad_size",
        "destination_average_age",
        "destination_foreigners_percentage",
        "destination_national_team_players",
        "destination_stadium_seats",
        "source_matches_365d_pre",
        "source_win_rate_365d_pre",
        "source_points_per_match_365d_pre",
        "source_goal_diff_per_match_365d_pre",
        "destination_matches_365d_pre",
        "destination_win_rate_365d_pre",
        "destination_points_per_match_365d_pre",
        "destination_goal_diff_per_match_365d_pre",
        "source_api_cached_matches",
        "source_api_cached_win_rate",
        "source_api_cached_goal_diff_per_match",
        "source_api_cache_available",
        "destination_api_cached_matches",
        "destination_api_cached_win_rate",
        "destination_api_cached_goal_diff_per_match",
        "destination_api_cache_available",
        "is_same_league_move",
        "is_same_country_move",
        "season_start_year",
        "target_destination_matches_24m",
        "target_destination_minutes_24m",
        "target_destination_goals_24m",
        "target_destination_assists_24m",
        "target_end_market_value",
        "target_market_value_delta_24m",
        "transfer_success",
    ]

    available_columns = [column for column in ordered_columns if column in modeling_dataset.columns]
    remaining_columns = [column for column in modeling_dataset.columns if column not in available_columns]
    modeling_dataset = modeling_dataset[available_columns + remaining_columns]
    return modeling_dataset.sort_values("transfer_date", kind="stable").reset_index(drop=True)


def build_transfer_modeling_dataset(
    raw_dir: str | Path = "data/player_scores_data",
    *,
    output_dir: str | Path | None = None,
    cache_dir: str | Path = "data",
    start_date: str = DEFAULT_TRANSFER_START_DATE,
    end_date: str = DEFAULT_TRANSFER_END_DATE,
    follow_up_months: int = DEFAULT_FOLLOW_UP_MONTHS,
    big_five_leagues: tuple[str, ...] = BIG_FIVE_LEAGUES,
    minutes_threshold: int = DEFAULT_MINUTES_THRESHOLD,
) -> dict[str, Any]:
    tables = load_primary_tables(raw_dir)
    required_tables = ["transfers", "players", "player_valuations", "appearances", "games", "club_games", "clubs", "competitions"]
    missing_tables = [name for name in required_tables if name not in tables]
    if missing_tables:
        raise FileNotFoundError(f"Missing required raw tables for the transfer modeling dataset: {', '.join(missing_tables)}")

    cohort, excluded_transfers = _prepare_base_transfer_cohort(
        tables,
        start_date=start_date,
        end_date=end_date,
        big_five_leagues=big_five_leagues,
    )
    cohort = _attach_pre_transfer_valuation_features(cohort, tables["player_valuations"])
    cohort = _attach_player_performance_features(cohort, tables["appearances"])
    club_history = _prepare_club_form_history(tables["club_games"], tables["games"])
    cohort = _attach_club_form_features(cohort, club_history)
    cohort = _attach_optional_api_features(cohort, cache_dir)
    labeled_cohort = create_transfer_success_labels(
        cohort,
        tables["appearances"],
        tables["player_valuations"],
        follow_up_months=follow_up_months,
        minutes_threshold=minutes_threshold,
    )

    modeling_dataset = _finalize_modeling_dataset(labeled_cohort)
    label_eligibility_failures = labeled_cohort[~labeled_cohort["target_is_eligible"]].copy().reset_index(drop=True)

    audit_summary = pd.DataFrame(
        [
            {
                "raw_transfers": int(len(tables["transfers"])),
                "candidate_transfers": int(len(cohort)),
                "modeling_rows": int(len(modeling_dataset)),
                "excluded_rows": int(len(excluded_transfers)),
                "label_ineligible_rows": int(len(label_eligibility_failures)),
                "positive_class_rate": round(float(modeling_dataset["transfer_success"].mean()), 4) if not modeling_dataset.empty else pd.NA,
                "start_date": start_date,
                "end_date": end_date,
                "follow_up_months": int(follow_up_months),
                "minutes_threshold": int(minutes_threshold),
                "big_five_leagues": ",".join(big_five_leagues),
            }
        ]
    )

    outputs = {
        "modeling_dataset": modeling_dataset,
        "labeled_transfer_cohort": labeled_cohort,
        "excluded_transfers": excluded_transfers,
        "label_eligibility_failures": label_eligibility_failures,
        "audit_summary": audit_summary,
    }

    if output_dir is not None:
        save_transfer_modeling_dataset(outputs, output_dir)
    return outputs


def save_transfer_modeling_dataset(outputs: dict[str, Any], output_dir: str | Path = "data/processed") -> dict[str, str]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    file_map = {
        "modeling_dataset": output_root / "transfer_modeling_dataset.csv",
        "labeled_transfer_cohort": output_root / "transfer_labeled_cohort.csv",
        "excluded_transfers": output_root / "transfer_excluded_audit.csv",
        "label_eligibility_failures": output_root / "transfer_label_failures.csv",
        "audit_summary": output_root / "transfer_modeling_summary.csv",
    }

    saved_paths: dict[str, str] = {}
    for key, path in file_map.items():
        frame = outputs.get(key)
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(path, index=False)
            saved_paths[key] = str(path)
    return saved_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the ScoutRadar transfer-level modeling dataset.")
    parser.add_argument("--raw-dir", default="data/player_scores_data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--start-date", default=DEFAULT_TRANSFER_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_TRANSFER_END_DATE)
    parser.add_argument("--follow-up-months", type=int, default=DEFAULT_FOLLOW_UP_MONTHS)
    parser.add_argument("--minutes-threshold", type=int, default=DEFAULT_MINUTES_THRESHOLD)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    outputs = build_transfer_modeling_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        follow_up_months=args.follow_up_months,
        minutes_threshold=args.minutes_threshold,
    )
    saved_paths = save_transfer_modeling_dataset(outputs, args.output_dir)
    logger.info("Saved transfer modeling outputs: %s", json.dumps(saved_paths, indent=2))


if __name__ == "__main__":
    main()
