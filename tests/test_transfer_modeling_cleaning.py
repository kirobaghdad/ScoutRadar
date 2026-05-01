import numpy as np
import pandas as pd

from src.data.transfer_modeling_cleaning import clean_transfer_modeling_dataframe


def test_clean_transfer_modeling_dataframe_quarantines_invalid_rows_and_drops_sparse_columns():
    df = pd.DataFrame(
        {
            "transfer_key": ["t1", "t2", "t2"],
            "transfer_date": ["2021-07-01", "2021-08-01", None],
            "player_id": [1, 2, 3],
            "transfer_success": [1, 0, 2],
            "position": ["Attack", "Goalkeeper", "Striker"],
            "age_at_transfer": [22, 31, 50],
            "transfer_fee": [10.0, None, 5.0],
            "pre_transfer_market_value": [8.0, 3.0, 4.0],
            "player_minutes_180d_pre": [0.0, 90.0, 0.0],
            "player_goals_per90_180d_pre": [None, 1.0, None],
            "mostly_missing": [None, None, "x"],
        }
    )

    outputs = clean_transfer_modeling_dataframe(df, missingness_threshold=0.6)
    cleaned = outputs["cleaned_dataset"]
    quarantine = outputs["quarantine_dataset"]
    dropped = outputs["dropped_columns"]

    assert len(cleaned) == 2
    assert len(quarantine) == 1
    assert "invalid_target" in quarantine.loc[0, "cleaning_rejection_reason"]
    assert "mostly_missing" not in cleaned.columns
    assert "mostly_missing" in dropped["column"].tolist()


def test_clean_transfer_modeling_dataframe_applies_safe_corrections_and_imputation():
    df = pd.DataFrame(
        {
            "transfer_key": ["t1", "t2"],
            "transfer_date": ["2021-07-01", "2021-08-01"],
            "player_id": [1, 2],
            "transfer_success": [1, 0],
            "position": [" Attack ", "Midfield"],
            "age_at_transfer": [22, 31],
            "transfer_fee": [10.0, None],
            "pre_transfer_market_value": [8.0, 3.0],
            "player_minutes_180d_pre": [0.0, 90.0],
            "player_goals_per90_180d_pre": [None, 1.0],
            "country_of_citizenship": [None, "Spain"],
            "source_matches_365d_pre": [10.0, 12.0],
            "source_win_rate_365d_pre": [None, 0.5],
        }
    )

    outputs = clean_transfer_modeling_dataframe(df, missingness_threshold=0.95)
    cleaned = outputs["cleaned_dataset"]
    log = outputs["cleaning_log"]

    assert cleaned.loc[0, "position"] == "Attack"
    assert cleaned.loc[0, "player_goals_per90_180d_pre"] == 0.0
    assert cleaned["transfer_fee"].isna().sum() == 1
    assert cleaned["transfer_fee_missing_flag"].tolist() == [0, 1]
    assert cleaned["is_free_transfer"].tolist() == [0, 0]
    assert cleaned["transfer_fee_for_model"].tolist() == [10.0, 0.0]
    assert cleaned["transfer_fee_log1p"].round(6).tolist() == [round(float(np.log1p(10.0)), 6), 0.0]
    assert "pre_transfer_market_value_log1p" in cleaned.columns
    assert cleaned["country_of_citizenship"].isna().sum() == 0
    assert {"type_standardization", "accuracy", "completeness", "outliers"}.issubset(set(log["step"]))


def test_clean_transfer_modeling_dataframe_marks_real_free_transfers():
    df = pd.DataFrame(
        {
            "transfer_key": ["t1", "t2"],
            "transfer_date": ["2021-07-01", "2021-08-01"],
            "player_id": [1, 2],
            "transfer_success": [1, 0],
            "position": ["Attack", "Midfield"],
            "age_at_transfer": [22, 31],
            "transfer_fee": [0.0, 25.0],
            "pre_transfer_market_value": [8.0, 3.0],
            "player_minutes_180d_pre": [0.0, 90.0],
            "player_goals_per90_180d_pre": [None, 1.0],
        }
    )

    outputs = clean_transfer_modeling_dataframe(df)
    cleaned = outputs["cleaned_dataset"]

    assert cleaned["transfer_fee_missing_flag"].tolist() == [0, 0]
    assert cleaned["is_free_transfer"].tolist() == [1, 0]
    assert cleaned["transfer_fee_for_model"].tolist() == [0.0, 25.0]
