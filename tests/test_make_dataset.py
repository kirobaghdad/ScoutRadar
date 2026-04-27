import pandas as pd
import numpy as np

from src.data.make_dataset import cap_outliers_iqr, clean_table, handle_missing_values, process_data, standardize_data_types


def test_handle_missing_values():
    df = pd.DataFrame(
        {
            "high_missing": [1, np.nan, np.nan, np.nan, 5],
            "low_missing_num": [10, 20, np.nan, 40, 50],
            "low_missing_cat": ["A", "B", "A", np.nan, "B"],
        }
    )

    clean_df = handle_missing_values(df, drop_threshold=0.50)

    assert "high_missing" not in clean_df.columns
    assert "low_missing_num" in clean_df.columns
    assert clean_df["low_missing_num"].isna().sum() == 0
    assert clean_df["low_missing_num"].iloc[2] == 30.0
    assert clean_df["low_missing_cat"].isna().sum() == 0


def test_cap_outliers_iqr():
    df = pd.DataFrame(
        {
            "normal": [10, 12, 11, 13, 12, 10, 11, 12, 10, 11],
            "with_outlier": [10, 12, 11, 13, 12, 10, 11, 12, 1000, -1000],
            "categorical_encoded": [0, 1, 0, 1, 2, 0, 1, 2, 0, 1],
        }
    )

    clean_df = cap_outliers_iqr(df)

    assert clean_df["normal"].max() <= 13
    assert clean_df["with_outlier"].max() < 1000
    assert clean_df["with_outlier"].min() > -1000
    assert clean_df["categorical_encoded"].max() == 2


def test_standardize_data_types_converts_date_columns():
    df = pd.DataFrame(
        {
            "transfer_date": ["2020-01-01", "bad-date", "2020-03-01"],
            "name": ["A", "B", "C"],
        }
    )

    clean_df = standardize_data_types(df)

    assert pd.api.types.is_datetime64_any_dtype(clean_df["transfer_date"])
    assert clean_df["transfer_date"].isna().sum() == 0


def test_clean_table_runs_full_basic_preprocessing():
    df = pd.DataFrame(
        {
            "match_date": ["2020-01-01", "2020-01-01", "2020-02-01"],
            "score": [10, 10, np.nan],
            "team": ["A", "A", np.nan],
        }
    )

    clean_df = clean_table("matches", df)

    assert len(clean_df) == 2
    assert clean_df.isna().sum().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(clean_df["match_date"])


def test_process_data_writes_clean_csv_files(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()

    pd.DataFrame(
        {
            "game_date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "minutes": [90, 80, np.nan],
        }
    ).to_csv(input_dir / "games.csv", index=False)

    process_data(str(input_dir), str(output_dir))

    output_path = output_dir / "games_clean.csv"
    clean_df = pd.read_csv(output_path)

    assert output_path.exists()
    assert len(clean_df) == 3
    assert clean_df["minutes"].isna().sum() == 0
