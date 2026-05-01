import pandas as pd

from src.features.build_features import (
    build_features,
    chronological_split,
    load_preprocessor_artifact,
    save_preprocessor_artifact,
    transform_with_preprocessor_artifact,
)


def test_build_features_fits_only_on_training_categories_and_excludes_target_columns():
    train_df = pd.DataFrame(
        {
            "transfer_date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            "player_id": [1, 2, 3],
            "position": ["Attack", "Defender", "Attack"],
            "destination_competition_name": ["Premier League", "LaLiga", "Premier League"],
            "age_at_transfer": [21, 25, None],
            "transfer_fee": [1_000_000, None, 0],
            "transfer_fee_for_model": [1_000_000, 0, 0],
            "transfer_fee_log1p": [13.8155, 0.0, 0.0],
            "pre_transfer_market_value": [1_000_000, 1_500_000, 1_200_000],
            "pre_transfer_market_value_log1p": [13.8155, 14.2210, 13.9978],
            "player_current_market_value": [2_000_000, 3_000_000, 2_500_000],
            "highest_market_value_in_eur": [5_000_000, 6_000_000, 7_000_000],
            "player_minutes_365d_pre": [1200, 900, 1500],
            "target_destination_minutes_24m": [200, 2300, 1400],
            "transfer_success": [0, 1, 1],
        }
    )
    val_df = pd.DataFrame(
        {
            "transfer_date": pd.to_datetime(["2019-04-01"]),
            "player_id": [4],
            "position": ["Goalkeeper"],
            "destination_competition_name": ["Bundesliga"],
            "age_at_transfer": [27],
            "transfer_fee": [None],
            "transfer_fee_for_model": [0],
            "transfer_fee_log1p": [0.0],
            "pre_transfer_market_value": [900_000],
            "pre_transfer_market_value_log1p": [13.7102],
            "player_current_market_value": [1_100_000],
            "highest_market_value_in_eur": [3_500_000],
            "player_minutes_365d_pre": [800],
            "target_destination_minutes_24m": [300],
            "transfer_success": [0],
        }
    )

    payload = build_features(train_df, val_df, val_df)

    assert "target_destination_minutes_24m" not in payload["feature_columns"]
    assert "transfer_fee" not in payload["feature_columns"]
    assert "transfer_fee_for_model" not in payload["feature_columns"]
    assert "pre_transfer_market_value" not in payload["feature_columns"]
    assert "transfer_fee_log1p" in payload["feature_columns"]
    assert "pre_transfer_market_value_log1p" in payload["feature_columns"]
    assert "player_current_market_value" not in payload["feature_columns"]
    assert "highest_market_value_in_eur" not in payload["feature_columns"]
    assert payload["X_train"].shape[1] == payload["X_val"].shape[1] == payload["X_test"].shape[1]
    assert not any("Goalkeeper" in name for name in payload["feature_names"])
    assert payload["X_train"].isna().sum().sum() == 0


def test_chronological_split_keeps_time_order_and_summary():
    df = pd.DataFrame(
        {
            "transfer_date": pd.to_datetime(
                [
                    "2020-04-01",
                    "2020-01-01",
                    "2020-03-01",
                    "2020-02-01",
                    "2020-05-01",
                    "2020-06-01",
                ]
            ),
            "transfer_success": [1, 0, 1, 0, 1, 0],
        }
    )

    splits = chronological_split(df, train_ratio=0.50, val_ratio=0.25)

    assert list(splits["train"]["transfer_date"]) == list(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    assert list(splits["val"]["transfer_date"]) == list(pd.to_datetime(["2020-04-01"]))
    assert list(splits["test"]["transfer_date"]) == list(pd.to_datetime(["2020-05-01", "2020-06-01"]))
    assert list(splits["split_summary"]["split"]) == ["train", "val", "test"]


def test_preprocessor_artifact_can_be_saved_loaded_and_reused(tmp_path):
    train_df = pd.DataFrame(
        {
            "transfer_date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            "position": ["Attack", "Defender", "Attack"],
            "age_at_transfer": [21, 25, None],
            "player_minutes_365d_pre": [1200, 900, 1500],
            "transfer_success": [0, 1, 1],
        }
    )
    new_rows = pd.DataFrame(
        {
            "transfer_date": pd.to_datetime(["2019-04-01"]),
            "position": ["Goalkeeper"],
            "age_at_transfer": [27],
            "player_minutes_365d_pre": [800],
            "transfer_success": [0],
        }
    )

    payload = build_features(train_df)
    artifact_path = save_preprocessor_artifact(payload, tmp_path / "preprocessor.pkl")
    artifact = load_preprocessor_artifact(artifact_path)
    transformed = transform_with_preprocessor_artifact(artifact, new_rows)

    assert artifact_path.exists()
    assert artifact["feature_columns"] == payload["feature_columns"]
    assert list(transformed.columns) == payload["feature_names"]
    assert transformed.isna().sum().sum() == 0
