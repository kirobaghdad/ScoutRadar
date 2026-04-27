import pandas as pd

from src.features.build_features import build_features


def test_build_features_fits_only_on_training_categories_and_excludes_target_columns():
    train_df = pd.DataFrame(
        {
            "transfer_date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            "player_id": [1, 2, 3],
            "position": ["Attack", "Defender", "Attack"],
            "destination_competition_name": ["Premier League", "LaLiga", "Premier League"],
            "age_at_transfer": [21, 25, None],
            "pre_transfer_market_value": [1_000_000, 1_500_000, 1_200_000],
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
            "pre_transfer_market_value": [900_000],
            "player_minutes_365d_pre": [800],
            "target_destination_minutes_24m": [300],
            "transfer_success": [0],
        }
    )

    payload = build_features(train_df, val_df, val_df)

    assert "target_destination_minutes_24m" not in payload["feature_columns"]
    assert payload["X_train"].shape[1] == payload["X_val"].shape[1] == payload["X_test"].shape[1]
    assert not any("Goalkeeper" in name for name in payload["feature_names"])
    assert payload["X_train"].isna().sum().sum() == 0
