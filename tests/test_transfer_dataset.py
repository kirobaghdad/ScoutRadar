import pandas as pd

from src.data.transfer_dataset import BIG_FIVE_LEAGUES, build_transfer_modeling_dataset

from .synthetic_phase2_data import create_synthetic_phase2_raw_dir


def test_build_transfer_modeling_dataset_filters_and_audits(tmp_path):
    raw_dir = create_synthetic_phase2_raw_dir(tmp_path)

    outputs = build_transfer_modeling_dataset(raw_dir=raw_dir, cache_dir=tmp_path)
    dataset = outputs["modeling_dataset"]
    excluded = outputs["excluded_transfers"]
    failures = outputs["label_eligibility_failures"]

    assert not dataset.empty
    assert dataset["destination_competition_id"].isin(BIG_FIVE_LEAGUES).all()
    assert dataset["transfer_date"].min() >= pd.Timestamp("2018-07-01")
    assert dataset["transfer_date"].max() <= pd.Timestamp("2022-06-30")
    assert dataset["transfer_key"].is_unique
    assert dataset["destination_api_context_matched"].eq(1).any()
    assert {"source_api_context_matched", "source_api_cached_matches"}.issubset(dataset.columns)

    assert excluded["exclusion_reason"].str.contains("after_modeling_window").any()
    assert excluded["exclusion_reason"].str.contains("destination_not_big_five").any()
    assert excluded["exclusion_reason"].str.contains("duplicate_transfer_business_key").any()

    assert not failures.empty
    assert failures["target_failure_reason"].str.contains("no_market_value_inside_follow_up_window|incomplete_follow_up_window").any()


def test_transfer_success_labels_do_not_leak_beyond_follow_up_window(tmp_path):
    raw_dir = create_synthetic_phase2_raw_dir(tmp_path)
    outputs = build_transfer_modeling_dataset(raw_dir=raw_dir, cache_dir=tmp_path)

    leakage_guard_row = outputs["labeled_transfer_cohort"].loc[outputs["labeled_transfer_cohort"]["player_id"] == 1000].iloc[0]

    assert leakage_guard_row["target_destination_minutes_24m"] < 1800
    assert leakage_guard_row["target_end_market_value"] < leakage_guard_row["pre_transfer_market_value"]
    assert leakage_guard_row["transfer_success"] == 0


def test_transfer_dataset_requires_api_fixture_cache(tmp_path):
    raw_dir = create_synthetic_phase2_raw_dir(tmp_path)
    for cache_file in (tmp_path / "api_football").glob("api_football*.json"):
        cache_file.unlink()

    try:
        build_transfer_modeling_dataset(raw_dir=raw_dir, cache_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "Missing required API Football fixture cache" in str(exc)
    else:
        raise AssertionError("Expected missing API fixture cache to fail the transfer dataset build")


def test_transfer_dataset_rejects_irrelevant_api_fixture_cache(tmp_path):
    raw_dir = create_synthetic_phase2_raw_dir(tmp_path)
    api_cache_dir = tmp_path / "api_football"
    for cache_file in api_cache_dir.glob("api_football*.json"):
        cache_file.unlink()
    (api_cache_dir / "api_football_fixtures_wrong_period.json").write_text(
        '{"response":[{"fixture":{"id":1,"date":"2024-09-01T15:00:00+00:00","timestamp":1725202800,"status":{"short":"FT","long":"Match Finished","elapsed":90}},"league":{"id":39,"name":"Premier League","country":"England","season":2024},"teams":{"home":{"id":1,"name":"Premier Town","winner":true},"away":{"id":2,"name":"Opponent","winner":false}},"goals":{"home":2,"away":1}}]}',
        encoding="utf-8",
    )

    try:
        build_transfer_modeling_dataset(raw_dir=raw_dir, cache_dir=tmp_path)
    except ValueError as exc:
        assert "did not match the transfer cohort" in str(exc)
    else:
        raise AssertionError("Expected irrelevant API fixture cache to fail the transfer dataset build")
