import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.models.transfer_success_modeling import (
    CandidateModel,
    build_expanding_season_splits,
    combine_train_validation,
    compute_business_metrics,
    evaluate_model,
    run_model_selection,
    tune_decision_threshold,
    tune_model_with_season_cv,
)


def test_combine_train_validation_creates_predefined_holdout():
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    X_val = pd.DataFrame({"a": [4, 5]})
    y_val = pd.Series([1, 0])

    X_all, y_all, split = combine_train_validation(X_train, y_train, X_val, y_val)

    assert X_all.shape == (5, 1)
    assert y_all.tolist() == [0, 1, 0, 1, 0]
    assert split.test_fold.tolist() == [-1, -1, -1, 0, 0]


def test_build_expanding_season_splits_uses_earlier_seasons_only():
    seasons = pd.Series([2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021])

    splits = build_expanding_season_splits(seasons, min_train_seasons=2)

    assert len(splits) == 2
    first_train, first_val = splits[0]
    second_train, second_val = splits[1]
    assert set(seasons.iloc[first_train]) == {2018, 2019}
    assert set(seasons.iloc[first_val]) == {2020}
    assert set(seasons.iloc[second_train]) == {2018, 2019, 2020}
    assert set(seasons.iloc[second_val]) == {2021}


def test_compute_business_metrics_uses_transfer_fees():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([1, 0, 1, 0])
    fees = pd.Series([10.0, 20.0, 30.0, 40.0])

    metrics = compute_business_metrics(y_true, y_pred, fees)

    assert metrics["recommendation_rate"] == 0.5
    assert metrics["successful_recommendation_rate"] == 0.5
    assert metrics["captured_success_fee_rate"] == 10.0 / 30.0
    assert metrics["false_positive_fee_exposure"] == 30.0


def test_evaluate_model_returns_technical_and_business_metrics():
    X = pd.DataFrame({"a": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    fees = pd.Series([5.0, 8.0, 3.0, 6.0])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    metrics = evaluate_model(model, X, y, transfer_fee=fees)

    assert {"accuracy", "precision", "recall", "f1", "false_positive_fee_exposure", "recommendation_rate"}.issubset(
        metrics.keys()
    )


def test_tune_decision_threshold_returns_best_validation_threshold():
    X_train = pd.DataFrame({"a": [0, 0, 0, 1, 1, 1]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    X_val = pd.DataFrame({"a": [0, 0, 1, 1]})
    y_val = pd.Series([0, 0, 0, 1])

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    threshold_result = tune_decision_threshold(model, X_val, y_val, thresholds=(0.3, 0.5, 0.7))

    assert threshold_result["best_threshold"] in {0.3, 0.5, 0.7}
    assert threshold_result["best_f1"] >= 0.0
    assert set(threshold_result["threshold_metrics"]["threshold"]) == {0.3, 0.5, 0.7}
    assert threshold_result["threshold_metrics"]["f1"].is_monotonic_decreasing


def test_evaluate_model_uses_custom_threshold_when_scores_are_available():
    X_train = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    X_eval = pd.DataFrame({"a": [0, 1, 2]})
    y_eval = pd.Series([0, 0, 1])

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    default_metrics = evaluate_model(model, X_eval, y_eval)
    strict_metrics = evaluate_model(model, X_eval, y_eval, threshold=0.8)

    assert default_metrics["recommendation_rate"] >= strict_metrics["recommendation_rate"]


def test_tune_model_with_season_cv_returns_fold_summary():
    X = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2, 3, 3]})
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    seasons = pd.Series([2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021])

    result = tune_model_with_season_cv(
        LogisticRegression(random_state=42, max_iter=1000),
        {"C": [0.1, 1.0], "solver": ["lbfgs"]},
        X,
        y,
        seasons,
        min_train_seasons=2,
        n_jobs=1,
    )

    assert result["best_params"]["C"] in {0.1, 1.0}
    assert result["cv_type"] == "expanding_season"
    assert list(result["cv_summary"]["validation_season"]) == [2020, 2021]


def test_run_model_selection_skips_failed_candidates_and_finishes(tmp_path, monkeypatch):
    dataset = pd.DataFrame(
        {
            "transfer_date": pd.date_range("2018-01-01", periods=12, freq="180D"),
            "season_start_year": [2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023],
            "position": ["Attack", "Defender"] * 6,
            "age_at_transfer": [20, 24, 21, 25, 22, 26, 23, 27, 24, 28, 25, 29],
            "transfer_fee_for_model": [0.0, 1.0] * 6,
            "transfer_fee_log1p": [0.0, 0.69] * 6,
            "player_minutes_365d_pre": [100, 200, 110, 210, 120, 220, 130, 230, 140, 240, 150, 250],
            "transfer_success": [0, 1] * 6,
        }
    )
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    broken_candidate = CandidateModel(
        name="Broken Model",
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        param_grid={"not_a_real_param": [1]},
        balance_strategy="none",
    )
    working_candidate = CandidateModel(
        name="Working Model",
        estimator=DummyClassifier(strategy="most_frequent"),
        param_grid={"strategy": ["most_frequent"]},
        balance_strategy="none",
    )

    monkeypatch.setattr(
        "src.models.transfer_success_modeling.get_candidate_models",
        lambda y_train: [broken_candidate, working_candidate],
    )
    monkeypatch.setattr(
        "src.models.transfer_success_modeling._log_mlflow_run",
        lambda **kwargs: "test-run-id",
    )

    result = run_model_selection(
        dataset_path=dataset_path,
        output_dir=tmp_path / "artifacts",
        tracking_uri=None,
    )

    assert list(result["results"]["model_name"]) == ["Working Model"]
    assert result["failed_runs"][0]["name"] == "Broken Model"
    assert (tmp_path / "artifacts" / "failed_model_runs.json").exists()
