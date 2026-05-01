from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, PredefinedSplit
from sklearn.svm import LinearSVC

from src.features.build_features import build_feature_splits, save_preprocessor_artifact

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    XGBClassifier = None

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "Transfer Success Prediction"
DEFAULT_TARGET_COLUMN = "transfer_success"
DEFAULT_OUTPUT_DIR = Path("models/transfer_success")
DEFAULT_THRESHOLD_GRID = tuple(np.round(np.arange(0.20, 0.81, 0.05), 2))


@dataclass(frozen=True)
class CandidateModel:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]
    balance_strategy: str


@dataclass(frozen=True)
class FailedModelRun:
    name: str
    balance_strategy: str
    stage: str
    error: str


def _ensure_series(values: pd.Series | np.ndarray | list[Any]) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values).reset_index(drop=True)


def _ensure_frame(values: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.reset_index(drop=True)
    return pd.DataFrame(values).reset_index(drop=True)


def combine_train_validation(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, PredefinedSplit]:
    X_all = pd.concat([_ensure_frame(X_train), _ensure_frame(X_val)], ignore_index=True)
    y_all = pd.concat([_ensure_series(y_train), _ensure_series(y_val)], ignore_index=True)
    test_fold = np.concatenate(
        [
            np.full(len(_ensure_frame(X_train)), -1, dtype=int),
            np.zeros(len(_ensure_frame(X_val)), dtype=int),
        ]
    )
    return X_all, y_all, PredefinedSplit(test_fold=test_fold)


def build_expanding_season_splits(
    seasons: pd.Series | np.ndarray,
    *,
    min_train_seasons: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    season_series = _ensure_series(seasons)
    if season_series.empty:
        return []

    unique_seasons = sorted(season_series.dropna().astype(int).unique().tolist())
    if len(unique_seasons) <= min_train_seasons:
        return []

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for cutoff_index in range(min_train_seasons, len(unique_seasons)):
        train_seasons = set(unique_seasons[:cutoff_index])
        validation_season = unique_seasons[cutoff_index]

        train_idx = season_series.index[season_series.astype(int).isin(train_seasons)].to_numpy()
        val_idx = season_series.index[season_series.astype(int) == validation_season].to_numpy()
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def _positive_class_scores(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        probabilities = np.asarray(probabilities)
        return probabilities[:, 1] if probabilities.ndim == 2 else probabilities
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return None


def compute_business_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    transfer_fee: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    y_true_series = _ensure_series(y_true).astype(int)
    y_pred_series = _ensure_series(y_pred).astype(int)

    if transfer_fee is None:
        fee_series = pd.Series(np.zeros(len(y_true_series), dtype=float))
    else:
        fee_series = pd.to_numeric(_ensure_series(transfer_fee), errors="coerce").fillna(0.0)

    true_positive_mask = (y_true_series == 1) & (y_pred_series == 1)
    false_positive_mask = (y_true_series == 0) & (y_pred_series == 1)
    false_negative_mask = (y_true_series == 1) & (y_pred_series == 0)

    positive_fee_total = float(fee_series[y_true_series == 1].sum())

    return {
        "recommendation_rate": float((y_pred_series == 1).mean()),
        "successful_recommendation_rate": float(precision_score(y_true_series, y_pred_series, zero_division=0)),
        "captured_success_fee_rate": float(fee_series[true_positive_mask].sum() / positive_fee_total) if positive_fee_total else 0.0,
        "false_positive_fee_exposure": float(fee_series[false_positive_mask].sum()),
        "missed_success_fee": float(fee_series[false_negative_mask].sum()),
    }


def evaluate_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    transfer_fee: pd.Series | np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = _ensure_series(y).astype(int)
    scores = _positive_class_scores(model, X)

    if scores is None:
        y_pred = _ensure_series(model.predict(X)).astype(int)
    else:
        y_pred = pd.Series((scores >= threshold).astype(int))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if scores is not None and y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
        metrics["pr_auc"] = float(auc(recall_vals, precision_vals))

    metrics.update(compute_business_metrics(y_true, y_pred, transfer_fee))
    return metrics


def tune_decision_threshold(
    model: Any,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    *,
    transfer_fee: pd.Series | np.ndarray | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> dict[str, Any]:
    y_true = _ensure_series(y_val).astype(int)
    scores = _positive_class_scores(model, X_val)

    if scores is None:
        default_metrics = evaluate_model(model, X_val, y_true, transfer_fee=transfer_fee, threshold=0.5)
        threshold_metrics = pd.DataFrame([{"threshold": 0.5, **default_metrics}])
        return {"best_threshold": 0.5, "best_f1": default_metrics["f1"], "threshold_metrics": threshold_metrics}

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = pd.Series((scores >= threshold).astype(int))
        row = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        row.update(compute_business_metrics(y_true, y_pred, transfer_fee))
        rows.append(row)

    threshold_metrics = pd.DataFrame(rows).sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )
    best_row = threshold_metrics.iloc[0]
    return {
        "best_threshold": float(best_row["threshold"]),
        "best_f1": float(best_row["f1"]),
        "threshold_metrics": threshold_metrics,
    }


def tune_model_with_season_cv(
    estimator: Any,
    param_grid: dict[str, list[Any]],
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    seasons: pd.Series | np.ndarray,
    *,
    min_train_seasons: int = 2,
    scoring: str = "f1",
    n_jobs: int | None = None,
) -> dict[str, Any]:
    del n_jobs

    X_frame = _ensure_frame(X)
    y_series = _ensure_series(y).astype(int)
    season_series = _ensure_series(seasons).astype(int)
    splits = build_expanding_season_splits(season_series, min_train_seasons=min_train_seasons)
    if not splits:
        raise ValueError("Not enough distinct seasons to build expanding cross-validation folds.")

    all_results: list[dict[str, Any]] = []

    for params in ParameterGrid(param_grid):
        fold_scores: list[float] = []
        fold_rows: list[dict[str, Any]] = []

        for fold_index, (train_idx, val_idx) in enumerate(splits, start=1):
            candidate = clone(estimator).set_params(**params)
            candidate.fit(X_frame.iloc[train_idx], y_series.iloc[train_idx])
            fold_metrics = evaluate_model(candidate, X_frame.iloc[val_idx], y_series.iloc[val_idx])
            score_value = float(fold_metrics.get(scoring, 0.0))
            fold_scores.append(score_value)
            fold_rows.append(
                {
                    "fold": fold_index,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                    "validation_season": int(season_series.iloc[val_idx].iloc[0]),
                    scoring: score_value,
                }
            )

        all_results.append(
            {
                "params": params,
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores)),
                "fold_summary": pd.DataFrame(fold_rows),
            }
        )

    best_result = max(all_results, key=lambda row: (row["mean_score"], -row["std_score"]))
    best_estimator = clone(estimator).set_params(**best_result["params"])
    best_estimator.fit(X_frame, y_series)

    cv_results = pd.DataFrame(
        [
            {
                **row["params"],
                f"mean_{scoring}": row["mean_score"],
                f"std_{scoring}": row["std_score"],
            }
            for row in all_results
        ]
    ).sort_values(by=f"mean_{scoring}", ascending=False, ignore_index=True)

    return {
        "best_estimator": best_estimator,
        "best_params": best_result["params"],
        "best_score": best_result["mean_score"],
        "cv_results": cv_results,
        "cv_summary": best_result["fold_summary"],
        "cv_type": "expanding_season",
    }


def _class_ratio(y: pd.Series | np.ndarray) -> float:
    target = _ensure_series(y).astype(int)
    positives = int(target.sum())
    negatives = int((target == 0).sum())
    return float(negatives / positives) if positives else 1.0


def get_candidate_models(y_train: pd.Series | np.ndarray) -> list[CandidateModel]:
    scale_pos_weight = max(1.0, _class_ratio(y_train))

    candidates = [
        CandidateModel(
            name="Dummy Baseline",
            estimator=DummyClassifier(strategy="most_frequent"),
            param_grid={"strategy": ["most_frequent", "prior"]},
            balance_strategy="none",
        ),
        CandidateModel(
            name="Logistic Regression + SMOTE",
            estimator=ImbPipeline(
                steps=[
                    ("sampler", SMOTE(random_state=42)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            param_grid={
                "sampler__k_neighbors": [3, 5],
                "model__C": [0.3, 1.0, 3.0],
                "model__solver": ["lbfgs"],
            },
            balance_strategy="smote_train_only",
        ),
        CandidateModel(
            name="Random Forest",
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1,
            ),
            param_grid={
                "n_estimators": [200, 350],
                "max_depth": [None, 12],
                "min_samples_leaf": [1, 3],
            },
            balance_strategy="class_weight",
        ),
        CandidateModel(
            name="Extra Trees",
            estimator=ExtraTreesClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            param_grid={
                "n_estimators": [200, 350],
                "max_depth": [None, 12],
                "min_samples_leaf": [1, 3],
            },
            balance_strategy="class_weight",
        ),
        CandidateModel(
            name="Linear SVM",
            estimator=LinearSVC(
                class_weight="balanced",
                dual="auto",
                random_state=42,
            ),
            param_grid={
                "C": [0.3, 1.0, 3.0],
            },
            balance_strategy="class_weight",
        ),
    ]

    if XGBClassifier is not None:
        candidates.append(
            CandidateModel(
                name="XGBoost",
                estimator=XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    tree_method="hist",
                    n_jobs=1,
                ),
                param_grid={
                    "n_estimators": [200, 350],
                    "max_depth": [4, 6],
                    "learning_rate": [0.03, 0.08],
                },
                balance_strategy="scale_pos_weight",
            )
        )
    else:
        logger.warning("xgboost is not installed; skipping the XGBoost candidate.")

    return candidates


def _log_mlflow_run(
    *,
    experiment_name: str,
    tracking_uri: str | None,
    model_name: str,
    balance_strategy: str,
    best_params: dict[str, Any],
    threshold: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    cv_summary: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    model: Any,
) -> str:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("balance_strategy", balance_strategy)
        mlflow.log_param("decision_threshold", threshold)
        for key, value in best_params.items():
            mlflow.log_param(key, value)

        for prefix, metric_set in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            for metric_name, metric_value in metric_set.items():
                mlflow.log_metric(f"{prefix}_{metric_name}", metric_value)

        cv_path = Path("cv_summary.csv")
        threshold_path = Path("threshold_metrics.csv")
        cv_summary.to_csv(cv_path, index=False)
        threshold_metrics.to_csv(threshold_path, index=False)
        mlflow.log_artifact(str(cv_path))
        mlflow.log_artifact(str(threshold_path))
        mlflow.sklearn.log_model(model, name="model")

        run_id = mlflow.active_run().info.run_id

    cv_path.unlink(missing_ok=True)
    threshold_path.unlink(missing_ok=True)
    return run_id


def run_model_selection(
    dataset_path: str | Path,
    *,
    target_col: str = DEFAULT_TARGET_COLUMN,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    dataset = pd.read_csv(dataset_path, parse_dates=["transfer_date"])
    payload = build_feature_splits(dataset, target_col=target_col)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_preprocessor_artifact(payload, output_path / "preprocessor.pkl")
    payload["split_summary"].to_csv(output_path / "split_summary.csv", index=False)

    X_train = payload["X_train"]
    y_train = payload["y_train"]
    X_val = payload["X_val"]
    y_val = payload["y_val"]
    X_test = payload["X_test"]
    y_test = payload["y_test"]

    train_fees = payload["train_df"]["transfer_fee_for_model"].fillna(0.0).reset_index(drop=True)
    val_fees = payload["val_df"]["transfer_fee_for_model"].fillna(0.0).reset_index(drop=True)
    test_fees = payload["test_df"]["transfer_fee_for_model"].fillna(0.0).reset_index(drop=True)
    train_seasons = payload["train_df"]["season_start_year"].reset_index(drop=True)

    model_runs: list[dict[str, Any]] = []
    failed_runs: list[FailedModelRun] = []

    for candidate in get_candidate_models(y_train):
        logger.info("Tuning %s", candidate.name)
        try:
            tuning_result = tune_model_with_season_cv(
                candidate.estimator,
                candidate.param_grid,
                X_train,
                y_train,
                train_seasons,
                min_train_seasons=2,
                scoring="f1",
            )

            best_model = tuning_result["best_estimator"]
            threshold_result = tune_decision_threshold(
                best_model,
                X_val,
                y_val,
                transfer_fee=val_fees,
            )
            threshold = threshold_result["best_threshold"]

            train_metrics = evaluate_model(best_model, X_train, y_train, transfer_fee=train_fees, threshold=threshold)
            val_metrics = evaluate_model(best_model, X_val, y_val, transfer_fee=val_fees, threshold=threshold)
            test_metrics = evaluate_model(best_model, X_test, y_test, transfer_fee=test_fees, threshold=threshold)

            run_id = _log_mlflow_run(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                model_name=candidate.name,
                balance_strategy=candidate.balance_strategy,
                best_params=tuning_result["best_params"],
                threshold=threshold,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                cv_summary=tuning_result["cv_summary"],
                threshold_metrics=threshold_result["threshold_metrics"],
                model=best_model,
            )

            model_path = output_path / f"{candidate.name.lower().replace(' ', '_').replace('+', 'plus')}_summary.json"
            summary = {
                "model_name": candidate.name,
                "balance_strategy": candidate.balance_strategy,
                "best_params": tuning_result["best_params"],
                "cv_best_f1": tuning_result["best_score"],
                "decision_threshold": threshold,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "run_id": run_id,
            }
            model_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            model_runs.append(summary)
        except Exception as exc:
            logger.exception("Skipping %s because training failed.", candidate.name)
            failed_runs.append(
                FailedModelRun(
                    name=candidate.name,
                    balance_strategy=candidate.balance_strategy,
                    stage="training",
                    error=str(exc),
                )
            )

    if failed_runs:
        failed_runs_path = output_path / "failed_model_runs.json"
        failed_runs_path.write_text(
            json.dumps([failed_run.__dict__ for failed_run in failed_runs], indent=2),
            encoding="utf-8",
        )

    if not model_runs:
        failure_summary = "; ".join(f"{failed_run.name}: {failed_run.error}" for failed_run in failed_runs) or "unknown error"
        raise RuntimeError(f"Model training failed for every candidate. {failure_summary}")

    results_df = pd.DataFrame(
        [
            {
                "model_name": run["model_name"],
                "balance_strategy": run["balance_strategy"],
                "cv_best_f1": run["cv_best_f1"],
                "decision_threshold": run["decision_threshold"],
                "val_f1": run["val_metrics"].get("f1", 0.0),
                "val_pr_auc": run["val_metrics"].get("pr_auc", 0.0),
                "val_recall": run["val_metrics"].get("recall", 0.0),
                "test_f1": run["test_metrics"].get("f1", 0.0),
                "test_pr_auc": run["test_metrics"].get("pr_auc", 0.0),
                "test_recall": run["test_metrics"].get("recall", 0.0),
                "test_precision": run["test_metrics"].get("precision", 0.0),
                "test_false_positive_fee_exposure": run["test_metrics"].get("false_positive_fee_exposure", 0.0),
                "mlflow_run_id": run["run_id"],
            }
            for run in model_runs
        ]
    ).sort_values(by=["val_f1", "val_pr_auc", "test_f1"], ascending=False, ignore_index=True)

    best_model_name = results_df.iloc[0]["model_name"]
    best_run = next(run for run in model_runs if run["model_name"] == best_model_name)

    results_df.to_csv(output_path / "model_comparison.csv", index=False)
    (output_path / "best_model.json").write_text(json.dumps(best_run, indent=2), encoding="utf-8")

    return {
        "results": results_df,
        "best_model": best_run,
        "split_summary": payload["split_summary"],
        "output_dir": output_path,
        "failed_runs": [failed_run.__dict__ for failed_run in failed_runs],
    }
