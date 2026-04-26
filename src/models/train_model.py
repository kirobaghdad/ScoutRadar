from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from src.data.transfer_dataset import build_transfer_modeling_dataset, save_transfer_modeling_dataset
from src.features.build_features import build_feature_splits

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "ScoutRadar Phase 2"
DEFAULT_MODEL_OUTPUT_DIR = Path("models/generated")
DEFAULT_DATASET_PATH = Path("data/processed/transfer_modeling_dataset.csv")


def _resolve_tracking_uri(output_dir: Path, tracking_uri: str | None) -> str:
    if tracking_uri:
        if "://" in tracking_uri:
            return tracking_uri
        return Path(tracking_uri).expanduser().resolve().as_uri()
    return (output_dir / "mlruns").resolve().as_uri()


def _sample_evenly(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.copy().reset_index(drop=True)
    positions = sorted(set(int(position) for position in pd.Series(range(len(df))).sample(sample_size, random_state=42).tolist()))
    if len(positions) < sample_size:
        positions = list(range(sample_size))
    return df.iloc[positions].sort_values("transfer_date", kind="stable").reset_index(drop=True)


def build_model_registry(random_state: int = 42) -> dict[str, Any]:
    return {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
        "SVC": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state),
    }


def _supports_smote(y_train: pd.Series) -> bool:
    class_counts = y_train.value_counts()
    if len(class_counts) != 2:
        return False
    minority_count = int(class_counts.min())
    minority_share = float(class_counts.min() / class_counts.sum())
    return minority_share < 0.40 and minority_count >= 6


def _maybe_apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    use_smote: bool,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if not use_smote or not _supports_smote(y_train):
        return X_train, y_train, {"smote_applied": False, "smote_rows_added": 0}

    class_counts = y_train.value_counts()
    minority_count = int(class_counts.min())
    sampler = SMOTE(random_state=random_state, k_neighbors=min(5, minority_count - 1))
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    meta = {
        "smote_applied": True,
        "smote_rows_added": int(len(y_resampled) - len(y_train)),
    }
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled), meta


def _score_predictions(model: Any, X: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    if hasattr(model, "decision_function"):
        return pd.Series(model.decision_function(X), index=X.index)
    return pd.Series(model.predict(X), index=X.index)


def _precision_at_top_fraction(y_true: pd.Series, scores: pd.Series, top_fraction: float = 0.20) -> float:
    if len(y_true) == 0:
        return float("nan")
    cutoff = max(1, int(len(y_true) * top_fraction))
    ranked = pd.DataFrame({"y_true": y_true.reset_index(drop=True), "score": scores.reset_index(drop=True)}).sort_values(
        "score",
        ascending=False,
        kind="stable",
    )
    return float(ranked.head(cutoff)["y_true"].mean())


def _lift_at_top_fraction(y_true: pd.Series, scores: pd.Series, top_fraction: float = 0.20) -> float:
    baseline = float(y_true.mean()) if len(y_true) else float("nan")
    if pd.isna(baseline) or baseline == 0:
        return float("nan")
    return _precision_at_top_fraction(y_true, scores, top_fraction=top_fraction) / baseline


def evaluate_model(model: Any, X_eval: pd.DataFrame, y_eval: pd.Series, *, split_name: str) -> dict[str, float]:
    y_pred = pd.Series(model.predict(X_eval), index=y_eval.index)
    y_score = _score_predictions(model, X_eval)

    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()
    actual_positive = tp + fn
    actual_negative = tn + fp

    metrics: dict[str, float] = {
        f"{split_name}_accuracy": float(accuracy_score(y_eval, y_pred)),
        f"{split_name}_balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
        f"{split_name}_precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        f"{split_name}_recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        f"{split_name}_f1": float(f1_score(y_eval, y_pred, zero_division=0)),
        f"{split_name}_missed_success_rate": float(fn / actual_positive) if actual_positive else float("nan"),
        f"{split_name}_risky_recommendation_rate": float(fp / actual_negative) if actual_negative else float("nan"),
        f"{split_name}_top20_precision": float(_precision_at_top_fraction(y_eval, y_score, top_fraction=0.20)),
        f"{split_name}_top20_lift": float(_lift_at_top_fraction(y_eval, y_score, top_fraction=0.20)),
    }

    if y_eval.nunique() > 1:
        try:
            metrics[f"{split_name}_roc_auc"] = float(roc_auc_score(y_eval, y_score))
        except ValueError:
            metrics[f"{split_name}_roc_auc"] = float("nan")
    else:
        metrics[f"{split_name}_roc_auc"] = float("nan")
    return metrics


def _save_confusion_matrix(
    output_dir: Path,
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Path:
    figure_path = output_dir / f"{model_name}_{split_name}_confusion_matrix.png"
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Failure", "Success"])
    fig, ax = plt.subplots(figsize=(4, 4))
    display.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} - {split_name.title()} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    use_smote: bool = True,
) -> tuple[Any, dict[str, Any]]:
    registry = build_model_registry(random_state=random_state)
    if model_name not in registry:
        raise KeyError(f"Unknown model name: {model_name}")

    estimator = clone(registry[model_name])
    X_fit, y_fit, smote_meta = _maybe_apply_smote(X_train, y_train, use_smote=use_smote, random_state=random_state)
    estimator.fit(X_fit, y_fit)
    metadata = {
        "model_name": model_name,
        "training_rows": int(len(y_train)),
        "training_positive_rate": float(y_train.mean()),
        **smote_meta,
    }
    return estimator, metadata


def _load_or_build_dataset(
    *,
    dataset_path: str | Path | None,
    raw_dir: str | Path,
    processed_dir: str | Path,
) -> pd.DataFrame:
    candidate_path = Path(dataset_path or DEFAULT_DATASET_PATH)
    if candidate_path.exists():
        dataset = pd.read_csv(candidate_path, low_memory=False)
        dataset["transfer_date"] = pd.to_datetime(dataset["transfer_date"], errors="coerce")
        return dataset

    outputs = build_transfer_modeling_dataset(raw_dir=raw_dir, output_dir=processed_dir)
    save_transfer_modeling_dataset(outputs, processed_dir)
    return outputs["modeling_dataset"].copy()


def train_and_compare(
    *,
    dataset: pd.DataFrame | None = None,
    dataset_path: str | Path | None = None,
    raw_dir: str | Path = "data/player_scores_data",
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    sample_size: int | None = None,
    use_smote: bool = True,
    log_models: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    load_dotenv()

    model_output_dir = Path(output_dir).expanduser().resolve()
    model_output_dir.mkdir(parents=True, exist_ok=True)
    mlflow_tracking_uri = _resolve_tracking_uri(model_output_dir, tracking_uri)

    working_dataset = dataset.copy() if dataset is not None else _load_or_build_dataset(
        dataset_path=dataset_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    working_dataset["transfer_date"] = pd.to_datetime(working_dataset["transfer_date"], errors="coerce")
    working_dataset = working_dataset.dropna(subset=["transfer_date", "transfer_success"]).copy()
    working_dataset = working_dataset.sort_values("transfer_date", kind="stable").reset_index(drop=True)
    working_dataset = _sample_evenly(working_dataset, sample_size)

    feature_payload = build_feature_splits(working_dataset)
    summary_rows: list[dict[str, Any]] = []

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    for model_name in build_model_registry(random_state=random_state):
        with mlflow.start_run(run_name=model_name):
            model, training_metadata = train_model(
                model_name,
                feature_payload["X_train"],
                feature_payload["y_train"],
                random_state=random_state,
                use_smote=use_smote,
            )

            metrics: dict[str, float] = {}
            predictions: dict[str, pd.Series] = {}
            for split_name, X_key, y_key in [("train", "X_train", "y_train"), ("val", "X_val", "y_val"), ("test", "X_test", "y_test")]:
                split_metrics = evaluate_model(model, feature_payload[X_key], feature_payload[y_key], split_name=split_name)
                metrics.update(split_metrics)
                predictions[split_name] = pd.Series(model.predict(feature_payload[X_key]), index=feature_payload[y_key].index)

            mlflow.log_params(
                {
                    "model_name": model_name,
                    "sample_size": int(len(working_dataset)),
                    "feature_count": int(len(feature_payload["feature_names"])),
                    "train_end_date": str(feature_payload["split_summary"].loc[feature_payload["split_summary"]["split"] == "train", "end_date"].iloc[0]),
                    "val_end_date": str(feature_payload["split_summary"].loc[feature_payload["split_summary"]["split"] == "val", "end_date"].iloc[0]),
                    **{
                        key: value
                        for key, value in clone(build_model_registry(random_state=random_state)[model_name]).get_params(deep=False).items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    },
                    **training_metadata,
                }
            )
            mlflow.log_metrics({key: float(value) for key, value in metrics.items() if pd.notna(value)})

            artifact_dir = model_output_dir / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            confusion_paths = []
            for split_name in ["val", "test"]:
                confusion_path = _save_confusion_matrix(
                    artifact_dir,
                    model_name,
                    split_name,
                    feature_payload[f"y_{split_name}"],
                    predictions[split_name],
                )
                confusion_paths.append(confusion_path)
                mlflow.log_artifact(str(confusion_path))

            run_summary = {
                "model_name": model_name,
                **training_metadata,
                **metrics,
            }
            summary_rows.append(run_summary)

            metrics_path = artifact_dir / f"{model_name}_metrics.json"
            metrics_path.write_text(json.dumps(run_summary, indent=2, default=str), encoding="utf-8")
            mlflow.log_artifact(str(metrics_path))
            if log_models:
                mlflow.sklearn.log_model(model, artifact_path="model")

    summary = pd.DataFrame(summary_rows).sort_values(["val_f1", "test_f1", "model_name"], ascending=[False, False, True]).reset_index(drop=True)
    summary_path = model_output_dir / "phase2_model_comparison.csv"
    summary.to_csv(summary_path, index=False)

    best_model_summary = summary.iloc[0].to_dict() if not summary.empty else {}
    best_model_path = model_output_dir / "phase2_best_model.json"
    best_model_path.write_text(json.dumps(best_model_summary, indent=2, default=str), encoding="utf-8")
    feature_payload["split_summary"].to_csv(model_output_dir / "phase2_split_summary.csv", index=False)

    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "best_model_path": str(best_model_path),
        "tracking_uri": mlflow_tracking_uri,
        "split_summary": feature_payload["split_summary"],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and compare ScoutRadar transfer-success classifiers.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--raw-dir", default="data/player_scores_data")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_OUTPUT_DIR))
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--no-model-log", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = train_and_compare(
        dataset_path=args.dataset_path,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        sample_size=args.sample_size,
        use_smote=not args.no_smote,
        log_models=not args.no_model_log,
    )
    logger.info("Saved model comparison summary to %s", results["summary_path"])


if __name__ == "__main__":
    main()
