from pathlib import Path

from src.models.train_model import train_and_compare

from .synthetic_phase2_data import create_synthetic_phase2_raw_dir


def test_train_and_compare_runs_end_to_end_with_mlflow_artifacts(tmp_path):
    raw_dir = create_synthetic_phase2_raw_dir(tmp_path, n_transfers=28)
    output_dir = tmp_path / "artifacts"
    tracking_dir = tmp_path / "mlruns"

    results = train_and_compare(
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        output_dir=output_dir,
        tracking_uri=str(tracking_dir),
        experiment_name="ScoutRadar Synthetic Test",
        log_models=False,
    )

    summary = results["summary"]
    assert len(summary) == 5
    assert set(summary["model_name"]) == {
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "SVC",
    }
    assert Path(results["summary_path"]).exists()
    assert any((output_dir / "artifacts").glob("*_confusion_matrix.png"))
    assert any(tracking_dir.rglob("meta.yaml"))
