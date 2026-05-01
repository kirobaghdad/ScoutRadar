from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.models.transfer_success_modeling import run_model_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare transfer-success classification models.")
    parser.add_argument(
        "--dataset-path",
        default="data/processed/transfer_modeling_dataset_clean.csv",
        help="Path to the cleaned transfer modeling dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/transfer_success",
        help="Directory used for saved summaries and preprocessing artifacts.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Transfer Success Prediction",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to the local MLflow store.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    result = run_model_selection(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
    )

    print(result["results"].to_string(index=False))
    print(f"\nBest model: {result['best_model']['model_name']}")
    print(f"Artifacts: {result['output_dir']}")


if __name__ == "__main__":
    main()
