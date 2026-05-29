from __future__ import annotations

from pathlib import Path

import fire
import mlflow
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from disaster_tweet_classifier.data.loading import (
    load_test_data,
    load_train_data,
    validate_test_data,
    validate_train_data,
)
from disaster_tweet_classifier.data.splitting import add_stratified_folds
from disaster_tweet_classifier.inference.onnx_export import export_bertweet_to_onnx
from disaster_tweet_classifier.inference.predictor import create_submission
from disaster_tweet_classifier.preprocessing.datasets import (
    add_clean_text_column,
    build_text_cleaning_config,
    save_processed_dataframe,
)
from disaster_tweet_classifier.tracking.git import get_git_commit_hash
from disaster_tweet_classifier.tracking.mlflow_tracking import (
    configure_mlflow,
    log_artifacts,
    log_config_params,
    log_metrics,
)
from disaster_tweet_classifier.training.baseline_trainer import train_and_save_baseline_model
from disaster_tweet_classifier.training.transformer_trainer import train_transformer_model

console = Console()


def load_config(
    config_path: str = "../configs",
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load Hydra config for CLI commands."""
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name, overrides=overrides or [])


def _print_validation_report(dataset_name: str, report: object) -> None:
    table = Table(title=f"{dataset_name} validation report")

    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("num_rows", str(report.num_rows))
    table.add_row("num_columns", str(report.num_columns))
    table.add_row("missing_columns", str(report.missing_columns))
    table.add_row("duplicated_texts", str(report.duplicated_texts))
    table.add_row("missing_values", str(report.missing_values))
    table.add_row("class_distribution", str(report.class_distribution))
    table.add_row("is_valid", str(report.is_valid))

    console.print(table)


class Commands:
    """CLI commands for the Disaster Tweet Classifier project."""

    def validate_data(self) -> None:
        """Load and validate raw Kaggle data files."""
        config = load_config()

        train_dataframe = load_train_data(config=config)
        test_dataframe = load_test_data(config=config)

        train_report = validate_train_data(config=config, dataframe=train_dataframe)
        test_report = validate_test_data(config=config, dataframe=test_dataframe)

        _print_validation_report(dataset_name="train.csv", report=train_report)
        _print_validation_report(dataset_name="test.csv", report=test_report)

    def prepare_folds(self) -> None:
        """Create stratified folds and save them to processed data directory."""
        config = load_config()

        train_dataframe = load_train_data(config=config)
        dataframe_with_folds = add_stratified_folds(
            dataframe=train_dataframe,
            target_column=config.data.target_column,
            num_folds=config.data.num_folds,
            random_state=config.project.seed,
        )

        processed_dir = Path(config.data.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        output_path = processed_dir / "train_folds.csv"
        dataframe_with_folds.to_csv(output_path, index=False)

        console.print(f"[bold green]Saved processed folds to:[/bold green] {output_path}")

    def show_config(self) -> None:
        """Print loaded Hydra config."""
        config = load_config()
        console.print(OmegaConf.to_yaml(config))

    def version(self) -> None:
        """Print CLI version."""
        console.print("[bold green]disaster-tweet-classifier version 0.1.0[/bold green]")

    def prepare_clean_data(self) -> None:
        """Create cleaned processed dataset from train folds."""
        config = load_config()

        processed_dir = Path(config.data.processed_dir)
        input_path = processed_dir / config.data.train_folds_file
        output_path = processed_dir / config.data.train_folds_clean_file

        if not input_path.exists():
            message = (
                f"Input file `{input_path}` does not exist. "
                "Run `uv run disaster-tweet prepare-folds` first."
            )
            raise FileNotFoundError(message)

        dataframe = pd.read_csv(input_path)
        cleaning_config = build_text_cleaning_config(config=config)

        processed_dataframe = add_clean_text_column(
            dataframe=dataframe,
            text_column=config.data.text_column,
            clean_text_column=config.data.clean_text_column,
            cleaning_config=cleaning_config,
        )

        save_processed_dataframe(dataframe=processed_dataframe, output_path=output_path)

        console.print(f"[bold green]Saved cleaned data to:[/bold green] {output_path}")

    def train_baseline(self) -> None:
        """Train TF-IDF + Logistic Regression baseline model."""
        config = load_config(
            overrides=[
                "model=baseline_logreg",
                "training=baseline",
                "preprocessing=baseline",
            ]
        )

        input_path = Path(config.data.processed_dir) / config.data.train_folds_baseline_clean_file
        model_output_path = Path(config.training.outputs.model_path)
        metrics_output_path = Path(config.training.outputs.metrics_path)
        plots_dir = Path(config.training.outputs.plots_dir)

        if not input_path.exists():
            message = (
                f"Input file `{input_path}` does not exist. "
                "Run `uv run disaster-tweet prepare-baseline-data` first."
            )
            raise FileNotFoundError(message)

        configure_mlflow(
            tracking_uri=config.training.mlflow.tracking_uri,
            experiment_name=config.training.mlflow.experiment_name,
        )

        git_commit_hash = get_git_commit_hash()

        with mlflow.start_run(run_name=config.training.mlflow.run_name):
            mlflow.log_param("git_commit_hash", git_commit_hash)
            log_config_params(config=config)

            result = train_and_save_baseline_model(
                config=config,
                input_path=input_path,
                model_output_path=model_output_path,
                metrics_output_path=metrics_output_path,
                plots_dir=plots_dir,
                validation_fold=config.training.validation_fold,
            )

            log_metrics(metrics=result.metrics.to_dict())
            log_artifacts(
                paths=[
                    model_output_path,
                    metrics_output_path,
                    plots_dir,
                ]
            )

        console.print("[bold green]Baseline model trained successfully.[/bold green]")
        console.print(f"Model saved to: {model_output_path}")
        console.print(f"Metrics saved to: {metrics_output_path}")
        console.print(f"Plots saved to: {plots_dir}")
        console.print(result.metrics.to_dict())

    def prepare_baseline_data(self) -> None:
        """Create cleaned processed dataset for baseline model."""
        config = load_config(overrides=["preprocessing=baseline"])

        processed_dir = Path(config.data.processed_dir)
        input_path = processed_dir / config.data.train_folds_file
        output_path = processed_dir / "train_folds_baseline_clean.csv"

        if not input_path.exists():
            message = (
                f"Input file `{input_path}` does not exist. "
                "Run `uv run disaster-tweet prepare-folds` first."
            )
            raise FileNotFoundError(message)

        dataframe = pd.read_csv(input_path)
        cleaning_config = build_text_cleaning_config(config=config)

        processed_dataframe = add_clean_text_column(
            dataframe=dataframe,
            text_column=config.data.text_column,
            clean_text_column=config.data.clean_text_column,
            cleaning_config=cleaning_config,
        )

        save_processed_dataframe(dataframe=processed_dataframe, output_path=output_path)

        console.print(f"[bold green]Saved baseline cleaned data to:[/bold green] {output_path}")

    def train_bertweet(self) -> None:
        """Train BERTweet classifier with PyTorch Lightning."""
        config = load_config(
            overrides=[
                "model=bertweet",
                "training=bertweet",
                "preprocessing=bertweet",
            ]
        )

        input_path = Path(config.data.processed_dir) / config.data.train_folds_clean_file

        if not input_path.exists():
            message = (
                f"Input file `{input_path}` does not exist. "
                "Run `uv run disaster-tweet prepare-clean-data` first."
            )
            raise FileNotFoundError(message)

        metrics = train_transformer_model(config=config, data_path=input_path)

        console.print("[bold green]BERTweet model trained successfully.[/bold green]")
        console.print(metrics)

    def predict_submission(self) -> None:
        """Create Kaggle submission using trained BERTweet checkpoint."""
        config = load_config(
            overrides=[
                "model=bertweet",
                "training=bertweet",
                "preprocessing=bertweet",
            ]
        )

        test_path = Path(config.data.raw_dir) / config.data.test_file
        sample_submission_path = Path(config.data.raw_dir) / config.data.sample_submission_file
        checkpoint_path = Path(config.training.inference.checkpoint_path)
        output_path = Path(config.training.outputs.submission_path)

        if not test_path.exists():
            message = f"Test file `{test_path}` does not exist."
            raise FileNotFoundError(message)

        if not sample_submission_path.exists():
            message = f"Sample submission file `{sample_submission_path}` does not exist."
            raise FileNotFoundError(message)

        output_path = create_submission(
            config=config,
            test_path=test_path,
            sample_submission_path=sample_submission_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
        )

        console.print("[bold green]Submission created successfully.[/bold green]")
        console.print(f"Submission saved to: {output_path}")

    def serve_api(
        self,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 8000,
        reload: bool = False,
    ) -> None:
        """Serve FastAPI inference API."""
        import uvicorn

        uvicorn.run(
            "disaster_tweet_classifier.api.app:app",
            host=host,
            port=port,
            reload=reload,
        )

    def export_onnx(
        self,
        checkpoint_path: str | None = None,
        output_path: str = "artifacts/onnx/bertweet/model.onnx",
        metadata_path: str = "artifacts/onnx/bertweet/metadata.json",
        opset_version: int = 17,
    ) -> None:
        """Export trained BERTweet classifier to ONNX."""
        config = load_config(
            overrides=[
                "model=bertweet",
                "training=bertweet",
                "preprocessing=bertweet",
            ]
        )

        resolved_checkpoint_path = Path(
            checkpoint_path or config.training.inference.checkpoint_path
        )

        export_bertweet_to_onnx(
            config=config,
            checkpoint_path=resolved_checkpoint_path,
            output_path=Path(output_path),
            metadata_path=Path(metadata_path),
            opset_version=opset_version,
        )

        print(f"ONNX model saved to: {output_path}")
        print(f"ONNX metadata saved to: {metadata_path}")


def main() -> None:
    fire.Fire(Commands)
