from __future__ import annotations

from pathlib import Path

import fire
from hydra import compose, initialize
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from disaster_tweet_classifier.data.loading import (
    load_test_data,
    load_train_data,
    validate_test_data,
    validate_train_data,
)
from disaster_tweet_classifier.data.splitting import add_stratified_folds

console = Console()


def load_config(config_path: str = "../configs", config_name: str = "config") -> DictConfig:
    """Load Hydra config for CLI commands."""
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name)


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
        console.print(config)

    def version(self) -> None:
        """Print CLI version."""
        console.print("[bold green]disaster-tweet-classifier version 0.1.0[/bold green]")


def main() -> None:
    fire.Fire(Commands)
