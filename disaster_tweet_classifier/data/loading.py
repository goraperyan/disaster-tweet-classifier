from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from disaster_tweet_classifier.data.validation import (
    DatasetValidationReport,
    validate_dataset,
)


def get_raw_data_paths(config: DictConfig) -> dict[str, Path]:
    """Build raw dataset file paths from Hydra config."""
    raw_dir = Path(config.data.raw_dir)

    return {
        "train": raw_dir / config.data.train_file,
        "test": raw_dir / config.data.test_file,
        "sample_submission": raw_dir / config.data.sample_submission_file,
    }


def check_raw_data_exists(config: DictConfig) -> None:
    """Check that all required raw dataset files exist."""
    paths = get_raw_data_paths(config=config)
    missing_files = [str(path) for path in paths.values() if not path.exists()]

    if missing_files:
        missing_files_text = "\n".join(f"- {file_path}" for file_path in missing_files)
        message = (
            "Raw dataset files are missing.\n"
            "Please download the Kaggle competition files manually and place them into "
            f"`{config.data.raw_dir}`.\n\n"
            "Missing files:\n"
            f"{missing_files_text}"
        )
        raise FileNotFoundError(message)


def load_train_data(config: DictConfig) -> pd.DataFrame:
    """Load train.csv from the raw data directory."""
    check_raw_data_exists(config=config)
    train_path = get_raw_data_paths(config=config)["train"]
    return pd.read_csv(train_path)


def load_test_data(config: DictConfig) -> pd.DataFrame:
    """Load test.csv from the raw data directory."""
    check_raw_data_exists(config=config)
    test_path = get_raw_data_paths(config=config)["test"]
    return pd.read_csv(test_path)


def load_sample_submission(config: DictConfig) -> pd.DataFrame:
    """Load sample_submission.csv from the raw data directory."""
    check_raw_data_exists(config=config)
    sample_submission_path = get_raw_data_paths(config=config)["sample_submission"]
    return pd.read_csv(sample_submission_path)


def validate_train_data(config: DictConfig, dataframe: pd.DataFrame) -> DatasetValidationReport:
    """Validate train dataframe against expected Kaggle schema."""
    required_columns = [
        config.data.id_column,
        config.data.keyword_column,
        config.data.location_column,
        config.data.text_column,
        config.data.target_column,
    ]

    return validate_dataset(
        dataframe=dataframe,
        required_columns=required_columns,
        text_column=config.data.text_column,
        target_column=config.data.target_column,
    )


def validate_test_data(config: DictConfig, dataframe: pd.DataFrame) -> DatasetValidationReport:
    """Validate test dataframe against expected Kaggle schema."""
    required_columns = [
        config.data.id_column,
        config.data.keyword_column,
        config.data.location_column,
        config.data.text_column,
    ]

    return validate_dataset(
        dataframe=dataframe,
        required_columns=required_columns,
        text_column=config.data.text_column,
        target_column=None,
    )
