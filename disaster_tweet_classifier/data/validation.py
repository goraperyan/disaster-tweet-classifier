from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetValidationReport:
    """Summary of dataset validation checks."""

    num_rows: int
    num_columns: int
    missing_columns: list[str]
    duplicated_texts: int
    missing_values: dict[str, int]
    class_distribution: dict[int, int] | None

    @property
    def is_valid(self) -> bool:
        return len(self.missing_columns) == 0


def validate_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Return a list of required columns that are missing from the dataframe."""
    return [column for column in required_columns if column not in dataframe.columns]


def count_missing_values(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count missing values for each dataframe column."""
    return dataframe.isna().sum().astype(int).to_dict()


def count_duplicated_texts(dataframe: pd.DataFrame, text_column: str) -> int:
    """Count duplicated texts in the dataframe."""
    if text_column not in dataframe.columns:
        return 0
    return int(dataframe.duplicated(subset=[text_column]).sum())


def get_class_distribution(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[int, int] | None:
    """Return class distribution if target column exists."""
    if target_column not in dataframe.columns:
        return None

    distribution = dataframe[target_column].value_counts().sort_index()
    return {int(label): int(count) for label, count in distribution.items()}


def validate_dataset(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    text_column: str,
    target_column: str | None = None,
) -> DatasetValidationReport:
    """Validate dataset schema and basic data quality properties."""
    missing_columns = validate_columns(dataframe=dataframe, required_columns=required_columns)

    class_distribution = None
    if target_column is not None:
        class_distribution = get_class_distribution(
            dataframe=dataframe,
            target_column=target_column,
        )

    return DatasetValidationReport(
        num_rows=len(dataframe),
        num_columns=len(dataframe.columns),
        missing_columns=missing_columns,
        duplicated_texts=count_duplicated_texts(
            dataframe=dataframe,
            text_column=text_column,
        ),
        missing_values=count_missing_values(dataframe=dataframe),
        class_distribution=class_distribution,
    )
