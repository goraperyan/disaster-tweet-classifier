from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_train_validation_split(
    dataframe: pd.DataFrame,
    target_column: str,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train-validation split."""
    train_dataframe, validation_dataframe = train_test_split(
        dataframe,
        test_size=validation_size,
        random_state=random_state,
        stratify=dataframe[target_column],
    )

    return train_dataframe.reset_index(drop=True), validation_dataframe.reset_index(drop=True)


def add_stratified_folds(
    dataframe: pd.DataFrame,
    target_column: str,
    num_folds: int,
    random_state: int,
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Add stratified K-Fold indices to the dataframe."""
    dataframe_with_folds = dataframe.copy()
    dataframe_with_folds[fold_column] = -1

    splitter = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_state,
    )

    for fold_index, (_, validation_indices) in enumerate(
        splitter.split(dataframe_with_folds, dataframe_with_folds[target_column])
    ):
        dataframe_with_folds.loc[validation_indices, fold_column] = fold_index

    dataframe_with_folds[fold_column] = dataframe_with_folds[fold_column].astype(int)
    return dataframe_with_folds
