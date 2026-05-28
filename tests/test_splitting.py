import pandas as pd
from disaster_tweet_classifier.data.splitting import (
    add_stratified_folds,
    make_train_validation_split,
)


def test_make_train_validation_split_preserves_total_size() -> None:
    dataframe = pd.DataFrame(
        {
            "text": [f"text {index}" for index in range(20)],
            "target": [0, 1] * 10,
        }
    )

    train_dataframe, validation_dataframe = make_train_validation_split(
        dataframe=dataframe,
        target_column="target",
        validation_size=0.2,
        random_state=42,
    )

    assert len(train_dataframe) == 16
    assert len(validation_dataframe) == 4


def test_add_stratified_folds_creates_fold_column() -> None:
    dataframe = pd.DataFrame(
        {
            "text": [f"text {index}" for index in range(20)],
            "target": [0, 1] * 10,
        }
    )

    dataframe_with_folds = add_stratified_folds(
        dataframe=dataframe,
        target_column="target",
        num_folds=5,
        random_state=42,
    )

    assert "fold" in dataframe_with_folds.columns
    assert sorted(dataframe_with_folds["fold"].unique().tolist()) == [0, 1, 2, 3, 4]
