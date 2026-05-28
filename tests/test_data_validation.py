import pandas as pd
from disaster_tweet_classifier.data.validation import validate_dataset


def test_validate_dataset_returns_valid_report() -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 2],
            "keyword": ["fire", "storm"],
            "location": ["Canada", None],
            "text": [
                "Forest fire near La Ronge Sask. Canada",
                "This movie is on fire!",
            ],
            "target": [1, 0],
        }
    )

    report = validate_dataset(
        dataframe=dataframe,
        required_columns=["id", "keyword", "location", "text", "target"],
        text_column="text",
        target_column="target",
    )

    assert report.is_valid
    assert report.num_rows == 2
    assert report.num_columns == 5
    assert report.missing_columns == []
    assert report.duplicated_texts == 0
    assert report.class_distribution == {0: 1, 1: 1}
    assert report.missing_values["location"] == 1
