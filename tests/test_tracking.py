from disaster_tweet_classifier.tracking.mlflow_tracking import flatten_dict


def test_flatten_dict_flattens_nested_dictionary() -> None:
    data = {
        "model": {
            "name": "baseline",
            "vectorizer": {
                "min_df": 2,
            },
        },
        "seed": 42,
    }

    flattened = flatten_dict(data)

    assert flattened == {
        "model.name": "baseline",
        "model.vectorizer.min_df": 2,
        "seed": 42,
    }
