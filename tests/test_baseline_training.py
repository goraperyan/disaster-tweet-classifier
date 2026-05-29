import pandas as pd
from disaster_tweet_classifier.commands import load_config
from disaster_tweet_classifier.training.baseline_trainer import train_baseline_model


def test_train_baseline_model_returns_metrics() -> None:
    config = load_config(
        overrides=[
            "model=baseline_logreg",
            "training=baseline",
            "preprocessing=baseline",
            "model.vectorizer.min_df=1",
        ]
    )

    dataframe = pd.DataFrame(
        {
            "clean_text": [
                "fire in the city",
                "earthquake destroyed buildings",
                "nice weather today",
                "watching a movie",
                "flood warning issued",
                "having dinner with friends",
                "wildfire spreading fast",
                "new song released",
                "storm damaged houses",
                "reading a book",
            ],
            "target": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "fold": [0, 1, 0, 1, 2, 2, 3, 3, 4, 4],
        }
    )

    result = train_baseline_model(
        config=config,
        dataframe=dataframe,
        validation_fold=0,
    )

    metrics_dict = result.metrics.to_dict()

    assert set(metrics_dict) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics_dict.values())
