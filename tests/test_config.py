from disaster_tweet_classifier.commands import load_config


def test_hydra_config_loads() -> None:
    config = load_config()

    assert config.project.name == "disaster-tweet-classifier"
    assert config.project.seed == 42
    assert config.data.text_column == "text"
    assert config.data.target_column == "target"
