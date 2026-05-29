from disaster_tweet_classifier.preprocessing.text_cleaning import (
    TextCleaningConfig,
    clean_text,
    clean_texts,
)


def test_clean_text_replaces_url_and_mention() -> None:
    config = TextCleaningConfig(
        lowercase=False,
        replace_urls=True,
        replace_mentions=True,
        normalize_whitespace=True,
        remove_html_entities=True,
    )

    text = "Check this https://example.com @john"
    cleaned_text = clean_text(text=text, config=config)

    assert cleaned_text == "Check this HTTPURL @USER"


def test_clean_text_lowercases_for_baseline() -> None:
    config = TextCleaningConfig(
        lowercase=True,
        replace_urls=True,
        replace_mentions=True,
        normalize_whitespace=True,
        remove_html_entities=True,
    )

    text = "DISASTER near CITY!!!"
    cleaned_text = clean_text(text=text, config=config)

    assert cleaned_text == "disaster near city!!!"


def test_clean_texts_processes_multiple_texts() -> None:
    config = TextCleaningConfig(
        lowercase=True,
        replace_urls=False,
        replace_mentions=False,
        normalize_whitespace=True,
        remove_html_entities=True,
    )

    texts = [" First   text ", "Second   text"]
    cleaned_texts = clean_texts(texts=texts, config=config)

    assert cleaned_texts == ["first text", "second text"]
