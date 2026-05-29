from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from disaster_tweet_classifier.preprocessing.text_cleaning import (
    TextCleaningConfig,
    clean_text,
)


def build_text_cleaning_config(config: DictConfig) -> TextCleaningConfig:
    """Build text cleaning config from Hydra config."""
    return TextCleaningConfig(
        lowercase=bool(config.preprocessing.lowercase),
        replace_urls=bool(config.preprocessing.replace_urls),
        replace_mentions=bool(config.preprocessing.replace_mentions),
        normalize_whitespace=bool(config.preprocessing.normalize_whitespace),
        remove_html_entities=bool(config.preprocessing.remove_html_entities),
        remove_punctuation=bool(getattr(config.preprocessing, "remove_punctuation", False)),
    )


def add_clean_text_column(
    dataframe: pd.DataFrame,
    text_column: str,
    clean_text_column: str,
    cleaning_config: TextCleaningConfig,
) -> pd.DataFrame:
    """Add a cleaned text column to dataframe."""
    dataframe_with_clean_text = dataframe.copy()
    dataframe_with_clean_text[clean_text_column] = dataframe_with_clean_text[text_column].apply(
        lambda text: clean_text(text=text, config=cleaning_config)
    )
    return dataframe_with_clean_text


def save_processed_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save processed dataframe to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
