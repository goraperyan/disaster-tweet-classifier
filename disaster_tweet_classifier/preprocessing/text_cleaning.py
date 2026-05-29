from __future__ import annotations

import html
import re
import string
from dataclasses import dataclass

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class TextCleaningConfig:
    """Configuration for text cleaning."""

    lowercase: bool
    replace_urls: bool
    replace_mentions: bool
    normalize_whitespace: bool
    remove_html_entities: bool
    remove_punctuation: bool = False


def clean_text(text: str, config: TextCleaningConfig) -> str:
    """Clean a single text according to the provided configuration."""
    cleaned_text = str(text)

    if config.remove_html_entities:
        cleaned_text = html.unescape(cleaned_text)

    if config.replace_urls:
        cleaned_text = URL_PATTERN.sub("HTTPURL", cleaned_text)

    if config.replace_mentions:
        cleaned_text = MENTION_PATTERN.sub("@USER", cleaned_text)

    if config.lowercase:
        cleaned_text = cleaned_text.lower()

    if config.remove_punctuation:
        cleaned_text = cleaned_text.translate(str.maketrans("", "", string.punctuation))

    if config.normalize_whitespace:
        cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()

    return cleaned_text


def clean_texts(texts: list[str], config: TextCleaningConfig) -> list[str]:
    """Clean multiple texts according to the provided configuration."""
    return [clean_text(text=text, config=config) for text in texts]
