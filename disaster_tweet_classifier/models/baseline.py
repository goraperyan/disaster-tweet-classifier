from __future__ import annotations

from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline_model(config: DictConfig) -> Pipeline:
    """Build TF-IDF + Logistic Regression baseline model."""
    vectorizer = TfidfVectorizer(
        min_df=config.model.vectorizer.min_df,
        max_df=config.model.vectorizer.max_df,
        max_features=config.model.vectorizer.max_features,
        ngram_range=tuple(config.model.vectorizer.ngram_range),
    )

    classifier = LogisticRegression(
        max_iter=config.model.classifier.max_iter,
        class_weight=config.model.classifier.class_weight,
        random_state=config.project.seed,
    )

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )
