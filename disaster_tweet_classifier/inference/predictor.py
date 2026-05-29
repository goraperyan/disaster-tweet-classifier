from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from disaster_tweet_classifier.data.torch_datasets import TweetClassificationDataset
from disaster_tweet_classifier.models.transformer_classifier import TransformerTweetClassifier
from disaster_tweet_classifier.preprocessing.text_cleaning import clean_text


def find_latest_checkpoint(checkpoint_path: Path) -> Path:
    """Find latest checkpoint in a file or directory path."""
    if checkpoint_path.is_file():
        return checkpoint_path

    checkpoint_files = sorted(
        checkpoint_path.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not checkpoint_files:
        message = f"No checkpoint files found in `{checkpoint_path}`."
        raise FileNotFoundError(message)

    return checkpoint_files[0]


def prepare_test_dataframe(config: DictConfig, test_path: Path) -> pd.DataFrame:
    """Load and preprocess test dataframe."""
    dataframe = pd.read_csv(test_path)
    dataframe[config.data.clean_text_column] = dataframe[config.data.text_column].apply(
        lambda text: clean_text(text=text, config=config.preprocessing)
    )

    return dataframe


def predict_probabilities(
    config: DictConfig,
    dataframe: pd.DataFrame,
    checkpoint_path: Path,
) -> list[float]:
    """Predict disaster probabilities for input dataframe."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_model_name,
        use_fast=False,
    )

    inference_dataframe = dataframe.copy()
    inference_dataframe[config.data.target_column] = 0

    dataset = TweetClassificationDataset(
        dataframe=inference_dataframe,
        text_column=config.data.clean_text_column,
        target_column=config.data.target_column,
        tokenizer=tokenizer,
        max_length=config.model.tokenizer.max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    model = TransformerTweetClassifier.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        pretrained_model_name=config.model.pretrained_model_name,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        freeze_backbone=config.model.freeze_backbone,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    probabilities: list[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(batch_probabilities.cpu().tolist())

    return probabilities


def create_submission(
    config: DictConfig,
    test_path: Path,
    sample_submission_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    """Create Kaggle submission file."""
    resolved_checkpoint_path = find_latest_checkpoint(checkpoint_path=checkpoint_path)

    test_dataframe = prepare_test_dataframe(config=config, test_path=test_path)
    sample_submission = pd.read_csv(sample_submission_path)

    probabilities = predict_probabilities(
        config=config,
        dataframe=test_dataframe,
        checkpoint_path=resolved_checkpoint_path,
    )

    predictions = [
        int(probability >= config.training.inference.threshold) for probability in probabilities
    ]

    submission = sample_submission.copy()
    submission[config.data.target_column] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    return output_path
