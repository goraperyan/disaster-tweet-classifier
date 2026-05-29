from __future__ import annotations

from pathlib import Path

import lightning as L  # noqa: N812
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from disaster_tweet_classifier.data.torch_datasets import TweetClassificationDataset


class TweetDataModule(L.LightningDataModule):
    """LightningDataModule for tweet classification."""

    def __init__(
        self,
        config: DictConfig,
        data_path: Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.data_path = data_path
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.train_dataset: TweetClassificationDataset | None = None
        self.validation_dataset: TweetClassificationDataset | None = None

    def prepare_data(self) -> None:
        """Download tokenizer files if needed."""
        AutoTokenizer.from_pretrained(
            self.config.model.pretrained_model_name,
            use_fast=False,
        )

    def setup(self, stage: str | None = None) -> None:
        """Create train and validation datasets."""
        dataframe = pd.read_csv(self.data_path)

        train_dataframe = dataframe[
            dataframe[self.config.data.fold_column] != self.config.training.validation_fold
        ].reset_index(drop=True)

        validation_dataframe = dataframe[
            dataframe[self.config.data.fold_column] == self.config.training.validation_fold
        ].reset_index(drop=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.pretrained_model_name,
            use_fast=False,
        )

        self.train_dataset = TweetClassificationDataset(
            dataframe=train_dataframe,
            text_column=self.config.data.clean_text_column,
            target_column=self.config.data.target_column,
            tokenizer=self.tokenizer,
            max_length=self.config.model.tokenizer.max_length,
        )

        self.validation_dataset = TweetClassificationDataset(
            dataframe=validation_dataframe,
            text_column=self.config.data.clean_text_column,
            target_column=self.config.data.target_column,
            tokenizer=self.tokenizer,
            max_length=self.config.model.tokenizer.max_length,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        if self.train_dataset is None:
            message = "Train dataset is not initialized. Call setup() first."
            raise RuntimeError(message)

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.validation_dataset is None:
            message = "Validation dataset is not initialized. Call setup() first."
            raise RuntimeError(message)

        return DataLoader(
            self.validation_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
        )
