from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TweetClassificationDataset(Dataset):
    """Torch dataset for tweet binary classification."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
        target_column: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts = dataframe[text_column].astype(str).tolist()
        self.targets = dataframe[target_column].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return tokenized sample."""
        encoding = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.targets[index], dtype=torch.long),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item
