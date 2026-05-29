import pandas as pd
import torch
from disaster_tweet_classifier.data.torch_datasets import TweetClassificationDataset


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def test_tweet_classification_dataset_returns_tokenized_item() -> None:
    dataframe = pd.DataFrame(
        {
            "clean_text": ["fire in city"],
            "target": [1],
        }
    )

    dataset = TweetClassificationDataset(
        dataframe=dataframe,
        text_column="clean_text",
        target_column="target",
        tokenizer=FakeTokenizer(),
        max_length=8,
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["labels"].item() == 1
