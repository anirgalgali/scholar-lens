import torch
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict,Dataset
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding
from src.config import TrainingConfig

class ArxivDataset(Dataset):
    def __init__(self, tokenized_dataset: Dataset) -> None:
        valid_columns_names = [
            "input_ids",
            "attention_mask",
            "label",
            "input",
            "category",
        ]
        self.dataset = tokenized_dataset.remove_columns(
            column_names=[
                k
                for k in tokenized_dataset.column_names
                if k not in valid_columns_names
            ]
        )

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return dict(
            input_ids=torch.tensor(item["input_ids"]),  # convert to pytorch tensor
            attention_mask=torch.tensor(item["attention_mask"]),
            labels=torch.tensor(item["label"]),
        )

    def __len__(self):
        return self.dataset.num_rows


def create_dataloaders(tokenized_data:DatasetDict, 
                       tokenizer: PreTrainedTokenizerFast,
                       config: TrainingConfig) -> tuple:

    data_collator = DataCollatorWithPadding(tokenizer)

    # Create pytorch dataloaders for train, val splits from the loaded dataset
    train_dataloader = DataLoader(
        ArxivDataset(tokenized_data["train"]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=data_collator,
    )

    val_dataloader = DataLoader(
        ArxivDataset(tokenized_data["validation"]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=data_collator,
    )

    test_dataloader = DataLoader(
        ArxivDataset(tokenized_data["test"]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=data_collator,
    )

    return train_dataloader, val_dataloader, test_dataloader