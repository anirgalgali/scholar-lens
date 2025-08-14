import torch
from src.config import ModelConfig, TrainingConfig, DataConfig, RunConfig
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
    get_scheduler,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.dataset import ArxivDataset
from src.data_processing import get_tokenized_dataset
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config: RunConfig):

    # Load processed data from disk
    num_classes = len(config.data.subjects) * config.data.num_categories_per_subject
    model_config = ModelConfig(num_classes=num_classes)
    config.model = model_config  # monkey patch the changed model config class
    processed_dataset = get_tokenized_dataset(config.data, config.model)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Create pytorch dataloaders for train, val splits from the loaded dataset
    train_dataloader = DataLoader(
        ArxivDataset(processed_dataset["train"]),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        collate_fn=data_collator,
    )

    val_dataloader = DataLoader(
        ArxivDataset(processed_dataset["validation"]),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        collate_fn=data_collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name, num_labels=config.model.num_classes
    )

    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)

    model.to(device)

    # Checking if a single batch fits without issues (for debugging)
    batch_iter = iter(train_dataloader)
    progress_bar = tqdm(range(config.train.num_epochs))
    for epoch in range(config.train.num_epochs):
        batch = {k: v.to(device) for k, v in next(batch_iter).items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        print(f"Epoch {epoch} - Loss = {loss.item()}")


if __name__ == "__main__":
    config = RunConfig()
    train(config)
