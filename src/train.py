import torch
from src.config import ModelConfig, TrainingConfig, DataConfig, RunConfig
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
    get_scheduler,
    PreTrainedModel,
)
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from src.dataset import ArxivDataset
from src.data_processing import get_tokenized_dataset
from tqdm.auto import tqdm
import wandb


class Trainer:

    def __init__(
        self,
        model: PreTrainedModel,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr_scheduler: LRScheduler,
        config: RunConfig,
    ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = lr_scheduler
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wandb.init(
            project="scholar-lens",
            config={
                "learning_rate": self.config.train.learning_rate,
                "epochs": self.config.train.num_epochs,
                "batch_size": self.config.train.batch_size,
                "model_name": self.config.model.model_name,
                "dataset_id": self.config.data.version_id,
            },
        )

        self.model.to(self.device)
        self.best_f1_score = 0.0

    def train_one_epoch(self) -> float:
        """Performs a single epoch of training"""

        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            model_logits = self.model(**batch)
            batch_loss = model_logits.loss
            batch_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += batch_loss.item()

        return total_loss / len(self.train_loader)


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

    for epoch in range(config.train.num_epochs):

        progress_bar.update(1)
        print(f"Epoch {epoch} - Loss = {loss.item()}")


if __name__ == "__main__":
    config = RunConfig()
    train(config)
