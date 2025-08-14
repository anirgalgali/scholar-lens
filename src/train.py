import torch
from src.config import ModelConfig, RunConfig
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
    PreTrainedModel,
)
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from src.dataset import ArxivDataset
from src.data_processing import get_tokenized_dataset
from tqdm.auto import tqdm
import wandb
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics import MetricCollection


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
        self.start_epoch = 0
        self.run_id = None
        self.best_f1_score = 0.0
        self.model.to(self.device)

        if self.config.train.resume_from_checkpoint:
            ckpt_path = self.config.train.resume_from_checkpoint
            self.run_id = ckpt_path.split("_")[0].split("-")[1]
            self.load_from_checkpoint(ckpt_path)

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

        if self.run_id is None:
            self.run_id = wandb.run.id

    def _train_one_epoch(self) -> float:
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
            wandb.log({"train_batch_loss": batch_loss.item()})
            total_loss += batch_loss.item()

        return total_loss / len(self.train_loader)

    def _evaluate(self) -> float:

        self.model.eval()

        metric_collection = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=self.config.model.num_classes, average="macro"
                ),
                "macro_precision": MulticlassPrecision(
                    num_classes=self.config.model.num_classes, average="macro"
                ),
                "macro_recall": MulticlassRecall(
                    num_classes=self.config.model.num_classes, average="macro"
                ),
                "macro_f1": MulticlassF1Score(
                    num_classes=self.config.model.num_classes, average="macro"
                ),
            }
        ).to(self.device)

        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                model_logits = self.model(**batch).logits
                predicted_labels = torch.argmax(model_logits, dim=-1)
                metric_collection.update(predicted_labels, batch["labels"])
                all_predictions.append(predicted_labels)
                all_labels.append(batch["labels"])

        all_predictions = torch.cat(all_predictions).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        final_metrics = metric_collection.compute()
        return final_metrics, all_predictions, all_labels

    def _save_checkpoint(self, current_f1_score: float, epoch: int):

        if current_f1_score > self.best_f1_score:

            self.best_f1_score = current_f1_score
            print(
                f"New best F1 score: {self.best_f1_score:.4f}. Saving model checkpoint..."
            )
            checkpoint = {
                "epoch": epoch,
                "best_f1_score": self.best_f1_score,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            }

            torch.save(checkpoint, f"run-{self.run_id}_best_model_checkpoint.pth")

    def load_from_checkpoint(self, checkpoint_path: str):
        print(f" Loading ckpt from : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_f1_score = checkpoint["best_f1_score"]
        print(f"Resuming training from epoch {self.start_epoch}")

    def train(self):
        print(f" Starting training...")
        for epoch in range(self.config.train.num_epochs):
            avg_train_loss = self._train_one_epoch()
            val_metrics, ypred_val, ytrue_val = self._evaluate()
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    "validation_metrics": val_metrics,
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        preds=ypred_val, y_true=ytrue_val, class_names=self.class_names
                    ),
                }
            )
            self._save_checkpoint(val_metrics["macro_f1"], epoch + 1)


def run():
    pass


# def train(config: RunConfig):

#     # Load processed data from disk
#     num_classes = len(config.data.subjects) * config.data.num_categories_per_subject
#     model_config = ModelConfig(num_classes=num_classes)
#     config.model = model_config  # monkey patch the changed model config class
#     processed_dataset = get_tokenized_dataset(config.data, config.model)

#     tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
#     data_collator = DataCollatorWithPadding(tokenizer)

#     # Create pytorch dataloaders for train, val splits from the loaded dataset
#     train_dataloader = DataLoader(
#         ArxivDataset(processed_dataset["train"]),
#         batch_size=64,
#         shuffle=True,
#         num_workers=0,
#         collate_fn=data_collator,
#     )

#     val_dataloader = DataLoader(
#         ArxivDataset(processed_dataset["validation"]),
#         batch_size=64,
#         shuffle=True,
#         num_workers=0,
#         collate_fn=data_collator,
#     )

#     model = AutoModelForSequenceClassification.from_pretrained(
#         config.model.model_name, num_labels=config.model.num_classes
#     )

#     optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)

#     model.to(device)

#     for epoch in range(config.train.num_epochs):

#         progress_bar.update(1)
#         print(f"Epoch {epoch} - Loss = {loss.item()}")


# if __name__ == "__main__":
#     config = RunConfig()
#     train(config)
