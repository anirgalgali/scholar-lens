import os
import json
import torch
from src.config import ModelConfig, RunConfig, TrainingConfig
from src.models.model import create_model
from transformers import AutoTokenizer, PreTrainedModel
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LRScheduler, LinearLR
from src.dataset import create_dataloaders
from src.data_processing import get_tokenized_dataset
from src.plotting import plot_confusion_matrix_heatmap
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
        class_names: list[str],
    ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = lr_scheduler
        self.config = config
        self.class_names = class_names

        self.run_id = None
        self.start_epoch = 0
        self.global_step = 0
        self.best_f1_score = 0.0

        if self.config.train.resume_from_checkpoint:
            ckpt_path = self.config.train.resume_from_checkpoint
            self.load_from_checkpoint(ckpt_path)

        wandb.init(
            project="scholar-lens",
            entity=os.getenv("WANDB_ENTITY"),
            id=self.run_id,
            resume="allow",
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.metric_collection = MetricCollection({
            'macro_accuracy': MulticlassAccuracy(num_classes=config.model.num_classes, average='macro'),
            'macro_f1': MulticlassF1Score(num_classes=config.model.num_classes, average='macro'),
            'macro_precision': MulticlassPrecision(num_classes=config.model.num_classes, average='macro'),
            'macro_recall': MulticlassRecall(num_classes=config.model.num_classes, average='macro') }).to(self.device)


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
            wandb.log({"train_batch_loss": batch_loss.item()}, step=self.global_step)
            self.global_step += 1
            batch_size = batch['input_ids'].shape[0]
            total_loss += batch_loss.item() * batch_size

            if self.global_step % self.config.train.eval_every_n_steps == 0:
                print(f"--- Intra-epoch eval at step {self.global_step}---")
                eval_results = self._evaluate()
                wandb.log(eval_results,
                step=self.global_step)

        return total_loss / len(self.train_loader.dataset) # This is the average per_sample loss

    def _evaluate(self, compute_all_metrics: bool = False) -> float:

        self.model.eval()

        if compute_all_metrics:
            self.metric_collection.reset()
            all_predictions = []
            all_labels = []

        total_val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                model_outputs = self.model(**batch)
                val_loss = model_outputs.loss
                batch_size = batch['input_ids'].shape[0]
                total_val_loss += val_loss.item() * batch_size

                if compute_all_metrics:
                    predicted_labels = torch.argmax(model_outputs.logits, dim=-1)
                    self.metric_collection.update(predicted_labels, batch["labels"])
                    all_predictions.append(predicted_labels)
                    all_labels.append(batch["labels"]) 

        avg_val_loss = total_val_loss / len(self.val_loader.dataset)  # This is the average per_sample loss
        results = {"avg_val_loss": avg_val_loss}
        if compute_all_metrics:
            results["validation_metrics"] = {k: v.item() for k, v in self.metric_collection.compute().items()}
            results["val_pred_labels"] = torch.cat(all_predictions).cpu().numpy()
            results["val_true_labels"] = torch.cat(all_labels).cpu().numpy()
        
        return results

    def _save_checkpoint(self, current_f1_score: float, epoch: int):

        if current_f1_score > self.best_f1_score:

            self.best_f1_score = current_f1_score
            print(
                f"New best F1 score: {self.best_f1_score:.4f}. Saving model checkpoint..."
            )
            checkpoint = {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_f1_score": self.best_f1_score,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "run_id": self.run_id,
            }

            torch.save(
                checkpoint, f"./models/checkpoints/run-{self.run_id}_best_model.pth"
            )

    def load_from_checkpoint(self, checkpoint_path: str):
        print(f" Loading ckpt from : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if not self.config.train.reset_scheduler_on_load:
            print(f"Scheduler state loaded from checkpoint.")
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print(f"Scheduler state NOT loaded. A new scheduler will be used.")

        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_f1_score = checkpoint["best_f1_score"]
        self.run_id = checkpoint["run_id"]
        print(f"Resuming training from epoch {self.start_epoch}")

    def train(self):
        print(f" Starting training...")
        for epoch in range(self.config.train.num_epochs):
            avg_train_loss = self._train_one_epoch()
            eval_results = self._evaluate(compute_all_metrics=True)
            cm_image = plot_confusion_matrix_heatmap(eval_results['val_true_labels'],
                                                     eval_results['val_pred_labels'],
                                                     self.class_names,
                                                     do_log = True)
            wandb.log(
                {"epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": eval_results['avg_val_loss'],
                    "validation_metrics": eval_results['validation_metrics'],
                    "confusion_matrix": cm_image},
                step=self.global_step
            )

            print( f"Epoch-{epoch}: Avg_train_loss={avg_train_loss}, Avg_val_loss={eval_results['avg_val_loss']}")
            self._save_checkpoint(eval_results["validation_metrics"]["macro_f1"], epoch + 1)

        wandb.finish()
        print("Training complete.")


def run(config: RunConfig):

    processed_cache_path = f"./data/processed/{config.data.version_id}"
    try:
        with open(os.path.join(processed_cache_path, "id2label.json"), "r") as f:
            id2label = json.load(f)
    except FileNotFoundError as e:
        return

    class_names = list(id2label.keys())
    num_classes = len(class_names)

    model_config = ModelConfig(num_classes=num_classes)
    config.model = model_config  # update the num_classes in model_config

    dataset = get_tokenized_dataset(config.data, config.model)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset, tokenizer, config.train
    )
    model = create_model(config.model)
    if config.train.weight_decay is not None:
        optimizer = AdamW(model.parameters(),weight_decay=config.train.weight_decay, lr=config.train.learning_rate)

    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)
    num_total_grad_steps = config.train.num_epochs * len(train_dataloader)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_total_grad_steps),
        num_training_steps=num_total_grad_steps,
    )

    trainer = Trainer(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        config,
        class_names=class_names,
    )

    trainer.train()


if __name__ == "__main__":
    config = RunConfig()
    run(config)
