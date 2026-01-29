"""Training utilities for PM models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict, Callable, List
import os
import logging
from tqdm import tqdm
import numpy as np

from .metrics import MetricTracker, compute_classification_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Generic trainer for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=5e-5,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Create directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Metrics
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

    def train_epoch(self) -> Dict:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            # Update metrics
            self.train_metrics.update({"loss": loss.item() * self.gradient_accumulation_steps})

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": self.train_metrics.get_value("loss"),
                "lr": self.optimizer.param_groups[0]["lr"]
            })

        return self.train_metrics.get_average()

    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.val_metrics.reset()

        all_predictions = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs["loss"]

            self.val_metrics.update({"loss": loss.item()})

            # Collect predictions for classification metrics
            if "predictions" in outputs:
                all_predictions.extend(outputs["predictions"].cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        metrics = self.val_metrics.get_average()

        # Compute classification metrics if applicable
        if all_predictions:
            cls_metrics = compute_classification_metrics(
                np.array(all_predictions),
                np.array(all_labels)
            )
            metrics.update({
                "accuracy": cls_metrics["accuracy"],
                "f1": cls_metrics["f1"]
            })

        return metrics

    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        eval_every: int = 1
    ) -> Dict:
        """Run full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save best model
            eval_every: Evaluate every N epochs

        Returns:
            Dictionary of training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            logger.info(f"Val batches: {len(self.val_loader)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}"
            )

            # Validate
            if self.val_loader and (epoch + 1) % eval_every == 0:
                val_metrics = self.validate()
                history["val_loss"].append(val_metrics.get("loss", 0))
                history["val_accuracy"].append(val_metrics.get("accuracy", 0))

                logger.info(
                    f"  Val Loss: {val_metrics.get('loss', 0):.4f}, "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
                )

                # Save best model
                if save_best and val_metrics.get("loss", float("inf")) < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best_model.pt")
                    logger.info("  Saved best model!")

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # Save final model
        self.save_checkpoint("final_model.pt")

        return history

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Checkpoint loaded from {path}")


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    num_training_steps: int,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0
) -> tuple:
    """Create optimizer and scheduler.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        num_training_steps: Total training steps
        warmup_steps: Number of warmup steps
        warmup_ratio: Ratio of warmup steps (used if warmup_steps is 0)

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Calculate warmup steps
    if warmup_steps == 0 and warmup_ratio > 0:
        warmup_steps = int(num_training_steps * warmup_ratio)

    # Create optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler
