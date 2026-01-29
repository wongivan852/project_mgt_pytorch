#!/usr/bin/env python3
"""Training script for PM Knowledge Area Classifier."""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import PMClassifier
from src.data.dataset import create_classification_dataloaders
from src.training.trainer import Trainer, create_optimizer_and_scheduler
from src.training.metrics import (
    compute_classification_metrics,
    print_classification_report
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            all_predictions.extend(outputs["predictions"].cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    # Compute metrics
    metrics = compute_classification_metrics(
        predictions,
        labels,
        label_names=model.LABEL_NAMES
    )

    # Print detailed report
    print_classification_report(predictions, labels, model.LABEL_NAMES)

    return metrics


def main(args):
    """Main training function."""
    # Load config
    config = load_config(args.config)
    cls_config = config["classifier"]
    data_config = config["data"]

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"]["use_cuda"]
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create model
    logger.info(f"Creating classifier: {cls_config['model_name']}")
    model = PMClassifier(
        model_name=cls_config["model_name"],
        num_labels=cls_config["num_labels"],
        device=device
    )

    total_params, trainable_params = model.get_num_parameters()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_classification_dataloaders(
        dataset_dir=data_config["dataset_dir"],
        tokenizer=model.tokenizer,
        batch_size=cls_config["batch_size"],
        max_length=data_config["max_length"],
        min_confidence=0.2,
        num_workers=args.num_workers
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Calculate training steps
    num_training_steps = len(train_loader) * cls_config["num_epochs"]

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        learning_rate=cls_config["learning_rate"],
        weight_decay=cls_config["weight_decay"],
        num_training_steps=num_training_steps,
        warmup_ratio=cls_config["warmup_ratio"]
    )

    # Create output directories
    output_dir = Path(config["output"]["model_dir"]) / "classifier"
    checkpoint_dir = Path(config["output"]["checkpoints_dir"]) / "classifier"
    log_dir = Path(config["output"]["logs_dir"]) / "classifier"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=1,
        max_grad_norm=cls_config["max_grad_norm"],
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir)
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        num_epochs=cls_config["num_epochs"],
        save_best=True
    )

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)

    logger.info("\nTest Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")

    # Save final model
    logger.info(f"\nSaving model to {output_dir}")
    model.save(str(output_dir))

    # Test predictions
    logger.info("\nTesting predictions...")
    test_texts = [
        "The project schedule needs to account for all task dependencies and milestones.",
        "We need to identify potential risks and create mitigation strategies.",
        "The budget variance indicates we are over cost projections.",
        "Team members should be assigned based on their skills and availability."
    ]

    for text in test_texts:
        result = model.predict(text, return_probs=True)
        logger.info(f"\nText: {text[:60]}...")
        logger.info(f"Predicted: {result['label']}")
        top_3 = sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for label, prob in top_3:
            logger.info(f"  {label}: {prob:.4f}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PM Knowledge Area Classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers"
    )

    args = parser.parse_args()
    main(args)
