#!/usr/bin/env python3
"""Training script for PM Language Model (GPT-2 fine-tuning)."""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.language_model import PMLanguageModel
from src.data.dataset import create_language_model_dataloaders
from src.training.trainer import Trainer, create_optimizer_and_scheduler

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


def main(args):
    """Main training function."""
    # Load config
    config = load_config(args.config)
    lm_config = config["language_model"]
    data_config = config["data"]

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"]["use_cuda"]
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create model
    logger.info(f"Creating model: {lm_config['model_name']}")
    model = PMLanguageModel(
        model_name=lm_config["model_name"],
        device=device
    )

    total_params, trainable_params = model.get_num_parameters()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_language_model_dataloaders(
        dataset_dir=data_config["dataset_dir"],
        tokenizer=model.tokenizer,
        batch_size=lm_config["batch_size"],
        max_length=data_config["max_length"],
        num_workers=args.num_workers
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Calculate training steps
    num_training_steps = (
        len(train_loader) // lm_config["gradient_accumulation_steps"]
    ) * lm_config["num_epochs"]

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model.model,
        learning_rate=lm_config["learning_rate"],
        weight_decay=lm_config["weight_decay"],
        num_training_steps=num_training_steps,
        warmup_steps=lm_config["warmup_steps"]
    )

    # Create output directories
    output_dir = Path(config["output"]["model_dir"]) / "language_model"
    checkpoint_dir = Path(config["output"]["checkpoints_dir"]) / "language_model"
    log_dir = Path(config["output"]["logs_dir"]) / "language_model"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = Trainer(
        model=model.model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=lm_config["gradient_accumulation_steps"],
        max_grad_norm=lm_config["max_grad_norm"],
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir)
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        num_epochs=lm_config["num_epochs"],
        save_best=True
    )

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    model.save(str(output_dir))

    # Test generation
    logger.info("\nTesting generation...")
    test_prompts = [
        "Project management is",
        "Risk mitigation strategies include",
        "The critical path method"
    ]

    for prompt in test_prompts:
        generated = model.generate(prompt, max_length=100)
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated[0][:200]}...")

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PM Language Model"
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
