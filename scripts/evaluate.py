#!/usr/bin/env python3
"""Evaluation script for PM models."""

import argparse
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import PMClassifier
from src.models.language_model import PMLanguageModel
from src.data.dataset import (
    create_classification_dataloaders,
    create_language_model_dataloaders
)
from src.training.metrics import (
    compute_classification_metrics,
    print_classification_report,
    get_confusion_matrix
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


def evaluate_classifier(model_path: str, config: dict, device: torch.device):
    """Evaluate the classifier model."""
    logger.info(f"Loading classifier from {model_path}")
    model = PMClassifier.load(model_path, device=device)

    # Create test loader
    _, _, test_loader = create_classification_dataloaders(
        dataset_dir=config["data"]["dataset_dir"],
        tokenizer=model.tokenizer,
        batch_size=config["classifier"]["batch_size"],
        max_length=config["data"]["max_length"],
        min_confidence=0.2
    )

    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate
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
        predictions, labels, model.LABEL_NAMES
    )

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("CLASSIFIER EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")

    print_classification_report(predictions, labels, model.LABEL_NAMES)

    # Per-class results
    logger.info("\nPer-class Performance:")
    for label, class_metrics in metrics["per_class"].items():
        logger.info(
            f"  {label}: F1={class_metrics['f1']:.3f}, "
            f"Support={class_metrics['support']}"
        )

    return metrics


def evaluate_language_model(model_path: str, config: dict, device: torch.device):
    """Evaluate the language model."""
    logger.info(f"Loading language model from {model_path}")
    model = PMLanguageModel.load(model_path, device=device)

    # Create test loader
    _, _, test_loader = create_language_model_dataloaders(
        dataset_dir=config["data"]["dataset_dir"],
        tokenizer=model.tokenizer,
        batch_size=config["language_model"]["batch_size"],
        max_length=config["data"]["max_length"]
    )

    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate perplexity
    model.model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # Count non-padding tokens
            mask = batch["labels"] != -100
            num_tokens = mask.sum().item()

            total_loss += outputs["loss"].item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("LANGUAGE MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Perplexity:   {perplexity:.4f}")

    # Test generation
    logger.info("\nGeneration Examples:")
    test_prompts = [
        "Project management involves",
        "To mitigate risks in a project",
        "The work breakdown structure",
        "Stakeholder engagement is important because"
    ]

    for prompt in test_prompts:
        generated = model.generate(
            prompt,
            max_length=100,
            temperature=0.8
        )
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated[0][:150]}...")

    return {"loss": avg_loss, "perplexity": perplexity}


def main(args):
    """Main evaluation function."""
    config = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"]["use_cuda"]
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    if args.model_type == "classifier":
        evaluate_classifier(args.model_path, config, device)
    elif args.model_type == "language_model":
        evaluate_language_model(args.model_path, config, device)
    else:
        # Evaluate both if both exist
        classifier_path = Path(config["output"]["model_dir"]) / "classifier"
        lm_path = Path(config["output"]["model_dir"]) / "language_model"

        if classifier_path.exists():
            evaluate_classifier(str(classifier_path), config, device)

        if lm_path.exists():
            evaluate_language_model(str(lm_path), config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PM models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["classifier", "language_model", "both"],
        default="both",
        help="Type of model to evaluate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model (overrides config)"
    )

    args = parser.parse_args()

    # Set default model paths if not specified
    if args.model_path is None:
        if args.model_type == "classifier":
            args.model_path = "models/classifier"
        elif args.model_type == "language_model":
            args.model_path = "models/language_model"

    main(args)
