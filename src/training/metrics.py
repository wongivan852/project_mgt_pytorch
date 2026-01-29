"""Evaluation metrics for PM models."""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict:
    """Compute classification metrics.

    Args:
        predictions: Model predictions
        labels: True labels
        label_names: Names of labels for reporting

    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(labels, predictions)

    # Compute precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(
            labels,
            predictions,
            average=None,
            zero_division=0
        )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": {}
    }

    # Add per-class metrics
    num_classes = len(precision_per_class)
    for i in range(num_classes):
        class_name = label_names[i] if label_names else str(i)
        metrics["per_class"][class_name] = {
            "precision": precision_per_class[i],
            "recall": recall_per_class[i],
            "f1": f1_per_class[i],
            "support": int(support_per_class[i])
        }

    return metrics


def compute_language_model_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> Dict:
    """Compute language model metrics.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: True labels [batch_size, seq_len]
        ignore_index: Index to ignore in loss calculation

    Returns:
        Dictionary of metrics
    """
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)

    # Create mask for valid positions
    mask = labels != ignore_index

    # Compute accuracy only on non-masked positions
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    # Compute perplexity from loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    perplexity = torch.exp(loss)

    return {
        "accuracy": accuracy.item(),
        "perplexity": perplexity.item(),
        "loss": loss.item()
    }


def print_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str]
):
    """Print detailed classification report.

    Args:
        predictions: Model predictions
        labels: True labels
        label_names: Names of labels
    """
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)


def get_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None
) -> np.ndarray:
    """Get confusion matrix.

    Args:
        predictions: Model predictions
        labels: True labels
        label_names: Names of labels

    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(labels, predictions)
    return cm


class MetricTracker:
    """Track metrics during training."""

    def __init__(self):
        """Initialize metric tracker."""
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict, count: int = 1):
        """Update tracked metrics.

        Args:
            metrics: Dictionary of metric values
            count: Number of samples in this update
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value * count
            self.counts[key] += count

    def get_average(self) -> Dict:
        """Get average of all tracked metrics.

        Returns:
            Dictionary of averaged metrics
        """
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
            if self.counts[key] > 0
        }

    def get_value(self, key: str) -> float:
        """Get average value for a specific metric.

        Args:
            key: Metric name

        Returns:
            Average metric value
        """
        if key not in self.metrics or self.counts[key] == 0:
            return 0.0
        return self.metrics[key] / self.counts[key]
