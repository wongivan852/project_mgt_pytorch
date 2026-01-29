"""PyTorch Dataset classes for project management models."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2Tokenizer
from typing import List, Dict, Optional, Tuple
import random
from sklearn.model_selection import train_test_split

from .preprocessing import load_and_preprocess_data


class PMLanguageModelDataset(Dataset):
    """Dataset for GPT-2 language model fine-tuning."""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: GPT2Tokenizer,
        max_length: int = 512
    ):
        """Initialize the language model dataset.

        Args:
            samples: List of preprocessed text samples
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]["text"]

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # For language modeling, labels are same as input_ids
        # We set padding tokens to -100 so they're ignored in loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class PMClassificationDataset(Dataset):
    """Dataset for PM knowledge area classification."""

    LABEL_TO_ID = {
        "Integration Management": 0,
        "Scope Management": 1,
        "Schedule Management": 2,
        "Cost Management": 3,
        "Quality Management": 4,
        "Resource Management": 5,
        "Communications Management": 6,
        "Risk Management": 7,
        "Procurement Management": 8,
        "Stakeholder Management": 9
    }

    ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        min_confidence: float = 0.0
    ):
        """Initialize the classification dataset.

        Args:
            samples: List of preprocessed text samples with labels
            tokenizer: Tokenizer (e.g., DistilBERT)
            max_length: Maximum sequence length
            min_confidence: Minimum label confidence to include sample
        """
        # Filter by confidence if specified
        self.samples = [
            s for s in samples
            if s.get("label_confidence", 1.0) >= min_confidence
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        label = sample["label"]

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        label_id = self.LABEL_TO_ID.get(label, 0)

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

    @classmethod
    def get_num_labels(cls) -> int:
        """Get the number of classification labels."""
        return len(cls.LABEL_TO_ID)


def create_data_splits(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples into train/val/test sets.

    Args:
        samples: List of all samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (val + test)
    train_samples, temp_samples = train_test_split(
        samples,
        train_size=train_ratio,
        random_state=random_seed
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        train_size=val_ratio_adjusted,
        random_state=random_seed
    )

    return train_samples, val_samples, test_samples


def create_language_model_dataloaders(
    dataset_dir: str,
    tokenizer: GPT2Tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for language model training.

    Args:
        dataset_dir: Path to dataset directory
        tokenizer: GPT-2 tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load and preprocess data
    samples = load_and_preprocess_data(dataset_dir)

    # Split data
    train_samples, val_samples, test_samples = create_data_splits(samples)

    # Create datasets
    train_dataset = PMLanguageModelDataset(train_samples, tokenizer, max_length)
    val_dataset = PMLanguageModelDataset(val_samples, tokenizer, max_length)
    test_dataset = PMLanguageModelDataset(test_samples, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def create_classification_dataloaders(
    dataset_dir: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    min_confidence: float = 0.2,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for classification training.

    Args:
        dataset_dir: Path to dataset directory
        tokenizer: Tokenizer for classification model
        batch_size: Batch size
        max_length: Maximum sequence length
        min_confidence: Minimum label confidence threshold
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load and preprocess data
    samples = load_and_preprocess_data(dataset_dir)

    # Split data
    train_samples, val_samples, test_samples = create_data_splits(samples)

    # Create datasets
    train_dataset = PMClassificationDataset(
        train_samples, tokenizer, max_length, min_confidence
    )
    val_dataset = PMClassificationDataset(
        val_samples, tokenizer, max_length, min_confidence
    )
    test_dataset = PMClassificationDataset(
        test_samples, tokenizer, max_length, min_confidence
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test datasets
    from transformers import GPT2Tokenizer, AutoTokenizer

    print("Testing Language Model Dataset...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_loader, val_loader, test_loader = create_language_model_dataloaders(
        "dataset",
        gpt2_tokenizer,
        batch_size=2
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get a sample batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")

    print("\nTesting Classification Dataset...")
    bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader, val_loader, test_loader = create_classification_dataloaders(
        "dataset",
        bert_tokenizer,
        batch_size=4
    )
    print(f"Train batches: {len(train_loader)}")

    batch = next(iter(train_loader))
    print(f"Labels: {batch['labels']}")
