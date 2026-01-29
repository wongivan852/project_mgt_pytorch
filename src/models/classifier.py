"""Text classification model for PM knowledge areas."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMClassifier(nn.Module):
    """Project Management Knowledge Area Classifier based on DistilBERT."""

    LABEL_NAMES = [
        "Integration Management",
        "Scope Management",
        "Schedule Management",
        "Cost Management",
        "Quality Management",
        "Resource Management",
        "Communications Management",
        "Risk Management",
        "Procurement Management",
        "Stakeholder Management"
    ]

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 10,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """Initialize the classifier.

        Args:
            model_name: Name of the pretrained encoder model
            num_labels: Number of classification labels
            dropout: Dropout probability
            device: Device to use for computation
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load pretrained encoder
        logger.info(f"Loading {model_name} encoder...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self.to(self.device)
        logger.info(f"Classifier loaded on {self.device}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Classification labels [batch_size]

        Returns:
            Dictionary containing loss, logits, and predictions
        """
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)

        return {
            "loss": loss,
            "logits": logits,
            "predictions": predictions
        }

    def predict(
        self,
        text: str,
        return_probs: bool = False
    ) -> Dict:
        """Predict the knowledge area for a text.

        Args:
            text: Input text to classify
            return_probs: Whether to return class probabilities

        Returns:
            Dictionary with prediction results
        """
        self.eval()

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        logits = outputs["logits"]
        prediction_id = outputs["predictions"].item()
        prediction_label = self.LABEL_NAMES[prediction_id]

        result = {
            "label": prediction_label,
            "label_id": prediction_id
        }

        if return_probs:
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            result["probabilities"] = {
                self.LABEL_NAMES[i]: float(probs[i])
                for i in range(len(self.LABEL_NAMES))
            }

        return result

    def predict_batch(
        self,
        texts: List[str]
    ) -> List[Dict]:
        """Predict knowledge areas for multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of prediction dictionaries
        """
        self.eval()

        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        predictions = outputs["predictions"].cpu().numpy()

        results = []
        for pred_id in predictions:
            results.append({
                "label": self.LABEL_NAMES[pred_id],
                "label_id": int(pred_id)
            })

        return results

    def save(self, path: str):
        """Save model.

        Args:
            path: Directory path to save to
        """
        import os
        os.makedirs(path, exist_ok=True)

        # Save encoder and tokenizer
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save classification head
        torch.save({
            "classifier_state_dict": self.classifier.state_dict(),
            "dropout": self.dropout.p,
            "num_labels": self.num_labels,
            "model_name": self.model_name
        }, os.path.join(path, "classifier_head.pt"))

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "PMClassifier":
        """Load a saved model.

        Args:
            path: Directory path to load from
            device: Device to load model to

        Returns:
            Loaded PMClassifier instance
        """
        import os

        # Load classifier head config
        head_path = os.path.join(path, "classifier_head.pt")
        head_config = torch.load(head_path, map_location="cpu")

        # Create instance
        instance = cls(
            model_name=path,  # Load from saved path
            num_labels=head_config["num_labels"],
            dropout=head_config["dropout"],
            device=device
        )

        # Load classifier head weights
        instance.classifier.load_state_dict(head_config["classifier_state_dict"])

        logger.info(f"Model loaded from {path}")
        return instance

    def get_num_parameters(self) -> Tuple[int, int]:
        """Get the number of model parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_pm_classifier(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 10,
    device: Optional[torch.device] = None
) -> PMClassifier:
    """Factory function to create a PM classifier.

    Args:
        model_name: Name of pretrained encoder model
        num_labels: Number of classification labels
        device: Device to use

    Returns:
        PMClassifier instance
    """
    return PMClassifier(
        model_name=model_name,
        num_labels=num_labels,
        device=device
    )


if __name__ == "__main__":
    # Test the model
    print("Creating PM Classifier...")
    model = create_pm_classifier()

    total, trainable = model.get_num_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Test prediction
    test_text = "The project schedule should account for all dependencies and milestones."
    print(f"\nTest text: {test_text}")

    result = model.predict(test_text, return_probs=True)
    print(f"Predicted label: {result['label']}")
    print("\nTop 3 probabilities:")
    sorted_probs = sorted(
        result["probabilities"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for label, prob in sorted_probs:
        print(f"  {label}: {prob:.4f}")
