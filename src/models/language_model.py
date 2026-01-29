"""GPT-2 based language model for project management text generation."""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMLanguageModel(nn.Module):
    """Project Management Language Model based on GPT-2."""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[torch.device] = None
    ):
        """Initialize the language model.

        Args:
            model_name: Name of the pretrained model (gpt2, gpt2-medium, etc.)
            device: Device to use for computation
        """
        super().__init__()

        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load pretrained model and tokenizer
        logger.info(f"Loading {model_name} model...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")

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
            labels: Labels for language modeling [batch_size, seq_len]

        Returns:
            Dictionary containing loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits
        }

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        num_return_sequences: int = 1,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.2
    ) -> List[str]:
        """Generate text based on a prompt.

        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repeating tokens

        Returns:
            List of generated text sequences
        """
        self.model.eval()

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def save(self, path: str):
        """Save model and tokenizer.

        Args:
            path: Directory path to save to
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "PMLanguageModel":
        """Load a saved model.

        Args:
            path: Directory path to load from
            device: Device to load model to

        Returns:
            Loaded PMLanguageModel instance
        """
        instance = cls.__new__(cls)
        super(PMLanguageModel, instance).__init__()

        instance.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        instance.model = GPT2LMHeadModel.from_pretrained(path)
        instance.tokenizer = GPT2Tokenizer.from_pretrained(path)
        instance.model_name = "custom"

        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token

        instance.model.to(instance.device)
        logger.info(f"Model loaded from {path}")

        return instance

    def get_num_parameters(self) -> Tuple[int, int]:
        """Get the number of model parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable


def create_pm_language_model(
    model_name: str = "gpt2",
    device: Optional[torch.device] = None
) -> PMLanguageModel:
    """Factory function to create a PM language model.

    Args:
        model_name: Name of pretrained model
        device: Device to use

    Returns:
        PMLanguageModel instance
    """
    return PMLanguageModel(model_name=model_name, device=device)


if __name__ == "__main__":
    # Test the model
    print("Creating PM Language Model...")
    model = create_pm_language_model("gpt2")

    total, trainable = model.get_num_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Test generation
    prompt = "Project management is"
    print(f"\nPrompt: {prompt}")
    generated = model.generate(prompt, max_length=100)
    print(f"Generated: {generated[0]}")
