#!/usr/bin/env python3
"""Text generation script using trained PM Language Model."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.language_model import PMLanguageModel


def main(args):
    """Generate text using the trained model."""
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = PMLanguageModel.load(args.model_path)
    print("Model loaded successfully!")

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 50)
        print("PM Language Model - Interactive Generation")
        print("=" * 50)
        print("Enter a prompt and press Enter to generate text.")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not prompt:
                continue

            generated = model.generate(
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_sequences
            )

            print("\nGenerated:")
            for i, text in enumerate(generated):
                print(f"\n--- Sequence {i + 1} ---")
                print(text)
            print()

    # Single prompt mode
    else:
        if not args.prompt:
            print("Error: Please provide a prompt with --prompt or use --interactive mode")
            return

        print(f"\nPrompt: {args.prompt}")
        print("-" * 50)

        generated = model.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences
        )

        print("\nGenerated Text:")
        for i, text in enumerate(generated):
            print(f"\n--- Sequence {i + 1} ---")
            print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using PM Language Model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/language_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1,
        help="Number of sequences to generate"
    )

    args = parser.parse_args()
    main(args)
