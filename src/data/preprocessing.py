"""Text preprocessing utilities for project management corpus."""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocessor for project management text data."""

    # Keywords for each PM knowledge area (for classification labeling)
    KNOWLEDGE_AREA_KEYWORDS = {
        "Integration Management": [
            "integration", "project charter", "project management plan",
            "direct and manage", "monitor and control", "integrated change control",
            "close project", "phase gate"
        ],
        "Scope Management": [
            "scope", "requirements", "wbs", "work breakdown structure",
            "scope baseline", "scope creep", "decomposition", "deliverables",
            "scope statement", "scope verification"
        ],
        "Schedule Management": [
            "schedule", "timeline", "milestone", "gantt", "critical path",
            "duration", "sequencing", "dependencies", "float", "slack",
            "schedule baseline", "schedule compression"
        ],
        "Cost Management": [
            "cost", "budget", "estimate", "earned value", "evm",
            "cost baseline", "funding", "expenditure", "variance",
            "cost performance index", "cpi"
        ],
        "Quality Management": [
            "quality", "quality assurance", "quality control", "inspection",
            "audit", "metrics", "standards", "defect", "rework",
            "continuous improvement", "six sigma"
        ],
        "Resource Management": [
            "resource", "team", "staff", "human resource", "skills",
            "responsibility", "raci", "resource leveling", "resource calendar",
            "team development", "conflict resolution"
        ],
        "Communications Management": [
            "communication", "stakeholder communication", "reporting",
            "information distribution", "status report", "meeting",
            "communication plan", "escalation"
        ],
        "Risk Management": [
            "risk", "threat", "opportunity", "risk register", "mitigation",
            "contingency", "risk response", "probability", "impact",
            "risk assessment", "monte carlo"
        ],
        "Procurement Management": [
            "procurement", "contract", "vendor", "supplier", "bid",
            "rfp", "rfq", "make-or-buy", "source selection",
            "contract administration"
        ],
        "Stakeholder Management": [
            "stakeholder", "stakeholder engagement", "stakeholder analysis",
            "power interest", "influence", "expectations",
            "stakeholder register"
        ]
    }

    def __init__(self, min_text_length: int = 50):
        """Initialize preprocessor.

        Args:
            min_text_length: Minimum length of text to keep after cleaning
        """
        self.min_text_length = min_text_length

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove PMI license notices
        text = re.sub(
            r"(Licensed To:.*?reproduction\.|PMI Member benefit.*?reproduction\.)",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove page numbers and headers
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Page \d+.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,;:!?'\"-]", " ", text)

        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to split
            chunk_size: Target size of each chunk (in words)
            overlap: Number of overlapping words between chunks

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []

        if len(words) <= chunk_size:
            if len(words) >= self.min_text_length:
                return [text]
            return []

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])

            if len(chunk.split()) >= self.min_text_length:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def classify_text(self, text: str) -> Tuple[str, float]:
        """Classify text into a PM knowledge area based on keywords.

        Args:
            text: Text to classify

        Returns:
            Tuple of (knowledge_area, confidence_score)
        """
        text_lower = text.lower()
        scores = {}

        for area, keywords in self.KNOWLEDGE_AREA_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            scores[area] = score

        if max(scores.values()) == 0:
            return "Integration Management", 0.0  # Default

        best_area = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_area] / total_score if total_score > 0 else 0.0

        return best_area, confidence

    def process_file(self, file_path: str) -> List[Dict]:
        """Process a single text file.

        Args:
            file_path: Path to text file

        Returns:
            List of processed text samples with metadata
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        cleaned_text = self.clean_text(raw_text)
        chunks = self.split_into_chunks(cleaned_text)

        samples = []
        file_name = Path(file_path).stem

        for i, chunk in enumerate(chunks):
            label, confidence = self.classify_text(chunk)
            samples.append({
                "text": chunk,
                "source_file": file_name,
                "chunk_id": i,
                "label": label,
                "label_confidence": confidence
            })

        return samples

    def process_directory(
        self,
        directory: str,
        exclude_files: Optional[List[str]] = None
    ) -> List[Dict]:
        """Process all text files in a directory.

        Args:
            directory: Path to directory containing text files
            exclude_files: List of filenames to exclude

        Returns:
            List of all processed samples
        """
        exclude_files = exclude_files or []
        all_samples = []

        txt_files = list(Path(directory).glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files in {directory}")

        for file_path in txt_files:
            if file_path.name in exclude_files:
                logger.info(f"Skipping excluded file: {file_path.name}")
                continue

            logger.info(f"Processing: {file_path.name}")
            try:
                samples = self.process_file(str(file_path))
                all_samples.extend(samples)
                logger.info(f"  -> Generated {len(samples)} samples")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")

        logger.info(f"Total samples generated: {len(all_samples)}")
        return all_samples


def load_and_preprocess_data(
    dataset_dir: str,
    min_text_length: int = 50,
    chunk_size: int = 512
) -> List[Dict]:
    """Main function to load and preprocess all data.

    Args:
        dataset_dir: Path to dataset directory
        min_text_length: Minimum text length to keep
        chunk_size: Size of text chunks

    Returns:
        List of preprocessed samples
    """
    preprocessor = TextPreprocessor(min_text_length=min_text_length)

    # Exclude combined files to avoid duplicates
    exclude = ["pmp_combined.txt", "obsolete_pmp_combined.txt"]

    samples = preprocessor.process_directory(dataset_dir, exclude_files=exclude)

    return samples


if __name__ == "__main__":
    # Test preprocessing
    import json

    samples = load_and_preprocess_data("dataset")

    # Print statistics
    print(f"\nTotal samples: {len(samples)}")

    # Label distribution
    from collections import Counter
    labels = Counter(s["label"] for s in samples)
    print("\nLabel distribution:")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")

    # Save sample for inspection
    with open("sample_processed.json", "w") as f:
        json.dump(samples[:5], f, indent=2)
    print("\nSaved 5 samples to sample_processed.json")
