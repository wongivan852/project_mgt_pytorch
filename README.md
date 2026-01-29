# Project Management PyTorch Models

AI models for project management using PyTorch, trained on PMI (Project Management Institute) publications and standards.

## Overview

This project implements two AI models for project management:

1. **PM Language Model**: A GPT-2 based language model fine-tuned on project management literature for text generation
2. **PM Classifier**: A DistilBERT-based classifier for categorizing text into PM knowledge areas

## Features

- **Text Generation**: Generate project management advice, documentation, and content
- **Knowledge Area Classification**: Classify text into 10 PM knowledge areas:
  - Integration Management
  - Scope Management
  - Schedule Management
  - Cost Management
  - Quality Management
  - Resource Management
  - Communications Management
  - Risk Management
  - Procurement Management
  - Stakeholder Management

## Project Structure

```
project_mgt_pytorch/
├── config/
│   └── config.yaml              # Training configuration
├── dataset/                     # PMI publications (text files)
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   └── preprocessing.py     # Text preprocessing
│   ├── models/
│   │   ├── language_model.py    # GPT-2 language model
│   │   └── classifier.py        # Text classifier
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/
│       └── helpers.py           # Utility functions
├── scripts/
│   ├── train_language_model.py  # Train language model
│   ├── train_classifier.py      # Train classifier
│   ├── evaluate.py              # Evaluate models
│   └── generate.py              # Generate text
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wongivan852/project_mgt_pytorch.git
cd project_mgt_pytorch
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Language Model

```bash
python scripts/train_language_model.py --config config/config.yaml
```

### Training the Classifier

```bash
python scripts/train_classifier.py --config config/config.yaml
```

### Generating Text

```bash
# Single prompt
python scripts/generate.py --model-path models/language_model --prompt "Project risk management"

# Interactive mode
python scripts/generate.py --model-path models/language_model --interactive
```

### Evaluating Models

```bash
# Evaluate both models
python scripts/evaluate.py --config config/config.yaml

# Evaluate specific model
python scripts/evaluate.py --model-type classifier --model-path models/classifier
```

## Configuration

Edit `config/config.yaml` to customize:

- Model parameters (learning rate, batch size, epochs)
- Data processing settings (max length, train/val/test splits)
- Output directories

## Dataset

The dataset consists of PMI publications including:

- PMBOK Guide (6th, 7th, 8th Editions)
- Agile Practice Guide
- Standard for Program/Portfolio Management
- Various Practice Guides (Risk, Scheduling, Estimating, etc.)
- AI guides for Project Professionals

Total: ~35 text files, ~780,000 lines, ~32MB of text

## Model Details

### Language Model (GPT-2)

- Base: GPT-2 (124M parameters)
- Fine-tuned on PM corpus
- Generates contextually relevant PM text

### Classifier (DistilBERT)

- Base: DistilBERT (66M parameters)
- 10-class classification
- Keyword-enhanced training labels

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU training)

## License

This project is for educational purposes. The dataset consists of PMI copyrighted materials used for research.

## Acknowledgments

- Project Management Institute (PMI) for the source materials
- Hugging Face for the Transformers library
