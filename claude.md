# Claude Code Documentation

This document provides guidance for AI assistants working with the Project Management PyTorch codebase.

## Project Overview

This project implements AI models for project management using PyTorch, trained on PMI (Project Management Institute) publications and standards.

### Two Main Models

1. **PM Language Model** (`src/models/language_model.py`)
   - GPT-2 based text generation
   - Fine-tuned on PM corpus
   - Use for generating PM advice, documentation, content

2. **PM Classifier** (`src/models/classifier.py`)
   - DistilBERT based classification
   - 10 PM knowledge areas
   - Use for categorizing PM content

## Project Structure

```
project_mgt_pytorch/
├── config/
│   └── config.yaml              # All hyperparameters and settings
├── dataset/                     # PMI text files (DO NOT MODIFY)
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Text cleaning, chunking, labeling
│   │   └── dataset.py           # PyTorch Dataset/DataLoader classes
│   ├── models/
│   │   ├── language_model.py    # PMLanguageModel class (GPT-2)
│   │   └── classifier.py        # PMClassifier class (DistilBERT)
│   ├── training/
│   │   ├── trainer.py           # Generic Trainer class
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/
│       └── helpers.py           # Utility functions
├── scripts/
│   ├── train_language_model.py  # Entry point for LM training
│   ├── train_classifier.py      # Entry point for classifier training
│   ├── evaluate.py              # Model evaluation
│   └── generate.py              # Text generation interface
├── models/                      # Saved models (created after training)
├── checkpoints/                 # Training checkpoints
├── logs/                        # Training logs
└── tests/                       # Unit tests
```

## Key Classes and Functions

### Data Pipeline

```python
# Preprocessing
from src.data.preprocessing import TextPreprocessor, load_and_preprocess_data

# Load all data
samples = load_and_preprocess_data("dataset")
# Returns: List[Dict] with keys: text, source_file, chunk_id, label, label_confidence

# Datasets
from src.data.dataset import (
    PMLanguageModelDataset,      # For GPT-2 training
    PMClassificationDataset,      # For classifier training
    create_language_model_dataloaders,
    create_classification_dataloaders
)
```

### Models

```python
# Language Model
from src.models.language_model import PMLanguageModel, create_pm_language_model

model = create_pm_language_model("gpt2")
generated = model.generate("Project risk management", max_length=100)
model.save("models/language_model")
model = PMLanguageModel.load("models/language_model")

# Classifier
from src.models.classifier import PMClassifier, create_pm_classifier

model = create_pm_classifier("distilbert-base-uncased", num_labels=10)
result = model.predict("The schedule has critical path issues", return_probs=True)
model.save("models/classifier")
model = PMClassifier.load("models/classifier")
```

### Training

```python
from src.training.trainer import Trainer, create_optimizer_and_scheduler

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device
)
history = trainer.train(num_epochs=5)
```

## Common Tasks

### Adding a New Knowledge Area

1. Edit `src/data/preprocessing.py`:
   - Add to `KNOWLEDGE_AREA_KEYWORDS` dict
   - Add relevant keywords

2. Edit `src/models/classifier.py`:
   - Add to `LABEL_NAMES` list
   - Update `LABEL_TO_ID` dict

3. Update `config/config.yaml`:
   - Add to `knowledge_areas` list
   - Update `num_labels`

### Modifying Model Architecture

**Language Model**: Edit `src/models/language_model.py`
- Change `model_name` parameter for different GPT-2 sizes (gpt2, gpt2-medium, gpt2-large)

**Classifier**: Edit `src/models/classifier.py`
- Modify `__init__` to add layers
- Update `forward` method for new architecture

### Adjusting Training Parameters

Edit `config/config.yaml`:

```yaml
language_model:
  learning_rate: 5.0e-5    # Lower for stability
  batch_size: 4            # Increase if GPU memory allows
  num_epochs: 3            # More epochs for better convergence

classifier:
  learning_rate: 2.0e-5
  batch_size: 16
  num_epochs: 5
```

### Adding New Data

1. Add `.txt` files to `dataset/` directory
2. If PDF, extract text first:
   ```python
   import fitz
   doc = fitz.open("file.pdf")
   text = "\n".join(page.get_text() for page in doc)
   ```
3. Re-run preprocessing - it auto-discovers new files

## PM Knowledge Areas

The classifier uses these 10 PMI knowledge areas:

| ID | Knowledge Area | Key Concepts |
|----|---------------|--------------|
| 0 | Integration Management | Project charter, change control, phase gates |
| 1 | Scope Management | WBS, requirements, deliverables |
| 2 | Schedule Management | Critical path, milestones, Gantt |
| 3 | Cost Management | Budget, EVM, cost baseline |
| 4 | Quality Management | QA, QC, inspections, audits |
| 5 | Resource Management | Team, RACI, resource leveling |
| 6 | Communications Management | Reporting, stakeholder comms |
| 7 | Risk Management | Risk register, mitigation, contingency |
| 8 | Procurement Management | Contracts, vendors, RFP/RFQ |
| 9 | Stakeholder Management | Engagement, analysis, expectations |

## Dataset Information

- **Source**: PMI publications (2007-2025)
- **Files**: 35 text files
- **Size**: ~32 MB, ~780,000 lines
- **Key Documents**:
  - PMBOK Guide (7th & 8th Editions)
  - Agile Practice Guide
  - Process Groups Practice Guide
  - Various standards and practice guides

## Running Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train language model
python scripts/train_language_model.py --config config/config.yaml

# Train classifier
python scripts/train_classifier.py --config config/config.yaml

# Interactive text generation
python scripts/generate.py --model-path models/language_model --interactive

# Evaluate models
python scripts/evaluate.py --model-type both

# Test preprocessing
python -m src.data.preprocessing
```

## Testing

```bash
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_dataset.py -v
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps`
- Use smaller model (gpt2 instead of gpt2-medium)

### Slow Training

- Enable CUDA: ensure `use_cuda: true` in config
- Increase `num_workers` for data loading
- Use mixed precision: `mixed_precision: true`

### Poor Generation Quality

- Train for more epochs
- Adjust temperature (lower = more focused)
- Increase `repetition_penalty`

### Low Classification Accuracy

- Increase `min_confidence` threshold for training data
- Balance the dataset across classes
- Train for more epochs with lower learning rate

## Code Style

- Type hints on all function signatures
- Docstrings in Google format
- Logging via Python `logging` module
- Constants in UPPER_SNAKE_CASE
- Classes in PascalCase
- Functions/variables in snake_case

## Dependencies

Core:
- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.30.0` - Hugging Face models
- `datasets>=2.14.0` - Data utilities

Data:
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Metrics, splitting

Utils:
- `pyyaml>=6.0` - Config loading
- `tqdm>=4.65.0` - Progress bars
- `pymupdf>=1.23.0` - PDF extraction
