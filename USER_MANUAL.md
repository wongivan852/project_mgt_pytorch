# Project Management AI Models - User Manual

## Quick Start

### 1. Setup Environment

```bash
cd ~/Desktop/pytorch-project-management-model-93a8M
source venv/bin/activate
```

### 2. Launch UAT Interface

```bash
streamlit run scripts/uat_app.py
```

Open your browser to: **http://localhost:8501**

---

## UAT Web Interface Guide

### Home Page

The home page displays:
- Overview of both AI models
- List of 10 PM knowledge areas
- Quick navigation sidebar

### Language Model Testing

**Purpose:** Generate project management content from prompts.

**How to use:**
1. Navigate to "Language Model" in the sidebar
2. Enter a prompt (e.g., "Project risk management is")
3. Adjust parameters:
   - **Max Length:** How long the generated text should be (50-500)
   - **Temperature:** Creativity level (0.1=focused, 2.0=creative)
   - **Top-K:** Limits word choices (1-100)
   - **Top-P:** Nucleus sampling threshold (0.1-1.0)
4. Click "Generate Text"
5. View generated output

**Sample Prompts:**
- "The critical path method is used to"
- "Stakeholder engagement strategies include"
- "Earned value management helps project managers"

### Classifier Testing

**Purpose:** Classify text into PM knowledge areas.

**How to use:**
1. Navigate to "Classifier" in the sidebar
2. Enter project management text
3. Click "Classify Text"
4. View results:
   - Predicted knowledge area
   - Confidence score
   - Probability chart for all 10 areas

**Knowledge Areas:**
| ID | Area |
|----|------|
| 0 | Integration Management |
| 1 | Scope Management |
| 2 | Schedule Management |
| 3 | Cost Management |
| 4 | Quality Management |
| 5 | Resource Management |
| 6 | Communications Management |
| 7 | Risk Management |
| 8 | Procurement Management |
| 9 | Stakeholder Management |

### Batch Testing

**Purpose:** Test multiple inputs at once using CSV files.

**For Classifier:**
1. Create a CSV file with columns: `text` (required), `expected` (optional)
2. Upload the file
3. Click "Run Batch Test"
4. Download results

**CSV Example (Classifier):**
```csv
text,expected
"The schedule shows delays",Schedule Management
"Risk mitigation needed",Risk Management
```

**For Language Model:**
1. Create a CSV file with column: `prompt`
2. Upload the file
3. Click "Run Batch Test"
4. Download generated results

**CSV Example (Language Model):**
```csv
prompt
"Project management is"
"Risk assessment involves"
```

### Model Info

View detailed information about:
- Model parameters
- Training configuration
- Dataset statistics
- UAT test history (current session)

---

## Command Line Interface

### Text Generation (CLI)

```bash
# Interactive mode
python scripts/generate.py --model-path models/language_model --interactive

# Single prompt
python scripts/generate.py --model-path models/language_model \
    --prompt "Project risk management" \
    --max-length 200 \
    --temperature 0.8
```

### Model Evaluation

```bash
# Evaluate both models
python scripts/evaluate.py --model-type both

# Evaluate classifier only
python scripts/evaluate.py --model-type classifier

# Evaluate language model only
python scripts/evaluate.py --model-type language_model
```

---

## Troubleshooting

### "Error loading language model"

**Solution:** Make sure you activate the virtual environment first:
```bash
source venv/bin/activate
streamlit run scripts/uat_app.py
```

### Models not loading

**Check if models exist:**
```bash
ls -la models/language_model/
ls -la models/classifier/
```

If missing, train the models:
```bash
python scripts/train_language_model.py --config config/config.yaml
python scripts/train_classifier.py --config config/config.yaml
```

### Slow performance

- The models run on CPU by default
- First load may take 30-60 seconds
- Subsequent requests are faster due to caching

### Port already in use

```bash
# Use a different port
streamlit run scripts/uat_app.py --server.port 8502
```

---

## Model Performance Summary

### Language Model (GPT-2)
- **Parameters:** 124 million
- **Training Data:** PMI publications (2007-2025)
- **Best For:** Generating PM documentation, advice, content

### Classifier (DistilBERT)
- **Parameters:** 66 million
- **Accuracy:** 61.41%
- **Best Performing Areas:**
  - Schedule Management (F1: 0.77)
  - Risk Management (F1: 0.71)
  - Cost Management (F1: 0.67)

---

## Tips for Best Results

### Language Model
1. Start with clear, specific prompts
2. Use lower temperature (0.5-0.8) for factual content
3. Use higher temperature (1.0-1.5) for creative content
4. Include PM terminology in your prompts

### Classifier
1. Provide sufficient context (2-3 sentences minimum)
2. Use clear PM terminology
3. Check confidence scores - low confidence may indicate ambiguous text
4. Review top 3 predictions for borderline cases

---

## Support

For technical issues, refer to:
- `claude.md` - Developer documentation
- `README.md` - Project overview
- `config/config.yaml` - Configuration options
