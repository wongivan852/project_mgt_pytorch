#!/usr/bin/env python3
"""
UAT (User Acceptance Testing) Web Application for PM Models.

A Streamlit-based interface for testing the Project Management
Language Model and Classifier.

Run with: streamlit run scripts/uat_app.py
"""

import sys
import os
from pathlib import Path

# Get project root - handle both direct run and streamlit run
def get_PROJECT_ROOT():
    """Get the project root directory."""
    # Try from __file__ first
    if '__file__' in dir():
        script_path = Path(__file__).resolve()
        if script_path.parent.name == 'scripts':
            return script_path.parent.parent

    # Try from current working directory
    cwd = Path.cwd()
    if (cwd / 'config' / 'config.yaml').exists():
        return cwd

    # Try parent of cwd
    if (cwd.parent / 'config' / 'config.yaml').exists():
        return cwd.parent

    # Fallback to explicit path
    explicit_path = Path('/Users/wongivan/Desktop/pytorch-project-management-model-93a8M')
    if explicit_path.exists():
        return explicit_path

    return cwd

PROJECT_ROOT = get_PROJECT_ROOT()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import yaml
import pandas as pd
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PM Model UAT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_config():
    """Load configuration file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_language_model():
    """Load the language model."""
    try:
        from src.models.language_model import PMLanguageModel
        model_path = PROJECT_ROOT / "models" / "language_model"
        if not model_path.exists():
            st.warning(f"Model path not found: {model_path}")
            return None
        model = PMLanguageModel.load(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading language model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


@st.cache_resource
def load_classifier():
    """Load the classifier model."""
    try:
        from src.models.classifier import PMClassifier
        model_path = PROJECT_ROOT / "models" / "classifier"
        if not model_path.exists():
            st.warning(f"Model path not found: {model_path}")
            return None
        model = PMClassifier.load(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def render_sidebar():
    """Render the sidebar with navigation and info."""
    with st.sidebar:
        st.title("PM Model UAT")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["Home", "Language Model", "Classifier", "Batch Testing", "Model Info"],
            index=0
        )

        st.markdown("---")

        # Device info
        device = get_device()
        st.markdown(f"**Device:** `{device}`")

        # Model status
        st.markdown("### Model Status")
        lm_path = PROJECT_ROOT / "models" / "language_model"
        cls_path = PROJECT_ROOT / "models" / "classifier"

        if lm_path.exists():
            st.success("Language Model: Loaded")
        else:
            st.warning("Language Model: Not trained")

        if cls_path.exists():
            st.success("Classifier: Loaded")
        else:
            st.warning("Classifier: Not trained")

        st.markdown("---")
        st.markdown("### UAT Session")
        st.markdown(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        return page


def render_home():
    """Render the home page."""
    st.title("Project Management Model - UAT Dashboard")

    st.markdown("""
    Welcome to the User Acceptance Testing (UAT) interface for the Project Management AI Models.

    ## Available Models

    ### 1. PM Language Model (GPT-2)
    - Fine-tuned on PMI publications (2007-2025)
    - Generates project management content
    - Use for: advice, documentation, content generation

    ### 2. PM Classifier (DistilBERT)
    - Classifies text into 10 PM knowledge areas
    - Trained on labeled PM corpus
    - Use for: content categorization, topic identification

    ## How to Use

    1. **Language Model Tab**: Enter a prompt and generate PM-related text
    2. **Classifier Tab**: Enter text to classify into knowledge areas
    3. **Batch Testing Tab**: Upload CSV for bulk testing
    4. **Model Info Tab**: View model parameters and configuration

    ## Knowledge Areas
    """)

    config = load_config()
    knowledge_areas = config.get("knowledge_areas", [])

    cols = st.columns(2)
    for i, area in enumerate(knowledge_areas):
        cols[i % 2].markdown(f"- {area}")


def render_language_model():
    """Render the language model testing page."""
    st.title("Language Model Testing")

    model = load_language_model()

    if model is None:
        st.error("Language model not available. Please train the model first.")
        st.code("python scripts/train_language_model.py --config config/config.yaml")
        return

    st.success("Model loaded successfully!")

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            value="Project risk management is essential because",
            height=100,
            help="Enter the beginning of your text. The model will continue it."
        )

    with col2:
        st.markdown("### Generation Parameters")
        max_length = st.slider("Max Length", 50, 500, 200)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        top_k = st.slider("Top-K", 1, 100, 50)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)
        num_sequences = st.slider("Number of Sequences", 1, 5, 1)

    # Generate button
    if st.button("Generate Text", type="primary"):
        if prompt.strip():
            with st.spinner("Generating..."):
                start_time = time.time()

                generated = model.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_sequences
                )

                elapsed = time.time() - start_time

            st.markdown("### Generated Text")
            st.info(f"Generation time: {elapsed:.2f}s")

            for i, text in enumerate(generated):
                with st.expander(f"Sequence {i + 1}", expanded=(i == 0)):
                    st.markdown(text)

            # Log for UAT
            if "lm_tests" not in st.session_state:
                st.session_state.lm_tests = []

            st.session_state.lm_tests.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "temperature": temperature,
                "max_length": max_length,
                "output_preview": generated[0][:100] + "..."
            })
        else:
            st.warning("Please enter a prompt.")

    # Sample prompts
    st.markdown("---")
    st.markdown("### Sample Prompts")

    sample_prompts = [
        "The critical path method is used to",
        "Stakeholder engagement strategies include",
        "Earned value management helps project managers",
        "Agile project management differs from traditional",
        "Risk mitigation strategies for software projects"
    ]

    cols = st.columns(len(sample_prompts))
    for i, sample in enumerate(sample_prompts):
        if cols[i].button(f"Try {i+1}", key=f"sample_lm_{i}"):
            st.session_state.sample_prompt = sample
            st.rerun()


def render_classifier():
    """Render the classifier testing page."""
    st.title("Classifier Testing")

    model = load_classifier()

    if model is None:
        st.error("Classifier not available. Please train the model first.")
        st.code("python scripts/train_classifier.py --config config/config.yaml")
        return

    st.success("Classifier loaded successfully!")

    # Input section
    text = st.text_area(
        "Enter text to classify:",
        value="The project schedule needs to be updated to reflect the new milestone dates and dependencies.",
        height=150,
        help="Enter project management related text to classify into knowledge areas."
    )

    # Classify button
    if st.button("Classify Text", type="primary"):
        if text.strip():
            with st.spinner("Classifying..."):
                start_time = time.time()
                result = model.predict(text, return_probs=True)
                elapsed = time.time() - start_time

            st.markdown("### Classification Result")
            st.info(f"Classification time: {elapsed:.3f}s")

            # Primary prediction
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Predicted Category", result["label"])
                st.metric("Confidence", f"{result['confidence']:.1%}")

            with col2:
                # Probability distribution
                probs_df = pd.DataFrame([
                    {"Knowledge Area": k, "Probability": v}
                    for k, v in result["probabilities"].items()
                ]).sort_values("Probability", ascending=False)

                st.bar_chart(
                    probs_df.set_index("Knowledge Area")["Probability"],
                    use_container_width=True
                )

            # Detailed probabilities
            with st.expander("Detailed Probabilities"):
                st.dataframe(
                    probs_df.style.format({"Probability": "{:.4f}"}),
                    use_container_width=True
                )

            # Log for UAT
            if "cls_tests" not in st.session_state:
                st.session_state.cls_tests = []

            st.session_state.cls_tests.append({
                "timestamp": datetime.now().isoformat(),
                "text": text[:100] + "..." if len(text) > 100 else text,
                "predicted": result["label"],
                "confidence": result["confidence"]
            })
        else:
            st.warning("Please enter text to classify.")

    # Sample texts
    st.markdown("---")
    st.markdown("### Sample Texts")

    sample_texts = {
        "Schedule": "The critical path analysis shows that task dependencies will delay the project by two weeks.",
        "Risk": "We identified three high-priority risks that need immediate mitigation strategies.",
        "Cost": "The earned value analysis indicates a cost variance of -$50,000 against the baseline.",
        "Quality": "The quality audit revealed several non-conformances that require corrective action.",
        "Stakeholder": "Key stakeholders need to be engaged early to ensure project buy-in and support."
    }

    cols = st.columns(len(sample_texts))
    for i, (name, sample) in enumerate(sample_texts.items()):
        if cols[i].button(name, key=f"sample_cls_{i}"):
            st.session_state.sample_text = sample
            st.rerun()


def render_batch_testing():
    """Render the batch testing page."""
    st.title("Batch Testing")

    st.markdown("""
    Upload a CSV file to perform batch testing on either model.

    ### CSV Format

    **For Classifier:**
    - Column: `text` - The text to classify
    - Optional: `expected` - Expected knowledge area (for accuracy calculation)

    **For Language Model:**
    - Column: `prompt` - The prompt for generation
    """)

    model_type = st.radio("Select Model", ["Classifier", "Language Model"])

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Preview")
        st.dataframe(df.head())

        if st.button("Run Batch Test", type="primary"):
            if model_type == "Classifier":
                model = load_classifier()
                if model is None:
                    st.error("Classifier not available.")
                    return

                if "text" not in df.columns:
                    st.error("CSV must have a 'text' column.")
                    return

                results = []
                progress = st.progress(0)

                for i, row in df.iterrows():
                    result = model.predict(row["text"], return_probs=True)
                    results.append({
                        "text": row["text"][:50] + "...",
                        "predicted": result["label"],
                        "confidence": result["confidence"]
                    })
                    progress.progress((i + 1) / len(df))

                results_df = pd.DataFrame(results)

                # Calculate accuracy if expected column exists
                if "expected" in df.columns:
                    results_df["expected"] = df["expected"]
                    results_df["correct"] = results_df["predicted"] == results_df["expected"]
                    accuracy = results_df["correct"].mean()
                    st.metric("Accuracy", f"{accuracy:.1%}")

                st.markdown("### Results")
                st.dataframe(results_df)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "batch_results.csv",
                    "text/csv"
                )

            else:  # Language Model
                model = load_language_model()
                if model is None:
                    st.error("Language model not available.")
                    return

                if "prompt" not in df.columns:
                    st.error("CSV must have a 'prompt' column.")
                    return

                results = []
                progress = st.progress(0)

                for i, row in df.iterrows():
                    generated = model.generate(row["prompt"], max_length=100)
                    results.append({
                        "prompt": row["prompt"],
                        "generated": generated[0]
                    })
                    progress.progress((i + 1) / len(df))

                results_df = pd.DataFrame(results)

                st.markdown("### Results")
                st.dataframe(results_df)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "generation_results.csv",
                    "text/csv"
                )


def render_model_info():
    """Render the model information page."""
    st.title("Model Information")

    config = load_config()

    # Language Model Info
    st.markdown("## Language Model (GPT-2)")

    lm = load_language_model()
    if lm is not None:
        total, trainable = lm.get_num_parameters()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Parameters", f"{total:,}")
        col2.metric("Trainable Parameters", f"{trainable:,}")
        col3.metric("Model Type", config["language_model"]["model_name"])

        with st.expander("Training Configuration"):
            st.json(config["language_model"])
    else:
        st.warning("Language model not trained yet.")

    st.markdown("---")

    # Classifier Info
    st.markdown("## Classifier (DistilBERT)")

    cls = load_classifier()
    if cls is not None:
        total, trainable = cls.get_num_parameters()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Parameters", f"{total:,}")
        col2.metric("Trainable Parameters", f"{trainable:,}")
        col3.metric("Number of Classes", config["classifier"]["num_labels"])

        with st.expander("Training Configuration"):
            st.json(config["classifier"])

        with st.expander("Knowledge Areas"):
            for i, area in enumerate(config["knowledge_areas"]):
                st.markdown(f"{i}. {area}")
    else:
        st.warning("Classifier not trained yet.")

    st.markdown("---")

    # Dataset Info
    st.markdown("## Dataset Information")

    dataset_path = PROJECT_ROOT / "dataset"
    if dataset_path.exists():
        txt_files = list(dataset_path.glob("*.txt"))
        pdf_files = list(dataset_path.glob("*.pdf"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Text Files", len(txt_files))
        col2.metric("PDF Files", len(pdf_files))

        total_size = sum(f.stat().st_size for f in txt_files)
        col3.metric("Total Size (Text)", f"{total_size / 1024 / 1024:.1f} MB")

        with st.expander("Dataset Files"):
            for f in sorted(txt_files)[:20]:
                st.markdown(f"- {f.name}")
            if len(txt_files) > 20:
                st.markdown(f"... and {len(txt_files) - 20} more files")

    st.markdown("---")

    # UAT Test History
    st.markdown("## UAT Test History (This Session)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Language Model Tests")
        if "lm_tests" in st.session_state and st.session_state.lm_tests:
            st.dataframe(pd.DataFrame(st.session_state.lm_tests))
        else:
            st.info("No tests recorded yet.")

    with col2:
        st.markdown("### Classifier Tests")
        if "cls_tests" in st.session_state and st.session_state.cls_tests:
            st.dataframe(pd.DataFrame(st.session_state.cls_tests))
        else:
            st.info("No tests recorded yet.")


def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Language Model":
        render_language_model()
    elif page == "Classifier":
        render_classifier()
    elif page == "Batch Testing":
        render_batch_testing()
    elif page == "Model Info":
        render_model_info()


if __name__ == "__main__":
    main()
