import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models
from transformers import AutoModelForCausalLM, AutoTokenizer

# Docling models can be prefetched for offline use
download_models()

artifacts_path = str(Path.home() / '.cache' / 'docling' / 'models')

def get_device():
    """Automatically detect the best available device in order of priority: MPS, CUDA, CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@st.cache_resource
def load_model(device):
    """Load model and tokenizer at application startup."""
    model_path = "ibm-granite/granite-4.0-h-tiny"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def convert(source, doc_converter):
    """Convert a source file to a Docling document and export to Markdown."""
    result = doc_converter.convert(
        source=source,
        max_num_pages=100,
        max_file_size=20971520
    )
    doc = result.document
    doc_markdown = doc.export_to_markdown()
    return doc_markdown

def translate(doc_markdown, target_language, model, tokenizer, device):
    """Translate the source text to the target language with a transformers model."""
    prompt = f"""Translate {doc_markdown} to {target_language}. Only output the translation, and nothing else."""
    chat = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)
    input_length = input_tokens["input_ids"].shape[1]
    # max_new_tokens = input_length * 2

    with torch.no_grad():
        output = model.generate(
            **input_tokens,
            max_new_tokens=200
            # max_new_tokens=max_new_tokens
        )

    generated_tokens = output[0][input_length:]
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return translation

st.title("Translation Pipeline")
st.write("Translate English text to target languages with IBM Granite 4.0 language models.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, tokenizer = load_model(device)

selected_model_path = "ibm-granite/granite-4.0-h-tiny"

st.subheader("Target Languages")
target_language = st.selectbox(
    "Select target language",
    options=["Arabic", "Chinese", "Czech", "Dutch", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Spanish"],
    index=0
)

if st.button("Translate", type='primary'):
    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_table_structure=True
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Converting document..."):
                # Save uploaded file temporarily for Docling to process                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                doc_markdown = convert(tmp_file_path, doc_converter)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)

            with st.spinner("Translating..."):
                start_time = time.time_ns()
                translation = translate(doc_markdown, target_language, model, tokenizer, device)
                end_time = time.time_ns()
                total_duration_ns = end_time - start_time

            st.success("Done.")

            st.subheader("Metrics")

            st.metric("Model", selected_model_path)
            st.metric("Total Duration (nanoseconds)", total_duration_ns)
            
            # Prepare JSON for download
            translation_data = {
                "model": selected_model_path,
                "total_duration_ns": total_duration_ns,
                "translation": translation
            }
            
            json_str = json.dumps(translation_data, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{uploaded_file.name}_translation.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Syntax error: {str(e)}")
    else:
        st.warning("Upload a PDF file.")
