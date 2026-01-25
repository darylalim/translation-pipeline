import json
import time

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = "google/madlad400-10b-mt"

# Supported languages with BCP-47 codes
LANGUAGES = {
    "Cantonese": "yue",
    "Mandarin Chinese": "zh",
    "Chuukese": "chk",
    "Hawaiian": "haw",
    "Ilocano": "ilo",
    "Japanese": "ja",
    "Korean": "ko",
    "Marshallese": "mh",
    "Samoan": "sm",
    "Spanish": "es",
    "Tagalog": "fil",
    "Thai": "th",
    "Vietnamese": "vi",
}


def get_device():
    """Automatically detect the best available device in order of priority: MPS, CUDA, CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@st.cache_resource
def load_model():
    """Load model and tokenizer at application startup."""
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.float16)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def translate(text, target_lang_code, model, tokenizer, device):
    """Translate text to the target language using MADLAD-400.

    Returns a dict with the response and timing/token metrics.
    """
    # MADLAD-400 uses language tag prefix format: <2xx> text
    input_text = f"<2{target_lang_code}> {text}"

    # Tokenize and measure prompt evaluation
    prompt_eval_start = time.perf_counter_ns()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    prompt_eval_count = inputs["input_ids"].shape[1]
    prompt_eval_end = time.perf_counter_ns()
    prompt_eval_duration = prompt_eval_end - prompt_eval_start

    # Generate and measure token generation
    eval_start = time.perf_counter_ns()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    eval_end = time.perf_counter_ns()
    eval_duration = eval_end - eval_start

    eval_count = outputs.shape[1]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "response": response,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
    }


st.title("Translation Pipeline")
st.write("Translate English text to supported languages using the MADLAD-400 model.")

device = get_device()

load_start = time.perf_counter_ns()
with st.spinner(f"Loading model on {device.upper()}..."):
    model, tokenizer = load_model()
    model.to(device)
load_end = time.perf_counter_ns()
load_duration = load_end - load_start

st.subheader("Input")
input_text = st.text_area("Enter English text to translate", height=150)

st.subheader("Target Language")
target_language = st.selectbox(
    "Select target language",
    options=list(LANGUAGES.keys()),
    index=0
)

if st.button("Translate", type="primary"):
    if input_text.strip():
        with st.spinner("Translating..."):
            target_code = LANGUAGES[target_language]
            total_start = time.perf_counter_ns()
            result = translate(input_text, target_code, model, tokenizer, device)
            total_end = time.perf_counter_ns()
            total_duration = total_end - total_start

        st.subheader("Translation")
        st.write(result["response"])

        # Build download data
        download_data = {
            "model": MODEL_NAME,
            "response": result["response"],
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_count": result["prompt_eval_count"],
            "prompt_eval_duration": result["prompt_eval_duration"],
            "eval_count": result["eval_count"],
            "eval_duration": result["eval_duration"],
        }

        # Display metrics (all fields except response)
        st.subheader("Metrics")
        st.metric("Model", MODEL_NAME)
        st.metric("Total Duration (ns)", total_duration)
        st.metric("Load Duration (ns)", load_duration)
        st.metric("Prompt Eval Count", result["prompt_eval_count"])
        st.metric("Prompt Eval Duration (ns)", result["prompt_eval_duration"])
        st.metric("Eval Count", result["eval_count"])
        st.metric("Eval Duration (ns)", result["eval_duration"])

        # Download button
        st.download_button(
            label="Download JSON",
            data=json.dumps(download_data, indent=2),
            file_name="translation.json",
            mime="application/json",
        )
    else:
        st.warning("Please enter text to translate.")
