import json
import logging
import os
import time

import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import AutoModelForImageTextToText, AutoProcessor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "google/translategemma-4b-it"
MAX_NEW_TOKENS = 512

# Language name → (BCP-47 code, bidirectional with English)
LANGUAGES: dict[str, tuple[str, bool]] = {
    "English": ("en", True),
    "Cantonese": ("yue", True),
    "Chinese": ("zh-CN", True),
    "Chuukese": ("chk", True),
    "Ilocano": ("ilo", True),
    "Japanese": ("ja", True),
    "Korean": ("ko", True),
    "Marshallese": ("mh", True),
    "Spanish": ("es", True),
    "Thai": ("th", True),
    "Tonga (Tonga Islands)": ("to", True),
    "Vietnamese": ("vi", True),
    "Filipino": ("fil", False),
    "Hawaiian": ("haw", False),
    "Samoan": ("sm", False),
}

SOURCE_LANGS: list[str] = sorted(name for name, (_, bi) in LANGUAGES.items() if bi)


def get_target_languages(source: str) -> list[str]:
    """Get available target languages for a given source language."""
    if source == "English":
        return sorted(name for name in LANGUAGES if name != "English")
    return ["English"]


@st.cache_resource
def load_model() -> tuple[AutoModelForImageTextToText, AutoProcessor, int]:
    token = os.getenv("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, token=token
    )
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    return model, processor, eos_token_id


def translate(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    eos_token_id: int,
) -> dict[str, str | int]:
    instruction = (
        f"You are a professional {src_lang} ({src_code}) to {tgt_lang} "
        f"({tgt_code}) translator. Your goal is to accurately convey the meaning and "
        f"nuances of the original {src_lang} text while adhering to {tgt_lang} grammar, "
        f"vocabulary, and cultural sensitivities. Produce only the {tgt_lang} "
        f"translation, without any additional explanations or commentary. Please translate "
        f"the following {src_lang} text into {tgt_lang}:\n\n\n{text}"
    )
    prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

    t0 = time.perf_counter_ns()
    inputs = processor.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    prompt_eval_duration = time.perf_counter_ns() - t0

    t1 = time.perf_counter_ns()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            # Override model's default generation config to suppress warnings
            # when greedy decoding (do_sample=False) is used
            top_p=None,
            top_k=None,
            eos_token_id=eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    eval_duration = time.perf_counter_ns() - t1

    generated = output[0][input_len:]
    return {
        "response": processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip(),
        "prompt_eval_count": input_len,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": int(generated.shape[0]),
        "eval_duration": eval_duration,
    }


st.title("Translation Pipeline")

if not os.getenv("HF_TOKEN"):
    st.error("HF_TOKEN not found. Create a .env file with your Hugging Face token.")
    st.stop()

try:
    t0 = time.perf_counter_ns()
    with st.spinner("Loading model..."):
        model, processor, eos_token_id = load_model()
    elapsed = time.perf_counter_ns() - t0
    if "load_duration" not in st.session_state:
        st.session_state.load_duration = elapsed
except Exception as e:
    logger.exception("Failed to load model")
    st.error(f"Failed to load model: {e}")
    st.stop()
load_duration: int = st.session_state.load_duration

col1, col2 = st.columns(2)
source = col1.selectbox("Source language", SOURCE_LANGS)
targets = get_target_languages(source)
target = col2.selectbox("Target language", targets)

text = st.text_area(f"Enter {source} text", height=150)

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please enter text to translate.")
    else:
        try:
            with st.spinner("Translating..."):
                t0 = time.perf_counter_ns()
                result = translate(
                    text,
                    source,
                    LANGUAGES[source][0],
                    target,
                    LANGUAGES[target][0],
                    model,
                    processor,
                    eos_token_id,
                )
                total_duration = time.perf_counter_ns() - t0

            st.subheader("Translation")
            st.write(result["response"])

            data = {
                "model": MODEL_ID,
                "total_duration": total_duration,
                "load_duration": load_duration,
                **result,
            }

            st.subheader("Metrics")
            metrics = [
                ("Model", MODEL_ID),
                ("Total Duration", f"{total_duration / 1e9:.2f}s"),
                ("Load Duration", f"{load_duration / 1e9:.2f}s"),
                ("Prompt Eval Count", result["prompt_eval_count"]),
                (
                    "Prompt Eval Duration",
                    f"{result['prompt_eval_duration'] / 1e9:.2f}s",
                ),
                ("Eval Count", result["eval_count"]),
                ("Eval Duration", f"{result['eval_duration'] / 1e9:.2f}s"),
            ]
            cols = st.columns(4)
            for i, (label, value) in enumerate(metrics):
                cols[i % 4].metric(label, value)

            st.download_button(
                "Download JSON", json.dumps(data, indent=2), "translation.json"
            )
        except Exception as e:
            logger.exception("Translation failed")
            st.error(f"Translation failed: {e}")
