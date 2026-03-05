import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import AutoModelForImageTextToText, AutoProcessor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_ID = "google/translategemma-4b-it"
MAX_NEW_TOKENS = 512
ACCEPTED_IMAGE_TYPES: list[str] = ["png", "jpg", "jpeg", "webp"]

LANGUAGES: dict[str, str] = {
    "English": "en",
    "Chinese": "zh-CN",
    "Dutch": "nl",
    "French": "fr",
    "German": "de",
    "Indonesian": "id",
    "Italian": "it",
    "Spanish": "es",
    "Vietnamese": "vi",
}

SOURCE_LANGS: list[str] = sorted(LANGUAGES.keys())


@dataclass(frozen=True)
class TranslationResult:
    response: str
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


def build_prompt(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
) -> str:
    instruction = (
        f"You are a professional {src_lang} ({src_code}) to {tgt_lang} "
        f"({tgt_code}) translator. Your goal is to accurately convey the meaning and "
        f"nuances of the original {src_lang} text while adhering to {tgt_lang} grammar, "
        f"vocabulary, and cultural sensitivities. Produce only the {tgt_lang} "
        f"translation, without any additional explanations or commentary. Please translate "
        f"the following {src_lang} text into {tgt_lang}:\n\n\n{text}"
    )
    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


@st.cache_resource
def load_model() -> tuple[Any, Any, int, int]:
    t0 = time.perf_counter_ns()
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16
    )
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    load_duration = time.perf_counter_ns() - t0
    return model, processor, eos_token_id, load_duration


def translate(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
) -> TranslationResult:
    prompt = build_prompt(text, src_lang, src_code, tgt_lang, tgt_code)
    model, processor, eos_token_id, _ = load_model()

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
    return TranslationResult(
        response=processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip(),
        prompt_eval_count=input_len,
        prompt_eval_duration=prompt_eval_duration,
        eval_count=int(generated.shape[0]),
        eval_duration=eval_duration,
    )


def translate_image(
    image: Image.Image,
    src_code: str,
    tgt_code: str,
) -> TranslationResult:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_lang_code": src_code,
                    "target_lang_code": tgt_code,
                    "image": image,
                },
            ],
        }
    ]
    model, processor, eos_token_id, _ = load_model()

    t0 = time.perf_counter_ns()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[1]
    prompt_eval_duration = time.perf_counter_ns() - t0

    t1 = time.perf_counter_ns()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            top_p=None,
            top_k=None,
            eos_token_id=eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    eval_duration = time.perf_counter_ns() - t1

    generated = output[0][input_len:]
    return TranslationResult(
        response=processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip(),
        prompt_eval_count=input_len,
        prompt_eval_duration=prompt_eval_duration,
        eval_count=int(generated.shape[0]),
        eval_duration=eval_duration,
    )


st.set_page_config(page_title="Translation Pipeline", page_icon="\U0001f310")
st.title("Translation Pipeline")

# --- Token check ---
if not os.environ.get("HF_TOKEN"):
    st.error("HF_TOKEN not found. Add it to your `.env` file.")
    st.stop()

# --- Model loading ---
try:
    with st.spinner("Loading model..."):
        model, processor, eos_token_id, load_duration = load_model()
except Exception as e:
    logger.exception("Failed to load model")
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Session state defaults ---
if "source_lang" not in st.session_state:
    st.session_state["source_lang"] = "English"
if "target_lang" not in st.session_state:
    st.session_state["target_lang"] = "Spanish"


def _swap_languages() -> None:
    src = st.session_state["source_lang"]
    tgt = st.session_state["target_lang"]
    st.session_state["source_lang"] = tgt
    st.session_state["target_lang"] = src


# --- Language selectors with swap button ---
col1, col_swap, col2 = st.columns([5, 1, 5])

source = col1.selectbox(
    "Source language",
    SOURCE_LANGS,
    key="source_lang",
)

# Valid targets: all languages except the selected source
valid_targets = sorted(n for n in LANGUAGES if n != source)
if st.session_state["target_lang"] not in valid_targets:
    st.session_state["target_lang"] = valid_targets[0]

target = col2.selectbox(
    "Target language",
    valid_targets,
    key="target_lang",
)

with col_swap:
    st.markdown("<div style='height: 1.8em'></div>", unsafe_allow_html=True)
    st.button(
        "\u21c4",
        use_container_width=True,
        on_click=_swap_languages,
    )

# --- Tabs for text and image input ---
uploaded_file = None
image = None
text_tab, image_tab = st.tabs(["Text", "Image"])

# --- Text tab ---
with text_tab:
    left_col, right_col = st.columns(2)

    with left_col:
        text = st.text_area(
            "Source text",
            height=150,
            placeholder=f"Enter {source} text to translate...",
        )
        st.caption(f"{len(text)} characters")
        translate_text_clicked = st.button(
            "Translate", type="primary", use_container_width=True, key="translate_text"
        )

    prev_response = (
        st.session_state["translation_result"].response
        if "translation_result" in st.session_state
        else ""
    )

    with right_col:
        st.markdown(
            "<label style='font-size: 0.875rem;'>Translation</label>",
            unsafe_allow_html=True,
        )
        if prev_response:
            st.code(prev_response, language=None)
        else:
            st.caption("Translation will appear here...")

# --- Image tab ---
with image_tab:
    left_col, right_col = st.columns(2)

    with left_col:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=ACCEPTED_IMAGE_TYPES,
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        translate_image_clicked = st.button(
            "Translate", type="primary", use_container_width=True, key="translate_image"
        )

    prev_image_response = (
        st.session_state["image_translation_result"].response
        if "image_translation_result" in st.session_state
        else ""
    )

    with right_col:
        st.markdown(
            "<label style='font-size: 0.875rem;'>Translation</label>",
            unsafe_allow_html=True,
        )
        if prev_image_response:
            st.code(prev_image_response, language=None)
        else:
            st.caption("Translation will appear here...")

# --- Text translation handler ---
if translate_text_clicked:
    if not text.strip():
        st.warning("Please enter text to translate.")
    else:
        try:
            with st.status("Translating...", expanded=True) as status:
                st.write("Running locally...")
                t0 = time.perf_counter_ns()
                result = translate(
                    text,
                    source,
                    LANGUAGES[source],
                    target,
                    LANGUAGES[target],
                )
                total_duration = time.perf_counter_ns() - t0
                status.update(
                    label=f"Translated in {total_duration / 1e9:.2f}s",
                    state="complete",
                    expanded=False,
                )

            st.session_state["translation_result"] = result
            st.session_state["total_duration"] = total_duration
            st.session_state["load_duration"] = load_duration
            st.session_state["active_mode"] = "text"
            st.rerun()
        except Exception as e:
            logger.exception("Translation failed")
            st.error(f"Translation failed: {e}")

# --- Image translation handler ---
if translate_image_clicked:
    if uploaded_file is None:
        st.warning("Please upload an image to translate.")
    else:
        try:
            with st.status("Translating image...", expanded=True) as status:
                st.write("Running locally...")
                t0 = time.perf_counter_ns()
                result = translate_image(
                    image,
                    LANGUAGES[source],
                    LANGUAGES[target],
                )
                total_duration = time.perf_counter_ns() - t0
                status.update(
                    label=f"Translated in {total_duration / 1e9:.2f}s",
                    state="complete",
                    expanded=False,
                )

            st.session_state["image_translation_result"] = result
            st.session_state["total_duration"] = total_duration
            st.session_state["load_duration"] = load_duration
            st.session_state["active_mode"] = "image"
            st.rerun()
        except Exception as e:
            logger.exception("Image translation failed")
            st.error(f"Image translation failed: {e}")

# --- Metrics in expander ---
active_mode = st.session_state.get("active_mode")
if active_mode == "image":
    active_result = st.session_state.get("image_translation_result")
elif active_mode == "text":
    active_result = st.session_state.get("translation_result")
else:
    active_result = None

if active_result is not None:
    total_duration = st.session_state["total_duration"]
    load_duration = st.session_state["load_duration"]

    data = {
        "model": MODEL_ID,
        "total_duration": total_duration,
        "load_duration": load_duration,
        **asdict(active_result),
    }
    metrics = [
        ("Total Time", f"{total_duration / 1e9:.2f}s"),
        ("Model Load Time", f"{load_duration / 1e9:.2f}s"),
        ("Input Tokens", active_result.prompt_eval_count),
        ("Input Processing Time", f"{active_result.prompt_eval_duration / 1e9:.2f}s"),
        ("Output Tokens", active_result.eval_count),
        ("Generation Time", f"{active_result.eval_duration / 1e9:.2f}s"),
    ]

    with st.expander("Performance details"):
        st.caption(f"Model: {MODEL_ID}")
        cols = st.columns(4)
        for i, (label, value) in enumerate(metrics):
            cols[i % 4].metric(label, value)

        st.download_button(
            "Download JSON", json.dumps(data, indent=2), "translation.json"
        )
