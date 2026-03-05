import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
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
TRANSLATION_LABEL: str = "<label style='font-size: 0.875rem;'>Translation</label>"

LANGUAGES: dict[str, str] = {
    "English": "en",
    "Chinese": "zh",
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


@dataclass(frozen=True)
class HistoryEntry:
    mode: str
    source_lang: str
    source_code: str
    target_langs: list[str]
    target_codes: list[str]
    source_text: str
    results: list[TranslationResult]
    total_duration: int
    load_duration: int
    timestamp: str


def compute_tokens_per_sec(eval_count: int, eval_duration: int) -> float:
    if eval_duration == 0 or eval_count == 0:
        return 0.0
    return eval_count / (eval_duration / 1e9)


def compute_char_ratio(source: str, target: str) -> float:
    if len(source) == 0 or len(target) == 0:
        return 0.0
    return len(target) / len(source)


def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


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


def _generate_and_decode(
    model: Any,
    processor: Any,
    inputs: Any,
    eos_token_id: int,
    prompt_eval_duration: int,
) -> TranslationResult:
    input_len = inputs["input_ids"].shape[1]

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
    prompt_eval_duration = time.perf_counter_ns() - t0

    return _generate_and_decode(
        model, processor, inputs, eos_token_id, prompt_eval_duration
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
    prompt_eval_duration = time.perf_counter_ns() - t0

    return _generate_and_decode(
        model, processor, inputs, eos_token_id, prompt_eval_duration
    )


def translate_multi(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_langs: list[str],
    tgt_codes: list[str],
) -> list[tuple[str, str, TranslationResult]]:
    results: list[tuple[str, str, TranslationResult]] = []
    for tgt_lang, tgt_code in zip(tgt_langs, tgt_codes, strict=True):
        result = translate(text, src_lang, src_code, tgt_lang, tgt_code)
        results.append((tgt_lang, tgt_code, result))
    return results


st.set_page_config(page_title="Translation Pipeline", page_icon="\U0001f310")
st.title("Translation Pipeline")

# --- Sidebar ---
with st.sidebar:
    st.header("Translation Pipeline")
    st.caption(f"Model: {MODEL_ID}")
    st.divider()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    history: list[dict[str, Any]] = st.session_state["history"]

    if history:
        st.subheader("History")
        for i, entry in enumerate(reversed(history)):
            idx = len(history) - 1 - i
            targets = ", ".join(entry["target_codes"]).upper()
            label = f"{entry['source_code'].upper()} \u2192 {targets}"
            preview = entry["source_text"][:40]
            if len(entry["source_text"]) > 40:
                preview += "..."
            icon = "\U0001f5bc\ufe0f" if entry["mode"] == "image" else "\U0001f4dd"
            if st.button(
                f"{icon} {label}: {preview}",
                key=f"history_{idx}",
                use_container_width=True,
            ):
                st.session_state["source_lang"] = entry["source_lang"]
                if len(entry["target_langs"]) == 1:
                    st.session_state["target_lang"] = entry["target_langs"][0]
                if entry["mode"] == "text" and len(entry["results"]) == 1:
                    st.session_state["translation_result"] = TranslationResult(
                        **entry["results"][0]
                    )
                    st.session_state["total_duration"] = entry["total_duration"]
                    st.session_state["load_duration"] = entry["load_duration"]
                    st.session_state["active_mode"] = "text"
                elif entry["mode"] == "image" and len(entry["results"]) == 1:
                    st.session_state["image_translation_result"] = TranslationResult(
                        **entry["results"][0]
                    )
                    st.session_state["total_duration"] = entry["total_duration"]
                    st.session_state["load_duration"] = entry["load_duration"]
                    st.session_state["active_mode"] = "image"
                st.rerun()

        st.divider()
        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()
        with col_export:
            st.download_button(
                "Export JSON",
                json.dumps(history, indent=2),
                "history.json",
                use_container_width=True,
            )
    else:
        st.caption("No translations yet. Results will appear here.")

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
if "history" not in st.session_state:
    st.session_state["history"] = []


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

# --- Multi-pair toggle (text mode only) ---
multi_pair = st.checkbox("Translate to multiple languages", key="multi_pair_mode")

if multi_pair:
    selected_targets = st.multiselect(
        "Target languages",
        sorted(n for n in LANGUAGES if n != source),
        key="selected_targets",
    )
else:
    selected_targets = [target]

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
        st.caption(f"{word_count(text)} words \u00b7 {len(text)} characters")
        translate_text_clicked = st.button(
            "Translate", type="primary", use_container_width=True, key="translate_text"
        )
        st.caption("Ctrl+Enter to translate")

    prev_response = (
        st.session_state["translation_result"].response
        if "translation_result" in st.session_state
        else ""
    )

    with right_col:
        st.markdown(TRANSLATION_LABEL, unsafe_allow_html=True)
        if prev_response:
            st.code(prev_response, language=None)
        else:
            st.caption("Enter text and click Translate to see results.")

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
            st.image(image, width="stretch")
        translate_image_clicked = st.button(
            "Translate", type="primary", use_container_width=True, key="translate_image"
        )

    prev_image_response = (
        st.session_state["image_translation_result"].response
        if "image_translation_result" in st.session_state
        else ""
    )

    with right_col:
        st.markdown(TRANSLATION_LABEL, unsafe_allow_html=True)
        if prev_image_response:
            st.code(prev_image_response, language=None)
        else:
            st.caption("Upload an image and click Translate to see results.")

# --- Text translation handler ---
if translate_text_clicked:
    if not text.strip():
        st.warning("Please enter text to translate.")
    elif multi_pair and not selected_targets:
        st.warning("Please select at least one target language.")
    elif multi_pair:
        # Multi-pair translation
        try:
            tgt_codes = [LANGUAGES[t] for t in selected_targets]
            with st.status(
                f"Translating to {len(selected_targets)} languages...", expanded=True
            ) as status:
                t0 = time.perf_counter_ns()
                multi_results = translate_multi(
                    text,
                    source,
                    LANGUAGES[source],
                    selected_targets,
                    tgt_codes,
                )
                total_duration = time.perf_counter_ns() - t0
                status.update(
                    label=f"Translated to {len(selected_targets)} languages in {total_duration / 1e9:.2f}s",
                    state="complete",
                    expanded=False,
                )

            st.session_state["multi_pair_results"] = [
                {"target_lang": lang, "target_code": code, "result": asdict(r)}
                for lang, code, r in multi_results
            ]
            st.session_state["total_duration"] = total_duration
            st.session_state["load_duration"] = load_duration
            st.session_state["active_mode"] = "multi"
            st.toast(
                f"Translated to {len(selected_targets)} languages "
                f"in {total_duration / 1e9:.1f}s"
            )
            st.session_state["history"].append(
                {
                    "mode": "text",
                    "source_lang": source,
                    "source_code": LANGUAGES[source],
                    "target_langs": selected_targets,
                    "target_codes": tgt_codes,
                    "source_text": text,
                    "results": [asdict(r) for _, _, r in multi_results],
                    "total_duration": total_duration,
                    "load_duration": load_duration,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            st.rerun()
        except Exception as e:
            logger.exception("Multi-pair translation failed")
            st.error(f"Translation failed: {e}")
    else:
        # Single-pair translation
        try:
            with st.status("Translating...", expanded=True) as status:
                st.write("Tokenizing input...")
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
            st.toast(
                f"{LANGUAGES[source].upper()} \u2192 {LANGUAGES[target].upper()} "
                f"translated in {total_duration / 1e9:.1f}s"
            )
            st.session_state["history"].append(
                {
                    "mode": "text",
                    "source_lang": source,
                    "source_code": LANGUAGES[source],
                    "target_langs": [target],
                    "target_codes": [LANGUAGES[target]],
                    "source_text": text,
                    "results": [asdict(result)],
                    "total_duration": total_duration,
                    "load_duration": load_duration,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            st.rerun()
        except Exception as e:
            logger.exception("Translation failed")
            st.error(f"Translation failed: {e}")

# --- Image translation handler ---
if translate_image_clicked:
    if image is None:
        st.warning("Please upload an image to translate.")
    else:
        try:
            with st.status("Translating image...", expanded=True) as status:
                st.write("Processing image and tokenizing...")
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
            st.toast(
                f"Image {LANGUAGES[source].upper()} \u2192 {LANGUAGES[target].upper()} "
                f"translated in {total_duration / 1e9:.1f}s"
            )
            filename = uploaded_file.name if uploaded_file else "image"
            st.session_state["history"].append(
                {
                    "mode": "image",
                    "source_lang": source,
                    "source_code": LANGUAGES[source],
                    "target_langs": [target],
                    "target_codes": [LANGUAGES[target]],
                    "source_text": filename,
                    "results": [asdict(result)],
                    "total_duration": total_duration,
                    "load_duration": load_duration,
                    "timestamp": datetime.now().isoformat(),
                }
            )
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

if active_result is not None and active_mode in ("text", "image"):
    total_duration = st.session_state["total_duration"]
    load_duration = st.session_state["load_duration"]

    tok_sec = compute_tokens_per_sec(
        active_result.eval_count, active_result.eval_duration
    )

    data = {
        "model": MODEL_ID,
        "total_duration": total_duration,
        "load_duration": load_duration,
        "tokens_per_sec": round(tok_sec, 2),
        **asdict(active_result),
    }

    metrics = [
        ("Total Time", f"{total_duration / 1e9:.2f}s"),
        ("Model Load Time", f"{load_duration / 1e9:.2f}s"),
        ("Input Tokens", active_result.prompt_eval_count),
        ("Input Processing Time", f"{active_result.prompt_eval_duration / 1e9:.2f}s"),
        ("Output Tokens", active_result.eval_count),
        ("Generation Time", f"{active_result.eval_duration / 1e9:.2f}s"),
        ("Tokens/sec", f"{tok_sec:.1f}"),
    ]

    # Add character ratio for text mode
    if active_mode == "text" and "translation_result" in st.session_state:
        source_text = text if text else ""
        char_ratio = compute_char_ratio(source_text, active_result.response)
        metrics.append(("Char Ratio (tgt/src)", f"{char_ratio:.2f}"))

    with st.expander("Performance details"):
        st.caption(f"Model: {MODEL_ID}")
        cols = st.columns(4)
        for i, (label, value) in enumerate(metrics):
            cols[i % 4].metric(label, value)

        st.download_button(
            "Download JSON", json.dumps(data, indent=2), "translation.json"
        )

# --- Multi-pair results table ---
if st.session_state.get("active_mode") == "multi":
    multi_results = st.session_state.get("multi_pair_results", [])
    if multi_results:
        total_duration = st.session_state["total_duration"]
        total_tokens = sum(r["result"]["eval_count"] for r in multi_results)
        total_gen_time = sum(r["result"]["eval_duration"] for r in multi_results)
        avg_tok_sec = compute_tokens_per_sec(total_tokens, total_gen_time)

        st.subheader("Multi-pair Results")
        st.caption(
            f"Total: {total_duration / 1e9:.2f}s \u00b7 "
            f"Avg: {avg_tok_sec:.1f} tok/s \u00b7 "
            f"{len(multi_results)} languages"
        )

        for r in multi_results:
            result = r["result"]
            tok_sec = compute_tokens_per_sec(
                result["eval_count"], result["eval_duration"]
            )
            with st.expander(
                f"**{r['target_lang']}** ({r['target_code']}) \u00b7 "
                f"{result['eval_duration'] / 1e9:.2f}s \u00b7 "
                f"{tok_sec:.1f} tok/s"
            ):
                st.code(result["response"], language=None)
