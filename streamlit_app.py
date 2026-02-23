import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import streamlit as st
import torch
from huggingface_hub import InferenceClient
from transformers import AutoModelForImageTextToText, AutoProcessor

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


def _build_target_langs() -> dict[str, list[str]]:
    targets: dict[str, list[str]] = {}
    for name, (_, bi) in LANGUAGES.items():
        if not bi:
            continue
        if name == "English":
            targets[name] = sorted(n for n in LANGUAGES if n != name)
        else:
            targets[name] = ["English"]
    return targets


TARGET_LANGS: dict[str, list[str]] = _build_target_langs()


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


def has_gpu() -> bool:
    return torch.cuda.is_available() or torch.backends.mps.is_available()


@st.cache_resource
def load_model(token: str) -> tuple[Any, Any, int, int]:
    t0 = time.perf_counter_ns()
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, token=token
    )
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    load_duration = time.perf_counter_ns() - t0
    return model, processor, eos_token_id, load_duration


def _translate_local(prompt: str, token: str) -> TranslationResult:
    model, processor, eos_token_id, _ = load_model(token)

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


def _translate_api(prompt: str, token: str) -> TranslationResult:
    client = InferenceClient(model=MODEL_ID, token=token)
    output = client.text_generation(
        prompt, max_new_tokens=MAX_NEW_TOKENS, details=True, return_full_text=False
    )
    return TranslationResult(
        response=output.generated_text.strip(),
        prompt_eval_count=len(output.details.prefill),
        prompt_eval_duration=0,
        eval_count=output.details.generated_tokens,
        eval_duration=0,
    )


def translate(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
    token: str,
) -> TranslationResult:
    prompt = build_prompt(text, src_lang, src_code, tgt_lang, tgt_code)
    if has_gpu():
        return _translate_local(prompt, token)
    return _translate_api(prompt, token)


st.set_page_config(page_title="Translation Pipeline", page_icon="\U0001f310")
st.title("Translation Pipeline")

# --- Authentication ---
hf_token = st.secrets.get("HF_TOKEN")
if not hf_token:
    st.info(
        "A [Hugging Face token](https://huggingface.co/settings/tokens) is required "
        "to access the TranslateGemma model. Create a token with **Read** access."
    )
    hf_token = st.text_input("Hugging Face Token", type="password")
if not hf_token:
    st.stop()

# --- Backend detection & model loading ---
if has_gpu():
    try:
        with st.spinner("Loading model..."):
            model, processor, eos_token_id, load_duration = load_model(hf_token)
    except Exception as e:
        logger.exception("Failed to load model")
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    load_duration = 0

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


# Swap disabled when current target is unidirectional (not a valid source)
_cur_target = st.session_state["target_lang"]
can_swap = LANGUAGES[_cur_target][1] and _cur_target in SOURCE_LANGS

# --- Language selectors with swap button ---
col1, col_swap, col2 = st.columns([5, 1, 5])

source = col1.selectbox(
    "Source language",
    SOURCE_LANGS,
    key="source_lang",
)

# Validate target: if current target isn't valid for the new source, reset it
valid_targets = TARGET_LANGS[source]
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
        disabled=not can_swap,
        use_container_width=True,
        on_click=_swap_languages,
    )

# --- Side-by-side input / output ---
left_col, right_col = st.columns(2)

with left_col:
    text = st.text_area(
        "Source text",
        height=150,
        placeholder=f"Enter {source} text to translate...",
    )
    st.caption(f"{len(text)} characters")
    translate_clicked = st.button("Translate", type="primary", use_container_width=True)

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

# --- Translation handler ---
if translate_clicked:
    if not text.strip():
        st.warning("Please enter text to translate.")
    else:
        try:
            backend_msg = (
                "Running on local GPU..."
                if has_gpu()
                else "Calling HF Inference API..."
            )
            with st.status("Translating...", expanded=True) as status:
                st.write(backend_msg)
                t0 = time.perf_counter_ns()
                result = translate(
                    text,
                    source,
                    LANGUAGES[source][0],
                    target,
                    LANGUAGES[target][0],
                    hf_token,
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
            st.session_state["used_gpu"] = has_gpu()
            st.rerun()
        except Exception as e:
            logger.exception("Translation failed")
            st.error(f"Translation failed: {e}")

# --- Metrics in expander ---
if "translation_result" in st.session_state:
    result = st.session_state["translation_result"]
    total_duration = st.session_state["total_duration"]
    load_duration = st.session_state["load_duration"]

    used_gpu = st.session_state.get("used_gpu", False)

    if used_gpu:
        data = {
            "model": MODEL_ID,
            "total_duration": total_duration,
            "load_duration": load_duration,
            **asdict(result),
        }
        metrics = [
            ("Total Time", f"{total_duration / 1e9:.2f}s"),
            ("Model Load Time", f"{load_duration / 1e9:.2f}s"),
            ("Input Tokens", result.prompt_eval_count),
            ("Input Processing Time", f"{result.prompt_eval_duration / 1e9:.2f}s"),
            ("Output Tokens", result.eval_count),
            ("Generation Time", f"{result.eval_duration / 1e9:.2f}s"),
        ]
    else:
        data = {
            "model": MODEL_ID,
            "total_duration": total_duration,
            "response": result.response,
            "prompt_eval_count": result.prompt_eval_count,
            "eval_count": result.eval_count,
        }
        metrics = [
            ("Total Time", f"{total_duration / 1e9:.2f}s"),
            ("Input Tokens", result.prompt_eval_count),
            ("Output Tokens", result.eval_count),
        ]

    with st.expander("Performance details"):
        st.caption(f"Model: {MODEL_ID}")
        cols = st.columns(4)
        for i, (label, value) in enumerate(metrics):
            cols[i % 4].metric(label, value)

        st.download_button(
            "Download JSON", json.dumps(data, indent=2), "translation.json"
        )
