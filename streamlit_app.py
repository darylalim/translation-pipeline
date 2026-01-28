import json
import logging
import os
import time
from enum import StrEnum
from typing import TypedDict

import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import AutoModelForImageTextToText, AutoProcessor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationResult(TypedDict):
    """Result from the translate function with timing metrics."""
    response: str
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


MODEL_ID = "google/translategemma-4b-it"

# Maximum tokens to generate in translation response.
# 512 is sufficient for most translations while preventing runaway generation.
MAX_NEW_TOKENS = 512


class Lang(StrEnum):
    """Supported language names. Using StrEnum catches typos at development time."""
    ENGLISH = "English"
    # Bidirectional with English
    ARABIC = "Arabic"
    CANTONESE = "Cantonese"
    CHINESE = "Chinese"
    CHUUKESE = "Chuukese"
    FIJIAN = "Fijian"
    FRENCH = "French"
    HINDI = "Hindi"
    ILOCANO = "Ilocano"
    INDONESIAN = "Indonesian"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    LINGALA = "Lingala"
    MARSHALLESE = "Marshallese"
    NEPALBHASA = "Nepalbhasa (Newari)"
    RUSSIAN = "Russian"
    SPANISH = "Spanish"
    SWAHILI = "Swahili"
    TAHITIAN = "Tahitian"
    THAI = "Thai"
    TONGA = "Tonga (Tonga Islands)"
    VIETNAMESE = "Vietnamese"
    # English-to-only
    ARMENIAN = "Armenian"
    BURMESE = "Burmese"
    CHINESE_TAIWAN = "Chinese (Taiwan)"
    FILIPINO = "Filipino"
    HAWAIIAN = "Hawaiian"
    KHMER = "Khmer"
    LAO = "Lao"
    MALAYALAM = "Malayalam"
    MARATHI = "Marathi"
    MONGOLIAN = "Mongolian"
    PERSIAN = "Persian"
    PORTUGUESE_BRAZIL = "Portuguese (Brazil)"
    PORTUGUESE_PORTUGAL = "Portuguese (Portugal)"
    PUNJABI = "Punjabi"
    SAMOAN = "Samoan"
    TELUGU = "Telugu"
    # Non-English pair only
    TAIWANESE_MANDARIN = "Taiwanese Mandarin"

# All supported languages with their BCP-47 codes
# Note: Chinese has multiple variants with different codes per TranslateGemma documentation
LANGUAGES: dict[Lang, str] = {
    Lang.ENGLISH: "en",
    # Bidirectional with English (zh-CN for Chinese per docs)
    Lang.ARABIC: "ar",
    Lang.CANTONESE: "yue",
    Lang.CHINESE: "zh-CN",
    Lang.CHUUKESE: "chk",
    Lang.FIJIAN: "fj",
    Lang.FRENCH: "fr",
    Lang.HINDI: "hi",
    Lang.ILOCANO: "ilo",
    Lang.INDONESIAN: "id",
    Lang.JAPANESE: "ja",
    Lang.KOREAN: "ko",
    Lang.LINGALA: "ln",
    Lang.MARSHALLESE: "mh",
    Lang.NEPALBHASA: "new",
    Lang.RUSSIAN: "ru",
    Lang.SPANISH: "es",
    Lang.SWAHILI: "sw",
    Lang.TAHITIAN: "ty",
    Lang.THAI: "th",
    Lang.TONGA: "to",
    Lang.VIETNAMESE: "vi",
    # English-to-only
    Lang.ARMENIAN: "hy",
    Lang.BURMESE: "my",
    Lang.CHINESE_TAIWAN: "zh-TW",
    Lang.FILIPINO: "fil",
    Lang.HAWAIIAN: "haw",
    Lang.KHMER: "km",
    Lang.LAO: "lo",
    Lang.MALAYALAM: "ml",
    Lang.MARATHI: "mr",
    Lang.MONGOLIAN: "mn",
    Lang.PERSIAN: "fa",
    Lang.PORTUGUESE_BRAZIL: "pt-BR",
    Lang.PORTUGUESE_PORTUGAL: "pt-PT",
    Lang.PUNJABI: "pa",
    Lang.SAMOAN: "sm",
    Lang.TELUGU: "te",
    # Non-English pair only (Cantonese ↔ Taiwanese Mandarin)
    Lang.TAIWANESE_MANDARIN: "zh-Hant",
}

# Languages that support bidirectional translation with English
BIDIRECTIONAL_WITH_ENGLISH: set[Lang] = {
    Lang.ARABIC, Lang.CANTONESE, Lang.CHINESE, Lang.CHUUKESE, Lang.FIJIAN,
    Lang.FRENCH, Lang.HINDI, Lang.ILOCANO, Lang.INDONESIAN, Lang.JAPANESE,
    Lang.KOREAN, Lang.LINGALA, Lang.MARSHALLESE, Lang.NEPALBHASA, Lang.RUSSIAN,
    Lang.SPANISH, Lang.SWAHILI, Lang.TAHITIAN, Lang.THAI, Lang.TONGA,
    Lang.VIETNAMESE,
}

# Languages that can only be translated TO from English
ENGLISH_TO_ONLY: set[Lang] = {
    Lang.ARMENIAN, Lang.BURMESE, Lang.CHINESE_TAIWAN, Lang.FILIPINO, Lang.HAWAIIAN,
    Lang.KHMER, Lang.LAO, Lang.MALAYALAM, Lang.MARATHI, Lang.MONGOLIAN,
    Lang.PERSIAN, Lang.PORTUGUESE_BRAZIL, Lang.PORTUGUESE_PORTUGAL, Lang.PUNJABI,
    Lang.SAMOAN, Lang.TELUGU,
}

# Non-English language pairs: source -> set of targets
# Per CLAUDE.md: Cantonese pairs with Chinese (zh) and Taiwanese Mandarin (zh-Hant)
# Chinese (zh-CN) pairs with Japanese, Chinese (zh) pairs with Swahili
# Using "Chinese" (zh-CN) for all Chinese variants as model may treat zh/zh-CN equivalently
NON_ENGLISH_PAIRS: dict[Lang, set[Lang]] = {
    Lang.ARABIC: {Lang.SWAHILI},
    Lang.CANTONESE: {Lang.CHINESE, Lang.TAIWANESE_MANDARIN},
    Lang.CHINESE: {Lang.JAPANESE, Lang.SWAHILI},
    Lang.SWAHILI: {Lang.ARABIC, Lang.CHINESE},
    Lang.JAPANESE: {Lang.CHINESE},
    Lang.TAIWANESE_MANDARIN: {Lang.CANTONESE},
}

# Source languages: English + all bidirectional + sources from non-English pairs
_non_english_only_sources = {lang for lang in NON_ENGLISH_PAIRS if lang not in BIDIRECTIONAL_WITH_ENGLISH}
SOURCE_LANGS: list[str] = sorted(
    [Lang.ENGLISH] + list(BIDIRECTIONAL_WITH_ENGLISH) + list(_non_english_only_sources)
)

# Validate all source languages exist in LANGUAGES dict
for _lang in SOURCE_LANGS:
    if _lang not in LANGUAGES:
        raise ValueError(f"Source language '{_lang}' not found in LANGUAGES dict")


def get_target_languages(source: str) -> list[str]:
    """Get available target languages for a given source language."""
    targets: set[Lang] = set()
    if source == Lang.ENGLISH:
        targets = BIDIRECTIONAL_WITH_ENGLISH | ENGLISH_TO_ONLY
    elif source in BIDIRECTIONAL_WITH_ENGLISH:
        targets.add(Lang.ENGLISH)
    if source in NON_ENGLISH_PAIRS:
        targets.update(NON_ENGLISH_PAIRS[source])
    return sorted(targets)


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
    text: str, src_lang: str, src_code: str, tgt_lang: str, tgt_code: str,
    model: AutoModelForImageTextToText, processor: AutoProcessor, eos_token_id: int,
) -> TranslationResult:
    instruction = (
        f"You are a professional {src_lang} ({src_code}) to {tgt_lang} ({tgt_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src_lang} text "
        f"while adhering to {tgt_lang} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {tgt_lang} translation, without any additional explanations or commentary. "
        f"Please translate the following {src_lang} text into {tgt_lang}:\n\n\n{text}"
    )
    prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

    t0 = time.perf_counter_ns()
    inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
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

    generated = output[0, input_len:]
    return {
        "response": processor.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "prompt_eval_count": input_len,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": generated.shape[0],
        "eval_duration": eval_duration,
    }


st.title("Translation Pipeline")

if not os.getenv("HF_TOKEN"):
    st.error("HF_TOKEN not found. Create a .env file with your Hugging Face token.")
    st.stop()

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

t0 = time.perf_counter_ns()
try:
    with st.spinner("Loading model..."):
        model, processor, eos_token_id = load_model()
except Exception as e:
    logger.exception("Failed to load model")
    st.error(f"Failed to load model: {e}")
    st.stop()
load_duration = time.perf_counter_ns() - t0

model_cached = st.session_state.model_loaded
st.session_state.model_loaded = True

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
                result = translate(text, source, LANGUAGES[source], target, LANGUAGES[target], model, processor, eos_token_id)
                total_duration = time.perf_counter_ns() - t0

            st.subheader("Translation")
            st.write(result["response"])

            data = {"model": MODEL_ID, "total_duration": total_duration, "load_duration": load_duration, **result}

            st.subheader("Metrics")
            total_secs = total_duration / 1e9
            input_tps = result["prompt_eval_count"] / max(result["prompt_eval_duration"] / 1e9, 1e-9)
            output_tps = result["eval_count"] / max(result["eval_duration"] / 1e9, 1e-9)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Time", f"{total_secs:.2f}s")
            c2.metric("Output Tokens", result["eval_count"])
            c3.metric("Output Speed", f"{output_tps:.1f} tok/s")

            with st.expander("Detailed Metrics"):
                d1, d2 = st.columns(2)
                d1.metric("Input Tokens", result["prompt_eval_count"])
                d1.metric("Processing Time", f"{result['prompt_eval_duration'] / 1e9:.2f}s")
                d1.metric("Input Speed", f"{input_tps:.1f} tok/s")
                d2.metric("Generation Time", f"{result['eval_duration'] / 1e9:.2f}s")
                load_label = "Model Load Time (cached)" if model_cached else "Model Load Time"
                d2.metric(load_label, f"{load_duration / 1e9:.2f}s")

            st.download_button("Download JSON", json.dumps(data, indent=2), "translation.json")
        except Exception as e:
            logger.exception("Translation failed")
            st.error(f"Translation failed: {e}")
