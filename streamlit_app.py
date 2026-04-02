import json
import logging
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from mlx_lm import generate, load

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/translategemma-4b-it-8bit"
MAX_NEW_TOKENS = 512

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
def load_model() -> tuple[Any, Any]:
    return load(MODEL_ID)  # type: ignore[return-value]


def translate(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
) -> str:
    prompt = build_prompt(text, src_lang, src_code, tgt_lang, tgt_code)
    model, tokenizer = load_model()
    result = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_NEW_TOKENS)
    # Strip <end_of_turn> and any trailing garbage — the manual prompt
    # doesn't let mlx_lm know to stop at the end-of-turn token.
    return result.split("<end_of_turn>", 1)[0].strip()


st.set_page_config(page_title="TranslateGemma Translate", page_icon="\U0001f310")
st.title("TranslateGemma Translate")

# --- Session state defaults ---
st.session_state.setdefault("source_lang", "English")
st.session_state.setdefault("target_lang", "Spanish")

# --- Model loading ---
try:
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
except Exception as e:
    logger.exception("Failed to load model")
    st.error(f"Failed to load model: {e}")
    st.stop()


def _swap_languages() -> None:
    state = st.session_state
    state["source_lang"], state["target_lang"] = (
        state["target_lang"],
        state["source_lang"],
    )
    if "translation_result" in state:
        state["source_text"] = state.pop("translation_result")


# --- Language selectors ---
col1, col_swap, col2 = st.columns([10, 1, 10])
source = col1.selectbox(
    "Source language",
    SOURCE_LANGS,
    key="source_lang",
    label_visibility="collapsed",
)

valid_targets = sorted(n for n in LANGUAGES if n != source)
if st.session_state["target_lang"] not in valid_targets:
    st.session_state["target_lang"] = valid_targets[0]

target = col2.selectbox(
    "Target language",
    valid_targets,
    key="target_lang",
    label_visibility="collapsed",
)

with col_swap:
    st.button(
        ":material/swap_horiz:",
        type="tertiary",
        use_container_width=True,
        on_click=_swap_languages,
        help="Swap languages",
    )

# --- Text areas ---
left_col, right_col = st.columns(2)

with left_col:
    text = st.text_area(
        "Source text",
        height=300,
        max_chars=5000,
        key="source_text",
        label_visibility="collapsed",
    )

prev_response = st.session_state.get("translation_result", "")

with right_col:
    st.session_state["text_output"] = prev_response
    st.text_area(
        "Translation output",
        placeholder="Translation",
        disabled=True,
        height=300,
        label_visibility="collapsed",
        key="text_output",
    )

# --- Buttons ---
btn_left_col, btn_right_col = st.columns(2)

with btn_left_col:
    btn_translate_col, btn_clear_col = st.columns([3, 1])
    with btn_translate_col:
        translate_clicked = st.button(
            "Translate",
            type="primary",
            key="translate_text",
        )
    with btn_clear_col:
        if st.button(
            ":material/close:",
            type="tertiary",
            help="Clear source text",
            key="clear_text",
        ):
            st.session_state["source_text"] = ""
            st.session_state.pop("translation_result", None)
            st.rerun()

with btn_right_col:
    _, copy_col, download_col = st.columns([6, 1, 1])
    with copy_col:
        if st.button(
            ":material/content_copy:",
            type="tertiary",
            help="Copy translation",
            key="copy_text",
        ):
            if prev_response:
                # json.dumps escapes quotes, backslashes, and control characters,
                # producing a valid JS string literal. We also escape < to \u003c
                # to prevent </script> from prematurely closing the script tag.
                safe_js = json.dumps(prev_response).replace("<", "\\u003c")
                components.html(
                    "<script>"
                    "try{window.parent.navigator.clipboard.writeText("
                    f"{safe_js});}}"
                    "catch(e){var t=document.createElement('textarea');"
                    f"t.value={safe_js};"
                    "document.body.appendChild(t);t.select();"
                    "document.execCommand('copy');"
                    "document.body.removeChild(t);}"
                    "</script>",
                    height=0,
                )
    with download_col:
        if prev_response:
            st.download_button(
                label=":material/download:",
                type="tertiary",
                help="Download translation",
                data=prev_response,
                file_name="translation.txt",
                mime="text/plain",
                key="download_text",
            )

if translate_clicked:
    if not text.strip():
        st.warning("Please enter text to translate.")
    else:
        try:
            result = translate(
                text,
                source,
                LANGUAGES[source],
                target,
                LANGUAGES[target],
            )
            st.session_state["translation_result"] = result
            st.rerun()
        except Exception as e:
            logger.exception("Translation failed")
            st.error(f"Translation failed: {e}")
