# Local-Only Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove HF Inference API backend and cloud deployment code; keep local GPU inference only with no auth UI.

**Architecture:** Single `translate()` function with inline local inference logic. `load_model()` cached with `@st.cache_resource`, no token param. No API fallback, no `has_gpu()` branching.

**Tech Stack:** Streamlit, PyTorch, Transformers, Accelerate

---

### Task 1: Refactor streamlit_app.py

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Write the refactored streamlit_app.py**

Replace the entire file with this (deletions: `huggingface_hub` import, `has_gpu()`, `_translate_local()`, `_translate_api()`, auth UI, `used_gpu` branching; modifications: `load_model()` and `translate()` drop `token` param, translate inlines local logic, metrics always show full GPU path):

```python
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import streamlit as st
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "google/translategemma-4b-it"
MAX_NEW_TOKENS = 512

# Language name -> (BCP-47 code, bidirectional with English)
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
            with st.status("Translating...", expanded=True) as status:
                st.write("Running on local GPU...")
                t0 = time.perf_counter_ns()
                result = translate(
                    text,
                    source,
                    LANGUAGES[source][0],
                    target,
                    LANGUAGES[target][0],
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
            st.rerun()
        except Exception as e:
            logger.exception("Translation failed")
            st.error(f"Translation failed: {e}")

# --- Metrics in expander ---
if "translation_result" in st.session_state:
    result = st.session_state["translation_result"]
    total_duration = st.session_state["total_duration"]
    load_duration = st.session_state["load_duration"]

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

    with st.expander("Performance details"):
        st.caption(f"Model: {MODEL_ID}")
        cols = st.columns(4)
        for i, (label, value) in enumerate(metrics):
            cols[i % 4].metric(label, value)

        st.download_button(
            "Download JSON", json.dumps(data, indent=2), "translation.json"
        )
```

**Step 2: Verify file saved correctly**

Run: `head -5 streamlit_app.py`
Expected: Should show `import json` as first line, no `huggingface_hub` import.

---

### Task 2: Remove huggingface_hub dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Remove huggingface_hub from dependencies**

In `pyproject.toml`, remove the `"huggingface_hub",` line from `dependencies`.

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Resolves without `huggingface_hub` in the dependency tree.

---

### Task 3: Update test fixtures

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Write the updated conftest.py**

Remove `mock_huggingface_hub`, `patched_translate_api` fixture, and simplify `patched_translate_local` to `patched_translate` (no `has_gpu` patch, no token):

```python
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def app_module():
    """Import streamlit_app with all heavy dependencies mocked."""
    mock_st = MagicMock()
    mock_st.cache_resource = lambda f: f
    mock_st.session_state = {}

    col1, col_swap, col2 = MagicMock(), MagicMock(), MagicMock()
    col1.selectbox.return_value = "English"
    col2.selectbox.return_value = "Spanish"
    left_col, right_col = MagicMock(), MagicMock()
    mock_st.columns.side_effect = [
        (col1, col_swap, col2),
        (left_col, right_col),
    ]
    mock_st.button.return_value = False

    mock_torch = MagicMock()
    mock_transformers = MagicMock()

    patches = {
        "streamlit": mock_st,
        "torch": mock_torch,
        "transformers": mock_transformers,
    }

    originals = {}
    for mod_name, mock_obj in patches.items():
        originals[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = mock_obj

    if "streamlit_app" in sys.modules:
        del sys.modules["streamlit_app"]
    module = importlib.import_module("streamlit_app")

    for mod_name, orig in originals.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig

    return module


@pytest.fixture()
def mock_processor():
    """MagicMock processor with configured tokenizer."""
    processor = MagicMock()

    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 10)

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: (
        mock_input_ids if key == "input_ids" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs

    processor.tokenizer.return_value = mock_inputs
    processor.tokenizer.decode.return_value = "  translated text  "
    processor.tokenizer.pad_token_id = 0
    processor.tokenizer.convert_tokens_to_ids.return_value = 107

    return processor


@pytest.fixture()
def mock_model():
    """MagicMock model with configured generate output."""
    model = MagicMock()
    model.device = "cpu"

    mock_generated = MagicMock()
    mock_generated.shape = (5,)

    mock_output_sequence = MagicMock()
    mock_output_sequence.__getitem__ = lambda self, key: mock_generated
    mock_output = MagicMock()
    mock_output.__getitem__ = lambda self, key: mock_output_sequence

    model.generate.return_value = mock_output

    return model


@pytest.fixture()
def patched_translate(app_module, mock_model, mock_processor):
    """Patch load_model for translation tests."""
    with patch.object(
        app_module,
        "load_model",
        return_value=(mock_model, mock_processor, 107, 5_000_000),
    ):
        yield {
            "translate": app_module.translate,
            "model": mock_model,
            "processor": mock_processor,
        }
```

---

### Task 4: Update tests

**Files:**
- Modify: `tests/test_streamlit_app.py`

**Step 1: Write the updated test file**

Delete `TestHasGpu`, `TestTranslateApi`, `TestTranslateDispatch`. Rename `TestTranslateLocal` to `TestTranslate`, use `patched_translate` fixture, remove token arg from all `translate()` calls. Update `TestLoadModel` to test without token param.

```python
from dataclasses import asdict, is_dataclass
from unittest.mock import MagicMock, patch


class TestConstants:
    def test_model_id(self, app_module):
        assert app_module.MODEL_ID == "google/translategemma-4b-it"

    def test_max_new_tokens(self, app_module):
        assert app_module.MAX_NEW_TOKENS == 512


class TestLanguageConfiguration:
    def test_languages_has_15_entries(self, app_module):
        assert len(app_module.LANGUAGES) == 15

    def test_languages_values_are_str_bool_tuples(self, app_module):
        for name, value in app_module.LANGUAGES.items():
            assert isinstance(value, tuple), f"{name}: expected tuple"
            code, bi = value
            assert isinstance(code, str), f"{name}: code should be str"
            assert isinstance(bi, bool), f"{name}: bidirectional should be bool"

    def test_12_bidirectional_languages(self, app_module):
        bi_count = sum(1 for _, (_, bi) in app_module.LANGUAGES.items() if bi)
        assert bi_count == 12

    def test_3_unidirectional_languages(self, app_module):
        uni_count = sum(1 for _, (_, bi) in app_module.LANGUAGES.items() if not bi)
        assert uni_count == 3

    def test_filipino_is_unidirectional(self, app_module):
        assert app_module.LANGUAGES["Filipino"][1] is False

    def test_hawaiian_is_unidirectional(self, app_module):
        assert app_module.LANGUAGES["Hawaiian"][1] is False

    def test_samoan_is_unidirectional(self, app_module):
        assert app_module.LANGUAGES["Samoan"][1] is False

    def test_bcp47_codes(self, app_module):
        expected = {
            "English": "en",
            "Cantonese": "yue",
            "Chinese": "zh-CN",
            "Chuukese": "chk",
            "Ilocano": "ilo",
            "Japanese": "ja",
            "Korean": "ko",
            "Marshallese": "mh",
            "Spanish": "es",
            "Filipino": "fil",
        }
        for name, expected_code in expected.items():
            assert app_module.LANGUAGES[name][0] == expected_code, (
                f"{name} code mismatch"
            )

    def test_source_langs_has_12_entries(self, app_module):
        assert len(app_module.SOURCE_LANGS) == 12

    def test_source_langs_is_sorted(self, app_module):
        assert app_module.SOURCE_LANGS == sorted(app_module.SOURCE_LANGS)

    def test_source_langs_contains_english(self, app_module):
        assert "English" in app_module.SOURCE_LANGS

    def test_source_langs_excludes_unidirectional(self, app_module):
        for name in ("Filipino", "Hawaiian", "Samoan"):
            assert name not in app_module.SOURCE_LANGS

    def test_target_langs_keys_match_source_langs(self, app_module):
        assert set(app_module.TARGET_LANGS.keys()) == set(app_module.SOURCE_LANGS)

    def test_english_targets_14_sorted_languages(self, app_module):
        targets = app_module.TARGET_LANGS["English"]
        assert len(targets) == 14
        assert targets == sorted(targets)
        assert "Filipino" in targets
        assert "Hawaiian" in targets
        assert "Samoan" in targets

    def test_non_english_targets_only_english(self, app_module):
        for name, targets in app_module.TARGET_LANGS.items():
            if name != "English":
                assert targets == ["English"], f"{name} should only target English"


class TestTranslationResult:
    def test_field_access(self, app_module):
        result = app_module.TranslationResult(
            response="hello",
            prompt_eval_count=10,
            prompt_eval_duration=100,
            eval_count=5,
            eval_duration=200,
        )
        assert result.response == "hello"
        assert result.prompt_eval_count == 10
        assert result.prompt_eval_duration == 100
        assert result.eval_count == 5
        assert result.eval_duration == 200

    def test_asdict(self, app_module):
        result = app_module.TranslationResult(
            response="hello",
            prompt_eval_count=10,
            prompt_eval_duration=100,
            eval_count=5,
            eval_duration=200,
        )
        d = asdict(result)
        assert d == {
            "response": "hello",
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100,
            "eval_count": 5,
            "eval_duration": 200,
        }

    def test_equality(self, app_module):
        kwargs = {
            "response": "hello",
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100,
            "eval_count": 5,
            "eval_duration": 200,
        }
        assert app_module.TranslationResult(**kwargs) == app_module.TranslationResult(
            **kwargs
        )

    def test_is_dataclass(self, app_module):
        assert is_dataclass(app_module.TranslationResult)


class TestBuildPrompt:
    def test_contains_language_names(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert "English" in prompt
        assert "Spanish" in prompt

    def test_contains_language_codes(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert "(en)" in prompt
        assert "(es)" in prompt

    def test_contains_source_text(self, app_module):
        prompt = app_module.build_prompt(
            "Translate me", "English", "en", "Japanese", "ja"
        )
        assert "Translate me" in prompt

    def test_uses_gemma_chat_format(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert prompt.startswith("<start_of_turn>user\n")
        assert "<end_of_turn>\n<start_of_turn>model\n" in prompt

    def test_returns_string(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert isinstance(prompt, str)


class TestTranslate:
    def test_returns_translation_result(self, app_module, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        assert isinstance(result, app_module.TranslationResult)

    def test_response_is_stripped(self, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        assert result.response == "translated text"

    def test_tokenizer_called_with_correct_args(self, patched_translate):
        patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        call_kwargs = patched_translate["processor"].tokenizer.call_args[1]
        assert call_kwargs["return_tensors"] == "pt"
        assert call_kwargs["add_special_tokens"] is True

    def test_inputs_moved_to_model_device(self, patched_translate):
        patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        patched_translate[
            "processor"
        ].tokenizer.return_value.to.assert_called_with("cpu")

    def test_generate_called_with_correct_args(self, patched_translate):
        patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        call_kwargs = patched_translate["model"].generate.call_args[1]
        assert call_kwargs["do_sample"] is False
        assert call_kwargs["max_new_tokens"] == 512
        assert call_kwargs["top_p"] is None
        assert call_kwargs["top_k"] is None
        assert call_kwargs["eos_token_id"] == 107
        assert call_kwargs["pad_token_id"] == 0

    def test_decode_uses_skip_special_tokens(self, patched_translate):
        patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        call_kwargs = patched_translate["processor"].tokenizer.decode.call_args[1]
        assert call_kwargs["skip_special_tokens"] is True

    def test_prompt_eval_count_matches_input_length(self, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        assert result.prompt_eval_count == 10

    def test_eval_count_matches_generated_length(self, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        assert result.eval_count == 5

    def test_timing_durations_are_non_negative_ints(self, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        assert isinstance(result.prompt_eval_duration, int)
        assert isinstance(result.eval_duration, int)
        assert result.prompt_eval_duration >= 0
        assert result.eval_duration >= 0

    def test_generate_called_exactly_once(self, patched_translate):
        patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es",
        )
        assert patched_translate["model"].generate.call_count == 1


class TestLoadModel:
    def test_returns_4_tuple(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            result = app_module.load_model()
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_processor_loaded_with_correct_args(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            app_module.load_model()
        call_args, call_kwargs = MockAutoProc.from_pretrained.call_args
        assert call_args[0] == "google/translategemma-4b-it"
        assert call_kwargs["use_fast"] is True
        assert "token" not in call_kwargs

    def test_model_loaded_with_correct_args(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            app_module.load_model()
        call_args, call_kwargs = MockAutoModel.from_pretrained.call_args
        assert call_args[0] == "google/translategemma-4b-it"
        assert call_kwargs["device_map"] == "auto"
        assert call_kwargs["dtype"] == app_module.torch.bfloat16
        assert "token" not in call_kwargs

    def test_no_token_passed(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            app_module.load_model()
        assert "token" not in MockAutoProc.from_pretrained.call_args[1]
        assert "token" not in MockAutoModel.from_pretrained.call_args[1]

    def test_eos_token_extracted(self, app_module):
        mock_proc = self._make_processor(eos_id=999)
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            _, _, eos_token_id, _ = app_module.load_model()
        mock_proc.tokenizer.convert_tokens_to_ids.assert_called_with("<end_of_turn>")
        assert eos_token_id == 999

    def test_load_duration_is_non_negative_int(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            _, _, _, load_duration = app_module.load_model()
        assert isinstance(load_duration, int)
        assert load_duration >= 0

    def test_controlled_timing(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
            patch.object(app_module.time, "perf_counter_ns", side_effect=[1000, 6000]),
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            _, _, _, load_duration = app_module.load_model()
        assert load_duration == 5000

    @staticmethod
    def _make_processor(eos_id=107):
        proc = MagicMock()
        proc.tokenizer.convert_tokens_to_ids.return_value = eos_id
        return proc

    @staticmethod
    def _make_model():
        return MagicMock()
```

**Step 2: Run tests**

Run: `uv run pytest -v`
Expected: All tests pass. No `TestHasGpu`, `TestTranslateApi`, or `TestTranslateDispatch` classes.

---

### Task 5: Lint and format

**Files:** All modified files

**Step 1: Run ruff format**

Run: `uv run ruff format .`

**Step 2: Run ruff check**

Run: `uv run ruff check .`
Expected: No errors.

**Step 3: Run type check**

Run: `uv run ty check`
Expected: No new errors.

**Step 4: Re-run tests after formatting**

Run: `uv run pytest -v`
Expected: All tests still pass.

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Apply these changes to the existing CLAUDE.md:

- **Dependencies section**: Remove `huggingface_hub` line
- **Authentication section**: Replace with "`HF_TOKEN` loaded from `.env` via `python-dotenv`. Startup check with `st.error`/`st.stop` if missing."
- **Backend Detection section**: Remove entirely (no `has_gpu()`)
- **Model Loading section**: Update `load_model()` signature to no params, remove "takes an HF token" text
- **Translation section**: Remove dispatch logic description, describe `translate()` as directly running local inference. Remove `_translate_local()` and `_translate_api()` bullet points. Drop `token` from signature.
- **UI Layout section**: Remove `used_gpu` from session_state keys list
- **Output section**: Remove "Metrics and JSON fields vary by backend" and the API bullet. Keep only the local/GPU fields. Remove `st.session_state["used_gpu"]` reference.

---

### Task 7: Commit

**Step 1: Stage and commit all changes**

```bash
git add streamlit_app.py pyproject.toml tests/conftest.py tests/test_streamlit_app.py CLAUDE.md
git commit -m "Refactor to local GPU inference only, remove API backend and auth UI"
```
