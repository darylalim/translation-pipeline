# Translation Pipeline

Streamlit web application for translating text using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) model.

## Commands

- `uv sync` — install dependencies
- `uv run streamlit run streamlit_app.py` — run application
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run ty check` — typecheck
- `uv run pytest` — run tests
- `uv run pytest tests/path_to_test.py::test_name -v` — run single test

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations for all parameters and returns
- Dataclasses for structured data
- Formatting and import sorting handled by ruff

## Dependencies

- `transformers` — model loading
- `torch` — tensor operations
- `streamlit` — web UI
- `accelerate` — multi-device support
- `huggingface_hub` — HF Inference API client

## Architecture

### Language Configuration

`LANGUAGES` maps language name to `(BCP-47 code, bidirectional)` tuple. `SOURCE_LANGS` and `TARGET_LANGS` are derived from `LANGUAGES`.

- **Bidirectional with English**: Cantonese (yue), Chinese (zh-CN), Chuukese (chk), Ilocano (ilo), Japanese (ja), Korean (ko), Marshallese (mh), Spanish (es), Thai (th), Tonga (to), Vietnamese (vi)
- **English-only target**: Filipino (fil), Hawaiian (haw), Samoan (sm)
- Chinese is Mandarin Chinese; Filipino is Tagalog

### Authentication

`HF_TOKEN` is resolved from `st.secrets` (`.streamlit/secrets.toml` locally, Streamlit Secrets on cloud). If not found, the UI prompts the user to enter a token via `st.text_input`. The app stops until a valid token is provided.

### Prompt Construction

`build_prompt(text, src_lang, src_code, tgt_lang, tgt_code)` constructs the full Gemma chat-formatted prompt with the translation instruction and source text. Shared by both backends.

### Backend Detection

`has_gpu()` returns `True` if `torch.cuda.is_available()` or `torch.backends.mps.is_available()`. Used to auto-select the inference backend.

### Model Loading

`load_model(token)` takes an HF token and returns `(model, processor, eos_token_id, load_duration)`. Only called on the local (GPU) path.

- Cached with `@st.cache_resource`
- Uses `device_map="auto"` and `dtype=torch.bfloat16`

### Translation

`translate(...)` calls `build_prompt()`, then dispatches to `_translate_local()` or `_translate_api()` based on `has_gpu()`. Returns a frozen `TranslationResult` dataclass with fields: `response`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`.

- **`_translate_local(prompt, token)`** — loads model via `load_model()`, tokenizes, runs `model.generate()`. All timing fields populated.
- **`_translate_api(prompt, token)`** — uses `InferenceClient.text_generation()` with `details=True`. `prompt_eval_duration` and `eval_duration` are `0` (API doesn't expose timing). `prompt_eval_count` from `len(output.details.prefill)`, `eval_count` from `output.details.generated_tokens`.

### UI Layout

- `st.set_page_config(page_title="Translation Pipeline", page_icon="🌐")` must be the first Streamlit call
- Language selectors use a 3-column `[5, 1, 5]` layout with a swap button in the middle column
- Input/output uses a 2-column side-by-side layout; input is `st.text_area`, output is `st.code(language=None)` for built-in copy button
- Labels use native Streamlit widget labels; the translation label uses an inline HTML `<label>` styled at `0.875rem` to match
- `st.session_state` keys: `source_lang`, `target_lang`, `translation_result`, `total_duration`, `load_duration`, `used_gpu`
- Swap button is disabled when the target language is unidirectional (English-only target)

### Output

Metrics and JSON fields vary by backend. `st.session_state["used_gpu"]` determines which set is shown.

- **Local (GPU)** — all fields: `model`, `response`, `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`
- **API** — omits zero-valued duration fields: `model`, `response`, `total_duration`, `prompt_eval_count`, `eval_count`

All durations are `int` in nanoseconds via `time.perf_counter_ns()`. Wrapped in `st.expander("Performance details")` with model name as `st.caption`, a 4-column grid of `st.metric` widgets, and a download button.

**Metric display labels** use friendly names; **JSON keys** use the original field names:

| Display Label | JSON Key |
|---|---|
| Total Time | `total_duration` |
| Model Load Time | `load_duration` |
| Input Tokens | `prompt_eval_count` |
| Input Processing Time | `prompt_eval_duration` |
| Output Tokens | `eval_count` |
| Generation Time | `eval_duration` |

## Known Issues

### Do NOT use `processor.apply_chat_template`

Fails at runtime for TranslateGemma. Structured content raises `AttributeError`, plain text is rejected. Manually construct the prompt and tokenize with `processor.tokenizer`:

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
```

### Do NOT use `processor.decode`

Use `processor.tokenizer(...)` and `processor.tokenizer.decode(...)`. Do not use `processor.decode(...)` or `processor.apply_chat_template(...)`.

### Override `top_p` and `top_k` for greedy decoding

Pass `top_p=None, top_k=None` to `model.generate()` when using `do_sample=False` to suppress warnings.

## Prompt Template

Variables: `source_lang` (e.g. English), `src_lang_code` (e.g. en), `target_lang` (e.g. German), `tgt_lang_code` (e.g. de-DE), `text`.

```
You are a professional {source_lang} ({src_lang_code}) to {target_lang}
({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and
nuances of the original {source_lang} text while adhering to {target_lang} grammar,
vocabulary, and cultural sensitivities. Produce only the {target_lang}
translation, without any additional explanations or commentary. Please translate
the following {source_lang} text into {target_lang}:\n\n\n{text}
```

## Example

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "google/translategemma-4b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
eos_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")

instruction = (
    "You are a professional Czech (cs) to German (de-DE) translator. "
    "Your goal is to accurately convey the meaning and nuances of the original Czech text "
    "while adhering to German grammar, vocabulary, and cultural sensitivities. "
    "Produce only the German translation, without any additional explanations or commentary. "
    "Please translate the following Czech text into German:\n\n\n"
    "V nejhorším případě i k prasknutí čočky."
)
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
input_len = inputs["input_ids"].shape[1]

with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        top_p=None,
        top_k=None,
        eos_token_id=eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

generated = output[0][input_len:]
decoded = processor.tokenizer.decode(generated, skip_special_tokens=True)
print(decoded)
```

## Resources

- [Technical Report](https://arxiv.org/pdf/2601.09012)
- [Gemma Cookbook](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Research/[TranslateGemma]Example.ipynb)
