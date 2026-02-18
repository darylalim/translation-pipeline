# Translation Pipeline

Streamlit web application for translating text using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) model.

## Commands

- **Install dependencies**: `uv sync`
- **Run application**: `uv run streamlit run streamlit_app.py`
- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Run tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/path_to_test.py::test_name -v`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations for all parameters and returns
- Use dataclasses for structured data
- Formatting and import sorting handled by ruff

## Dependencies

- `transformers` — model loading
- `torch` — tensor operations
- `streamlit` — web UI
- `accelerate` — multi-device support
- `python-dotenv` — environment variables

## Architecture

### Language Configuration

`LANGUAGES` maps language name to `(BCP-47 code, bidirectional)` tuple. `SOURCE_LANGS` and `TARGET_LANGS` are derived from `LANGUAGES`.

Bidirectional with English: Cantonese (yue), Chinese (zh-CN), Chuukese (chk), Ilocano (ilo), Japanese (ja), Korean (ko), Marshallese (mh), Spanish (es), Thai (th), Tonga (to), Vietnamese (vi)

English-only target: Filipino (fil), Hawaiian (haw), Samoan (sm)

Notes: Chinese is Mandarin Chinese. Filipino is Tagalog.

### Model Loading

`load_model()` returns `(model, processor, eos_token_id, load_duration)`. Cached with `@st.cache_resource`. Uses `device_map="auto"` and `dtype=torch.bfloat16`. Reads `HF_TOKEN` from environment.

### Translation

`translate()` calls `load_model()` internally (cached) and returns a frozen `TranslationResult` dataclass with fields: `response`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`.

### Download JSON

Includes: `model` (string), `response` (string), `total_duration` (int, ns), `load_duration` (int, ns), `prompt_eval_count` (int), `prompt_eval_duration` (int, ns), `eval_count` (int), `eval_duration` (int, ns).

All durations measured with `time.perf_counter_ns()`.

### Metrics

Displays all JSON fields except `response` using `st.metric`.

## Known Issues

### Do NOT use `processor.apply_chat_template`

Fails at runtime for TranslateGemma. Structured content raises `AttributeError`, plain text is rejected. Instead, manually construct the prompt and tokenize with `processor.tokenizer`:

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
```

### Override `top_p` and `top_k` for greedy decoding

Pass `top_p=None, top_k=None` to `model.generate()` when using `do_sample=False` to suppress warnings.

### Use `processor.tokenizer` for tokenization and decoding

Use `processor.tokenizer(...)` and `processor.tokenizer.decode(...)`. Do not use `processor.decode(...)` or `processor.apply_chat_template(...)`.

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
