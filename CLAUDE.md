# Translation Pipeline

Streamlit app for translating text using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) model with local GPU inference.

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
- `python-dotenv` — `.env` file loading

## Architecture

### Authentication

`HF_TOKEN` is loaded from `.env` via `python-dotenv`. The app checks for it at startup and stops with `st.error` if missing. The `transformers` library reads it automatically from the environment — no explicit token passing in app code.

### Languages

`LANGUAGES` maps language name to `(BCP-47 code, bidirectional)` tuple. `SOURCE_LANGS` and `TARGET_LANGS` are derived from `LANGUAGES`.

- **Bidirectional**: Cantonese (yue), Chinese (zh-CN), Chuukese (chk), Ilocano (ilo), Japanese (ja), Korean (ko), Marshallese (mh), Spanish (es), Thai (th), Tonga (to), Vietnamese (vi)
- **English-only target**: Filipino (fil), Hawaiian (haw), Samoan (sm)
- Chinese is Mandarin Chinese; Filipino is Tagalog

### Model Loading

`load_model()` returns `(model, processor, eos_token_id, load_duration)`. Cached with `@st.cache_resource`. Uses `device_map="auto"` and `dtype=torch.bfloat16`.

### Translation

`translate(text, src_lang, src_code, tgt_lang, tgt_code)` builds the prompt, loads the model, tokenizes, and runs `model.generate()`. Returns a frozen `TranslationResult` dataclass:

- `response` — translated text
- `prompt_eval_count` — input token count
- `prompt_eval_duration` — tokenization time (ns)
- `eval_count` — output token count
- `eval_duration` — generation time (ns)

### UI

- Language selectors: 3-column `[5, 1, 5]` layout with swap button in the middle
- Input/output: 2-column side-by-side; `st.text_area` for input, `st.code(language=None)` for output
- Translation label uses inline HTML `<label>` styled at `0.875rem` to match native widget labels
- Swap button disabled when target is unidirectional (English-only target)
- `st.session_state` keys: `source_lang`, `target_lang`, `translation_result`, `total_duration`, `load_duration`

### Output

Performance details in `st.expander` with `st.metric` widgets (4-column grid) and JSON download.

| Display Label | JSON Key |
|---|---|
| Total Time | `total_duration` |
| Model Load Time | `load_duration` |
| Input Tokens | `prompt_eval_count` |
| Input Processing Time | `prompt_eval_duration` |
| Output Tokens | `eval_count` |
| Generation Time | `eval_duration` |

All durations are `int` nanoseconds via `time.perf_counter_ns()`.

## Known Issues

### Do NOT use `processor.apply_chat_template`

Fails at runtime for TranslateGemma. Manually construct the prompt and tokenize with `processor.tokenizer`:

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
```

### Do NOT use `processor.decode`

Use `processor.tokenizer(...)` and `processor.tokenizer.decode(...)`.

### Override `top_p` and `top_k` for greedy decoding

Pass `top_p=None, top_k=None` to `model.generate()` with `do_sample=False` to suppress warnings.

## Prompt Template

```
You are a professional {source_lang} ({src_lang_code}) to {target_lang}
({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and
nuances of the original {source_lang} text while adhering to {target_lang} grammar,
vocabulary, and cultural sensitivities. Produce only the {target_lang}
translation, without any additional explanations or commentary. Please translate
the following {source_lang} text into {target_lang}:\n\n\n{text}
```

## Resources

- [Technical Report](https://arxiv.org/pdf/2601.09012)
- [Gemma Cookbook](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Research/[TranslateGemma]Example.ipynb)
