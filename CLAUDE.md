# TranslateGemma Translate

Streamlit app for translating text using [TranslateGemma 4B 8-bit](https://huggingface.co/mlx-community/translategemma-4b-it-8bit) with Apple Silicon inference via MLX.

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
- Formatting and import sorting handled by ruff

## Dependencies

- `streamlit` — web UI
- `mlx-lm` — model loading and inference on Apple Silicon

## Architecture

### Languages

`LANGUAGES` maps language name to BCP-47 code. `SOURCE_LANGS` is derived as a sorted list of all language names.

All languages are bidirectional with English: Chinese (zh), Dutch (nl), French (fr), German (de), Indonesian (id), Italian (it), Spanish (es), Vietnamese (vi).

### Model Loading

`load_model()` returns `(model, tokenizer)`. Cached with `@st.cache_resource`. Uses `mlx_lm.load()`. Registers `<end_of_turn>` as an EOS token via `tokenizer.add_eos_token()` so generation stops early instead of running to `MAX_NEW_TOKENS`.

### Translation

`translate(text, src_lang, src_code, tgt_lang, tgt_code)` builds the prompt, loads the model, and runs `mlx_lm.generate()`. Generation stops at `<end_of_turn>` via the registered EOS token. A safety-net split on `<end_of_turn>` strips the token if it leaks into the output string. Returns a `str` — the translated text. `MAX_NEW_TOKENS` (512) limits the maximum output length.

### UI

- Language selectors: 3-column `[10, 1, 10]` layout with swap button (`:material/swap_horiz:`) in the middle, labels hidden via `label_visibility="collapsed"`
- Swap button moves translation output to source input and clears the result
- 2-column side-by-side; `st.text_area` (no placeholder, `max_chars=5000`, height 300) for input, disabled `st.text_area` (placeholder "Translation", height 300) for output
- Output text areas use `st.session_state` to set value (not the `value` parameter) to avoid stale widget state
- Button groups inside their respective panel columns — left (inside `left_col`): translate button (primary) + clear button (`:material/close:`) in `[3, 1]`; right (inside `right_col`): copy button (`:material/content_copy:`) + download button (`:material/download:`) in `[1, 1]`
- All icon buttons use `type="tertiary"` (no outline) with tooltip via `help`
- Copy uses `streamlit.components.v1.html` with JS clipboard API
- Download uses `st.download_button` with `mime="text/plain"`
- `st.session_state` keys: `source_lang`, `target_lang`, `translation_result`, `source_text`, `text_output`

## Known Issues

### Do NOT use `tokenizer.apply_chat_template` for text translation

Fails at runtime for TranslateGemma text translation. Manually construct the prompt string.

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
```

### Strip `<end_of_turn>` from model output

`load_model()` registers `<end_of_turn>` as an EOS token so generation stops early. The `translate()` split on `<end_of_turn>` is kept as a safety net in case the token leaks into the output string.

### Use `zh` not `zh-CN` for Chinese

TranslateGemma's chat template does not include `zh-CN` in its language dict. Use `zh` instead.

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
