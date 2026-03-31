# TranslateGemma Translate

Streamlit app for translating text and images using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) model with local GPU inference.

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

- `streamlit` — web UI
- `transformers` — model loading
- `torch` — tensor operations
- `torchvision` — fast image processor
- `accelerate` — multi-device support
- `python-dotenv` — `.env` file loading
- `Pillow` — image processing

## Architecture

### Authentication

`HF_TOKEN` is loaded from `.env` via `python-dotenv`. The app checks for it at startup and stops with `st.error` if missing. The `transformers` library reads it automatically from the environment — no explicit token passing in app code.

### Languages

`LANGUAGES` maps language name to BCP-47 code. `SOURCE_LANGS` is derived as a sorted list of all language names.

All languages are bidirectional with English: Chinese (zh), Dutch (nl), French (fr), German (de), Indonesian (id), Italian (it), Spanish (es), Vietnamese (vi).

### Model Loading

`load_model()` returns `(model, processor, eos_token_id, load_duration)`. Cached with `@st.cache_resource`. Uses `device_map="auto"` and `dtype=torch.bfloat16`.

### Translation

`translate(text, src_lang, src_code, tgt_lang, tgt_code)` builds the prompt, loads the model, tokenizes, and runs `model.generate()`. Returns a frozen `TranslationResult` dataclass:

- `response` — translated text
- `prompt_eval_count` — input token count
- `prompt_eval_duration` — tokenization time (ns)
- `eval_count` — output token count
- `eval_duration` — generation time (ns)

### Image Translation

`translate_image(image, src_code, tgt_code)` uses `processor.apply_chat_template` with the image message format. Unlike text translation (which manually constructs the prompt due to `apply_chat_template` failures), image translation uses the template's image code path which works correctly. The image is passed inside the message content dict — do NOT pass `images=` separately, as `apply_chat_template` extracts images automatically when `tokenize=True`.

Input images are normalized to 896x896 resolution and encoded to 256 tokens each. Total input context is 2000 tokens.

Accepted image types: PNG, JPG, JPEG, WEBP.

### UI

- Input mode: `st.tabs(["Text", "Images"])` — tabs appear first, language selectors are inside each tab
- Language selectors: 3-column `[10, 1, 10]` layout with swap button (`:material/swap_horiz:`) in the middle, labels hidden via `label_visibility="collapsed"`
- Language state shared between tabs via canonical `source_lang`/`target_lang` session state with tab-specific widget keys synced via `on_change` callbacks
- Swap button moves translation output to source input and clears the result
- Text tab: 2-column side-by-side; `st.text_area` (no placeholder, `max_chars=5000`, height 300) for input, disabled `st.text_area` (placeholder "Translation", height 300) for output
- Output text areas use `st.session_state` to set value (not the `value` parameter) to avoid stale widget state
- Below text input: translate button (primary, left) and clear button (`:material/close:`, right) on same row `[3, 1, 6]`
- Below text/image output: copy button (`:material/content_copy:`), download button (`:material/download:`), right-aligned `[18, 1, 1]`
- All icon buttons use `type="tertiary"` (no outline) with tooltip via `help`
- Image tab: `st.file_uploader` + caption for supported types + `st.image` preview (left), disabled `st.text_area` output (right)
- Translate button: primary, left-aligned (both tabs)
- Copy uses `streamlit.components.v1.html` with JS clipboard API
- Download uses `st.download_button` with `mime="text/plain"`
- `st.session_state` keys: `source_lang`, `target_lang`, `translation_result`, `image_translation_result`, `source_text`, `text_output`, `image_output`
- `st.session_state` keys (tab-specific widget keys): `text_source_lang`, `text_target_lang`, `image_source_lang`, `image_target_lang`

## Known Issues

### Do NOT use `processor.apply_chat_template` for text translation

Fails at runtime for TranslateGemma text translation. Manually construct the prompt and tokenize with `processor.tokenizer`. Note: `apply_chat_template` **does** work for image translation (different code path).

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
```

### Do NOT use `processor.decode`

Use `processor.tokenizer(...)` and `processor.tokenizer.decode(...)`.

### Do NOT pass `images=` separately to `apply_chat_template`

When `tokenize=True`, the processor extracts images from the content items automatically. Passing `images=[image]` as a separate parameter causes a conflict.

### Use `zh` not `zh-CN` for Chinese

TranslateGemma's chat template does not include `zh-CN` in its language dict. Use `zh` instead.

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
