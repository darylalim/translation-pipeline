# Local-Only Refactor Design

## Goal

Remove HF Inference API backend and cloud deployment support. Keep local GPU inference only. Remove auth UI. Make the codebase performant and concise.

## Decisions

- **Auth**: `HF_TOKEN` loaded from `.env` via `python-dotenv`. Startup check with `st.error`/`st.stop` if missing.
- **Metrics**: Keep all 5 `TranslationResult` fields (all meaningful for local inference).
- **Approach**: Surgical deletion — remove dead code, keep clean function boundaries.

## Changes

### streamlit_app.py

**Delete:**
- `from huggingface_hub import InferenceClient`
- `has_gpu()` function
- `_translate_api()` function
- `_translate_local()` function (logic moves into `translate()`)
- Auth UI block (secrets check, token input, `st.stop()`)
- `used_gpu` session state key and all branching on it
- `has_gpu()` call in backend detection section
- API-specific metrics path (the `else` branch in metrics display)

**Modify:**
- `load_model()`: drop `token` param, remove `token=` from `from_pretrained` calls
- `translate()`: drop `token` param, inline the local inference logic directly
- Backend detection section: unconditionally call `load_model()` (no `has_gpu()` check)
- Translation handler: remove `token` arg from `translate()` call, remove `used_gpu` from session state
- Metrics display: remove `used_gpu` branching, always show full 6-metric GPU layout

### pyproject.toml

- Remove `huggingface_hub` from dependencies

### tests/conftest.py

- Remove `mock_huggingface_hub` from patches
- Remove `patched_translate_api` fixture
- Remove `has_gpu` patch from `patched_translate_local` (no longer exists)
- Remove `token` from fixture yields
- Update `load_model` mock to match new 3-tuple return `(model, processor, eos_token_id)`

### tests/test_streamlit_app.py

- Delete `TestHasGpu` class
- Delete `TestTranslateApi` class
- Delete `TestTranslateDispatch` class
- Update `TestTranslateLocal` to call `translate()` without token param
- Update `TestLoadModel` to test without token param, 3-tuple return

### CLAUDE.md

- Remove API backend references
- Remove `huggingface_hub` from dependencies list
- Remove auth section about `st.secrets` and token input
- Remove backend detection section
- Remove API-specific output/metrics documentation
- Simplify `translate()` and `load_model()` signatures

## Files touched

1. `streamlit_app.py`
2. `pyproject.toml`
3. `tests/conftest.py`
4. `tests/test_streamlit_app.py`
5. `CLAUDE.md`
