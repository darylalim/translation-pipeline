# TranslateGemma Studio

Streamlit application for translation using [Google TranslateGemma](https://huggingface.co/google/translategemma-4b-it) on Apple Silicon with MLX.

## Commands

- `uv sync` — install dependencies
- `uv run streamlit run streamlit_app.py` — run application
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run ty check` — typecheck
- `uv run pytest` — run tests
- `uv run pytest tests/path_to_test.py::test_name -v` — run single test
- `uv run pytest -m live` — run the live-model test (loads the real quant; deselected by default)
- `uv run pytest --cov` — run tests with coverage (sources configured in `pyproject.toml`)

## Code Style

- `snake_case` for functions and variables, `PascalCase` for classes
- Type annotations on all parameters and returns
- Formatting and import sorting handled by ruff
- Ruff lint rules beyond the defaults are set in `[tool.ruff.lint]` via `extend-select`: `I` (import sorting), `UP` (pyupgrade), `B` (bugbear), `C4` (comprehensions), `RUF` (ruff-specific), `SIM` (simplify) — enforced by `uv run ruff check .` and CI
- When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed

## Dependencies

- `streamlit>=1.58` — web UI; 1.58 is the release the UI conventions were written against
- `mlx-lm>=0.31.3` — model loading and inference on Apple Silicon; see Known Issues for why the floor is a patch version

Dev-group floors (`pytest>=8.4`, `ruff>=0.16`, `ty>=0.0.69`) pin the tools whose output *is* the CI contract — a lower `ruff` can format differently and a lower `ty` can emit different diagnostics, either of which fails the build.

Floors record the oldest version verified to work; `uv.lock` still pins the exact resolution. Verify a floor change with `uv lock --resolution lowest-direct && uv sync --frozen`, then run the gate (plain `uv sync` silently re-resolves to highest and hides the floor).

## Architecture

### Languages

Two dicts in `languages.py` from the TranslateGemma Technical Report (Tables 5 and 6):

- `BIDIRECTIONAL` (225) — pair with English in both directions
- `FROM_ENGLISH_ONLY` (70) — receive translations from English only

Derived constants: `ALL_LANGUAGES` (merged for name → code lookup), `SOURCE_LANGS` (sorted bidirectional names), `TARGET_LANGS_FOR_ENGLISH` (sorted non-English names from both dicts).

Directionality: bidirectional languages pair only with English (not with each other). The swap button is disabled when swapping would produce an invalid pair.

### Model Loading

`load_model()` returns `(model, tokenizer)`, cached with `@st.cache_resource`. Loads `mlx-community/translategemma-4b-it-8bit` via `mlx_lm.load()` and registers `<end_of_turn>` as an EOS token so generation stops early instead of running to the `max_tokens` cap.

The module configures `logging.basicConfig(INFO)` (silencing `httpx` to `WARNING`); both the model-load and translation failure paths call `logger.exception(...)` alongside their `st.error` callouts.

### Translation

`_prepare_generation()` builds the prompt, loads the model, enforces the token budget, and returns `(model, tokenizer, prompt, max_tokens)` — shared by both entry points:

- `translate(...)` — runs `mlx_lm.generate()`, returns `str`
- `translate_stream(...)` — generator running `mlx_lm.stream_generate()`, yields segment-by-segment

`_strip_eos_token()` removes `<end_of_turn>` from the output as a safety net for the rare case it leaks past the registered EOS.

### Context window

- `CONTEXT_WINDOW = 2048` — total context, shared by prompt and output
- `MAX_PROMPT_TOKENS = 1024` — prompt cap; `_prepare_generation()` raises `ValueError` when exceeded
- `max_tokens = CONTEXT_WINDOW - prompt_tokens` — translation gets all remaining room (EOS still stops it early)
- `MAX_INPUT_CHARS = 5000` — coarse text-area backstop; the token counter is the real, language-aware limit

`count_prompt_tokens(prompt, tokenizer)` returns the token length of the wrapped prompt — the Gemma chat scaffold (`<start_of_turn>user...`) is included, since that's what `build_prompt()` returns. The UI shows a live token count under the input and disables Translate when over budget.

### UI

- **Header** — `st.title` only
- **Layout** — default centered (no `layout=` kwarg); a focused two-panel tool relies on the readable-width cap, guarded by `test_page_layout_is_centered`
- **Language selectors** — `[10, 1, 10]` column layout with the swap button (`:material/swap_horiz:`) in the middle; labels collapsed
- **Swap button** — calls `_swap_languages()` to swap source/target and move the previous translation into the source area; disabled when target is `FROM_ENGLISH_ONLY` (the only invalid swap, since non-English sources always pair with English)
- **Body** — two side-by-side columns:
  - **Left** — `st.text_area` (`key="source_text"`, height 300, `max_chars=MAX_INPUT_CHARS`); live token counter caption with a red over-budget `st.badge`; Translate button (primary, full-width, disabled when over budget)
  - **Right** — `st.empty()` placeholder holding either the disabled output `st.text_area` (height 300) or the streaming container during generation; alignment-spacer caption; Download button (secondary, `mime="text/plain"`, disabled when no result)
- **Streaming** — Translate feeds `translate_stream()` into a fixed-height (300) `st.container`, updated token-by-token via `st.text` (raw text, not markdown — matches the text area and the `text/plain` download). On completion the result is saved to `st.session_state["translation_result"]` and `st.rerun()` reverts the placeholder to the settled text area.
- **Session state keys** — `source_lang`, `target_lang`, `translation_result`, `source_text`, `text_output`
- **State seeding** — output text areas are populated via session state (not the `value=` parameter) to avoid stale widget state
- **1.61 conventions** — buttons size with `width="stretch"` (replacing the deprecated `use_container_width`, which 1.61 still accepts but plans to remove); the page icon and the `st.error`/`st.warning` callouts use Material Symbols (`:material/...:`)

### Theme

`.streamlit/config.toml` applies a Material Design 3 theme (violet `#6750A4` primary, Roboto via Google Fonts) with `[theme.light]` and `[theme.dark]` variants, which gives the in-app light/dark switcher. The file is git-tracked — `.gitignore` keeps `config.toml` while ignoring `secrets.toml` and the rest of `.streamlit/`.

## Testing

Two mocked layers plus a config guard, ~1s combined for 87 tests at 100% coverage, and one opt-in live test that runs against the real model:

- **Import-time tests** — swap `sys.modules["streamlit"]` and `sys.modules["mlx_lm"]` for `MagicMock`s, import `streamlit_app.py`, then assert on captured `st.*` calls. No Streamlit runtime runs. Covers pure functions, layout, token counting, EOS stripping.
- **End-to-end tests** (`TestStreamingClickPath`) — drive the real script via `streamlit.testing.v1.AppTest` with only `mlx_lm` mocked. Reaches branches the import-time tests can't: streaming click path, model-load failure, runtime target filtering, swap-button wiring, empty-text warning.
- **Theme-config guard** (`TestThemeConfig`) — validates `.streamlit/config.toml` keys against Streamlit's option template (`config._config_options_template`), the same lookup the runtime uses. Catches invalid theme keys (e.g. a per-variant `base`) that Streamlit only *logs* a warning for, so they'd otherwise slip past the suite.
- **Live-model test** (`tests/test_live_model.py`, `@pytest.mark.live`) — the only test with `mlx_lm` unmocked; drives AppTest against the real 3.9 GB quant and asserts a non-empty, EOS-free translation. **Deselected by default** via `-m "not live"` in `addopts`, so neither `uv run pytest` nor CI touches it. Run it with `uv run pytest -m live` after any `mlx`/`mlx-lm` bump — the mocked layers cannot see a stack that returns empty output.

Because the app catches generation failures and renders `st.error`, the live test asserts on `at.error` as well as `at.exception`; checking only the latter turns a real failure into a downstream `KeyError`.

**Fixtures (`tests/conftest.py`):**

- `_clear_streamlit_caches` (autouse) — clears `st.cache_resource` before each test; required because Streamlit's resource cache is process-global
- `app_module` (session) — mocked-import setup for the import-time tests
- `mock_tokenizer` — `encode()` returns 50 tokens, under the budget cap
- `patched_translate` — patches `load_model`, `generate`, `stream_generate`; exposes the mocks for per-test configuration
- `fake_mlx_lm` — `mlx_lm` mock injected into `sys.modules` for AppTest fixtures
- `app_test` — AppTest pre-run to its settled state
- `app_test_unrun` — AppTest not yet run; for tests that configure mocks before the first `.run()` (e.g. load failure)

**Pytest config (`pyproject.toml`):** `addopts = ["-ra", "--strict-markers", "--strict-config"]`, `xfail_strict = true`, `filterwarnings = ["error"]`. Coverage sources in `[tool.coverage.run]`.

**CI (`.github/workflows/ci.yml`):** two jobs, `test` then `release`.

`test` — `uv sync --locked`, then `ruff check` + `ruff format --check` + `ty` + `pytest` — on `macos-latest` (an Apple Silicon image is required for `mlx-lm`) for every push to `main` and PR. `--locked` is the gate that catches a `pyproject.toml` floor bump landing without a matching `uv.lock`; `enable-cache: false` is deliberate, since `setup-uv` v9 defaults it to `"auto"` with `prune-cache` off and the unpruned cache is ~620 MB against a ~4 s install. `setup-uv` publishes no bare major tag past v7, so the version is pinned in full (`@v9.0.0`).

`release` — see below.

## Releases

Releases are cut by the `release` job in `.github/workflows/ci.yml`. **Bumping `version` in `pyproject.toml` is the entire release action** — push the bump to `main` and, once `test` passes, the job tags the commit and publishes a public GitHub Release with auto-generated notes. There is nothing to run by hand, which is the point: pushes via GitHub Desktop skip tags, so tagging had to move into CI.

- **Trigger** — `needs: test` plus `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`. PRs reach the job and skip it, and a red `test` blocks the release entirely.
- **Bump detection is tag existence, not a diff.** The job reads `[project].version` with `tomllib` and asks the remote whether `v$VERSION` is already tagged (`git ls-remote`, since checkout is shallow and fetches no tags). Diffing `pyproject.toml` against the parent commit would break on workflow re-runs, on squash merges, and on the bumps here that ride along with unrelated changes in a single commit. Tag existence answers the real question — *is this version released?* — and is idempotent, so every push to `main` between bumps is a no-op.
- **Version/lockfile drift is already covered.** `uv.lock` records the project's own version, so a bump without a matching `uv lock` fails `uv sync --locked` in `test`; the release job needs no check of its own.
- **`permissions: contents: write` is required at the job level.** The repo's `default_workflow_permissions` is `read`, so the token is read-only unless a job asks for more; without it `gh release create` fails with a 403.
- **`concurrency: {group: release, cancel-in-progress: false}`** queues rather than cancels, so two pushes landing together cannot race to create the same tag and a half-finished release is never killed.
- **`gh release create --target "$GITHUB_SHA"` creates the tag as part of the release** — a lightweight tag, matching `v0.13.1`/`v0.14.0`. `--generate-notes` builds the body from commits since the previous release.
- The tag value reaches the shell through `env:` rather than `${{ }}` interpolation, so a crafted `pyproject.toml` version cannot break out into the run script.

To reword a release afterwards, edit it in the GitHub UI — the job never touches a release that already exists.

## Hooks

`.claude/settings.json` is git-tracked, so its hooks apply to every clone rather than one machine. Two hooks, both sub-100ms. A file watcher picks up edits to the file mid-session; `/hooks` shows what is actually live and which settings file it came from.

- **`PreToolUse` on `Edit|Write`** — denies writes to `uv.lock`, `.env`, and `.streamlit/secrets.toml`. Change `uv.lock` through uv (`uv add` / `uv lock` / `uv sync`); the two gitignored secret files are edited by hand. The `case` matches the bare filename with an optional directory prefix, so `.env.example` and `uv.lock.bak` pass through.
- **`PostToolUse` on `Edit|Write`** — runs `ruff format` then `ruff check --fix` on the edited file when it ends in `.py`, covering two of the four CI gates. Python files are therefore already formatted and auto-fixed after an edit; a follow-up `ruff format` pass is redundant.

**No hook runs the tests or the type checker.** Two `Stop` hooks used to, and were removed deliberately: `Stop` fires once per *turn* rather than once per *change*, so conversational turns ran the full suite and a whole-project `ty check` against code nobody touched — and `exit 2` on `Stop` prevents the turn from ending, letting an unrelated or pre-existing failure hijack the conversation. Run `uv run pytest` and `uv run ty check` explicitly after changing Python; otherwise CI is the first thing that sees a failure. Do not reinstate them as `Stop` hooks.

Hooks are the one part of this repo with no test and no CI signal — nothing validates the shell embedded in `settings.json`, and it survives two layers of escaping. After editing one, replay it from the file rather than from the string you meant to write:

```sh
CMD=$(jq -r '.hooks.PreToolUse[0].hooks[0].command' .claude/settings.json)
printf '{"tool_input":{"file_path":"uv.lock"}}' | sh -c "$CMD"   # expect a deny payload
printf '{"tool_input":{"file_path":"pyproject.toml"}}' | sh -c "$CMD"  # expect no output
```

## Known Issues

### Do NOT use `tokenizer.apply_chat_template`

TranslateGemma's chat template requires `content` as a list with exactly one structured mapping (`type`, `source_lang_code`, `target_lang_code`, `text`). A plain string trips the `content | length != 1` guard:

```
jinja2.exceptions.TemplateError: User role must provide `content` as an
iterable with exactly one item.
```

The structured form works, but this app builds the prompt as a raw string instead — keeping it explicit and independent of the MLX quant's bundled template:

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
```

### `<end_of_turn>` safety-net strip

The registered EOS token usually stops generation before `<end_of_turn>` appears. `_strip_eos_token()` is kept as a safety net for the rare case the token leaks into the decoded output.

### Chinese uses `zh-CN`, not `zh`

The locale code matches the TranslateGemma Technical Report (Table 5). Since prompts are built manually, the code is inserted as text — and the model was trained with these locale codes.

### Theme variant keys are top-level-only

`base` and `chartCategoricalColors` are valid only in the top-level `[theme]` section, not inside `[theme.light]`/`[theme.dark]`. Streamlit only *logs* a warning for an invalid config key rather than raising, so `TestThemeConfig` validates every `config.toml` key against Streamlit's option template to catch regressions.

### `mlx-lm` below 0.31.3 breaks on Streamlit's thread

With `mlx` 0.32, `mlx-lm` 0.31.1 and 0.31.2 raise `RuntimeError: There is no Stream(gpu, 0) in current thread` from `wired_limit()` in `mlx_lm/generate.py` — generation runs on Streamlit's ScriptRunner thread, not the main thread. Translation returns empty; the app itself loads fine. Hence the `mlx-lm>=0.31.3` floor.

The mocked layers cannot catch this: they replace `mlx_lm` with a `MagicMock`, so no real generation ever runs. `tests/test_live_model.py` exists for exactly this failure — run `uv run pytest -m live` after any `mlx`/`mlx-lm` bump.

## Prompt Template

```
You are a professional {source_lang} ({src_lang_code}) to {target_lang}
({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and
nuances of the original {source_lang} text while adhering to {target_lang} grammar,
vocabulary, and cultural sensitivities.\nProduce only the {target_lang}
translation, without any additional explanations or commentary. Please translate
the following {source_lang} text into {target_lang}:\n\n\n{text}
```

## Resources

- [Technical Report](https://arxiv.org/pdf/2601.09012)
- [Gemma Cookbook](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Research/[TranslateGemma]Example.ipynb)
- [Streamlit AppTest reference](https://docs.streamlit.io/develop/api-reference/app-testing)
