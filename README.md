# Translation Pipeline

Streamlit web application for translating text using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) model.

## Supported Languages

- **Bidirectional with English**: Cantonese, Chinese, Chuukese, Ilocano, Japanese, Korean, Marshallese, Spanish, Thai, Tonga (Tonga Islands), Vietnamese
- **English-only target**: Filipino, Hawaiian, Samoan

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Add your [Hugging Face token](https://huggingface.co/settings/tokens) to `.streamlit/secrets.toml`:

   ```toml
   HF_TOKEN = "your_token_here"
   ```

3. Run the application:

   ```bash
   uv run streamlit run streamlit_app.py
   ```

## Development

```bash
uv run ruff check .   # lint
uv run ruff format .  # format
uv run ty check       # typecheck
uv run pytest         # run tests
```

## Requirements

- Python 3.12+
- **GPU (CUDA or Apple Silicon MPS)**: runs locally with TranslateGemma 4B (~8GB RAM)
- **CPU-only**: falls back to HF Inference API (requires network access)
