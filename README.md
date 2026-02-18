# Translation Pipeline

A Streamlit web application for translating text using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) model.

## Supported Languages

**Bidirectional with English:**
Cantonese, Chinese, Chuukese, Ilocano, Japanese, Korean, Marshallese, Spanish, Thai, Tonga (Tonga Islands), Vietnamese

**From English only:**
Filipino, Hawaiian, Samoan

## Setup

```bash
uv sync
```

Create a `.env` file with your [Hugging Face token](https://huggingface.co/settings/tokens):

```bash
echo "HF_TOKEN=your_token_here" > .env
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

1. Select source and target languages
2. Enter text to translate
3. Click **Translate**
4. View translation, metrics, and download results as JSON

## Development

```bash
uv run ruff check .   # lint
uv run ruff format .  # format
uv run ty check       # typecheck
uv run pytest         # run tests
```

## Requirements

- Python 3.12+
- ~8GB RAM for model loading
- GPU recommended (CUDA, Apple Silicon MPS, or CPU fallback)
