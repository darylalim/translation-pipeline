# TranslateGemma Translate

Streamlit app for translating text and images using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) model with local GPU inference.

## Features

- **Text translation** — translate text between supported languages (up to 5,000 characters)
- **Image translation** — extract and translate text from uploaded images (PNG, JPG, JPEG, WEBP)
- **Swap languages** — swap source and target languages, moving translation output to source input
- **Copy to clipboard** — copy translation output with one click
- **Download as text** — download translation output as a `.txt` file

## Supported Languages

Chinese, Dutch, French, German, Indonesian, Italian, Spanish, Vietnamese — all bidirectional with English.

## Requirements

- Python 3.12+
- GPU with CUDA or Apple Silicon MPS (~8GB VRAM)

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Add your [Hugging Face token](https://huggingface.co/settings/tokens) to `.env`:

   ```
   HF_TOKEN=your_token_here
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
