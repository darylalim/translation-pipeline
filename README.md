# Translation Pipeline

A Streamlit web application for translating text using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) model.

## Supported Languages

**Bidirectional with English (21 languages):**
Arabic, Cantonese, Chinese, Chuukese, Fijian, French, Hindi, Ilocano, Indonesian, Japanese, Korean, Lingala, Marshallese, Nepalbhasa (Newari), Russian, Spanish, Swahili, Tahitian, Thai, Tonga, Vietnamese

**English to only (16 languages):**
Armenian, Burmese, Chinese (Taiwan), Filipino, Hawaiian, Khmer, Lao, Malayalam, Marathi, Mongolian, Persian, Portuguese (Brazil), Portuguese (Portugal), Punjabi, Samoan, Telugu

**Non-English pairs:**
- Arabic ↔ Swahili
- Cantonese ↔ Chinese, Taiwanese Mandarin
- Chinese ↔ Japanese, Swahili
- Japanese → Chinese

## Installation

```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Hugging Face token:

```bash
echo "HF_TOKEN=your_token_here" > .env
```

Get your token at https://huggingface.co/settings/tokens

## Usage

```bash
streamlit run streamlit_app.py
```

## Testing

```bash
pytest tests/ -v
```

## Requirements

- Python 3.12+
- ~8GB RAM for model loading
- GPU recommended (supports CUDA, Apple Silicon MPS, or CPU fallback)
