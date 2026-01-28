# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

A Streamlit web application that translates text to supported languages using [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) translation model from Google. The pipeline translates text using a PyTorch model with the Hugging Face Transformers library.

## Directory Structure

The application is a single file Streamlit app (`streamlit_app.py`).

## Main Dependencies

- `streamlit` - Web user interface framework
- `transformers` - Hugging Face model loading
- `torch` - Tensor operations

## Architecture

### Components in `streamlit_app.py`

- **Translation**: Uses Hugging Face Transformers with a model for text translation 
- **Device detection**: Automatically selects MPS (Apple Silicon) > CUDA > CPU
- **Caching**: Uses `@st.cache_resource` for model loading

### With direct initialization

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "google/translategemma-4b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")


# ---- Text Translation ----
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "cs",
                "target_lang_code": "de-DE",
                "text": "V nejhorším případě i k prasknutí čočky.",
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)
input_len = len(inputs['input_ids'][0])

with torch.inference_mode():
    generation = model.generate(**inputs, do_sample=False)

generation = generation[0][input_len:]
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

# ---- Text Extraction and Translation ----
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source_lang_code": "cs",
                "target_lang_code": "de-DE",
                "url": "https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

with torch.inference_mode():
    generation = model.generate(**inputs, do_sample=False)

generation = generation[0][input_len:]
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
```

### Preferred Prompt

Preferred prompt when using the model. source_lang refers to the source language name,
e.g. English, src_lang_code to the source language code, e.g. en-US, target_lang to the target
language, e.g. German, and tgt_lang_code to the target language code, i.e. de-DE.

```python
"""
You are a professional {source_lang} ({src_lang_code}) to {target_lang}
({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and
nuances of the original {source_lang} text while adhering to {target_lang} grammar,
vocabulary, and cultural sensitivities. Produce only the {target_lang}
translation, without any additional explanations or commentary. Please translate
the following {source_lang} text into {target_lang}:\n\n\n{text}
"""
```

### Constants

- `model_id`: Translation model identifier

### Supported Languages

Source language and target language are configurable via selectbox widgets.

Languages paired with English in both directions:

- Arabic, Modern Standard (ar)
- Cantonese (yue)
- Chinese (zh-CN)
- Chuukese (chk)
- Fijian (fj)
- French (fr)
- Hindi (hi)
- Ilocano (ilo)
- Indonesian (id)
- Japanese (ja)
- Korean (ko)
- Lingala (ln)
- Marshallese (mh)
- Nepalbhasa (Newari) (new)
- Russian (ru)
- Spanish (es)
- Swahili (sw)
- Tahitian (ty)
- Tonga (Tonga Islands) (to)
- Thai (th)
- Vietnamese (vi)

Languages from English:

- Armenian (hy)
- Burmese (my)
- Chinese (Taiwan) (zh-TW)
- Filipino (fil)
- Hawaiian (haw)
- Khmer (km)
- Lao (lo)
- Malayalam (ml)
- Marathi (mr)
- Mongolian (mn)
- Persian (fa)
- Portuguese (Brazil) (pt-BR)
- Portuguese (Portugal) (pt-PT)
- Punjabi (pa)
- Samoan (sm)
- Telugu (te)

Non-English language pairs:

- Arabic (ar) - Swahili (sw)
- Cantonese (yue) - Chinese (zh)
- Cantonese (yue) - Taiwanese Mandarin (zh-Hant)
- Chinese (zh-CN) - Japanese (ja)
- Chinese (zh) - Swahili (sw)

Visayan is not supported.

### Download

Include these items in the response JSON file for download.

- model (string): Model name
- response (string): The model's generated text response
- total_duration (integer): Time spent generating the response in nanoseconds
- load_duration (integer): Time spent loading the model in nanoseconds
- prompt_eval_count (integer): Number of input tokens in the prompt
- prompt_eval_duration (integer): Time spent evaluating the prompt in nanoseconds
- eval_count (integer): Number of output tokens generated in the response
- eval_duration (integer): Time spent generating tokens in nanoseconds

Use `time.perf_counter_ns()` to measure duration and return time in nanoseconds.

### Metrics

Display metrics for the model's generated response including all of the items in the JSON body except for the text response.

## Standards

- Type hints required on all functions
- pytest for testing (fixtures in `tests/conftest.py`)
- PEP 8 with 100 character lines
- pylint for static code analysis

## Commands

```bash
# Setup
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Resources and Technical Documentation

- [Technical Report](https://arxiv.org/pdf/2601.09012)
- [Gemma Cookbook](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Research/[TranslateGemma]Example.ipynb)
