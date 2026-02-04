# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit web application for translating text to supported languages using Google's [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) translation model.

## Setup & Development

- **Setup environment**: `python3.12 -m venv streamlit_env`
- **Activate environment**: `source streamlit_env/bin/activate`
- **Install dependencies**: `pip install -r requirements.txt`
- **Run application**: `streamlit run streamlit_app.py`

## Testing & Code Quality

- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Typecheck**: `pyright`
- **Run tests**: `pytest`
- **Run single test**: `pytest tests/path_to_test.py::test_name -v`

## Code Style

- **Python**: snake_case for functions/variables, PascalCase for classes
- **Imports**: Use isort with combine-as-imports
- **Error handling**: Use custom ToolError for tool errors
- **Types**: Add type annotations for all parameters and returns
- **Classes**: Use dataclasses and abstract base classes

## Dependencies

- **Hugging Face model loading**: `transformers`
- **Tensor operations**: `torch`
- **Web user interface**: `streamlit`

## Architecture

- **Device detection**: Use best available device (MPS > CUDA > CPU)
- **Translation**: Use `transformers` with translation model
- **Caching**: Use `@st.cache_resource` to load model
- **Supported languages**: Use `st.selectbox` for source language and target language options
- **Metrics**: Use `st.metric` to display metrics
- **Download JSON**: Use `st.download_button` to download a JSON file

### Supported Languages

Languages paired with English in both directions:

- Cantonese (yue)
- Chinese (zh-CN)
- Chuukese (chk)
- Ilocano (ilo)
- Japanese (ja)
- Korean (ko)
- Marshallese (mh)
- Spanish (es)
- Tonga (Tonga Islands) (to)
- Thai (th)
- Vietnamese (vi)

Languages from English:

- Filipino (fil)
- Hawaiian (haw)
- Samoan (sm)

Notes:

- Chinese is Mandarin Chinese
- Filipino is Tagalog

### Download

Include these items in the response JSON file:

- model `string`: Model name
- response `string`: The model's generated text response
- total_duration `integer`: Time spent generating the response in nanoseconds
- load_duration `integer`: Time spent loading the model in nanoseconds
- prompt_eval_count `integer`: Number of input tokens in the prompt
- prompt_eval_duration `integer`: Time spent evaluating the prompt in nanoseconds
- eval_count `integer`: Number of output tokens generated in the response
- eval_duration `integer`: Time spent generating tokens in nanoseconds

Use `time.perf_counter_ns()` to measure duration and return time in nanoseconds.

### Metrics

Display metrics for the model's generated response including all items in the JSON body except for the text response.

## Known Issues

### Do NOT use `processor.apply_chat_template` for TranslateGemma

`processor.apply_chat_template` fails at runtime for this model:

- **Structured content format** (with `source_lang_code`/`target_lang_code` fields) raises `AttributeError: 'dict' object has no attribute '<lang_code>'` during template rendering, regardless of `tokenize=True` or `tokenize=False`.
- **Plain text content** (string instead of list) is rejected by the template with `User role must provide content as an iterable with exactly one item`.

Instead, manually construct the prompt using Gemma chat format tokens and the preferred prompt template, then tokenize directly with `processor.tokenizer`:

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
```

### Always override `top_p` and `top_k` for greedy decoding

When using `do_sample=False`, explicitly pass `top_p=None, top_k=None` to `model.generate()` to override the model's default generation config and suppress warnings.

### Use `processor.tokenizer` for tokenization and decoding

Use `processor.tokenizer(...)` for tokenization and `processor.tokenizer.decode(...)` for decoding. Do not use `processor.decode(...)` or `processor.apply_chat_template(...)`.

## Usage

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

### Text Translation

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "google/translategemma-4b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
eos_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")

instruction = (
    "You are a professional Czech (cs) to German (de-DE) translator. "
    "Your goal is to accurately convey the meaning and nuances of the original Czech text "
    "while adhering to German grammar, vocabulary, and cultural sensitivities. "
    "Produce only the German translation, without any additional explanations or commentary. "
    "Please translate the following Czech text into German:\n\n\n"
    "V nejhorším případě i k prasknutí čočky."
)
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
input_len = inputs["input_ids"].shape[1]

with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        top_p=None,
        top_k=None,
        eos_token_id=eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

generated = output[0][input_len:]
decoded = processor.tokenizer.decode(generated, skip_special_tokens=True)
print(decoded)
```

## Resources

- [Technical Report](https://arxiv.org/pdf/2601.09012)
- [Gemma Cookbook](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Research/[TranslateGemma]Example.ipynb)
