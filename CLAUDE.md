# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit web application that translates English text to supported languages using the Google [MADLAD-400-10B-MT](https://huggingface.co/google/madlad400-10b-mt) language model. The pipeline translates text using a PyTorch model with the Hugging Face Transformers library.

## Commands

```bash
# Setup
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Architecture

The application is a single file Streamlit app (`streamlit_app.py`) with these key components:

- **Translation**: Uses Hugging Face Transformers with a model for text translation 
- **Device detection**: Automatically selects MPS (Apple Silicon) > CUDA > CPU
- **Caching**: Uses `@st.cache_resource` for model loading

### Key Constants

- `MODEL_NAME`: Language model identifier

### Supported Languages

This is a list of target languages with their BCP-47 codes in parentheses.

- Cantonese (yue)
- Mandarin Chinese (zh)
- Chuukese (chk)
- Hawaiian (haw)
- Ilocano (ilo)
- Japanese (ja)
- Korean (ko)
- Marshallese (mh)
- Samoan (sm)
- Spanish (es)
- Tagalog (fil)
- Thai (th)
- Vietnamese (vi)

Visayan is not supported by the MADLAD-400 model. This language translation will be implemented with another model.  

### Download

This is a list of items to include in the body of the JSON file for download.

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

The application displays metrics for the model's generated response including all of the items in the JSON body except for the text response.

## Resources

Resources for more information on the MADLAD-400 model:

- [Research paper](https://arxiv.org/abs/2309.04662)
- [GitHub Repo](https://github.com/google-research/t5x)
- [Hugging Face MADLAD-400 Docs (Similar to T5)](https://huggingface.co/docs/transformers/model_doc/MADLAD-400)
