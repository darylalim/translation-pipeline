# Translation Pipeline
Translation pipeline for documents.

## Installation
- Download and install [Python](https://www.python.org/downloads/)
- Download and install [Ollama](https://ollama.com/download)

Run the following commands in the terminal.

- Download an Ollama model: `ollama pull granite4`
- Set up a Python virtual environment: `python3 -m venv translation_pipeline_env`
- Activate the virtual environment: `source translation_pipeline_env/bin/activate` (Mac)
- Install the required Python packages: `pip install -r requirements.txt`
- Run the application in a web browser: `streamlit run streamlit_app.py`

## Notes
- Text input is limited to a maximum of 100 characters.
- Translated text is generated using an [IBM Granite 4.0 language model](https://github.com/ibm-granite/granite-4.0-language-models).
- Languages supported: English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese (Simplified).
