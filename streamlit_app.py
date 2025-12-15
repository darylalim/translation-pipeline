# Import packages
from ollama import generate
import streamlit as st

# Function to translate text to target language
def translate(text: str, target_language: str = "French"):
    translated = generate(
        model='granite4', 
        prompt=f'Translate {text} to {target_language}. Only output the translation, and nothing else.'
    )
    return translated['response']

st.title("Translation Pipeline")
st.write("Generate text translations using an Ollama model.")

# User inputs text here
text_input = st.text_input("Enter text to translate", max_chars=100)

# Target language options
target_language = st.selectbox("Select target language", ["Arabic", "Chinese", "Czech", "Dutch", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Spanish"])

# Button for translation
if st.button("Translate", type='primary'):
    if text_input:
        # Perform translation
        translation = translate(text_input, target_language)
        # Display the result
        st.write("Translation:")
        st.write(translation)
    else:
        st.warning("Please enter some text to translate")
