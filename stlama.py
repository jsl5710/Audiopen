# Import necessary libraries and modules
import re
import os
import time
import streamlit as st
import openai

from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI

# Set OpenAI API key
openai.api_key = st.secrets['OPENAI_API_KEY']

# Set Streamlit page configuration
st.set_page_config(page_title="streamlit_audio_recorder")
st.markdown("""<style>.css-1egvi7u {margin-top: -3rem;}</style>""", unsafe_allow_html=True)

# Define Streamlit app header
st.header(":violet[Audio]  :orange[Whisper] :headphones:")
st.write(":violet[Unleash Ideas through Audio]")
st.write("\n\n")

# Function to transcribe audio and process text
@st.cache_data
def process_audio(audio_file):
    start_time = time.time()
    try:
        # Transcribe the audio
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        x = transcript["text"]

        # LangChain template for text processing
        template = """
        You are an expert in converting messy thoughts into clear text.
        Messy text: {x}

        - Give it a nice headline and clear text
        - Output should be a list of headline and clear text
        """

        sprompt = PromptTemplate.from_template(template)

        # Initialize the models
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets['OPENAI_API_KEY'])
        llm_chain = LLMChain(prompt=sprompt, llm=llm)

        # Process the text
        z = llm_chain.run(x)
        headline_match = re.search(r"Headline:\s*(.*?)\n", z, re.DOTALL)
        clear_text_match = re.search(r"Clear Text:\s*(.*?)$", z, re.DOTALL)

        # Check if matches were found and extract the text
        headline = headline_match.group(1).strip() if headline_match else ""
        clear_text = clear_text_match.group(1).strip() if clear_text_match else ""

        end_time = time.time()
        execution_time = end_time - start_time
        st.write("Execution time:", execution_time, "seconds")

        return headline, clear_text, transcript

    except openai.error.InvalidRequestError as e:
        st.error("Error: The audio file is too short. Minimum audio length is 0.1 seconds.")
        return "", "", None

def audiorec_demo_app():
    # Record audio using the custom component
    wav_audio_data = st.file_uploader("Record audio", type=["wav"])

    # Process audio data if available
    if wav_audio_data is not None:
        # Save the audio to a WAV file
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        wav_file_path = os.path.join(upload_dir, "recorded_audio.wav")
        with open(wav_file_path, "wb") as wav_file:
            wav_file.write(wav_audio_data.read())

        # Transcribe and process the audio
        audio_file = open(wav_file_path, "rb")

        # Error handling for processing
        try:
            with st.spinner("Processing Your Ideas..."):
                headline, clear_text, transcript = process_audio(audio_file)

            with st.expander("Original Transcription"):
                st.write(transcript)

            if headline and clear_text:
                from streamlit_extras.colored_header import colored_header

                colored_header(
                    label=headline,
                    description=clear_text,
                    color_name="violet-70",
                )
        except Exception as e:
            st.error(f"An error occurred while processing: {e}")

        # Closing the file
        audio_file.close()
        # Removing the temporary file
        os.remove(wav_file_path)

if __name__ == "__main__":
    # Call the main function
    audiorec_demo_app()
