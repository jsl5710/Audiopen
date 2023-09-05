import re
import os
import time
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI

from langchain import PromptTemplate
from st_custom_components import st_audiorec
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
        You have the unique ability to convert spoken ideas into coherent text. 
        Imagine that you're assisting someone who has just recorded their thoughts in audio. 
        These thoughts might be a bit unorganized or lengthy. Your task is to transform this spoken content into two key parts:

        1. A concise headline: Capture the essence of the spoken ideas in a short, attention-grabbing sentence.
        2. Clear text: Explain the main points or details in a straightforward, easy-to-understand paragraph.
        """

        sprompt = PromptTemplate.from_template(template, x=x)

        # Initialize the models
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets['OPENAI_API_KEY'])
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
    wav_audio_data1 = st.file_uploader("Record audio", type=["wav"])
    wav_audio_data2 = st_audiorec()

    # Process audio data if available
    if wav_audio_data2 is not None:
        # Save the audio to a WAV file
        with open("recorded_audio.wav", "wb") as wav_file:
            wav_file.write(wav_audio_data2)

            # Transcribe and process the audio
            file = "./recorded_audio.wav"
            audio = open(file, "rb")
        try:
            with st.spinner("Processing Your Ideas..."):
                headline, clear_text, transcript = process_audio(audio)

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

    # Process audio data if available
    if wav_audio_data1 is not None:
        # Save the audio to a WAV file
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        wav_file_path = os.path.join(upload_dir, "recorded_audio.wav")
        with open(wav_file_path, "wb") as wav_file:
            wav_file.write(wav_audio_data1.read())

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
