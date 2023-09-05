import time
import re
from datetime import datetime
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import openai
import streamlit as st
from st_custom_components import st_audiorec


# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set Streamlit page configuration
st.set_page_config(page_title="streamlit_audio_recorder")
st.markdown(
    """<style>.css-1egvi7u {margin-top: -3rem;}</style>""", unsafe_allow_html=True
)


# Define Streamlit app header
def app_header():
    st.header(":violet[VoiceNote ] :orange[Vault] :headphones:", divider="violet")
    st.caption(":violet[Capturing Brilliance], One Sound at a Time. :microphone:")
    st.write("\n\n")


def process_audio(audio_file):
    start_time = time.time()

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
    llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])

    llm_chain = LLMChain(prompt=sprompt, llm=llm)

    # Process the text
    z = llm_chain.run(x)
    st.info(z)
    headline_match = re.search(r"Headline:\s*(.*?)\n", z, re.DOTALL | re.IGNORECASE)
    clear_text_match = re.search(r"Clear Text:\s*(.*?)$", z, re.DOTALL | re.IGNORECASE)

    # Check if matches were found and extract the text
    headline = headline_match.group(1).strip() if headline_match else ""
    clear_text = clear_text_match.group(1).strip() if clear_text_match else ""

    end_time = time.time()
    execution_time = end_time - start_time
    st.write("Execution time:", execution_time, "seconds")

    return headline, clear_text, transcript['text']

def mainfun():
    # Record audio using the custom component
    wav_audio_data = st_audiorec()

    # Process audio data if available
    if wav_audio_data is not None:
        # Save the audio to a WAV file
        with open("recorded_audio.wav", "wb") as wav_file:
            wav_file.write(wav_audio_data)

        # Transcribe and process the audio
        with st.status(""":rainbow[Processing Your Ideas... ]"""):
            headline, clear_text, transcribe_text = process_audio("recorded_audio.wav")

        # Display original voice note and final results
        expander = st.beta_expander("Original Voice Note")
        expander.write(transcribe_text)
        st.write(f"Headline: {headline}")
        st.write(f"Clear Text: {clear_text}")

app_header()
mainfun()
