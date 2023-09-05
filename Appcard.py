

import time
import re
from datetime import datetime
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI
import openai
import streamlit as st
from st_custom_components import st_audiorec
from langchain.chat_models import ChatOpenAI


# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set Streamlit page configuration
st.set_page_config(page_title="streamlit_audio_recorder")
st.markdown(
    """<style>.css-1egvi7u {margin-top: -3rem;}</style>""", unsafe_allow_html=True
)


# Define Streamlit app header


def card(source, context):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%b %d, %Y")

    st.markdown(
        f"""
    <div class="card" style="margin:6rem;">
        <div class="card-body">
            <h6 class="card-time">{formatted_time}</h6>
            <h3 class="card-title">{source}</h3>
            <h5 class="card-text">{context}</h5>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def app_header():
    st.header(":violet[VoiceNote ] :orange[Vault] :headphones:", divider="violet")
    st.caption(":violet[Capturing Brilliance], One Sound at a Time. :microphone:")
    st.write("\n\n")


# Function to transcribe audio and process text
@st.cache_data
def process_audio(audio_file):
    global headline, clear_text
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
    headline_match = re.search(r"Headline:\s*(.*?)\n", z, re.DOTALL)
    clear_text_match = re.search(r"Clear Text:\s*(.*?)$", z, re.DOTALL)

    # Check if matches were found and extract the text
    headline = headline_match.group(1).strip() if headline_match else ""

    end_time = time.time()
    execution_time = end_time - start_time
    st.write("Execution time:", execution_time, "seconds")

    return headline, clear_text_match, transcript


def mainfun(wav_audio_data):
    # Process audio data if available
    if wav_audio_data is not None:
        # Save the audio to a WAV file
        with open("recorded_audio.wav", "wb") as wav_file:
            wav_file.write(wav_audio_data)

            # Transcribe and process the audio
            file = "./recorded_audio.wav"
            audio = open(file, "rb")

            with st.status(""":rainbow[Processing Your Ideas... ]"""):
                headline, clear_text, transcript = process_audio(audio)

        expander = st.expander("Original Voice Note")
        expander.write(transcript["text"])
        card(headline, clear_text)


app_header()
# Record audio using the custom component
wav_audio_data = st_audiorec()
mainfun(wav_audio_data)
