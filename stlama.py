# Import necessary libraries and modules
import re
import config
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI
import ffmpeg
import openai
import streamlit as st
from st_custom_components import st_audiorec

# Set OpenAI API key
openai.api_key = st.secrets['OPENAI_API_KEY']

# Set Streamlit page configuration
st.set_page_config(page_title="streamlit_audio_recorder")
st.markdown(
    """<style>.css-1egvi7u {margin-top: -3rem;}</style>""", unsafe_allow_html=True
)

# Define Streamlit app header
st.header(":violet[Audio]  :orange[Whisper] :headphones:", divider="violet")
st.write(":violet[Unleash Ideas through Audio]")
st.write("\n\n")


# Function to transcribe audio and process text
# Function to transcribe audio and process text
@st.cache_data
def process_audio(audio_file):
    try:
        # Transcribe the audio
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        x = transcript["text"]

        # LangChain template for text processing
        template = """


You're a master at transforming disorganized thoughts into crystal-clear text. Imagine you're assisting someone who has just recorded their thoughts in audio. These thoughts might be a bit chaotic or lengthy. Your mission is to craft two essential components:
disorganized thought : {x}
1. headline: Summarize the core message of the spoken thoughts in a concise sentence.
2. text: Break down the key points or details into an easy-to-follow paragraph.

Your goal is to produce both the headline and text. Ensure they are brief, coherent, and accurately represent the disorganized thought.

        
        """

        sprompt = PromptTemplate.from_template(template)

        # Initialize the models
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets['OPENAI_API_KEY'], temperature=0)
        llm_chain = LLMChain(prompt=sprompt, llm=llm)

        # Process the text
        z = llm_chain.run(x)
        headline_match = re.search(r"headline:\s*(.*?)\n", z, re.DOTALL)
        clear_text_match = re.search(r"text:\s*(.*?)$", z, re.DOTALL)

        # Check if matches were found and extract the text
        headline = headline_match.group(1).strip() if headline_match else ""
        clear_text = clear_text_match.group(1).strip() if clear_text_match else ""

        return headline, clear_text, transcript

    except openai.error.InvalidRequestError as e:
        st.error("Error: The audio file is too short. Minimum audio length is 0.1 seconds.")
        return "", "", None


# Streamlit app
def audiorec_demo_app():
    # Record audio using the custom component
    wav_audio_data = st_audiorec()

    # Process audio data if available
    if wav_audio_data is not None:
        # Save the audio to a WAV file
        with open("recorded_audio.wav", "wb") as wav_file:
            wav_file.write(wav_audio_data)

        # Transcribe and process the audio
        file = "./recorded_audio.wav"
        audio_file = open(file, "rb")

        with st.status(""":rainbow[Processing Your Ideas... ]"""):
            headline, clear_text, transcript = process_audio(audio_file)
        # st.info(headline)
        # st.success(clear_text)
        from streamlit_extras.colored_header import colored_header

        with st.expander("Original Transcription"):
            st.write(transcript)

        colored_header(
            label=headline,
            description=clear_text,
            color_name="violet-70",
        )

        # from streamlit_card import card

        # card(
        #     title=headline,
        #     text=clear_text,
        #     image="http://placekitten.com/300/250",
        #     url="https://www.google.com",
        # )

        # Toggle to show the original transcription
        # on = st.toggle("Original Transcription")
        # if on:
        #     st.write(transcript)


if __name__ == "__main__":
    # Call the main function
    audiorec_demo_app()
