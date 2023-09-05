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
@st.cache_data
def process_audio(audio_file):
    try:
        # Transcribe the audio
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        x = transcript["text"]

        # LangChain template for text processing
        template = """

        You're an Expert in converting spoken ideas into coherent text. Imagine that you're assisting someone who has just recorded their thoughts in audio. These thoughts might be a bit unorganized or lengthy. Your task is to transform this spoken content, which is represented by the transcription below, into two key parts:

1. Headline: Capture the essence of the spoken ideas in a short, attention-grabbing sentence.
2. Clear text: Explain the main points or details in a straightforward, easy-to-understand paragraph.

Transcription: {x}

the output should consist of both the headline and the Clear text.

        """

        sprompt = PromptTemplate.from_template(template)

        # Initialize the models
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets['OPENAI_API_KEY'], temperature=0)
        llm_chain = LLMChain(prompt=sprompt, llm=llm)

        # Process the text
        z = llm_chain.run(x)
        headline_match = re.search(r"Headline:\s*(.*?)\n", z, re.DOTALL)
        clear_text_match = re.search(r"Clear text:\s*(.*?)$", z, re.DOTALL)

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
        os.remove(file)


if __name__ == "__main__":
    # Call the main function
    audiorec_demo_app()
