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
st.header(":blue[Ideation] from :orange[Sound] :headphones:", divider="rainbow")


st.caption("Keep track of YOUR thoughts	:sunglasses:")
st.write("\n\n")


# Function to transcribe audio and process text
@st.cache_data
def process_audio(audio_file):
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
    llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=config.OPENAI_API_KEY)
    # llm = CTransformers(
    #     model="/Users/prathapreddy/Documents/AUDIOPEN/llama-2-7b-chat.ggmlv3.q8_0.bin",
    #     model_type="llama",
    #     config={"max_new_tokens": 512, "temperature": 0.8},
    # )

    llm_chain = LLMChain(prompt=sprompt, llm=llm)

    # Process the text
    z = llm_chain.run(x)
    headline_match = re.search(r"Headline:\s*(.*?)\n", z, re.DOTALL)
    clear_text_match = re.search(r"Clear Text:\s*(.*?)$", z, re.DOTALL)

    # Check if matches were found and extract the text
    headline = headline_match.group(1).strip() if headline_match else ""
    clear_text = clear_text_match.group(1).strip() if clear_text_match else ""

    return headline, clear_text, transcript


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

        with st.spinner("Transcribing audio..."):
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
