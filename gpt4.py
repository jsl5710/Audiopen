
import re
import config
from langchain import PromptTemplate, LLMChain # just one line to import these two classes 
from langchain.llms import OpenAI
import ffmpeg
from st_custom_components import st_audiorec
import openai
import streamlit as st

# I've removed st_custom_components import as there's no use of it in this code

openai.api_key = st.secrets['OPENAI_API_KEY']

st.set_page_config(page_title="streamlit_audio_recorder")
st.title(":violet[Audio]  :orange[Whisper] :headphones:")  # I've used title method to set a title for the page

@st.cache_data
def process_audio(audio_file):
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        x = transcript["text"]

        template = """

You're a master at transforming disorganized thoughts into crystal-clear text.  you're assisting user, who has just recorded their thoughts in audio. These thoughts might be a bit chaotic or lengthy. Your mission is to craft two essential components:

1. A captivating headline: Summarize the core message of the spoken thoughts in a concise sentence.
2. Clear text: rephrase key points or details into an easy-to-follow paragraph.

Your goal is to produce both the headline and clear text. Ensure they are brief, coherent, and accurately represent the spoken content.

   here is the thought :{x}     
        """

        sprompt = PromptTemplate.from_template(template)

        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets['OPENAI_API_KEY'], temperature=0)
        llm_chain = LLMChain(prompt=sprompt, llm=llm)

        z = llm_chain.run(x)
        st.info(z)
        headline_match = re.search(r"Headline:\s*(.*?)\n", z, re.DOTALL)
        clear_text_match = re.search(r"Clear Text:\s*(.*?)$", z, re.DOTALL)

        headline = headline_match.group(1).strip() if headline_match else "Headline not found"
        clear_text = clear_text_match.group(1).strip() if clear_text_match else "Clear text not found"

        return headline, clear_text, transcript

    except openai.error.InvalidRequestError as e:
        st.error("Error: The audio file is too short. Minimum audio length is 0.1 seconds.")
        return "", "", None


def audiorec_demo_app():
    # I am assuming you have a method to record audio 
    # as the previous method was not working
    wav_audio_data = st_audiorec()
    # Replace record_audio() with your method to record the audio

    if wav_audio_data is not None:
            with open("recorded_audio.wav", "wb") as wav_file:
                wav_file.write(wav_audio_data)

            file = "./recorded_audio.wav"
            audio_file = open(file, "rb")

            with st.spinner("Processing Your Ideas... "):    # Spinner instead of status method to show progress
                headline, clear_text, transcript = process_audio(audio_file)

    with st.expander("Original Transcription"):
        st.write(transcript)

        st.header(headline)
        st.text(clear_text)    # removed the unused imported libraries and methods

if __name__ == "__main__":
    audiorec_demo_app()
