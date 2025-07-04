import tempfile
from pathlib import Path

import streamlit as st

from audify.constants import DEFAULT_LANGUAGE_LIST
from audify.text_to_speech import EpubSynthesizer, InspectSynthesizer, PdfSynthesizer
from audify.utils import get_file_extension

st.set_page_config(page_title="Audify", page_icon="🎙️")

st.title("Audify: Text to Speech")

# --- Sidebar for options ---
st.sidebar.header("Configuration")

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload a file (.pdf or .epub)", type=["pdf", "epub"]
)

# Language Selection
language = st.sidebar.selectbox(
    "Language of the text",
    options=DEFAULT_LANGUAGE_LIST,
    index=DEFAULT_LANGUAGE_LIST.index("en"),
    help="Select the language of the input text.",
)

# Translation Selection
translate_option = st.sidebar.checkbox(
    "Translate the text?",
    help="Check this box to translate the text before synthesis.",
)
translate_language = None
if translate_option:
    translate_language = st.sidebar.selectbox(
        "Translate to",
        options=DEFAULT_LANGUAGE_LIST,
        index=DEFAULT_LANGUAGE_LIST.index("en"),
        help="Select the language to translate the text to.",
    )

# TTS Engine
engine = st.sidebar.selectbox(
    "TTS Engine",
    options=["kokoro", "tts_models"],
    index=0,
    help="Select the TTS engine to use.",
)

# Model Selection
model = st.sidebar.text_input(
    "TTS Model",
    value="tts_models/multilingual/multi-dataset/xtts_v2",
    help="Path or name of the TTS model to use.",
)

# Voice Selection
voice = st.sidebar.text_input(
    "Speaker Voice Path",
    value="data/Jennifer_16khz.wav",
    help="Path to the speaker's .wav file.",
)

# Other options
save_text = st.sidebar.checkbox(
    "Save extracted text", help="Save the extracted text to a file."
)

# --- Main Area ---

if uploaded_file is not None:
    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / uploaded_file.name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write(f"Uploaded file: `{uploaded_file.name}`")

        if st.button("Synthesize Audiobook"):
            with st.spinner("Synthesizing... This may take a while."):
                try:
                    file_extension = get_file_extension(str(temp_file_path))
                    synthesizer = None

                    if file_extension == ".epub":
                        synthesizer = EpubSynthesizer(
                            str(temp_file_path),
                            language=language,
                            speaker=voice,
                            model_name=model,
                            translate=translate_language,
                            save_text=save_text,
                            engine=engine,
                            confirm=False,  # No confirmation needed in GUI
                        )
                    elif file_extension == ".pdf":
                        synthesizer = PdfSynthesizer(
                            str(temp_file_path),
                            language=language,
                            speaker=voice,
                            model_name=model,
                            translate=translate_language,
                            save_text=save_text,
                            engine=engine,
                        )

                    if synthesizer:
                        output_path = synthesizer.synthesize()
                        st.success("Synthesis complete!")
                        st.audio(output_path, format="audio/mp3")

                        # Provide a download link
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Audiobook",
                                data=f,
                                file_name=Path(output_path).name,
                                mime="audio/mp3",
                            )
                    else:
                        st.error("Unsupported file format.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a file to begin.")


# --- Information Expander ---
with st.expander("Show available models and languages"):
    inspector = InspectSynthesizer()
    st.subheader("Available Languages")
    st.json(inspector.model.languages)

    st.subheader("Available Models")
    st.json(inspector.model.models)
