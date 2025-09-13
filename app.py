import tempfile
from pathlib import Path

import streamlit as st
from typing_extensions import Literal

from audify.constants import (
    DEFAULT_LANGUAGE_LIST,
    DEFAULT_MODEL,
    KOKORO_DEFAULT_VOICE,
)
from audify.text_to_speech import BaseSynthesizer, EpubSynthesizer, PdfSynthesizer
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
engine: Literal["kokoro", "tts_models"] = st.sidebar.selectbox(
    "TTS Engine",
    options=["kokoro", "tts_models"],
    index=0,
    help="Select the TTS engine to use.",
)

# if engine == "tts_models", allow for tts model selection
if engine == "tts_models":
    # User needs to agree with terms and conditions of the xtts models
    st.sidebar.markdown(
        "By using the `tts_models` engine, you agree to the terms and conditions of the"
        " XTTS models."
    )
    st.sidebar.markdown(
        "You can find the terms and conditions [here](https://coqui.ai/cpml)."
    )
    st.sidebar.markdown(
        "The `tts_models` engine uses pre-trained models from the TTS library. "
        "You can specify the model name in the input below."
    )

    # Model Selection
    model = st.sidebar.text_input(
        "XTTS Model",
        value="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Path or name of the XTTS model to use.",
    )
else:
    model = ""


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
                    synthesizer: BaseSynthesizer | None = None

                    if file_extension == ".epub":
                        synthesizer = EpubSynthesizer(
                            str(temp_file_path),
                            language=language,
                            model_name=model
                            if engine == "tts_models"
                            else DEFAULT_MODEL,
                            translate=translate_language,
                            speaker=(
                                KOKORO_DEFAULT_VOICE
                                if engine == "kokoro"
                                else "data/Jennifer_16khz.wav"
                            ),
                            save_text=save_text,
                            engine=engine,
                            confirm=False,  # No confirmation needed in GUI
                        )
                    elif file_extension == ".pdf":
                        synthesizer = PdfSynthesizer(
                            str(temp_file_path),
                            language=language,
                            model_name=model
                            if engine == "tts_models"
                            else DEFAULT_MODEL,
                            translate=translate_language,
                            save_text=save_text,
                            engine=engine,
                        )

                    if synthesizer:
                        # Show terminal output in the app
                        output_path = synthesizer.synthesize()
                    else:
                        st.error(
                            "Unsupported file format. Please upload "
                            "a .pdf or .epub file."
                        )
                        st.stop()

                    # file exists check
                    if not output_path.exists():
                        st.error(
                            "Output file not found. Please check the synthesis process."
                        )
                    else:
                        st.success(
                            "Synthesis complete! Synthesis output path:  "
                            "`{output_path}`"
                        )
                    # Provide a download link
                    with open(str(output_path), "rb") as f:
                        st.download_button(
                            label="Download Audiobook",
                            data=f,
                            file_name=output_path.name,
                            mime="audio/mp3",
                        )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
