# %%
from pathlib import Path

import torch
from TTS.api import TTS

MODULE_PATH = Path(__file__).parents[1]

# Get device
device = "cuda" if torch.cuda.is_available() else False
# %%

LOADED_MODEL = TTS("tts_models/es/mai/tacotron2-DDC", gpu=device)
# %%


def sentence_to_speech(
    sentence: str,
    file_path: str = f"{MODULE_PATH}/data/output/speech.wav",
) -> None:
    if Path(file_path).parent.is_dir() is False:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        LOADED_MODEL.tts_to_file(
            text=sentence,
            file_path=file_path,
        )
    except Exception as e:
        error_message = "Error: " + str(e)
        LOADED_MODEL.tts_to_file(
            text=error_message,
            file_path=file_path,
        )
