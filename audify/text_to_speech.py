# %%
from pathlib import Path

import torch
from TTS.api import TTS

MODULE_PATH = Path(__file__).parents[1]

# Get device
device = "cuda" if torch.cuda.is_available() else False
# %%

# List available 🐸TTS models
TTS().models
# %%

LOADED_MODEL = TTS("tts_models/en/jenny/jenny", gpu=device)
# %%


def sentence_to_speech(
    sentence: str,
    file_path: str = f"{MODULE_PATH}/data/output/speech.mp3",
) -> None:
    if Path(file_path).parent.is_dir() is False:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    LOADED_MODEL.tts_to_file(
        text=sentence,
        file_path=file_path,
    )
