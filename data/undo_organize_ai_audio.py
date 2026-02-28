"""
Reverses organize_ai_audio.py — moves files back to their original locations.
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SRC = PROJECT_ROOT / "data" / "raw" / "ai_generated"

VOCODER_FOLDERS = [
    "common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech",
    "jsut_multi_band_melgan",
    "jsut_parallel_wavegan",
    "ljspeech_full_band_melgan",
    "ljspeech_hifiGAN",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_waveglow",
]

GENERATED_AUDIO_DIR = SCRIPT_DIR / "generated_audio"
ELEVENLABS_DIR = SCRIPT_DIR / "ai_generated"

moved = 0
errors = 0

for f in SRC.iterdir():
    if not f.is_file():
        continue

    matched = False
    for folder in VOCODER_FOLDERS:
        prefix = folder + "_"
        if f.name.startswith(prefix):
            original_name = f.name[len(prefix):]
            dest = GENERATED_AUDIO_DIR / folder / original_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dest)
            moved += 1
            matched = True
            break

    if not matched:
        dest = ELEVENLABS_DIR / f.name
        ELEVENLABS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), dest)
        moved += 1

print(f"Restored {moved} files. Errors: {errors}")
