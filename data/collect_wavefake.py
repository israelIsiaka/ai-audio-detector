"""
data/collect_wavefake.py — Download and organise WaveFake dataset.

WaveFake contains AI-generated speech from 6 vocoders (MelGAN, HiFi-GAN,
WaveGlow, etc.) and the original LJSpeech recordings as real speech.

Usage:
    python data/collect_wavefake.py
    python data/collect_wavefake.py --output data/raw --samples 200
    python data/collect_wavefake.py --dry-run   # show what would be downloaded

Produces:
    data/raw/natural/   ← LJSpeech originals
    data/raw/ai/        ← WaveFake generated samples
"""

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# WaveFake sources
# Source: https://github.com/RUB-SysSec/WaveFake
# The dataset is hosted on Zenodo — direct download links below.
# ---------------------------------------------------------------------------

WAVEFAKE_PARTS = [
    {
        "name": "ljspeech_original",
        "label": "natural",
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "filename": "LJSpeech-1.1.tar.bz2",
        "extract_subdir": "LJSpeech-1.1/wavs",
        "md5": None,  # ~2.6 GB — skip hash check to save time
    },
    {
        "name": "wavefake_melgan",
        "label": "ai_generated",
        "url": "https://zenodo.org/record/5642694/files/generated_audio.zip",
        "filename": "wavefake_generated.zip",
        "extract_subdir": None,  # flat zip
        "md5": None,
    },
]

# Smaller, faster alternative: pre-split WaveFake subset on HuggingFace
HUGGINGFACE_WAVEFAKE = "https://huggingface.co/datasets/MarcBrun/WaveFake-subset/resolve/main/"


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        mb = downloaded / 1_048_576
        print(f"\r  [{bar}] {pct:.0f}%  {mb:.1f} MB", end="", flush=True)


def _download(url: str, dest: Path) -> Path:
    print(f"  Downloading {dest.name} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    print()  # newline after progress bar
    return dest


def _extract_tar(archive: Path, dest: Path, subdir: str | None, max_files: int):
    print(f"  Extracting {archive.name} …")
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:bz2") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".wav"):
                continue
            if subdir and subdir not in member.name:
                continue
            if count >= max_files:
                break
            member.name = Path(member.name).name  # flatten path
            tf.extract(member, dest)
            count += 1
    print(f"  Extracted {count} wav files → {dest}")
    return count


def _copy_wavs(src: Path, dest: Path, max_files: int) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    wavs = sorted(src.rglob("*.wav"))[:max_files]
    for w in wavs:
        shutil.copy2(w, dest / w.name)
    return len(wavs)


def collect(output_dir: Path, samples_per_class: int, dry_run: bool):
    natural_dir = output_dir / "natural"
    ai_dir = output_dir / "ai_generated"
    tmp_dir = output_dir / "_tmp_wavefake"

    print("\n═══ WaveFake Collection ═══")
    print(f"  Target : {samples_per_class} samples per class")
    print(f"  Output : {output_dir.resolve()}")

    if dry_run:
        print("\n[dry-run] Would download:")
        for p in WAVEFAKE_PARTS:
            print(f"  • {p['name']} ({p['label']}) — {p['url']}")
        return

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Natural speech: LJSpeech ──────────────────────────────────────────
    ljspeech_archive = tmp_dir / "LJSpeech-1.1.tar.bz2"
    if not ljspeech_archive.exists():
        print("\n[1/2] Downloading LJSpeech (real speech) …")
        print("  Note: ~2.6 GB. This will take a while on slow connections.")
        print("  Tip : Run with --samples 100 to copy just 100 files after download.")
        _download(WAVEFAKE_PARTS[0]["url"], ljspeech_archive)

    ljspeech_wavs = tmp_dir / "ljspeech_wavs"
    if not ljspeech_wavs.exists() or not any(ljspeech_wavs.glob("*.wav")):
        _extract_tar(ljspeech_archive, ljspeech_wavs, "LJSpeech-1.1/wavs", samples_per_class)

    n_natural = _copy_wavs(ljspeech_wavs, natural_dir, samples_per_class)
    print(f"  ✅ Natural : {n_natural} files → {natural_dir}")

    # ── AI speech: WaveFake generated ─────────────────────────────────────
    print("\n[2/2] Downloading WaveFake generated audio …")
    wf_archive = tmp_dir / "wavefake_generated.zip"
    if not wf_archive.exists():
        _download(WAVEFAKE_PARTS[1]["url"], wf_archive)

    import zipfile
    wf_wavs = tmp_dir / "wavefake_wavs"
    if not wf_wavs.exists() or not any(wf_wavs.rglob("*.wav")):
        wf_wavs.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting {wf_archive.name} …")
        with zipfile.ZipFile(wf_archive) as zf:
            wav_members = [m for m in zf.namelist() if m.endswith(".wav")][:samples_per_class * 6]
            for m in wav_members:
                zf.extract(m, wf_wavs)
        print(f"  Extracted to {wf_wavs}")

    n_ai = _copy_wavs(wf_wavs, ai_dir, samples_per_class)
    print(f"  ✅ AI      : {n_ai} files → {ai_dir}")

    print(f"\n  Cleaning up temp files …")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  WaveFake collection complete.")
    print(f"  Natural : {n_natural}  |  AI : {n_ai}")
    return n_natural, n_ai


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Download and organise WaveFake dataset.")
    p.add_argument("--output", default="data/raw", help="Output root directory")
    p.add_argument("--samples", type=int, default=200, help="Max samples per class")
    p.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    args = p.parse_args()

    collect(
        output_dir=Path(args.output),
        samples_per_class=args.samples,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
