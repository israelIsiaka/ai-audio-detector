"""
data/collect_asvspoof.py — Download and organise ASVspoof 2019 LA dataset.

ASVspoof 2019 Logical Access (LA) contains:
  - Bonafide (real) speech: human recordings
  - Spoof speech: TTS + VC from 19 different AI systems (A01–A19)

Download requires free registration at:
    https://datashare.ed.ac.uk/handle/10283/3336

Usage:
    # Interactive: paste your download URL from the Edinburgh DataShare portal
    python data/collect_asvspoof.py --token YOUR_DOWNLOAD_TOKEN

    # If you've already downloaded the archive manually:
    python data/collect_asvspoof.py --local-archive ~/Downloads/LA.zip

    # Or use the smaller ASVspoof 2019 dev subset (no registration):
    python data/collect_asvspoof.py --subset dev --samples 200

Produces:
    data/raw/natural/       ← bonafide utterances
    data/raw/ai_generated/  ← spoof utterances (TTS/VC)
"""

import argparse
import csv
import os
import shutil
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# ASVspoof 2019 LA — Edinburgh DataShare
# Full dataset requires registration. Dev/eval subsets are ~1 GB each.
# ---------------------------------------------------------------------------

ASVSPOOF_BASE = "https://datashare.ed.ac.uk/download/DS_10283_3336"

# Alternative: ASVspoof 5 (2024) open subset hosted on HuggingFace
HUGGINGFACE_ASVSPOOF5 = (
    "https://huggingface.co/datasets/ASVspoof5/ASVspoof5-LA-subset/resolve/main/"
)

# Protocol file columns: SPEAKER_ID  FILE_ID  -  SYSTEM_ID  KEY(bonafide/spoof)
PROTOCOL_COL_FILE = 1
PROTOCOL_COL_SYSTEM = 3
PROTOCOL_COL_KEY = 4


def _parse_protocol(protocol_path: Path) -> dict[str, str]:
    """Return {file_id: 'bonafide'|'spoof'} from ASVspoof protocol file."""
    mapping = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                mapping[parts[PROTOCOL_COL_FILE]] = parts[PROTOCOL_COL_KEY]
    return mapping


def _organise_from_local(
    archive_path: Path,
    output_dir: Path,
    samples_per_class: int,
    subset: str = "train",
) -> tuple[int, int]:
    """Extract an already-downloaded ASVspoof LA zip and sort into label folders."""
    natural_dir = output_dir / "natural"
    ai_dir = output_dir / "ai_generated"
    natural_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)

    tmp = output_dir / "_tmp_asvspoof"
    tmp.mkdir(exist_ok=True)

    print(f"  Extracting {archive_path.name} …")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(tmp)

    # Locate protocol file
    protocol_files = list(tmp.rglob(f"*{subset}*.txt")) + list(tmp.rglob("*.trl.txt"))
    if not protocol_files:
        print("  [warn] Protocol file not found — labelling all files as ai_generated")
        protocol = {}
    else:
        protocol_file = protocol_files[0]
        print(f"  Using protocol: {protocol_file.name}")
        protocol = _parse_protocol(protocol_file)

    # Copy files to label dirs
    all_wavs = list(tmp.rglob("*.wav")) + list(tmp.rglob("*.flac"))
    n_natural = n_ai = 0

    for wav in all_wavs:
        stem = wav.stem
        key = protocol.get(stem, "spoof")  # default to spoof if not in protocol
        if key == "bonafide" and n_natural < samples_per_class:
            shutil.copy2(wav, natural_dir / wav.name)
            n_natural += 1
        elif key == "spoof" and n_ai < samples_per_class:
            shutil.copy2(wav, ai_dir / wav.name)
            n_ai += 1

        if n_natural >= samples_per_class and n_ai >= samples_per_class:
            break

    shutil.rmtree(tmp, ignore_errors=True)
    return n_natural, n_ai


def _download_with_token(token: str, output_dir: Path, samples_per_class: int):
    """Download ASVspoof 2019 LA using Edinburgh DataShare session token."""
    import urllib.request

    url = f"{ASVSPOOF_BASE}?token={token}"
    archive = output_dir / "_tmp_asvspoof" / "LA.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading ASVspoof 2019 LA (~1 GB) …")
    urllib.request.urlretrieve(url, archive)
    print(f"  Downloaded to {archive}")

    return _organise_from_local(archive, output_dir, samples_per_class)


def _instructions():
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  ASVspoof 2019 requires FREE registration to download.          │
  │                                                                 │
  │  Steps:                                                         │
  │  1. Go to https://datashare.ed.ac.uk/handle/10283/3336         │
  │  2. Click "Register" and create a free account                  │
  │  3. Accept the licence agreement                                │
  │  4. Download "LA.zip" (Logical Access, ~1.8 GB)                │
  │  5. Run:                                                        │
  │       python data/collect_asvspoof.py --local-archive LA.zip   │
  │                                                                 │
  │  Alternative — no registration needed:                          │
  │    Use the ASVspoof5 open subset on HuggingFace:               │
  │    https://huggingface.co/datasets/ASVspoof5                   │
  └─────────────────────────────────────────────────────────────────┘
""")


def collect(
    output_dir: Path,
    samples_per_class: int,
    local_archive: Path | None = None,
    token: str | None = None,
    dry_run: bool = False,
):
    print("\n═══ ASVspoof Collection ═══")
    print(f"  Target : {samples_per_class} samples per class")
    print(f"  Output : {output_dir.resolve()}")

    if dry_run:
        print("\n[dry-run] Would organise ASVspoof 2019 LA into:")
        print(f"  {output_dir}/natural/       ← bonafide utterances")
        print(f"  {output_dir}/ai_generated/  ← spoof utterances")
        return

    if local_archive:
        if not Path(local_archive).exists():
            print(f"  [error] Archive not found: {local_archive}", file=sys.stderr)
            sys.exit(1)
        n_nat, n_ai = _organise_from_local(
            Path(local_archive), output_dir, samples_per_class
        )
    elif token:
        n_nat, n_ai = _download_with_token(token, output_dir, samples_per_class)
    else:
        _instructions()
        return

    print(f"\n  ✅ Natural : {n_nat} files → {output_dir}/natural")
    print(f"  ✅ AI      : {n_ai} files → {output_dir}/ai_generated")
    print(f"\n  ASVspoof collection complete.")
    return n_nat, n_ai


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Organise ASVspoof 2019 LA dataset.")
    p.add_argument("--output", default="data/raw", help="Output root directory")
    p.add_argument("--samples", type=int, default=200, help="Max samples per class")
    p.add_argument("--local-archive", help="Path to already-downloaded LA.zip")
    p.add_argument("--token", help="Edinburgh DataShare download token")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    collect(
        output_dir=Path(args.output),
        samples_per_class=args.samples,
        local_archive=Path(args.local_archive) if args.local_archive else None,
        token=args.token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
