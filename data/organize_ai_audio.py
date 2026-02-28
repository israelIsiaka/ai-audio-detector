"""
data/organize_ai_audio.py — Move all AI-generated audio into data/raw/ai_generated/

Sources:
    data/generated_audio/**/*.wav   (WaveFake vocoders)
    data/ai_generated/*.mp3         (ElevenLabs previews)

Destination:
    data/raw/ai_generated/          (flat directory, prefixed to avoid name collisions)

Usage:
    python data/organize_ai_audio.py
    python data/organize_ai_audio.py --dry-run
    python data/organize_ai_audio.py --copy   # keep originals
"""

import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SOURCES = [
    {
        "path": SCRIPT_DIR / "generated_audio",
        "extensions": (".wav", ".mp3", ".flac", ".ogg"),
        "prefix_with_parent": True,   # avoids collisions across vocoder subfolders
    },
    {
        "path": SCRIPT_DIR / "ai_generated",
        "extensions": (".wav", ".mp3", ".flac", ".ogg"),
        "prefix_with_parent": False,
    },
]

DEST = PROJECT_ROOT / "data" / "raw" / "ai_generated"


def collect_files(source: dict) -> list[tuple[Path, str]]:
    """Return list of (file_path, destination_filename) for a source."""
    root: Path = source["path"]
    exts = source["extensions"]
    prefix = source["prefix_with_parent"]

    if not root.exists():
        print(f"  [skip] {root} — does not exist")
        return []

    results = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            if prefix:
                # e.g. ljspeech_hifiGAN_LJ001-0001_generated.wav
                dest_name = f"{f.parent.name}_{f.name}"
            else:
                dest_name = f.name
            results.append((f, dest_name))
    return results


def main():
    p = argparse.ArgumentParser(description="Move AI audio into data/raw/ai_generated/")
    p.add_argument("--dry-run", action="store_true", help="Show what would be moved, do nothing")
    p.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    args = p.parse_args()

    action = "copy" if args.copy else "move"
    print(f"\n{'═'*60}")
    print(f"  Organise AI audio — {action} to {DEST}")
    print(f"{'═'*60}")

    if args.dry_run:
        print("  [dry-run mode — no files will be moved]\n")

    if not args.dry_run:
        DEST.mkdir(parents=True, exist_ok=True)

    total_moved = 0
    total_skipped = 0
    total_conflict = 0

    for source in SOURCES:
        files = collect_files(source)
        print(f"\n  Source : {source['path']}")
        print(f"  Found  : {len(files)} audio files")

        for src_path, dest_name in files:
            dest_path = DEST / dest_name

            if dest_path.exists():
                print(f"  [skip]  {dest_name} — already exists in destination")
                total_skipped += 1
                continue

            if args.dry_run:
                print(f"  [dry]   {src_path.name} → {dest_name}")
                total_moved += 1
                continue

            try:
                if args.copy:
                    shutil.copy2(src_path, dest_path)
                else:
                    shutil.move(str(src_path), dest_path)
                total_moved += 1
            except Exception as e:
                print(f"  [error] {dest_name}: {e}")
                total_conflict += 1

    print(f"\n{'─'*60}")
    print(f"  {'Would move' if args.dry_run else 'Moved'} : {total_moved} files")
    if total_skipped:
        print(f"  Skipped  : {total_skipped} files (already in destination)")
    if total_conflict:
        print(f"  Errors   : {total_conflict} files")
    print(f"  Output   : {DEST}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
