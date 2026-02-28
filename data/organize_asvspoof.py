"""
data/organize_asvspoof.py — Copy ASVspoof 2019 LA files into raw data folders.

Reads the protocol files to identify bonafide (natural) vs spoof (AI-generated)
audio, then copies them into:
    data/raw/natural/        ← bonafide files
    data/raw/ai_generated/   ← spoof files

Usage:
    python data/organize_asvspoof.py
    python data/organize_asvspoof.py --dry-run
    python data/organize_asvspoof.py --splits train dev eval
"""

import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ASVSPOOF_DIR = SCRIPT_DIR / "DS_10283_3336" / "LA"

SPLITS = {
    "train": {
        "audio": ASVSPOOF_DIR / "ASVspoof2019_LA_train" / "flac",
        "protocol": ASVSPOOF_DIR / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
    },
    "dev": {
        "audio": ASVSPOOF_DIR / "ASVspoof2019_LA_dev" / "flac",
        "protocol": ASVSPOOF_DIR / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
    },
    "eval": {
        "audio": ASVSPOOF_DIR / "ASVspoof2019_LA_eval" / "flac",
        "protocol": ASVSPOOF_DIR / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt",
    },
}

NATURAL_DIR = PROJECT_ROOT / "data" / "raw" / "natural"
AI_DIR = PROJECT_ROOT / "data" / "raw" / "ai_generated"


def process_split(split_name: str, config: dict, dry_run: bool) -> dict:
    protocol: Path = config["protocol"]
    audio_dir: Path = config["audio"]

    if not protocol.exists():
        print(f"  [skip] Protocol not found: {protocol}")
        return {"bonafide": 0, "spoof": 0, "missing": 0}

    if not audio_dir.exists():
        print(f"  [skip] Audio dir not found: {audio_dir}")
        return {"bonafide": 0, "spoof": 0, "missing": 0}

    print(f"\n  Split : {split_name}")

    counts = {"bonafide": 0, "spoof": 0, "missing": 0, "skipped": 0}

    with open(protocol) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            filename = parts[1]   # e.g. LA_T_1138215
            label = parts[4]      # bonafide or spoof

            src = audio_dir / f"{filename}.flac"
            if not src.exists():
                counts["missing"] += 1
                continue

            dest_dir = NATURAL_DIR if label == "bonafide" else AI_DIR
            dest = dest_dir / f"asvspoof_{split_name}_{filename}.flac"

            if dest.exists():
                counts["skipped"] += 1
                continue

            if dry_run:
                print(f"  [dry] {label:<10} {src.name} → {dest_dir.name}/")
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

            counts[label] += 1

    print(f"    bonafide : {counts['bonafide']}")
    print(f"    spoof    : {counts['spoof']}")
    if counts["missing"]:
        print(f"    missing  : {counts['missing']}")
    if counts["skipped"]:
        print(f"    skipped  : {counts['skipped']} (already in destination)")

    return counts


def main():
    p = argparse.ArgumentParser(description="Organise ASVspoof 2019 LA into raw data folders.")
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "eval"],
                   default=["train", "dev", "eval"], help="Which splits to process")
    p.add_argument("--dry-run", action="store_true", help="Show plan without copying")
    args = p.parse_args()

    print(f"\n{'═'*60}")
    print("  ASVspoof 2019 LA — Organise into raw folders")
    print(f"{'═'*60}")
    print(f"  Source  : {ASVSPOOF_DIR}")
    print(f"  Natural : {NATURAL_DIR}")
    print(f"  AI      : {AI_DIR}")
    if args.dry_run:
        print("  [dry-run — no files will be copied]")

    total = {"bonafide": 0, "spoof": 0, "missing": 0, "skipped": 0}

    for split in args.splits:
        counts = process_split(split, SPLITS[split], args.dry_run)
        for k in total:
            total[k] += counts.get(k, 0)

    print(f"\n{'─'*60}")
    print(f"  Total natural copied : {total['bonafide']}")
    print(f"  Total spoof copied   : {total['spoof']}")
    if total["missing"]:
        print(f"  Missing files        : {total['missing']}")
    if total["skipped"]:
        print(f"  Skipped (existing)   : {total['skipped']}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
