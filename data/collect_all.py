"""
data/collect_all.py — Master data collection orchestrator.

Runs all three collectors in sequence, then auto-runs dataset_builder.py
and optionally retrains the model — giving you a retrain-ready CSV in one command.

Usage:
    # Full pipeline (WaveFake + ElevenLabs, skip ASVspoof until you have the archive)
    python data/collect_all.py --elevenlabs-key YOUR_KEY

    # Everything including ASVspoof (if you have the archive)
    python data/collect_all.py --elevenlabs-key YOUR_KEY --asvspoof-archive ~/Downloads/LA.zip

    # Dry run — show the plan
    python data/collect_all.py --dry-run

    # Skip download, just rebuild CSV + retrain from existing data/raw/
    python data/collect_all.py --skip-download --retrain

Produces:
    data/raw/natural/           ← all real speech
    data/raw/ai_generated/      ← all AI speech
    data/dataset.csv            ← labelled features CSV
    models/detector.pkl         ← retrained model (if --retrain)
    data/collection_report.json ← run summary
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Resolve project root (works whether run from project root or data/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_CSV = DATA_DIR / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def _run(cmd: list[str], desc: str, env: dict | None = None) -> tuple[bool, str]:
    """Run a subprocess, stream output, return (success, output)."""
    print(f"\n{'─'*60}")
    print(f"  ▶  {desc}")
    print(f"{'─'*60}")

    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=merged_env,
        capture_output=False,  # stream to terminal in real time
        text=True,
    )
    success = result.returncode == 0
    if not success:
        print(f"  [warn] Command exited with code {result.returncode}")
    return success, ""


def _count_files(directory: Path, extensions=(".wav", ".mp3", ".flac", ".ogg")) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in extensions)


def _print_status(label: str, value):
    print(f"  {'·'*2} {label:<30} {value}")


def collect_all(
    elevenlabs_key: str | None,
    asvspoof_archive: Path | None,
    samples_per_class: int,
    elevenlabs_samples: int,
    skip_wavefake: bool,
    skip_asvspoof: bool,
    skip_elevenlabs: bool,
    skip_download: bool,
    retrain: bool,
    dry_run: bool,
):
    start_time = time.time()
    report = {
        "run_at": datetime.now().isoformat(),
        "dry_run": dry_run,
        "steps": {},
    }

    print("\n" + "═" * 60)
    print("  AI Audio Detector — Data Collection Pipeline")
    print("═" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Output dir   : {RAW_DIR}")
    print(f"  Samples/class: {samples_per_class} (WaveFake/ASVspoof)")
    print(f"  ElevenLabs   : {elevenlabs_samples} samples")
    print(f"  Dry run      : {dry_run}")

    if dry_run:
        print("\n  [dry-run mode — no files will be downloaded or generated]\n")

    # ── Step 1: WaveFake ─────────────────────────────────────────────────
    if not skip_download and not skip_wavefake:
        args = [
            sys.executable, str(SCRIPT_DIR / "collect_wavefake.py"),
            "--output", str(RAW_DIR),
            "--samples", str(samples_per_class),
        ]
        if dry_run:
            args.append("--dry-run")

        ok, _ = _run(args, "Step 1/4 — WaveFake download + organise")
        report["steps"]["wavefake"] = "ok" if ok else "failed"
    else:
        print("\n  Step 1/4 — WaveFake  [SKIPPED]")
        report["steps"]["wavefake"] = "skipped"

    # ── Step 2: ASVspoof ─────────────────────────────────────────────────
    if not skip_download and not skip_asvspoof and asvspoof_archive:
        args = [
            sys.executable, str(SCRIPT_DIR / "collect_asvspoof.py"),
            "--output", str(RAW_DIR),
            "--samples", str(samples_per_class),
            "--local-archive", str(asvspoof_archive),
        ]
        if dry_run:
            args.append("--dry-run")

        ok, _ = _run(args, "Step 2/4 — ASVspoof organise")
        report["steps"]["asvspoof"] = "ok" if ok else "failed"
    else:
        reason = "no archive provided" if not asvspoof_archive else "skipped"
        print(f"\n  Step 2/4 — ASVspoof  [SKIPPED — {reason}]")
        if not asvspoof_archive:
            print("    → Download LA.zip from https://datashare.ed.ac.uk/handle/10283/3336")
            print("    → Then re-run with: --asvspoof-archive ~/Downloads/LA.zip")
        report["steps"]["asvspoof"] = "skipped"

    # ── Step 3: ElevenLabs ───────────────────────────────────────────────
    if not skip_download and not skip_elevenlabs:
        if not elevenlabs_key and not os.environ.get("ELEVENLABS_API_KEY"):
            print("\n  Step 3/4 — ElevenLabs  [SKIPPED — no API key]")
            print("    → Set ELEVENLABS_API_KEY or pass --elevenlabs-key YOUR_KEY")
            report["steps"]["elevenlabs"] = "skipped"
        else:
            env = {}
            if elevenlabs_key:
                env["ELEVENLABS_API_KEY"] = elevenlabs_key

            args = [
                sys.executable, str(SCRIPT_DIR / "collect_elevenlabs.py"),
                "--output", str(RAW_DIR),
                "--samples", str(elevenlabs_samples),
            ]
            if dry_run:
                args.append("--dry-run")

            ok, _ = _run(args, "Step 3/4 — ElevenLabs bulk generation", env=env)
            report["steps"]["elevenlabs"] = "ok" if ok else "failed"
    else:
        print("\n  Step 3/4 — ElevenLabs  [SKIPPED]")
        report["steps"]["elevenlabs"] = "skipped"

    # ── Dataset inventory ────────────────────────────────────────────────
    n_natural = _count_files(RAW_DIR / "natural")
    n_ai = _count_files(RAW_DIR / "ai_generated")

    print(f"\n{'─'*60}")
    print("  Data inventory after collection:")
    _print_status("Natural speech files", n_natural)
    _print_status("AI-generated files", n_ai)
    _print_status("Total", n_natural + n_ai)

    report["inventory"] = {"natural": n_natural, "ai_generated": n_ai}

    if n_natural == 0 and n_ai == 0 and not dry_run:
        print("\n  [warn] No audio files found. Skipping CSV build + retrain.")
        _write_report(report, start_time)
        return

    # ── Step 4: Build dataset CSV ────────────────────────────────────────
    if not dry_run:
        DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
        ok, _ = _run(
            [
                sys.executable, str(SRC_DIR / "dataset_builder.py"),
                str(RAW_DIR / "natural"),
                str(RAW_DIR / "ai_generated"),
                "--output", str(DATASET_CSV),
            ],
            "Step 4/5 — Build labelled CSV (dataset_builder.py)",
        )
        report["steps"]["dataset_builder"] = "ok" if ok else "failed"

        if ok and DATASET_CSV.exists():
            # Quick row count
            with open(DATASET_CSV) as f:
                rows = sum(1 for _ in f) - 1  # subtract header
            print(f"\n  ✅ Dataset CSV : {DATASET_CSV} ({rows} rows)")
            report["csv_rows"] = rows
    else:
        print("\n  Step 4/5 — Build CSV  [dry-run, skipped]")

    # ── Step 5: Retrain model ────────────────────────────────────────────
    if retrain and not dry_run:
        MODELS_DIR.mkdir(exist_ok=True)
        ok, _ = _run(
            [sys.executable, str(SRC_DIR / "model.py"), "--dataset", str(DATASET_CSV)],
            "Step 5/5 — Retrain model (model.py)",
        )
        report["steps"]["retrain"] = "ok" if ok else "failed"
        if ok:
            print(f"\n  ✅ Model saved to {MODELS_DIR}/detector.pkl")
    elif retrain:
        print("\n  Step 5/5 — Retrain  [dry-run, skipped]")
    else:
        print("\n  Step 5/5 — Retrain  [SKIPPED — pass --retrain to enable]")
        print(f"    → When ready: python src/model.py --dataset {DATASET_CSV}")
        report["steps"]["retrain"] = "skipped"

    _write_report(report, start_time)


def _write_report(report: dict, start_time: float):
    elapsed = round(time.time() - start_time, 1)
    report["elapsed_seconds"] = elapsed
    report_path = DATA_DIR / "collection_report.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'═'*60}")
    print(f"  Pipeline complete in {elapsed:.0f}s")
    print(f"  Report : {report_path}")
    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Master data collection pipeline — WaveFake + ASVspoof + ElevenLabs → CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect everything + build CSV (no retrain yet)
  python data/collect_all.py --elevenlabs-key sk_...

  # Full pipeline including ASVspoof + auto-retrain
  python data/collect_all.py \\
      --elevenlabs-key sk_... \\
      --asvspoof-archive ~/Downloads/LA.zip \\
      --retrain

  # Skip downloads (data already in data/raw/) — just rebuild CSV + retrain
  python data/collect_all.py --skip-download --retrain

  # Dry run
  python data/collect_all.py --dry-run
        """,
    )
    p.add_argument("--elevenlabs-key", help="ElevenLabs API key (or set ELEVENLABS_API_KEY)")
    p.add_argument("--asvspoof-archive", help="Path to ASVspoof 2019 LA.zip")
    p.add_argument("--samples", type=int, default=200, help="Samples per class for WaveFake/ASVspoof")
    p.add_argument("--elevenlabs-samples", type=int, default=100, help="ElevenLabs samples to generate")
    p.add_argument("--skip-wavefake", action="store_true")
    p.add_argument("--skip-asvspoof", action="store_true")
    p.add_argument("--skip-elevenlabs", action="store_true")
    p.add_argument("--skip-download", action="store_true", help="Skip all downloads, just build CSV")
    p.add_argument("--retrain", action="store_true", help="Auto-retrain model after building CSV")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    collect_all(
        elevenlabs_key=args.elevenlabs_key,
        asvspoof_archive=Path(args.asvspoof_archive) if args.asvspoof_archive else None,
        samples_per_class=args.samples,
        elevenlabs_samples=args.elevenlabs_samples,
        skip_wavefake=args.skip_wavefake,
        skip_asvspoof=args.skip_asvspoof,
        skip_elevenlabs=args.skip_elevenlabs,
        skip_download=args.skip_download,
        retrain=args.retrain,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()