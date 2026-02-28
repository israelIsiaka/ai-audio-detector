"""
data/collect_elevenlabs.py — Bulk-generate AI speech via ElevenLabs API.

Generates diverse AI speech samples across multiple voices, speaking styles,
and text content — maximising variety for training data.

Usage:
    export ELEVENLABS_API_KEY=your_key_here
    python data/collect_elevenlabs.py
    python data/collect_elevenlabs.py --samples 100 --output data/raw

Produces:
    data/raw/ai_generated/elevenlabs_*.mp3
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Default voices — a mix of styles to maximise acoustic variety
# Get voice IDs from: GET https://api.elevenlabs.io/v1/voices
# ---------------------------------------------------------------------------

DEFAULT_VOICES = [
    {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    {"id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
    {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
    {"id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
    {"id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
    {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold"},
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
]

# Varied texts — different lengths, punctuation styles, emotional content
# Diversity here improves generalisation
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Scientists have discovered a new species of deep-sea fish that produces its own light.",
    "Please remember to take your medication after breakfast every morning.",
    "The stock market closed higher today, with technology shares leading the gains.",
    "I really can't believe how quickly the time has passed this year.",
    "In 1969, Neil Armstrong became the first human to walk on the surface of the Moon.",
    "The weather forecast predicts heavy rainfall across the southern regions tomorrow.",
    "She walked slowly through the old library, running her fingers along the dusty spines.",
    "Welcome to the annual conference on artificial intelligence and machine learning.",
    "The patient presented with mild symptoms that resolved after two days of rest.",
    "Could you please pass the salt? This soup needs a little more seasoning.",
    "The new policy will take effect on the first day of next month.",
    "Three hundred students gathered in the auditorium for the graduation ceremony.",
    "He sat by the window watching the rain fall on the empty street below.",
    "Our team has been working on this project for the past six months.",
    "The ancient ruins were discovered by archaeologists working in the desert.",
    "I need you to submit the quarterly report by end of business Friday.",
    "The concert was absolutely spectacular — the crowd was on their feet all night.",
    "Air traffic controllers manage thousands of flights every single day.",
    "She paused for a moment, then smiled and continued walking toward the door.",
]

# Model settings
TTS_MODEL = "eleven_monolingual_v1"
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
}

# Rate limiting — ElevenLabs free tier: ~10k chars/month, ~2 req/sec
REQUEST_DELAY_SECONDS = 0.6


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        print(
            "[error] ELEVENLABS_API_KEY not set.\n"
            "  export ELEVENLABS_API_KEY=your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _list_voices(api_key: str) -> list[dict]:
    """Fetch available voices from ElevenLabs account."""
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": api_key, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return data.get("voices", [])


def _generate_speech(api_key: str, voice_id: str, text: str) -> bytes:
    """Call ElevenLabs TTS API, return raw MP3 bytes."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = json.dumps(
        {"text": text, "model_id": TTS_MODEL, "voice_settings": VOICE_SETTINGS}
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _get_subscription_info(api_key: str) -> dict:
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/user/subscription",
        headers={"xi-api-key": api_key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Collection logic
# ---------------------------------------------------------------------------

def collect(
    output_dir: Path,
    samples: int,
    voices: list[dict] | None = None,
    texts: list[str] | None = None,
    dry_run: bool = False,
):
    api_key = _get_api_key()
    ai_dir = output_dir / "ai_generated"

    if voices is None:
        voices = DEFAULT_VOICES
    if texts is None:
        texts = SAMPLE_TEXTS

    print("\n═══ ElevenLabs Collection ═══")

    # Check subscription headroom
    sub = _get_subscription_info(api_key)
    if sub:
        used = sub.get("character_count", 0)
        limit = sub.get("character_limit", 0)
        remaining = limit - used
        est_chars = sum(len(t) for t in texts[:samples]) + samples * 10
        print(f"  Quota remaining : {remaining:,} chars")
        print(f"  Estimated usage : ~{est_chars:,} chars for {samples} samples")
        if est_chars > remaining:
            print(f"  [warn] May exceed quota. Reduce --samples or upgrade plan.")

    print(f"  Voices  : {len(voices)}")
    print(f"  Texts   : {len(texts)}")
    print(f"  Target  : {samples} samples")
    print(f"  Output  : {ai_dir.resolve()}")

    if dry_run:
        print("\n[dry-run] Would generate:")
        for i, (v, t) in enumerate(
            _sample_combinations(voices, texts, samples)
        ):
            print(f"  [{i+1:03d}] voice={v['name']}  text={t[:50]}…")
        return

    ai_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed = 0
    manifest = []

    combos = list(_sample_combinations(voices, texts, samples))

    for i, (voice, text) in enumerate(combos, 1):
        filename = f"elevenlabs_{voice['name'].lower()}_{i:04d}.mp3"
        out_path = ai_dir / filename

        if out_path.exists():
            print(f"  [{i:03d}/{len(combos)}] skip  {filename} (exists)")
            generated += 1
            continue

        try:
            print(f"  [{i:03d}/{len(combos)}] gen   {voice['name']} — {text[:45]}…", end=" ")
            audio_bytes = _generate_speech(api_key, voice["id"], text)
            out_path.write_bytes(audio_bytes)
            size_kb = len(audio_bytes) / 1024
            print(f"✅ {size_kb:.0f} KB")
            manifest.append({"file": filename, "voice": voice["name"], "text": text})
            generated += 1
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            print(f"❌ HTTP {e.code}: {body[:80]}")
            failed += 1
            if e.code == 401:
                print("  [error] Invalid API key. Check ELEVENLABS_API_KEY.", file=sys.stderr)
                break
            if e.code == 429:
                print("  [warn] Rate limited. Waiting 5s …")
                time.sleep(5)
        except Exception as exc:
            print(f"❌ {exc}")
            failed += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    # Save manifest
    manifest_path = ai_dir / "elevenlabs_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  ✅ Generated : {generated} files → {ai_dir}")
    if failed:
        print(f"  ⚠️  Failed    : {failed} files")
    print(f"  Manifest   : {manifest_path}")
    return generated


def _sample_combinations(voices: list[dict], texts: list[str], n: int):
    """Round-robin voices × texts up to n samples."""
    combos = []
    i = 0
    while len(combos) < n:
        voice = voices[i % len(voices)]
        text = texts[i % len(texts)]
        combos.append((voice, text))
        i += 1
    return combos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Bulk-generate AI speech via ElevenLabs.")
    p.add_argument("--output", default="data/raw", help="Output root directory")
    p.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    p.add_argument("--list-voices", action="store_true", help="Print available voices and exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.list_voices:
        key = _get_api_key()
        voices = _list_voices(key)
        print(f"\nAvailable voices ({len(voices)}):")
        for v in voices:
            print(f"  {v['voice_id']}  {v['name']}  ({v.get('category', '?')})")
        return

    collect(
        output_dir=Path(args.output),
        samples=args.samples,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
