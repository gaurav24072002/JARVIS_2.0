#!/usr/bin/env python3
"""Enroll a voice clip as a cloned voice in VoiceStudio (J.A.R.V.I.S pipeline).

Usage:
    venv/bin/python enroll_voice.py path/to/jarvis_clip.mp3 [--name JARVIS] [--ref-text "..."]

VoiceStudio must be running (uv run python -m backend.main in ~/JARVIS/voicestudio).
The script prints the VOICESTUDIO_VOICE line to paste into .env, then restart JARVIS.
"""

import argparse
import sys
from pathlib import Path

import requests

from config import VOICESTUDIO_URL, VOICESTUDIO_TIMEOUT

ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".pcm", ".aac", ".wma"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Path to a clean voice clip (mp3/wav/m4a/flac/ogg)")
    parser.add_argument("--name", default="JARVIS", help="Profile name (default: JARVIS)")
    parser.add_argument("--ref-text", default="", help="Transcript of the clip (improves cloning)")
    parser.add_argument("--url", default=None, help=f"VoiceStudio base URL (default: {VOICESTUDIO_URL})")
    args = parser.parse_args()

    url = (args.url or VOICESTUDIO_URL).rstrip("/")
    path = Path(args.audio).expanduser()

    if not path.is_file():
        print(f"File not found: {path}")
        return 1
    if path.suffix.lower() not in ALLOWED_EXTS:
        print(f"Unsupported extension {path.suffix!r}. Convert to mp3 or wav first, sir.")
        return 1
    if path.stat().st_size > 25 * 1024 * 1024:
        print("Clip looks large (>25 MB). A short, clean 10-30 second sample clones best.")
        return 1

    print(f"Checking VoiceStudio at {url} ...")
    try:
        health = requests.get(f"{url}/health", timeout=5)
        health.raise_for_status()
        info = health.json()
        print(f"  device: {info.get('device', '?')} | version: {info.get('version', '?')}")
    except Exception as exc:
        print(f"VoiceStudio is not reachable ({exc}). Start it first:\n"
              "  cd ~/JARVIS/voicestudio && uv run python -m backend.main")
        return 1

    print(f"Enrolling {args.name!r} from {path.name} ({path.stat().st_size/1024:.0f} KB) ...")
    with path.open("rb") as fh:
        resp = requests.post(
            f"{url}/profiles",
            data={
                "name": args.name,
                "ref_text": args.ref_text,
                "language": "Auto",
                "kind": "clone",
            },
            files={"ref_audio": (path.name, fh, "application/octet-stream")},
            timeout=max(60, VOICESTUDIO_TIMEOUT),
        )

    if resp.status_code != 200:
        print(f"Enrollment failed (HTTP {resp.status_code}): {resp.text[:500]}")
        return 1

    profile = resp.json()
    profile_id = profile.get("id") or (profile.get("profile") or {}).get("id")
    if not profile_id:
        print(f"Unexpected response: {resp.text[:500]}")
        return 1

    print()
    print("Cloned voice enrolled. Add this line to .env:")
    print(f"  VOICESTUDIO_VOICE={profile_id}")
    print()
    print("Then restart JARVIS and say something worth a voice, sir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
