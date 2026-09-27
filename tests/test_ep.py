"""The Through the Wall EP is a split of the cached echo, not a new Atlas run."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atlas_music import (
    EP_DIR,
    FULL_PIECE,
    MUSIC_DIR,
    _join_piece_bytes,
    _movement_spans,
    ep_manifest,
    write_ep,
)

def _frames(path: Path) -> int:
    with wave.open(str(path)) as handle:
        return handle.getnframes()


def test_ep_splits_cached_audio_and_keeps_job_ids():
    echo = (MUSIC_DIR / "echo_roundtrip.wav").read_bytes()
    coda = (MUSIC_DIR / "mix_echo.wav").read_bytes()
    expected_piece = _join_piece_bytes(echo, coda)
    tracks = write_ep()
    by_file = {track["file"]: track for track in tracks}

    piece = (EP_DIR / "through_the_wall.wav").read_bytes()
    assert piece == expected_piece
    assert FULL_PIECE.read_bytes() == expected_piece

    with wave.open(str(MUSIC_DIR / "echo_roundtrip.wav")) as handle:
        spans = _movement_spans(handle.getnframes(), handle.getframerate())
    assert _frames(EP_DIR / "I-hydrogen.wav") == spans[0][1] - spans[0][0]
    assert _frames(EP_DIR / "II-deuterium.wav") == spans[1][1] - spans[1][0]
    assert _frames(EP_DIR / "III-tritium.wav") == spans[2][1] - spans[2][0]
    assert _frames(EP_DIR / "coda.wav") == _frames(MUSIC_DIR / "mix_echo.wav")
    assert (EP_DIR / "coda.wav").read_bytes() == coda

    notes = (EP_DIR / "liner_notes.md").read_text(encoding="utf-8")
    manifest = ep_manifest()
    assert "5.3 times as long" in notes
    assert "19.3 times as long" in notes
    for track in manifest:
        assert track["note"] in notes
        for job_id in (
            "484db3ab-4a72-4c55-9e69-bd26ac3246e8",
            "b6f554d7-089a-4204-9841-ef4ec8e8a25e",
            "3ce48680-cf94-47e7-9f38-369903d7ce21",
            "c4236288-d0a0-4728-9450-c4ecf2f18710",
        ):
            if job_id in track["jobs"]:
                assert job_id in notes
    assert by_file["II-deuterium.wav"]["note"].endswith("5.3 times as long.")
    assert by_file["III-tritium.wav"]["note"].endswith("19.3 times as long.")
    echo_id = json.loads((MUSIC_DIR / "echo_roundtrip.json").read_text(encoding="utf-8"))["job_id"]
    assert by_file["I-hydrogen.wav"]["jobs"].endswith(echo_id)
    assert "qrc-midi-v1" in by_file["coda.wav"]["jobs"]
    assert (EP_DIR / "cover.png").exists()
    assert (ROOT / "notebooks" / "media" / "through_the_wall" / "liner_notes.md").exists()
    print("ep checks passed")


if __name__ == "__main__":
    test_ep_splits_cached_audio_and_keeps_job_ids()
