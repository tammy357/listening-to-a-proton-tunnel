"""A thirty-second tunneling song, then a qubit-chain echo.

``retrocausal-echo-v1`` is a media engine. Its schema says that when no ``ir``
file is attached it measures the impulse response itself, so ``otoc-echo-v1``
is not a required first step. ``machine`` accepts ``aer`` or an IBM backend
name. No IBM backend is enumerated, and this module never sends a token, so
the saved runs use ``aer`` and are labelled a simulator.

``qrc-midi-v1`` is the documented MIDI path: one call learns a ``.mid`` and
writes a new one. ``qrc-train-v2`` / ``qrc-gen-v2`` speak token sequences, not
MIDI files. ``blur-midi-v1`` is the optional piano-roll blur.

Every engine result is cached under ``atlas_music/`` with a sidecar JSON
(engine, job_id, params, provider, backend). With no ``MOTH_API_KEY`` the
loaders read that cache and do not call the API.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.moth_client import MothClient, MothError
from src.moth_sonification import DEFAULT_PITCH_HZ, DEFAULT_SAMPLE_RATE, pcm_wav_bytes, tunneling_beat
from src.quantum_sim import evolve_exact

MUSIC_DIR = ROOT / "atlas_music"
BANK_PATH = ROOT / "src" / "atlas_bank.json"
SAMPLE_RATE = DEFAULT_SAMPLE_RATE
TRACK_SECONDS = 30.0
FADE_SECONDS = 1.0
BARRIER_EV = 0.15
HALF_WIDTH_A = 0.35
PERIODS = 4
TPQ = 480
BPM = 120
BARS = 16
# Quarter notes across 16 bars of 4/4.
CONTINUATION_NOTES = BARS * 4
PITCHES = {"H": 60, "D": 55, "T": 48}  # C4, G3, C3

# Schema fields, not guesses. exact=true is aer-only and ignores shots.
ECHO_PARAMS = {
    "machine": "aer",
    "exact": True,
    "emit": "audio",
    "include_tap_map": True,
    "mix": 1.0,
    "n_sites": 8,
    "lattice": "chain",
    "negative_mode": "invert",
}
QRC_PARAMS = {
    "bpm": BPM,
    "length": CONTINUATION_NOTES,
    "loop": True,
    "quality": "fast",
    "seed": 7,
    "variation": 1.0,
    "velocity": 72,
}
BLUR_PARAMS = {
    "strength": 0.35,
    "reach": 0.25,
    "qubits": 8,
    "threshold": 0.1,
    "resolution": 0,
}

CAPTION = (
    "The proton's round trip, echoed by a measurement that runs time backwards. "
    "The echo comes from a qubit chain, not from the proton."
)
SIMULATOR_NOTE = (
    "machine is aer with exact true. The schema also accepts an IBM backend "
    "name such as ibm_phoenix, but it does not enumerate one, and no IBM token "
    "is sent. This run is a simulator, not hardware."
)


def _key_set() -> bool:
    return bool(os.environ.get("MOTH_API_KEY", "").strip())


def join_roundtrip(segments: list[np.ndarray], sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Join three stereo clips into ``TRACK_SECONDS`` with ``FADE_SECONDS`` crossfades.

    Segment length is chosen so three pieces and two overlaps land on exactly
    thirty seconds at the given sample rate.
    """

    if len(segments) != 3:
        raise ValueError("The round trip needs hydrogen, deuterium, and tritium.")
    n_fade = int(round(FADE_SECONDS * sample_rate))
    n_total = int(round(TRACK_SECONDS * sample_rate))
    n_seg = (n_total + 2 * n_fade) // 3
    if 3 * n_seg - 2 * n_fade != n_total:
        raise ValueError("Sample rate does not divide the 30 s track with 1 s fades.")
    pieces = []
    for segment in segments:
        audio = np.asarray(segment, dtype=np.float64)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("Each clip must be stereo, shape (n, 2).")
        pieces.append(_fit_length(audio, n_seg))
    mixed = np.zeros((n_total, 2), dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, np.pi / 2.0, n_fade, endpoint=False))
    for index, piece in enumerate(pieces):
        window = np.ones(n_seg, dtype=np.float64)
        if index > 0:
            window[:n_fade] = ramp
        if index < 2:
            window[-n_fade:] = ramp[::-1]
        start = index * (n_seg - n_fade)
        mixed[start : start + n_seg] += piece * window[:, None]
    peak = float(np.max(np.abs(mixed))) if mixed.size else 1.0
    if peak > 0.98:
        mixed *= 0.98 / peak
    return mixed


def _fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    if audio.shape[0] == n_samples:
        return audio
    source_x = np.linspace(0.0, 1.0, audio.shape[0])
    target_x = np.linspace(0.0, 1.0, n_samples)
    fitted = np.column_stack(
        [np.interp(target_x, source_x, audio[:, channel]) for channel in range(2)]
    )
    return fitted


def barrier_crossings(bank: dict, periods: int = PERIODS) -> list[str]:
    """Isotope labels for each time the bank's ideal acceptor curve crosses 1/2.

    The bank covers one tunneling period. It is tiled ``periods`` times. A
    crossing is a sign change of ``ideal_acceptor - 1/2``. Order is H, then D,
    then T, eight crossings in four periods when the two-level curve is smooth.
    """

    points = sorted(bank["points"], key=lambda point: float(point["t_over_period"]))
    series = [float(point["ideal_acceptor"]) for point in points]
    if len(series) < 2:
        raise ValueError("atlas_bank.json needs at least two time points.")
    tiled = series * periods
    crossed = _crossing_indices(tiled)
    return ["H"] * len(crossed) + ["D"] * len(crossed) + ["T"] * len(crossed)


def _crossing_indices(series: list[float]) -> list[int]:
    hits: list[int] = []
    previous = series[0] - 0.5
    for index in range(1, len(series)):
        current = series[index] - 0.5
        if previous == 0.0:
            previous = current
            continue
        if previous * current < 0.0 or current == 0.0:
            hits.append(index)
            previous = 0.0 if current == 0.0 else current
            continue
        previous = current
    return hits


def pitches_for(isotopes: list[str]) -> list[int]:
    return [PITCHES[name] for name in isotopes]


def write_midi(pitches: list[int], path: Path, bpm: int = BPM) -> Path:
    """One quarter note per pitch, type-0 MIDI, a single tempo track."""

    if not pitches:
        raise ValueError("A MIDI file needs at least one note.")
    events: list[tuple[int, bytes]] = [(0, bytes([0xFF, 0x51, 0x03]) + _tempo_bytes(bpm))]
    for index, pitch in enumerate(pitches):
        start = index * TPQ
        end = start + TPQ
        events.append((start, bytes([0x90, int(pitch) & 0x7F, 80])))
        events.append((end, bytes([0x80, int(pitch) & 0x7F, 0])))
    events.append((len(pitches) * TPQ, bytes([0xFF, 0x2F, 0x00])))
    events.sort(key=lambda item: item[0])
    track = _encode_track(events)
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, TPQ)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + track)
    return path


def read_notes(path: Path) -> tuple[list[tuple[int, float, float]], int]:
    """Return ``(pitch, start_seconds, duration_seconds)`` and the file tempo in BPM."""

    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"{path} is not a MIDI file.")
    _length, _fmt, n_tracks, division = struct.unpack(">IHHH", data[4:14])
    if division & 0x8000:
        raise ValueError("SMPTE MIDI timing is not used here.")
    cursor = 14
    bpm = BPM
    notes: list[tuple[int, float, float]] = []
    open_notes: dict[int, tuple[int, int]] = {}
    for _track in range(n_tracks):
        if data[cursor : cursor + 4] != b"MTrk":
            break
        track_len = struct.unpack(">I", data[cursor + 4 : cursor + 8])[0]
        body = data[cursor + 8 : cursor + 8 + track_len]
        cursor += 8 + track_len
        tick = 0
        pos = 0
        status = 0
        while pos < len(body):
            delta, pos = _read_vlq(body, pos)
            tick += delta
            if pos >= len(body):
                break
            byte = body[pos]
            if byte >= 0x80:
                status = byte
                pos += 1
            if status == 0xFF:
                meta = body[pos]
                pos += 1
                size, pos = _read_vlq(body, pos)
                payload = body[pos : pos + size]
                pos += size
                if meta == 0x51 and size == 3:
                    microseconds = int.from_bytes(payload, "big")
                    if microseconds > 0:
                        bpm = int(round(60_000_000 / microseconds))
                continue
            if status in (0x90, 0x80):
                pitch = body[pos]
                velocity = body[pos + 1]
                pos += 2
                if status == 0x90 and velocity > 0:
                    open_notes[pitch] = (tick, velocity)
                else:
                    started = open_notes.pop(pitch, None)
                    if started is not None:
                        start_tick, _velocity = started
                        seconds_per_tick = (60.0 / bpm) / division
                        notes.append(
                            (
                                int(pitch),
                                start_tick * seconds_per_tick,
                                max(0.02, (tick - start_tick) * seconds_per_tick),
                            )
                        )
                continue
            # Skip other channel messages by their usual length.
            length = 1 if status in (0xC0, 0xD0) else 2
            pos += length
    notes.sort(key=lambda item: item[1])
    return notes, bpm


def render_midi(path: Path, sample_rate: int = SAMPLE_RATE) -> bytes:
    """A soft sine synth: each note decays, and the mix is peak-normalised."""

    notes, _bpm = read_notes(path)
    if not notes:
        raise ValueError(f"{path} has no notes to play.")
    end = max(start + duration for _pitch, start, duration in notes) + 0.4
    n_samples = int(np.ceil(end * sample_rate))
    mix = np.zeros(n_samples, dtype=np.float64)
    for pitch, start, duration in notes:
        start_i = int(start * sample_rate)
        length = max(1, int((duration + 0.25) * sample_rate))
        times = np.arange(length, dtype=np.float64) / sample_rate
        freq = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        envelope = np.exp(-times / max(duration * 0.45, 0.05))
        window = np.sin(np.pi * times / times[-1]) if length > 1 else np.ones(1)
        tone = 0.22 * np.sin(2.0 * np.pi * freq * times) * envelope * window
        stop = min(n_samples, start_i + length)
        mix[start_i:stop] += tone[: stop - start_i]
    peak = float(np.max(np.abs(mix))) or 1.0
    mix *= 0.9 / peak
    stereo = np.column_stack((mix, mix))
    return pcm_wav_bytes(stereo, sample_rate)


def build_dry_track(path: Path | None = None) -> Path:
    """H, then D, then T, each a tunneling beat, joined to thirty seconds."""

    destination = path or (MUSIC_DIR / "dry_roundtrip.wav")
    clips = []
    n_fade = int(round(FADE_SECONDS * SAMPLE_RATE))
    n_total = int(round(TRACK_SECONDS * SAMPLE_RATE))
    n_seg = (n_total + 2 * n_fade) // 3
    seconds = n_seg / SAMPLE_RATE
    for isotope in ("H", "D", "T"):
        _model, evolution = evolve_exact(
            barrier_ev=BARRIER_EV,
            asymmetry_ev=0.0,
            well_half_width_angstrom=HALF_WIDTH_A,
            mass=isotope,
            n_periods=1.0,
            n_steps=64,
        )
        clip = tunneling_beat(
            evolution.donor_population,
            evolution.acceptor_population,
            float(evolution.splitting_ev),
            base_pitch_hz=DEFAULT_PITCH_HZ,
            sample_rate=SAMPLE_RATE,
            seconds=seconds,
        )
        clips.append(_wav_samples(clip.wav_bytes))
    mixed = join_roundtrip(clips, SAMPLE_RATE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pcm_wav_bytes(mixed, SAMPLE_RATE))
    return destination


def build_rhythm(path: Path | None = None, bank_path: Path = BANK_PATH) -> tuple[Path, list[str]]:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    isotopes = barrier_crossings(bank, PERIODS)
    destination = path or (MUSIC_DIR / "tunnel_rhythm.mid")
    write_midi(pitches_for(isotopes), destination, BPM)
    return destination, isotopes


def load_song() -> dict:
    """Return the cached song, or submit the engines when ``MOTH_API_KEY`` is set."""

    if _key_set():
        print("path: live — MOTH_API_KEY is set, submitting the song engines")
        return write_song()
    print("path: cache — MOTH_API_KEY is not set, loading atlas_music/")
    return read_song()


def read_song() -> dict:
    required = [
        MUSIC_DIR / "dry_roundtrip.wav",
        MUSIC_DIR / "echo_roundtrip.wav",
        MUSIC_DIR / "echo_roundtrip.json",
        MUSIC_DIR / "tunnel_rhythm.mid",
        MUSIC_DIR / "qrc_continuation.mid",
        MUSIC_DIR / "qrc_continuation.json",
        MUSIC_DIR / "mix_dry.wav",
        MUSIC_DIR / "mix_echo.wav",
        MUSIC_DIR / "mix_echo.json",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "atlas_music cache is missing "
            + ", ".join(missing)
            + ". Set MOTH_API_KEY to build it."
        )
    song = {
        "dry_wav": required[0].read_bytes(),
        "echo_wav": required[1].read_bytes(),
        "echo": json.loads(required[2].read_text(encoding="utf-8")),
        "rhythm_mid": required[3],
        "qrc_mid": required[4],
        "qrc": json.loads(required[5].read_text(encoding="utf-8")),
        "mix_dry_wav": required[6].read_bytes(),
        "mix_echo_wav": required[7].read_bytes(),
        "mix_echo": json.loads(required[8].read_text(encoding="utf-8")),
        "taps": _read_json(MUSIC_DIR / "echo_roundtrip_taps.json"),
        "blur": _read_json(MUSIC_DIR / "blur_continuation.json"),
        "blur_mid": MUSIC_DIR / "blur_continuation.mid",
    }
    return song


def write_song() -> dict:
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    dry = build_dry_track()
    rhythm, isotopes = build_rhythm()
    (MUSIC_DIR / "tunnel_rhythm.json").write_text(
        json.dumps(
            {
                "source": "src/atlas_bank.json ideal_acceptor crossings",
                "periods": PERIODS,
                "isotopes": isotopes,
                "pitches": PITCHES,
                "bpm": BPM,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    client = MothClient()
    echo = _echo_file(client, dry, "echo_roundtrip")
    qrc_mid = _qrc_midi(client, rhythm)
    try:
        blur_mid = _blur_midi(client, qrc_mid)
    except MothError as exc:
        print(f"blur-midi-v1 skipped: {exc}", flush=True)
        blur_mid = qrc_mid
    source_mid = blur_mid if blur_mid.exists() else qrc_mid
    mix_dry = MUSIC_DIR / "mix_dry.wav"
    mix_dry.write_bytes(render_midi(source_mid))
    mix_echo = _echo_file(client, mix_dry, "mix_echo")
    song = read_song()
    song["echo"] = echo
    song["mix_echo"] = mix_echo
    return song


def _echo_file(client: MothClient, wav_path: Path, stem: str) -> dict:
    asset = client.upload_asset(wav_path, "audio/wav")
    job_id, payload = _submit_and_wait(
        client,
        "retrocausal-echo-v1",
        ECHO_PARAMS,
        {"audio": asset},
    )
    outputs = _index_outputs(payload)
    audio_url = _first_url(outputs, ("audio", "wav", "result", "output"))
    if audio_url is None:
        raise MothError(f"{job_id} returned no audio URL. Slots: {list(outputs)}")
    audio_path = MUSIC_DIR / f"{stem}.wav"
    _download(audio_url, audio_path)
    taps_path = MUSIC_DIR / f"{stem}_taps.json"
    taps_url = outputs.get("taps")
    if taps_url:
        _download(taps_url, taps_path)
    ir_url = outputs.get("ir")
    if ir_url:
        _download(ir_url, MUSIC_DIR / f"{stem}_ir.json")
    sidecar = _sidecar(
        "retrocausal-echo-v1",
        job_id,
        ECHO_PARAMS,
        payload,
        outputs,
        taps_path=taps_path,
    )
    (MUSIC_DIR / f"{stem}.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"saved {audio_path.name} job={job_id}", flush=True)
    return sidecar


def _qrc_midi(client: MothClient, midi_path: Path) -> Path:
    asset = client.upload_asset(midi_path, "audio/midi")
    job_id, payload = _submit_and_wait(
        client,
        "qrc-midi-v1",
        QRC_PARAMS,
        {"midi": asset},
    )
    outputs = _index_outputs(payload)
    midi_url = _first_url(outputs, ("midi", "result", "output"))
    if midi_url is None:
        raise MothError(f"{job_id} returned no MIDI URL. Slots: {list(outputs)}")
    destination = MUSIC_DIR / "qrc_continuation.mid"
    _download(midi_url, destination)
    sidecar = _sidecar("qrc-midi-v1", job_id, QRC_PARAMS, payload, outputs)
    (MUSIC_DIR / "qrc_continuation.json").write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {destination.name} job={job_id}", flush=True)
    return destination


def _blur_midi(client: MothClient, midi_path: Path) -> Path:
    asset = client.upload_asset(midi_path, "audio/midi")
    job_id, payload = _submit_and_wait(
        client,
        "blur-midi-v1",
        BLUR_PARAMS,
        {"midi": asset},
    )
    outputs = _index_outputs(payload)
    midi_url = _first_url(outputs, ("midi", "result", "output"))
    if midi_url is None:
        raise MothError(f"{job_id} returned no MIDI URL. Slots: {list(outputs)}")
    destination = MUSIC_DIR / "blur_continuation.mid"
    _download(midi_url, destination)
    sidecar = _sidecar("blur-midi-v1", job_id, BLUR_PARAMS, payload, outputs)
    (MUSIC_DIR / "blur_continuation.json").write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {destination.name} job={job_id}", flush=True)
    return destination


def _sidecar(
    engine: str,
    job_id: str,
    params: dict,
    payload: dict,
    outputs: dict,
    taps_path: Path | None = None,
) -> dict:
    inline = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if not inline and isinstance(payload.get("output"), dict):
        inline = payload["output"]
    taps = None
    for key in ("taps", "tap_map", "trajectory"):
        if key in payload:
            taps = payload[key]
            break
        if isinstance(inline, dict) and key in inline:
            taps = inline[key]
            break
    extras = payload.get("extras") if isinstance(payload.get("extras"), dict) else {}
    if taps is None and isinstance(extras.get("taps"), list):
        taps = extras["taps"]
    provider, backend, note = _provenance(engine, params)
    record = {
        "engine": engine,
        "job_id": job_id,
        "params": params,
        "provider": provider,
        "backend": backend,
        "execution": "simulator",
        "simulator_note": note,
        "output_slots": sorted(outputs),
    }
    if taps is not None:
        preview = taps[:12] if isinstance(taps, list) else taps
        record["tap_preview"] = preview
        if taps_path is not None and not taps_path.exists() and isinstance(taps, (list, dict)):
            taps_path.write_text(json.dumps(taps, indent=2) + "\n", encoding="utf-8")
    return record


def _provenance(engine: str, params: dict) -> tuple[str, str, str]:
    """Provider and backend taken from the engine schema, never invented hardware."""

    if engine == "retrocausal-echo-v1":
        machine = str(params.get("machine", "aer"))
        return "aer", machine, SIMULATOR_NOTE
    if engine == "qrc-midi-v1":
        return (
            "aer",
            "aer",
            "qrc-midi-v1 is the documented MIDI path. It has no machine field. "
            "The OpenAPI text says the reservoir work goes to qrc_core, and "
            "qrc-train-v2 states that trainer uses Qiskit Aer. length is a note "
            "count; 64 quarter notes are 16 bars of 4/4. This run is a simulator.",
        )
    if engine == "blur-midi-v1":
        return (
            "unspecified",
            "quantum simulator",
            "blur-midi-v1 has no machine or provider field. Its schema describes "
            "a quantum simulation of piano-roll tiles. This run is a simulator.",
        )
    return "unspecified", "unspecified", ""


def _submit_and_wait(
    client: MothClient,
    engine: str,
    params: dict,
    files: dict[str, str],
) -> tuple[str, dict]:
    job_id = ""
    for attempt in range(4):
        if not job_id:
            job_id = client.submit(engine, params, input_files=files)
            print(f"submitted {engine} job={job_id}", flush=True)
        status = client.wait(job_id, timeout_s=900)
        if status.get("status") == "completed":
            return job_id, client.result(job_id)
        error = status.get("error") if isinstance(status.get("error"), dict) else {}
        print(f"{engine} job={job_id} ended {status.get('status')}: {error}", flush=True)
        if not error.get("retryable") or attempt == 3:
            raise MothError(f"{job_id} ended {status.get('status')}: {error}")
        job_id = ""
    raise MothError(f"{engine} did not complete.")


def _index_outputs(payload: dict) -> dict[str, str]:
    found: dict[str, str] = {}
    for item in payload.get("outputs") or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url:
            continue
        slot = str(item.get("slot") or item.get("filename") or "output")
        found[slot] = url
    return found


def _first_url(outputs: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in outputs:
            return outputs[name]
    for slot, url in outputs.items():
        lowered = slot.lower()
        if any(name in lowered for name in names):
            return url
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    return None


def _download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=180)
    if response.status_code != 200 or not response.content:
        raise MothError(f"Download failed with HTTP {response.status_code}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _wav_samples(blob: bytes) -> np.ndarray:
    import io

    with wave.open(io.BytesIO(blob), "rb") as handle:
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32767.0
    if channels == 1:
        pcm = np.column_stack((pcm, pcm))
    else:
        pcm = pcm.reshape(-1, channels)[:, :2]
    return pcm


def _tempo_bytes(bpm: int) -> bytes:
    microseconds = int(round(60_000_000 / bpm))
    return microseconds.to_bytes(3, "big")


def _encode_track(events: list[tuple[int, bytes]]) -> bytes:
    blob = bytearray()
    previous = 0
    for tick, payload in events:
        blob.extend(_vlq(tick - previous))
        blob.extend(payload)
        previous = tick
    return b"MTrk" + struct.pack(">I", len(blob)) + bytes(blob)


def _vlq(value: int) -> bytes:
    buffer = [value & 0x7F]
    value >>= 7
    while value:
        buffer.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(buffer))


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


EP_DIR = ROOT / "notebooks" / "media" / "through_the_wall"
FULL_PIECE = ROOT / "notebooks" / "media" / "through_the_wall.wav"
# A few milliseconds, only on the cut tracks, so a splice in the crossfade does not click.
_EDGE_FADE_S = 0.01


def _period_ratios() -> tuple[float, float, float]:
    """Hydrogen period in femtoseconds, and the deuterium and tritium ratios to it."""

    physics = json.loads((ROOT / "src" / "experience_data.json").read_text(encoding="utf-8"))
    hydrogen = float(physics["isotopes"]["H"]["period_fs"])
    deuterium = float(physics["isotopes"]["D"]["period_fs"])
    tritium = float(physics["isotopes"]["T"]["period_fs"])
    return hydrogen, deuterium / hydrogen, tritium / hydrogen


def _job_id(name: str) -> str:
    return str(json.loads((MUSIC_DIR / name).read_text(encoding="utf-8"))["job_id"])


def ep_manifest() -> list[dict]:
    """Track titles, liner notes, and the cached Atlas job ids that shaped them.

    Reads sidecars and the saved periods. It does not call the API.
    """

    hydrogen_fs, deuterium_times, tritium_times = _period_ratios()
    echo_id = _job_id("echo_roundtrip.json")
    qrc_id = _job_id("qrc_continuation.json")
    blur_id = _job_id("blur_continuation.json")
    mix_id = _job_id("mix_echo.json")
    echo_jobs = f"retrocausal-echo-v1 · {echo_id}"
    coda_jobs = (
        f"qrc-midi-v1 · {qrc_id} · blur-midi-v1 · {blur_id} · retrocausal-echo-v1 · {mix_id}"
    )
    return [
        {
            "file": "I-hydrogen.wav",
            "title": "I. Hydrogen",
            "note": (
                "I. Hydrogen: the lightest nucleus, and the fastest crossing, "
                f"{hydrogen_fs:.0f} femtoseconds from well to well."
            ),
            "jobs": echo_jobs,
        },
        {
            "file": "II-deuterium.wav",
            "title": "II. Deuterium",
            "note": (
                "II. Deuterium: twice as heavy, so the crossing takes "
                f"{deuterium_times:.1f} times as long."
            ),
            "jobs": echo_jobs,
        },
        {
            "file": "III-tritium.wav",
            "title": "III. Tritium",
            "note": (
                "III. Tritium: three times as heavy, so the crossing takes "
                f"{tritium_times:.1f} times as long."
            ),
            "jobs": echo_jobs,
        },
        {
            "file": "coda.wav",
            "title": "Coda",
            "note": "Coda: the crossings continue as a melody, and an echo gives them a room.",
            "jobs": coda_jobs,
        },
        {
            "file": "through_the_wall.wav",
            "title": "Full piece",
            "note": "The four tracks in order, with one second of quiet between tritium and the coda.",
            "jobs": f"{echo_jobs} · {coda_jobs}",
        },
    ]


def liner_notes_markdown(tracks: list[dict] | None = None) -> str:
    tracks = tracks or ep_manifest()
    lines = [
        "# Through the Wall",
        "",
        "Listening to a proton tunnel",
        "",
        "a short piece in three movements and a coda",
        "",
        "Cover: the proton's crossing, position across and time down, coloured by phase.",
        "",
        "Through the Wall, cover and pipeline by Tammy. Engines by Moth; Retrocausal Echo by Spencer Topel.",
        "",
        "I. Hydrogen · II. Deuterium · III. Tritium · Coda",
        "",
    ]
    for track in tracks:
        lines.extend(["## " + track["title"], "", track["note"], "", track["jobs"], ""])
    return "\n".join(lines).rstrip() + "\n"


def _read_stereo(blob: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(blob)) as handle:
        rate = handle.getframerate()
        if handle.getnchannels() != 2 or handle.getsampwidth() != 2:
            raise ValueError("Expected 16-bit stereo WAV.")
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype="<i2").reshape(-1, 2).astype(np.float64) / 32768.0
    return audio, rate


def _write_stereo(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(clipped.tobytes())


def _edge_fade(audio: np.ndarray, fade_in: bool, fade_out: bool, fade: int) -> np.ndarray:
    out = np.array(audio, dtype=np.float64, copy=True)
    fade = min(int(fade), max(0, out.shape[0] // 4))
    if fade <= 0:
        return out
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
    if fade_in:
        out[:fade] *= ramp[:, None]
    if fade_out:
        out[-fade:] *= ramp[::-1, None]
    return out


def _movement_spans(n_frames: int, sample_rate: int) -> list[tuple[int, int]]:
    """Sample ranges of H, D, and T inside the echoed round trip, tail included on T."""

    n_fade = int(round(FADE_SECONDS * sample_rate))
    n_total = int(round(TRACK_SECONDS * sample_rate))
    n_seg = (n_total + 2 * n_fade) // 3
    start_ii = n_seg - n_fade
    start_iii = 2 * (n_seg - n_fade)
    if n_frames < n_total:
        raise ValueError("The echoed round trip is shorter than the thirty-second piece.")
    return [(0, start_ii), (start_ii, start_iii), (start_iii, n_frames)]


def _join_piece_bytes(echo: bytes, coda: bytes) -> bytes:
    with wave.open(io.BytesIO(echo)) as left:
        params = left.getparams()
        left_pcm = left.readframes(left.getnframes())
        silence = b"\x00" * (left.getframerate() * left.getsampwidth() * left.getnchannels())
    with wave.open(io.BytesIO(coda)) as right:
        right_pcm = right.readframes(right.getnframes())
    out = io.BytesIO()
    with wave.open(out, "wb") as dest:
        dest.setparams(params)
        dest.writeframes(left_pcm + silence + right_pcm)
    return out.getvalue()


def write_ep(destination: Path | None = None) -> list[dict]:
    """Split the cached echo and coda into four tracks plus the full piece.

    No engine is called. Job ids are copied from the sidecars already on disk.
    """

    echo_path = MUSIC_DIR / "echo_roundtrip.wav"
    coda_path = MUSIC_DIR / "mix_echo.wav"
    echo_bytes = echo_path.read_bytes()
    coda_bytes = coda_path.read_bytes()
    movements, sample_rate = _read_stereo(echo_bytes)
    spans = _movement_spans(movements.shape[0], sample_rate)
    fade = int(round(_EDGE_FADE_S * sample_rate))
    pieces = [
        _edge_fade(movements[spans[0][0] : spans[0][1]], False, True, fade),
        _edge_fade(movements[spans[1][0] : spans[1][1]], True, True, fade),
        _edge_fade(movements[spans[2][0] : spans[2][1]], True, False, fade),
    ]
    folder = destination or EP_DIR
    folder.mkdir(parents=True, exist_ok=True)
    names = ("I-hydrogen.wav", "II-deuterium.wav", "III-tritium.wav")
    for name, audio in zip(names, pieces):
        _write_stereo(folder / name, audio, sample_rate)
    (folder / "coda.wav").write_bytes(coda_bytes)
    piece = _join_piece_bytes(echo_bytes, coda_bytes)
    (folder / "through_the_wall.wav").write_bytes(piece)
    FULL_PIECE.parent.mkdir(parents=True, exist_ok=True)
    FULL_PIECE.write_bytes(piece)
    cover = ROOT / "atlas_tessa" / "carpet_aer.png"
    if not cover.exists():
        cover = ROOT / "atlas_tessa" / "carpet.png"
    if cover.exists():
        (folder / "cover.png").write_bytes(cover.read_bytes())
    tracks = ep_manifest()
    (folder / "liner_notes.md").write_text(liner_notes_markdown(tracks), encoding="utf-8")
    for track in tracks:
        track["path"] = str(folder / track["file"])
    return tracks


if __name__ == "__main__":
    song = load_song()
    print(song["echo"]["engine"], song["echo"]["job_id"], song["echo"]["execution"])
    print(song["qrc"]["engine"], song["qrc"]["job_id"])
    print(song["mix_echo"]["engine"], song["mix_echo"]["job_id"])
