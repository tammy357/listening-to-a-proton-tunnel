"""Listening modes for the tunneling doublet.

The installed ``quantumaudio`` entry points, confirmed against this
environment, are:

* ``quantumaudio.encode(data, scheme=None, **kwargs)`` — builds a circuit.
  QPAM's own method is ``encode(data, measure=True, verbose=1)``. ``data``
  must be a real ``numpy`` array already inside [-1, 1];
  ``validate_data`` rejects anything outside that range. The converter then
  maps ``(sample + 1) / 2`` onto probability amplitudes.
* ``quantumaudio.decode(circuit, **kwargs)`` — measures and restores audio.
  QPAM's method is ``decode(circuit, metadata=None, shots=8000, ...)``.
  ``shots`` is the number of projective measurements.
* ``quantumaudio.stream(data, scheme=None, **kwargs)`` — splits a long
  buffer into chunks (default ``chunk_size=64``) and runs encode then decode
  on each chunk. Measured-state mode calls encode and decode on each
  probability frame itself, then concatenates, so the shot count applies
  once per frame.

Mode A, "Tunneling beat", is an artistic mapping of real physical
quantities. The carrier is the chosen base pitch. The second tone sits
``f_beat`` above it, and ``f_beat`` is a strictly increasing function of
``log10(ΔE)``. The donor and acceptor populations, resampled to the audio
buffer, pan that tone left and right. None of this is a spectrum of the
Hamiltonian. It is a listening model of the splitting and of the
population transfer.

Mode B, "Measured state", takes frames of ``|ψ(x, t)|²``, scales each into
[-1, 1], and passes it through QPAM encode and decode. Fewer shots leave a
rougher reconstruction, which is the measurement uncertainty made audible.
"""

from __future__ import annotations

import contextlib
import io
import wave
from dataclasses import dataclass

import numpy as np
import quantumaudio

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_PITCH_HZ = 220.0
AUDIO_SECONDS = 4.0
MEASURED_FRAMES = 12

# log10(ΔE in eV) is mapped across this window onto the beat band below.
# The model's sidebar (barrier 0.1–0.5 eV, a 0.25–0.5 Å, H/D/T) produces
# splittings from roughly 10 µeV to 0.1 eV, which sits inside the window.
LOG10_SPLITTING_EV_MIN = -5.0
LOG10_SPLITTING_EV_MAX = -1.0
BEAT_HZ_MIN = 0.5
BEAT_HZ_MAX = 12.0


@dataclass(frozen=True)
class AudioClip:
    """16-bit PCM WAV and the samples that were written into it.

    Attributes:
        wav_bytes: WAV file bytes, mono or stereo, at least three seconds.
        samples: Floating waveform in [-1, 1]. Shape ``(n,)`` or ``(n, 2)``.
        sample_rate: Frames per second.
        mode: ``"tunneling_beat"`` or ``"measured_state"``.
        note: One sentence a caption can show next to the player.
    """

    wav_bytes: bytes
    samples: np.ndarray
    sample_rate: int
    mode: str
    note: str


def beat_frequency_hz(splitting_ev: float) -> float:
    """Map ``log10(ΔE)`` onto a beat in [0.5, 12] Hz.

    The map is affine in ``log10(ΔE/eV)`` between -5 and -1, and strictly
    increasing on that interval: a larger tunneling splitting is a faster
    beat. Splittings outside the window sit on the nearer endpoint so the
    musical range stays fixed. This is an artistic mapping of the real gap,
    not a conversion of ΔE into a frequency of the proton's motion.
    """

    if splitting_ev <= 0:
        raise ValueError("Tunneling splitting must be positive.")
    log_gap = float(np.log10(splitting_ev))
    span = LOG10_SPLITTING_EV_MAX - LOG10_SPLITTING_EV_MIN
    fraction = (log_gap - LOG10_SPLITTING_EV_MIN) / span
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return BEAT_HZ_MIN + fraction * (BEAT_HZ_MAX - BEAT_HZ_MIN)


def tunneling_beat(
    donor: np.ndarray,
    acceptor: np.ndarray,
    splitting_ev: float,
    base_pitch_hz: float = DEFAULT_PITCH_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seconds: float = AUDIO_SECONDS,
) -> AudioClip:
    """Synthesise the tunneling beat and pan it with the populations.

    Two sines play together, at ``f0`` and at ``f0 + f_beat``. ``f_beat``
    comes from :func:`beat_frequency_hz`. The donor population is the left
    amplitude and the acceptor population is the right amplitude, after both
    series are resampled to the audio length. This is an artistic mapping of
    real physical quantities: the pitch is chosen, the beat follows ΔE, and
    the stereo image follows the probability in each well.
    """

    duration = max(float(seconds), 3.0)
    n_samples = int(np.ceil(duration * sample_rate))
    pitch = max(float(base_pitch_hz), 20.0)
    beat = beat_frequency_hz(splitting_ev)
    times = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    tone = 0.5 * np.sin(2.0 * np.pi * pitch * times)
    tone += 0.5 * np.sin(2.0 * np.pi * (pitch + beat) * times)

    left_gain = _resample(np.clip(np.asarray(donor, dtype=np.float64), 0.0, 1.0), n_samples)
    right_gain = _resample(np.clip(np.asarray(acceptor, dtype=np.float64), 0.0, 1.0), n_samples)
    stereo = np.column_stack((tone * left_gain, tone * right_gain))
    stereo = _fade(stereo, sample_rate)
    stereo = _peak_normalize(stereo, peak=0.9)
    note = (
        f"Artistic mapping. Tones at {pitch:.1f} Hz and {pitch + beat:.2f} Hz "
        f"(beat {beat:.2f} Hz from log10(ΔE)). Left follows the donor, right the acceptor."
    )
    return AudioClip(
        wav_bytes=pcm_wav_bytes(stereo, sample_rate),
        samples=stereo,
        sample_rate=sample_rate,
        mode="tunneling_beat",
        note=note,
    )


def measured_state(
    densities: np.ndarray,
    shots: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seconds: float = AUDIO_SECONDS,
    n_frames: int = MEASURED_FRAMES,
) -> AudioClip:
    """Measure ``|ψ|²`` frames with QPAM and lay them end to end.

    ``densities`` has shape ``(n_times, n_sites)``. ``n_frames`` of those
    rows, spread across the run, are scaled independently onto [-1, 1] and
    passed through ``quantumaudio.encode(..., scheme="qpam")`` and
    ``quantumaudio.decode(..., shots=shots)``. The decoded frames are
    stretched so the concatenation lasts at least three seconds. A smaller
    shot count leaves a noisier frame, which is the audible measurement
    uncertainty.
    """

    if int(shots) < 1:
        raise ValueError("Shot count must be positive.")
    frames = _select_frames(np.asarray(densities, dtype=np.float64), int(n_frames))
    decoded_frames = [_qpam_roundtrip(_scale_unit(frame), int(shots)) for frame in frames]
    duration = max(float(seconds), 3.0)
    waveform = _concatenate_frames(decoded_frames, sample_rate, duration)
    waveform = _fade(waveform, sample_rate)
    waveform = _peak_normalize(waveform, peak=0.9)
    note = (
        f"QPAM encode then decode of {len(decoded_frames)} probability frames, "
        f"{int(shots)} shots each. Fewer shots leave a noisier sound."
    )
    return AudioClip(
        wav_bytes=pcm_wav_bytes(waveform, sample_rate),
        samples=waveform,
        sample_rate=sample_rate,
        mode="measured_state",
        note=note,
    )


def pcm_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Pack floating samples into a 16-bit PCM WAV held in ``io.BytesIO``.

    A 1-D buffer is mono. A 2-D buffer with shape ``(n, 2)`` is stereo,
    interleaved. The result is always at least three seconds long.
    """

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 1:
        channels = 1
        audio = _ensure_duration(audio, sample_rate, 3.0)
        pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    elif audio.ndim == 2 and audio.shape[1] in (1, 2):
        channels = int(audio.shape[1])
        audio = _ensure_duration(audio, sample_rate, 3.0)
        pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        pcm = np.ascontiguousarray(pcm)
    else:
        raise ValueError("Samples must have shape (n,) or (n, 2).")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _scale_unit(frame: np.ndarray) -> np.ndarray:
    """Map one real frame onto [-1, 1] by its own minimum and maximum."""

    values = np.asarray(frame, dtype=np.float64).reshape(-1)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-15:
        return np.zeros_like(values)
    scaled = 2.0 * (values - lo) / (hi - lo) - 1.0
    return np.clip(scaled, -1.0, 1.0)


def _select_frames(densities: np.ndarray, n_frames: int) -> list[np.ndarray]:
    if densities.ndim != 2 or densities.shape[0] < 1 or densities.shape[1] < 2:
        raise ValueError("densities must have shape (n_times, n_sites).")
    count = min(int(n_frames), int(densities.shape[0]))
    count = max(count, 1)
    indices = np.linspace(0, densities.shape[0] - 1, count)
    indices = np.unique(np.rint(indices).astype(int))
    return [densities[int(index)] for index in indices]


def _qpam_roundtrip(frame: np.ndarray, shots: int) -> np.ndarray:
    """Encode one real frame with QPAM and decode it at ``shots``."""

    samples = np.clip(np.asarray(frame, dtype=np.float64).reshape(-1), -1.0, 1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = quantumaudio.encode(samples, scheme="qpam", verbose=0)
        decoded = quantumaudio.decode(circuit, shots=int(shots))
    restored = np.asarray(decoded, dtype=np.float64).reshape(-1)
    if restored.size > samples.size:
        restored = restored[: samples.size]
    return np.clip(restored, -1.0, 1.0)


def _concatenate_frames(frames: list[np.ndarray], sample_rate: int, seconds: float) -> np.ndarray:
    n_total = int(np.ceil(max(seconds, 3.0) * sample_rate))
    n_each = int(np.ceil(n_total / len(frames)))
    pieces = []
    for frame in frames:
        source_x = np.linspace(0.0, 1.0, frame.size)
        target_x = np.linspace(0.0, 1.0, n_each)
        pieces.append(np.interp(target_x, source_x, frame))
    fade = min(int(0.01 * sample_rate), n_each // 4)
    if fade >= 2:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
        for piece in pieces:
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
    return np.concatenate(pieces)[:n_total]


def _resample(series: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.full(n_samples, float(values[0]))
    return np.interp(
        np.linspace(0.0, 1.0, n_samples),
        np.linspace(0.0, 1.0, values.size),
        values,
    )


def _fade(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.array(samples, dtype=np.float64, copy=True)
    fade = max(int(0.02 * sample_rate), 1)
    fade = min(fade, audio.shape[0] // 2)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
    audio[:fade] *= ramp if audio.ndim == 1 else ramp[:, None]
    audio[-fade:] *= ramp[::-1] if audio.ndim == 1 else ramp[::-1, None]
    return audio


def _peak_normalize(samples: np.ndarray, peak: float) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float64)
    maximum = float(np.max(np.abs(audio)))
    if maximum == 0.0:
        return audio
    return audio / maximum * peak


def _ensure_duration(samples: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    need = int(np.ceil(seconds * sample_rate))
    if samples.shape[0] >= need:
        return samples
    repeats = int(np.ceil(need / samples.shape[0]))
    if samples.ndim == 1:
        return np.tile(samples, repeats)[:need]
    return np.tile(samples, (repeats, 1))[:need]
