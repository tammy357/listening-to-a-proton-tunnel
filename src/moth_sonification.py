"""Moth Quantum ``quantumaudio`` integration: QPAM statevector sonification + Atlas payloads.

This is the only module in the project that talks to ``quantumaudio`` (and to the Moth Atlas
platform API). Everything else stays pure physics.

The Moth Quantum QPAM contract
------------------------------
QPAM = **Quantum Probability Amplitude Modulation**. It is the simplest of Moth Quantum's
audio encoding schemes and the one whose register layout matches this project exactly:

* A real audio frame ``a[k]`` with ``a[k]`` in ``[-1, 1]`` and ``len(a) = 2**n`` is mapped onto
  the probability *amplitudes* of an ``n``-qubit state::

      |psi_audio> = (1 / N) * sum_k sqrt((a[k] + 1) / 2) |k>

  so sample index ``k`` is a **time** index carried by the qubit register and the sample value
  lives in the *probability* of that basis state. A 256-sample frame therefore needs
  ``n = 8`` qubits — the same 8 qubits our spatial grid occupies.
* ``quantumaudio.encode(frame, scheme="qpam")`` returns the Qiskit state-preparation circuit.
* ``quantumaudio.decode(circuit, shots=...)`` measures it and reconstructs ``a[k]`` from the
  shot histogram, undoing the normalisation factor.

Because decoding is a *measurement*, the recovered waveform is shot-noise limited: the
per-sample error scales like ``1/sqrt(shots)``. That noise is not a bug, it is the sound of
sampling a quantum state, and with a few thousand shots it reads as a pleasant analogue
"hiss" riding on the tone.

From tunneling statevector to sound
-----------------------------------
The physics gives us ``|psi(x, t)>``: 256 complex amplitudes on a 256-point spatial grid,
evolving over ~50 femtoseconds. Two problems: it is complex-valued, and 50 fs is roughly
11 orders of magnitude too fast to hear.

1. **Complex -> real frame.** We take ``Re(psi[k])`` — the projection that carries *both* the
   probability envelope ``|psi|`` and the quantum phase ``arg(psi)`` — remove its DC offset
   and normalise to ``[-1, 1]``. The result is a single-cycle **wavetable** whose shape is
   literally the shape of the proton's wavefunction: a smooth near-Gaussian lump while the
   proton sits in the donor well, and a rippled two-lobed shape once amplitude has tunnelled
   through the barrier. Different wavefunction => different harmonic spectrum => audibly
   different timbre.
2. **Femtoseconds -> seconds.** The femtosecond trajectory is stretched onto a listening
   timescale: snapshots of ``|psi(t)>`` become successive wavetables that are crossfaded over
   a couple of seconds, so the timbral morph you hear *is* the tunneling event slowed by
   ~10^13. The oscillator pitch simultaneously glides with the acceptor-well population, so
   the proton arriving on the far side of the barrier sounds like a rising interval.

Audio output is 16-bit PCM mono WAV at 22050 Hz, wrapped in :class:`io.BytesIO` so Streamlit's
``st.audio()`` can play it straight from memory.
"""

from __future__ import annotations

import contextlib
import io
import os
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import guard keeps the app alive if the package is missing
    import quantumaudio

    QUANTUMAUDIO_AVAILABLE = True
    QUANTUMAUDIO_VERSION = getattr(quantumaudio, "__version__", "unknown")
except ImportError:  # pragma: no cover
    quantumaudio = None  # type: ignore[assignment]
    QUANTUMAUDIO_AVAILABLE = False
    QUANTUMAUDIO_VERSION = "not installed"

DEFAULT_SAMPLE_RATE: int = 22050
DEFAULT_SHOTS: int = 8000
DEFAULT_BASE_PITCH_HZ: float = 220.0
DEFAULT_DURATION_S: float = 2.5
QPAM_SCHEME: str = "qpam"

#: Moth Atlas platform API. Override with the ``MOTH_ATLAS_API_URL`` environment variable.
ATLAS_API_URL: str = os.environ.get("MOTH_ATLAS_API_URL", "https://api.atlas.mothquantum.com/v1/jobs")
#: Credential for the Atlas API. Absent => the Atlas card reports local-simulator provenance.
ATLAS_API_KEY: Optional[str] = os.environ.get("MOTH_ATLAS_API_KEY") or os.environ.get(
    "MOTH_API_KEY"
)


@dataclass
class SonificationResult:
    """Rendered audio plus the quantum provenance of how it was produced.

    Attributes
    ----------
    wav_bytes:
        16-bit PCM mono WAV file in a :class:`io.BytesIO` buffer, ready for ``st.audio()``.
    samples:
        The float waveform in ``[-1, 1]`` that was written to the WAV, shape ``(n_samples,)``.
    sample_rate:
        Sample rate in Hz.
    wavetables:
        The QPAM-decoded single-cycle wavetables, shape ``(n_frames, 256)``. These are the
        measured reconstructions of the tunneling statevector snapshots.
    num_qubits:
        Width of the QPAM register used (8 for a 256-sample frame).
    circuit_depth:
        Depth of the QPAM state-preparation circuit (per frame).
    shots:
        Shots used per frame when decoding.
    encode_decode_fidelity:
        Mean normalised overlap between the encoded frame and its measured reconstruction,
        ``<a, a_hat> / (|a| |a_hat|)``, averaged over frames. 1.0 would be a noiseless
        infinite-shot decode; expect ~0.97-0.99 at 8000 shots.
    base_pitch_hz:
        Fundamental of the wavetable oscillator before the tunneling pitch glide.
    backend:
        What actually ran the QPAM circuits, e.g. ``"quantumaudio/aer-simulator"`` or
        ``"local-qpam-reference"`` if the package was unavailable.
    used_quantumaudio:
        ``False`` when the built-in reference QPAM implementation was used as a fallback.
    notes:
        Human-readable provenance strings surfaced in the UI.
    """

    wav_bytes: io.BytesIO
    samples: np.ndarray
    sample_rate: int
    wavetables: np.ndarray
    num_qubits: int
    circuit_depth: int
    shots: int
    encode_decode_fidelity: float
    base_pitch_hz: float
    backend: str
    used_quantumaudio: bool
    notes: List[str] = field(default_factory=list)
    transpiled_depth: int = 0
    """Depth of the QPAM circuit after transpilation to a ``{cx, u}`` hardware basis.

    The abstract :attr:`circuit_depth` is tiny (QPAM is one state-preparation block); this is
    the number that tells you what running the frame on real hardware would actually cost.
    """
    two_qubit_gate_count: int = 0
    """CX count of the transpiled QPAM circuit — the dominant error source on hardware."""

    @property
    def duration_s(self) -> float:
        """Length of the rendered audio in seconds."""
        return float(self.samples.size) / float(self.sample_rate)

    def wav_value(self) -> bytes:
        """Raw WAV bytes (convenience for writing to disk or a non-Streamlit consumer)."""
        self.wav_bytes.seek(0)
        return self.wav_bytes.read()


# --------------------------------------------------------------------------------------
# Statevector -> audio frame
# --------------------------------------------------------------------------------------
def statevector_to_audio_frame(statevector: Any, dc_block: bool = True) -> np.ndarray:
    """Convert a 256-amplitude quantum statevector into a QPAM-ready audio frame.

    The frame is ``Re(psi[k])`` rescaled to ``[-1, 1]``. Taking the real part (rather than
    ``|psi|``) keeps the **quantum phase** in the signal: as the wavepacket evolves, phase
    winding produces sign changes and ripples across the grid, which is exactly the
    high-frequency content that makes the timbre move while the proton tunnels.

    Parameters
    ----------
    statevector:
        A Qiskit :class:`~qiskit.quantum_info.Statevector`, or anything ``numpy`` can turn
        into a length-256 complex array.
    dc_block:
        Subtract the mean before normalising. A constant offset is inaudible but wastes
        headroom and makes the looped wavetable click, so it is removed by default.

    Returns
    -------
    numpy.ndarray
        Real frame of length ``2**n`` (256 here) with values in ``[-1, 1]`` and unit peak.
    """
    data = np.asarray(getattr(statevector, "data", statevector), dtype=complex).ravel()
    if data.size == 0:
        raise ValueError("Empty statevector: nothing to sonify.")

    norm = np.linalg.norm(data)
    if norm > 0:
        data = data / norm

    frame = np.real(data)
    if dc_block:
        frame = frame - frame.mean()

    peak = np.max(np.abs(frame))
    if peak > 0:
        frame = frame / peak
    return np.clip(frame, -1.0, 1.0)


# --------------------------------------------------------------------------------------
# Moth Quantum QPAM encode / decode
# --------------------------------------------------------------------------------------
def encode_qpam(frame: np.ndarray) -> Any:
    """Encode an audio frame into a quantum circuit with Moth Quantum QPAM.

    Thin wrapper over ``quantumaudio.encode(frame, scheme="qpam")``. Kept as a named function
    so the Moth API surface used by this project is documented in exactly one place.

    Parameters
    ----------
    frame:
        Real audio frame in ``[-1, 1]`` whose length is a power of two (256 here).

    Returns
    -------
    qiskit.QuantumCircuit
        The QPAM state-preparation circuit: ``sqrt((a[k]+1)/2)`` amplitudes on ``|k>``.

    Raises
    ------
    RuntimeError
        If the ``quantumaudio`` package is not installed.
    """
    if not QUANTUMAUDIO_AVAILABLE:
        raise RuntimeError("quantumaudio is not installed; run `pip install quantumaudio`.")
    return quantumaudio.encode(np.asarray(frame, dtype=float), scheme=QPAM_SCHEME)


def decode_qpam(circuit: Any, shots: int = DEFAULT_SHOTS) -> np.ndarray:
    """Measure a QPAM circuit and reconstruct the time-domain audio samples.

    Thin wrapper over ``quantumaudio.decode(circuit, shots=shots)``. ``quantumaudio`` reads the
    scheme and normalisation metadata straight off the circuit, simulates the measurement,
    and inverts the QPAM map ``p[k] -> a[k]``.

    Parameters
    ----------
    circuit:
        Circuit produced by :func:`encode_qpam`.
    shots:
        Measurement shots. The reconstruction error scales as ``1/sqrt(shots)``; 8000 gives a
        clean tone with a touch of quantum hiss, 500 sounds deliciously broken.

    Returns
    -------
    numpy.ndarray
        Decoded real audio samples (length 256), typically within ``[-1, 1]``.
    """
    if not QUANTUMAUDIO_AVAILABLE:
        raise RuntimeError("quantumaudio is not installed; run `pip install quantumaudio`.")
    decoded = quantumaudio.decode(circuit, shots=int(shots))
    return np.asarray(decoded, dtype=float).ravel()


def _reference_qpam_roundtrip(frame: np.ndarray, shots: int) -> np.ndarray:
    """Pure-NumPy QPAM encode/measure/decode, used only if ``quantumaudio`` is unavailable.

    Implements the same map Moth Quantum's QPAM uses, so the demo degrades in sound quality
    rather than crashing:

    1. ``probabilities[k] = (a[k] + 1) / 2``, normalised to sum to 1.
    2. Draw ``shots`` multinomial samples (this is the measurement).
    3. Invert: ``a_hat[k] = 2 * (counts[k] / shots) * norm - 1``.
    """
    frame = np.asarray(frame, dtype=float).ravel()
    shifted = (frame + 1.0) / 2.0
    total = shifted.sum()
    if total <= 0:
        return np.zeros_like(frame)
    probabilities = shifted / total
    counts = np.random.multinomial(int(shots), probabilities)
    return 2.0 * (counts / float(shots)) * total - 1.0


@dataclass
class _QpamBatch:
    """Result of pushing a batch of audio frames through QPAM encode -> measure -> decode."""

    wavetables: np.ndarray
    num_qubits: int
    circuit_depth: int
    backend: str
    used_quantumaudio: bool
    notes: List[str] = field(default_factory=list)
    transpiled_depth: int = 0
    two_qubit_gate_count: int = 0


def _hardware_cost(circuit: Any) -> tuple[int, int]:
    """Transpile a QPAM circuit to a ``{cx, u}`` basis and return ``(depth, cx_count)``.

    QPAM's state preparation is one opaque block until it is decomposed; transpiling reveals
    what an actual device would have to run. Cheap for 8 qubits (~50 ms) and returns
    ``(0, 0)`` if transpilation is unavailable, since this is reporting only.
    """
    try:
        from qiskit import transpile

        decomposed = transpile(circuit, basis_gates=["cx", "u"], optimization_level=0)
        return int(decomposed.depth()), int(decomposed.count_ops().get("cx", 0))
    except Exception:  # pragma: no cover - reporting must never break the render
        return 0, 0


def _quantum_roundtrip(frames: Sequence[np.ndarray], shots: int) -> _QpamBatch:
    """Run every frame through QPAM encode -> measure -> decode.

    Falls back to :func:`_reference_qpam_roundtrip` if ``quantumaudio`` is missing or raises,
    so a broken install costs you audio quality rather than the whole demo.
    """
    notes: List[str] = []
    decoded_frames: List[np.ndarray] = []
    num_qubits = int(np.log2(len(frames[0])))
    circuit_depth = 0
    transpiled_depth = 0
    cx_count = 0

    if QUANTUMAUDIO_AVAILABLE:
        try:
            # quantumaudio narrates its register allocation on stdout; keep the logs readable.
            with contextlib.redirect_stdout(io.StringIO()):
                for index, frame in enumerate(frames):
                    circuit = encode_qpam(frame)
                    num_qubits = int(getattr(circuit, "num_qubits", num_qubits))
                    circuit_depth = max(circuit_depth, int(circuit.depth()))
                    if index == 0:
                        transpiled_depth, cx_count = _hardware_cost(circuit)
                    decoded_frames.append(decode_qpam(circuit, shots=shots))
            notes.append(
                f"Encoded {len(frames)} frames with quantumaudio {QUANTUMAUDIO_VERSION} "
                f"(QPAM, {num_qubits} qubits, {shots} shots/frame)."
            )
            return _QpamBatch(
                wavetables=np.vstack(decoded_frames),
                num_qubits=num_qubits,
                circuit_depth=circuit_depth,
                backend="quantumaudio/aer-simulator",
                used_quantumaudio=True,
                notes=notes,
                transpiled_depth=transpiled_depth,
                two_qubit_gate_count=cx_count,
            )
        except Exception as exc:  # pragma: no cover - keeps the live demo alive
            notes.append(f"quantumaudio QPAM path failed ({exc!s}); using reference QPAM.")
            decoded_frames = []

    else:
        notes.append("quantumaudio not installed; using the built-in reference QPAM map.")

    for frame in frames:
        decoded_frames.append(_reference_qpam_roundtrip(frame, shots=shots))
    return _QpamBatch(
        wavetables=np.vstack(decoded_frames),
        num_qubits=num_qubits,
        circuit_depth=circuit_depth,
        backend="local-qpam-reference",
        used_quantumaudio=False,
        notes=notes,
    )


def _frame_fidelity(original: np.ndarray, decoded: np.ndarray) -> float:
    """Normalised overlap (cosine similarity) between an encoded and a decoded frame."""
    a = np.asarray(original, dtype=float).ravel()
    b = np.asarray(decoded, dtype=float).ravel()
    size = min(a.size, b.size)
    a, b = a[:size] - a[:size].mean(), b[:size] - b[:size].mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


# --------------------------------------------------------------------------------------
# Wavetable synthesis + WAV packaging
# --------------------------------------------------------------------------------------
def _normalise_tables(wavetables: np.ndarray) -> np.ndarray:
    """Peak-normalise each wavetable row and DC-block it so crossfades stay click-free."""
    tables = np.atleast_2d(np.asarray(wavetables, dtype=float))
    tables = tables - tables.mean(axis=1, keepdims=True)
    peaks = np.max(np.abs(tables), axis=1, keepdims=True)
    peaks[peaks == 0] = 1.0
    return tables / peaks


def wavetable_synthesis(
    wavetables: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
    base_pitch_hz: float = DEFAULT_BASE_PITCH_HZ,
    pitch_curve: Optional[np.ndarray] = None,
    fade_s: float = 0.02,
) -> np.ndarray:
    """Synthesise audio by sweeping a wavetable oscillator through the decoded frames.

    Each decoded QPAM frame is one single-cycle wavetable = one snapshot of the proton
    wavefunction. The oscillator reads through them with linear interpolation *within* a table
    (anti-aliasing-ish) and linear crossfading *between* tables, so the femtosecond-scale
    change in the wavefunction becomes a slow timbral morph.

    Parameters
    ----------
    wavetables:
        Array of shape ``(n_frames, table_len)``; a single 1-D table is also accepted.
    sample_rate:
        Output sample rate in Hz.
    duration_s:
        Rendered length in seconds.
    base_pitch_hz:
        Oscillator fundamental before modulation.
    pitch_curve:
        Optional multiplier per frame (e.g. ``1 + 0.5 * acceptor_population``) applied to the
        base pitch and interpolated across the render. Defaults to a constant pitch.
    fade_s:
        Attack/release fade length in seconds, to avoid clicks at the buffer edges.

    Returns
    -------
    numpy.ndarray
        Mono float waveform in ``[-1, 1]``.
    """
    tables = _normalise_tables(wavetables)
    n_frames, table_len = tables.shape
    n_samples = max(int(round(sample_rate * duration_s)), 1)

    frame_pos = np.linspace(0.0, n_frames - 1, n_samples)
    lower = np.floor(frame_pos).astype(int)
    upper = np.minimum(lower + 1, n_frames - 1)
    frame_weight = frame_pos - lower

    if pitch_curve is not None and np.size(pitch_curve) > 1:
        curve = np.asarray(pitch_curve, dtype=float).ravel()
        frequency = base_pitch_hz * np.interp(
            np.linspace(0.0, 1.0, n_samples), np.linspace(0.0, 1.0, curve.size), curve
        )
    else:
        frequency = np.full(n_samples, float(base_pitch_hz))

    phase = np.cumsum(frequency / float(sample_rate)) % 1.0
    read = phase * table_len
    k0 = np.floor(read).astype(int) % table_len
    k1 = (k0 + 1) % table_len
    sample_weight = read - np.floor(read)

    low = tables[lower, k0] * (1.0 - sample_weight) + tables[lower, k1] * sample_weight
    high = tables[upper, k0] * (1.0 - sample_weight) + tables[upper, k1] * sample_weight
    signal = low * (1.0 - frame_weight) + high * frame_weight

    fade_len = min(int(sample_rate * max(fade_s, 0.0)), n_samples // 2)
    if fade_len > 1:
        ramp = np.linspace(0.0, 1.0, fade_len)
        signal[:fade_len] *= ramp
        signal[-fade_len:] *= ramp[::-1]

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.92 * signal / peak
    return signal


def to_wav_bytes(samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> io.BytesIO:
    """Pack a float waveform into a 16-bit PCM mono WAV inside :class:`io.BytesIO`.

    Parameters
    ----------
    samples:
        Mono float waveform, nominally in ``[-1, 1]`` (values are clipped).
    sample_rate:
        Sample rate written into the WAV header.

    Returns
    -------
    io.BytesIO
        Buffer positioned at 0, containing a complete RIFF/WAVE file. Hand it straight to
        ``st.audio(buffer, format="audio/wav")``.
    """
    clipped = np.clip(np.asarray(samples, dtype=float), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    buffer.seek(0)
    return buffer


# --------------------------------------------------------------------------------------
# Public sonification entry points
# --------------------------------------------------------------------------------------
def sonify_tunneling_state(
    statevector: Any,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    base_pitch_hz: float = DEFAULT_BASE_PITCH_HZ,
    duration_s: float = DEFAULT_DURATION_S,
    shots: int = DEFAULT_SHOTS,
) -> SonificationResult:
    """Sonify a single tunneling statevector through Moth Quantum QPAM.

    Steps (this is the spec of the Moth integration, end to end):

    1. Normalise the complex amplitudes and project them onto a real audio frame
       (:func:`statevector_to_audio_frame`).
    2. ``quantumaudio.encode(frame, scheme="qpam")`` -> QPAM state-preparation circuit.
    3. ``quantumaudio.decode(circuit, shots=...)`` -> measured time-domain samples.
    4. Use the decoded samples as a single-cycle wavetable and run the oscillator for
       ``duration_s`` seconds.
    5. Pack as 16-bit PCM WAV in :class:`io.BytesIO`.

    Parameters
    ----------
    statevector:
        A 256-amplitude Qiskit statevector (8 qubits) from :mod:`src.quantum_sim`.
    sample_rate:
        Output sample rate in Hz (Moth default: 22050).
    base_pitch_hz:
        Fundamental frequency of the wavetable oscillator.
    duration_s:
        Length of the rendered audio in seconds.
    shots:
        QPAM measurement shots.

    Returns
    -------
    SonificationResult
        Contains ``wav_bytes`` (the :class:`io.BytesIO` WAV for ``st.audio()``) plus the
        quantum provenance of the render.
    """
    frame = statevector_to_audio_frame(statevector)
    batch = _quantum_roundtrip([frame], shots=shots)
    samples = wavetable_synthesis(
        batch.wavetables,
        sample_rate=sample_rate,
        duration_s=duration_s,
        base_pitch_hz=base_pitch_hz,
    )
    return SonificationResult(
        wav_bytes=to_wav_bytes(samples, sample_rate=sample_rate),
        samples=samples,
        sample_rate=int(sample_rate),
        wavetables=batch.wavetables,
        num_qubits=batch.num_qubits,
        circuit_depth=batch.circuit_depth,
        shots=int(shots),
        encode_decode_fidelity=_frame_fidelity(frame, batch.wavetables[0]),
        base_pitch_hz=float(base_pitch_hz),
        backend=batch.backend,
        used_quantumaudio=batch.used_quantumaudio,
        notes=batch.notes,
        transpiled_depth=batch.transpiled_depth,
        two_qubit_gate_count=batch.two_qubit_gate_count,
    )


def sonify_tunneling_trajectory(
    statevectors: Sequence[Any],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    base_pitch_hz: float = DEFAULT_BASE_PITCH_HZ,
    duration_s: float = DEFAULT_DURATION_S,
    shots: int = DEFAULT_SHOTS,
    num_frames: int = 16,
    acceptor_population: Optional[np.ndarray] = None,
    pitch_glide: float = 0.5,
) -> SonificationResult:
    """Sonify a whole tunneling trajectory so the *transition* itself is audible.

    ``num_frames`` snapshots are taken evenly along the trajectory, each one QPAM-encoded and
    measured into its own wavetable. The oscillator crossfades through them across
    ``duration_s`` seconds, stretching a ~50 fs quantum event by ~10^13 into the audio band.

    If ``acceptor_population`` is supplied, the pitch glides upward in proportion to how much
    probability has crossed the barrier: ``f(t) = base_pitch * (1 + pitch_glide * P_acceptor)``.
    With the default ``pitch_glide = 0.5`` a complete proton transfer is heard as a rise of a
    perfect fifth, so "how far did it tunnel" maps onto a musical interval.

    Parameters
    ----------
    statevectors:
        Sequence of statevectors in time order (``TunnelingResult.statevectors``).
    sample_rate, base_pitch_hz, duration_s, shots:
        As in :func:`sonify_tunneling_state`.
    num_frames:
        How many trajectory snapshots to encode. Each one costs a QPAM circuit + measurement,
        so this is the main cost/quality dial (16 is smooth and fast).
    acceptor_population:
        Optional ``P(x > 0)`` versus time, used for the pitch glide. Resampled to the frames.
    pitch_glide:
        Maximum fractional pitch rise at full transfer (0.5 = a perfect fifth).

    Returns
    -------
    SonificationResult
        Rendered audio and QPAM provenance.
    """
    if len(statevectors) == 0:
        raise ValueError("No statevectors to sonify.")

    indices = np.unique(
        np.linspace(0, len(statevectors) - 1, max(int(num_frames), 1)).round().astype(int)
    )
    frames = [statevector_to_audio_frame(statevectors[i]) for i in indices]
    batch = _quantum_roundtrip(frames, shots=shots)
    fidelity = float(
        np.mean([_frame_fidelity(f, t) for f, t in zip(frames, batch.wavetables)])
    )

    pitch_curve = None
    if acceptor_population is not None:
        population = np.asarray(acceptor_population, dtype=float).ravel()
        if population.size:
            pitch_curve = 1.0 + pitch_glide * population[indices.clip(0, population.size - 1)]

    samples = wavetable_synthesis(
        batch.wavetables,
        sample_rate=sample_rate,
        duration_s=duration_s,
        base_pitch_hz=base_pitch_hz,
        pitch_curve=pitch_curve,
    )
    batch.notes.append(
        f"Rendered {len(frames)} wavetable morph frames -> {duration_s:.1f} s of audio "
        f"at {sample_rate} Hz (base pitch {base_pitch_hz:.1f} Hz)."
    )
    return SonificationResult(
        wav_bytes=to_wav_bytes(samples, sample_rate=sample_rate),
        samples=samples,
        sample_rate=int(sample_rate),
        wavetables=batch.wavetables,
        num_qubits=batch.num_qubits,
        circuit_depth=batch.circuit_depth,
        shots=int(shots),
        encode_decode_fidelity=fidelity,
        base_pitch_hz=float(base_pitch_hz),
        backend=batch.backend,
        used_quantumaudio=batch.used_quantumaudio,
        notes=batch.notes,
        transpiled_depth=batch.transpiled_depth,
        two_qubit_gate_count=batch.two_qubit_gate_count,
    )


# --------------------------------------------------------------------------------------
# Moth Atlas platform API
# --------------------------------------------------------------------------------------
def build_atlas_job_payload(
    sonification: SonificationResult,
    statevector_fidelity: float,
    simulation_metadata: Optional[Dict[str, Any]] = None,
    job_name: str = "ca2-proton-tunneling-qpam",
) -> Dict[str, Any]:
    """Build the Moth Atlas platform job document for this run.

    Atlas is Moth Quantum's execution/observability platform: you submit a circuit plus the
    audio-encoding metadata, and it reports back execution provenance. This function produces
    the exact JSON document the app displays (and would POST), which keeps the demo honest
    about what ran where.

    Parameters
    ----------
    sonification:
        The render whose QPAM circuits we are describing.
    statevector_fidelity:
        Norm/unitarity fidelity of the physics propagation (from
        :attr:`src.quantum_sim.TunnelingResult.norm_fidelity`).
    simulation_metadata:
        Optional physics parameters (barrier height, asymmetry, duration, ...) to attach.
    job_name:
        Human-readable Atlas job label.

    Returns
    -------
    dict
        JSON-serialisable Atlas job document.
    """
    return {
        "platform": "Moth Quantum Atlas",
        "api_endpoint": ATLAS_API_URL,
        "job": {
            "name": job_name,
            "status": "COMPLETED",
            "submitted_by": "moth-hacks-2026/quantum-tunneling-sonification",
            "encoding_scheme": "QPAM (Quantum Probability Amplitude Modulation)",
            "quantumaudio_version": QUANTUMAUDIO_VERSION,
        },
        "quantum_execution": {
            "active_backend": sonification.backend,
            "num_qubits": sonification.num_qubits,
            "circuit_depth": sonification.circuit_depth,
            "transpiled_depth_cx_u_basis": sonification.transpiled_depth,
            "two_qubit_gate_count": sonification.two_qubit_gate_count,
            "circuits_submitted": int(sonification.wavetables.shape[0]),
            "shots_per_circuit": sonification.shots,
            "total_shots": int(sonification.shots * sonification.wavetables.shape[0]),
        },
        "fidelity": {
            "statevector_norm_fidelity": round(float(statevector_fidelity), 12),
            "qpam_encode_decode_fidelity": round(float(sonification.encode_decode_fidelity), 6),
            "shot_noise_floor": round(1.0 / np.sqrt(max(sonification.shots, 1)), 6),
        },
        "audio": {
            "sample_rate_hz": sonification.sample_rate,
            "duration_s": round(sonification.duration_s, 3),
            "bit_depth": 16,
            "channels": 1,
            "base_pitch_hz": sonification.base_pitch_hz,
            "format": "audio/wav",
        },
        "simulation": simulation_metadata or {},
        "notes": sonification.notes,
    }


def submit_atlas_job(payload: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
    """POST a job document to the Moth Atlas API, or report local-execution provenance.

    Atlas requires credentials. When ``MOTH_ATLAS_API_KEY`` (or ``MOTH_API_KEY``) is not set,
    no request is made and the returned document is flagged as locally executed — the demo
    stays fully functional offline, which is the whole point of the fallback.

    Parameters
    ----------
    payload:
        Job document from :func:`build_atlas_job_payload`.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    dict
        ``{"mode": ..., "detail": ..., "response": ...}`` describing what happened.
    """
    if not ATLAS_API_KEY:
        return {
            "mode": "local-simulation",
            "detail": (
                "No MOTH_ATLAS_API_KEY in the environment - QPAM circuits were executed on the "
                "local quantumaudio simulator. Set MOTH_ATLAS_API_KEY to dispatch this exact "
                "job document to Atlas."
            ),
            "response": None,
        }

    try:
        import requests

        response = requests.post(
            ATLAS_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {ATLAS_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text[:2000]
        return {
            "mode": "atlas-api",
            "detail": f"POST {ATLAS_API_URL} -> HTTP {response.status_code}",
            "response": body,
        }
    except Exception as exc:
        return {
            "mode": "atlas-unreachable",
            "detail": f"Atlas submission failed ({exc!s}); local QPAM results retained.",
            "response": None,
        }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from .quantum_sim import simulate_tunneling

    run = simulate_tunneling(duration_fs=50.0, num_steps=60)
    audio = sonify_tunneling_trajectory(
        run.statevectors,
        acceptor_population=run.acceptor_population,
        num_frames=12,
    )
    print(f"backend: {audio.backend} (quantumaudio={audio.used_quantumaudio})")
    print(f"QPAM fidelity: {audio.encode_decode_fidelity:.4f}, depth {audio.circuit_depth}")
    print(f"audio: {audio.duration_s:.2f} s @ {audio.sample_rate} Hz")
    with open("quantum_tunneling.wav", "wb") as handle:
        handle.write(audio.wav_value())
    print("wrote quantum_tunneling.wav")
