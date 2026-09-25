"""QPAM sonification of a tunneling statevector, plus an Atlas job record.

Moth Quantum's ``quantumaudio`` package represents a digital waveform as a
quantum state. Quantum Probability Amplitude Modulation (QPAM), the
real-ket scheme, loads a real sample buffer with ``circuit.initialize``
after the map

    sample ∈ [-1, 1]  →  (sample + 1) / 2  →  ℓ²-normalised amplitudes.

``quantumaudio.decode`` measures that circuit in the computational basis
and inverts the map, so shot noise becomes part of the sound.

A tunneling statevector is complex and 256 samples long. QPAM cannot ingest
complex values (its converter casts to float). This module removes a global
phase, keeps both the strongest quadrature and the quadrature orthogonal to
it, and concatenates them into one real frame of length 512. That frame is
what ``quantumaudio.encode(..., scheme="qpam")`` receives. The decoded frame
is then played as a wavetable at the requested pitch: one spatial pass of
the proton amplitude is one cycle of the oscillator. Acceptor population,
when supplied, is stretched across the clip as a loudness contour, so the
arrival of amplitude in the acceptor well is audible as a swell.

The Atlas card posts the same job metadata to ``ATLAS_API_URL``. The alpha
platform is not assumed to accept anonymous jobs. Fidelity, circuit depth,
and the active backend always come from the local Aer execution that
produced the audio.
"""

from __future__ import annotations

import contextlib
import io
import os
import uuid
import wave
from dataclasses import dataclass

import numpy as np
import quantumaudio
import requests
from qiskit import transpile
from qiskit_aer import AerSimulator

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_PITCH_HZ = 220.0
CLIP_SECONDS = 2.8
SHOTS = 48_000
ATLAS_URL_ENV = "ATLAS_API_URL"
ATLAS_KEY_ENV = "ATLAS_API_KEY"
DEFAULT_ATLAS_URL = "https://api.mothquantum.com/v1/jobs"
BACKEND_NAME = "qiskit_aer.AerSimulator"


@dataclass(frozen=True)
class Sonification:
    """Browser-ready audio and the quantum-job facts behind it.

    Attributes:
        wav_bytes: 16-bit PCM WAV, mono, wrapped for ``st.audio``.
        decoded_samples: QPAM measurement reconstruction in [-1, 1].
        waveform: Pitched, enveloped signal that was written into the WAV.
        sample_rate: Samples per second.
        circuit_depth: Basis-gate depth after decomposing state preparation.
        circuit_size: Basis-gate count of that same circuit.
        num_qubits: Qubits in the QPAM index register.
        shots: Projective measurements used by ``quantumaudio.decode``.
        statevector_fidelity: |⟨v|v_decoded⟩|² of the QPAM amplitude vectors.
        backend: Name of the simulator that sampled the circuit.
        scheme: ``"QPAM"``.
        atlas_job: JSON-serialisable job record for the Atlas card.
    """

    wav_bytes: bytes
    decoded_samples: np.ndarray
    waveform: np.ndarray
    sample_rate: int
    circuit_depth: int
    circuit_size: int
    num_qubits: int
    shots: int
    statevector_fidelity: float
    backend: str
    scheme: str
    atlas_job: dict


def sonify_tunneling_state(
    statevector: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    base_pitch: float = DEFAULT_PITCH_HZ,
    envelope: np.ndarray | None = None,
    atlas_context: dict | None = None,
) -> Sonification:
    """Encode a tunneling statevector with QPAM and return WAV bytes.

    Steps, in order:

    1. Normalise the complex amplitudes.
    2. Project onto two real quadratures and scale them into [-1, 1].
    3. ``quantumaudio.encode(amplitudes, scheme="qpam")``.
    4. ``quantumaudio.decode`` on the local Aer simulator (shot noise kept).
    5. Loop the decoded frame at ``base_pitch`` and write 16-bit PCM.

    Args:
        statevector: Complex amplitudes, typically length 256.
        sample_rate: WAV sample rate. 22050 Hz matches the package demos.
        base_pitch: Wavetable frequency in hertz.
        envelope: Optional loudness contour, usually the acceptor population
            over the simulation. Resampled across the clip.
        atlas_context: Extra JSON fields (PDB id, slider values) merged into
            the Atlas job body. This function posts that body.

    Returns:
        Audio bytes plus fidelity, depth, backend, and the Atlas record.
    """

    samples = complex_amplitudes_to_qpam(statevector)
    circuit = _encode_qpam(samples)
    decoded = _decode_qpam(circuit, shots=SHOTS)
    decoded = np.clip(np.asarray(decoded, dtype=np.float64), -1.0, 1.0)
    fidelity = _amplitude_fidelity(samples, decoded)
    depth, size = _basis_gate_cost(circuit)
    waveform = render_pitched_waveform(
        decoded,
        sample_rate=sample_rate,
        pitch_hz=base_pitch,
        envelope=envelope,
    )
    wav_bytes = pcm_wav_bytes(waveform, sample_rate)
    job = _atlas_job(
        fidelity=fidelity,
        depth=depth,
        size=size,
        num_qubits=int(circuit.num_qubits),
        shots=SHOTS,
        context=atlas_context or {},
    )
    return Sonification(
        wav_bytes=wav_bytes,
        decoded_samples=decoded,
        waveform=waveform,
        sample_rate=sample_rate,
        circuit_depth=depth,
        circuit_size=size,
        num_qubits=int(circuit.num_qubits),
        shots=SHOTS,
        statevector_fidelity=fidelity,
        backend=BACKEND_NAME,
        scheme="QPAM",
        atlas_job=job,
    )


def complex_amplitudes_to_qpam(statevector: np.ndarray) -> np.ndarray:
    """Normalise ψ and pack two real quadratures into a QPAM frame.

    A single global phase is chosen to maximise the energy in one
    quadrature. The orthogonal quadrature is appended, so a donor/acceptor
    superposition that sits at a relative phase of π/2 still contributes
    a second hump to the oscillator cycle. Both halves are mean-centred
    together and peak-normalised into the digital audio range [-1, 1].
    """

    psi = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm == 0.0:
        raise ValueError("Cannot sonify a zero statevector.")
    psi = psi / norm
    real = np.real(psi)
    imag = np.imag(psi)
    energy_real = float(np.dot(real, real))
    energy_imag = float(np.dot(imag, imag))
    cross = float(np.dot(real, imag))
    phi = 0.5 * np.arctan2(2.0 * cross, energy_real - energy_imag)
    cosine = np.cos(phi)
    sine = np.sin(phi)
    primary = real * cosine + imag * sine
    orthogonal = -real * sine + imag * cosine
    frame = np.concatenate([primary, orthogonal]).astype(np.float64)
    frame = frame - float(np.mean(frame))
    peak = float(np.max(np.abs(frame)))
    if peak < 1e-15:
        return np.zeros_like(frame)
    frame = frame / peak
    return np.clip(frame, -1.0, 1.0)


def render_pitched_waveform(
    cycle: np.ndarray,
    sample_rate: int,
    pitch_hz: float,
    envelope: np.ndarray | None = None,
    seconds: float = CLIP_SECONDS,
) -> np.ndarray:
    """Repeat a decoded QPAM frame at ``pitch_hz`` for a few seconds.

    Linear interpolation stretches the spatial frame to one period of
    ``sample_rate / pitch_hz`` samples. An optional envelope, typically the
    acceptor-well population, rescales loudness over the clip without
    erasing the quiet parts: the floor is 0.45 so a proton that never
    tunnels is still audible, and arrival in the acceptor well opens it up.
    """

    cycle = np.asarray(cycle, dtype=np.float64).reshape(-1)
    if cycle.size < 2:
        raise ValueError("QPAM frame is too short to pitch.")
    pitch = max(float(pitch_hz), 20.0)
    period = max(int(round(sample_rate / pitch)), 8)
    source_x = np.linspace(0.0, 1.0, cycle.size, endpoint=False)
    period_x = np.linspace(0.0, 1.0, period, endpoint=False)
    one_cycle = np.interp(period_x, source_x, cycle)
    n_samples = int(seconds * sample_rate)
    repeats = int(np.ceil(n_samples / period))
    waveform = np.tile(one_cycle, repeats)[:n_samples]

    if envelope is not None and len(envelope) > 1:
        contour = np.asarray(envelope, dtype=np.float64).reshape(-1)
        contour = np.clip(contour, 0.0, 1.0)
        stretched = np.interp(
            np.linspace(0.0, 1.0, n_samples),
            np.linspace(0.0, 1.0, contour.size),
            contour,
        )
        waveform = waveform * (0.45 + 0.55 * stretched)

    fade = max(int(0.02 * sample_rate), 1)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
    waveform[:fade] *= ramp
    waveform[-fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = waveform / peak * 0.9
    return waveform


def pcm_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Pack floating samples into a 16-bit PCM WAV held in memory."""

    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _encode_qpam(samples: np.ndarray):
    """Call quantumaudio.encode while swallowing the scheme's stdout banner."""

    with contextlib.redirect_stdout(io.StringIO()):
        return quantumaudio.encode(
            np.asarray(samples, dtype=np.float64),
            scheme="qpam",
            verbose=0,
        )


def _decode_qpam(circuit, shots: int) -> np.ndarray:
    """Measure the QPAM circuit with quantumaudio's Aer-backed decoder."""

    with contextlib.redirect_stdout(io.StringIO()):
        decoded = quantumaudio.decode(
            circuit,
            shots=int(shots),
            optimization_level=0,
        )
    return np.asarray(decoded, dtype=np.float64)


def _amplitude_fidelity(original: np.ndarray, decoded: np.ndarray) -> float:
    """Squared overlap of the probability amplitudes QPAM actually loads.

    Both buffers are passed through the same (sample + 1) / 2 map the
    scheme uses, then ℓ²-normalised. The overlap is the fidelity between
    the encoded statevector and the statevector implied by the measured
    audio. Shot noise pulls it slightly below 1.
    """

    length = min(original.size, decoded.size)
    left = _qpam_amplitudes(original[:length])
    right = _qpam_amplitudes(decoded[:length])
    overlap = float(np.dot(left, right))
    return float(np.clip(overlap**2, 0.0, 1.0))


def _qpam_amplitudes(samples: np.ndarray) -> np.ndarray:
    shifted = (np.asarray(samples, dtype=np.float64) + 1.0) / 2.0
    shifted = np.clip(shifted, 0.0, None)
    norm = float(np.linalg.norm(shifted))
    if norm == 0.0:
        return np.zeros_like(shifted)
    return shifted / norm


def _basis_gate_cost(circuit) -> tuple[int, int]:
    """Depth and size after the initialize instruction is written as gates.

    Aer can execute ``initialize`` as one opaque box, which reports a depth
    of 2 and hides the preparation. Decomposing and transpiling exposes the
    basis-gate depth a hardware backend would have to run.
    """

    prepared = circuit.decompose(reps=4)
    compiled = transpile(prepared, AerSimulator(), optimization_level=1)
    return int(compiled.depth()), int(compiled.size())


def _atlas_job(
    fidelity: float,
    depth: int,
    size: int,
    num_qubits: int,
    shots: int,
    context: dict,
) -> dict:
    """Build the job document and POST it to the Atlas API."""

    url = os.environ.get(ATLAS_URL_ENV, DEFAULT_ATLAS_URL).strip() or DEFAULT_ATLAS_URL
    key = os.environ.get(ATLAS_KEY_ENV, "").strip()
    payload = {
        "job_id": f"local-{uuid.uuid4().hex[:12]}",
        "platform": "moth-atlas",
        "task": "qpam-statevector-sonification",
        "scheme": "QPAM",
        "scheme_name": "Quantum Probability Amplitude Modulation",
        "statevector_fidelity": round(float(fidelity), 6),
        "circuit_depth": int(depth),
        "circuit_size": int(size),
        "num_qubits": int(num_qubits),
        "shots": int(shots),
        "active_backend": BACKEND_NAME,
        "encoding": "quantumaudio.encode(scheme='qpam')",
        "decoding": "quantumaudio.decode",
    }
    payload.update(context)
    payload["atlas"] = dispatch_atlas_job(payload, url=url, api_key=key)
    return payload


def dispatch_atlas_job(payload: dict, url: str, api_key: str, timeout: float = 3.0) -> dict:
    """POST job metadata to Atlas and return the HTTP outcome, never a fake 200.

    Anonymous calls against the public alpha host usually fail. The caller
    still has a complete local record: fidelity, depth, and backend were
    measured before this request. Set ``ATLAS_API_KEY`` to send a bearer
    token, and ``ATLAS_API_URL`` to point at another route.
    """

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "moth-hacks-2026-tunneling/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # The payload already nested under "atlas" must not be echoed back into itself.
    body = {key: value for key, value in payload.items() if key != "atlas"}
    try:
        response = requests.post(url, json=body, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return {
            "dispatched": False,
            "endpoint": url,
            "http_status": None,
            "authenticated": bool(api_key),
            "detail": f"Atlas request failed: {exc}",
        }
    try:
        remote = response.json()
    except ValueError:
        remote = {"raw": response.text[:400]}
    return {
        "dispatched": bool(response.ok),
        "endpoint": url,
        "http_status": int(response.status_code),
        "authenticated": bool(api_key),
        "response": remote,
    }
