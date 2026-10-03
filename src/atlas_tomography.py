"""One-qubit tomography of the tunneling oscillation on Moth Atlas.

The circuit at time ``t_k = k T / 24`` is ``rx(2π t_k / T)`` on ``|0⟩``.
``|0⟩`` is the donor well and ``|1⟩`` is the acceptor well. OpenQASM 2 is
what ``tomography-api-v2`` asks for in ``circuit_qasm``.

The engine schema (moth-api v0.41.0) gives ``provider_name`` as a free
string whose default is ``aer``, and ``backend_name`` as a free string
whose default is ``automatic``. It does not list an IBM hardware provider.
``aer`` here is a simulator.

The Bloch vector is the Pauli expectations ``result.observables["0"]``
with keys ``X``, ``Y``, ``Z``, which a local ``rx(π/2)`` job matched.
Acceptor population is ``(1 - z) / 2``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from qiskit import QuantumCircuit, qasm2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.moth_client import MothClient
from src.quantum_sim import evolve_exact

BARRIER_EV = 0.15
HALF_WIDTH_A = 0.35
ASYMMETRY = 0.0
ISOTOPE = "H"
N_TIMES = 24
SHOTS = 4096
ENGINE = "tomography-api-v2"
# Schema defaults from moth-api v0.41.0. provider_name and backend_name are
# free strings. The document names no other values and has no enum.
PROVIDER = "aer"
BACKEND = "automatic"
CANONICAL_BANK = Path(__file__).resolve().parent / "atlas_bank.json"
MAX_POPULATION_ERROR = 0.02
HARDWARE_PENDING = "hardware run: pending QPU access"


def parse_backend_flag(value: str) -> tuple[str, str]:
    """Return ``(provider_name, backend_name)`` for ``--backend``.

    ``aer`` selects the schema defaults, ``aer`` and ``automatic``.
    Any other setting must be ``provider/backend``. Both halves are passed
    through as the free strings the schema defines. Nothing else is invented.
    """

    text = value.strip()
    if text == PROVIDER:
        return PROVIDER, BACKEND
    if "/" not in text:
        raise SystemExit(
            "tomography-api-v2 names only the defaults aer and automatic. "
            "Pass --backend aer, or --backend provider/backend."
        )
    provider, backend = text.split("/", 1)
    provider = provider.strip()
    backend = backend.strip()
    if not provider or not backend:
        raise SystemExit("Pass --backend aer, or --backend provider/backend.")
    return provider, backend


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def bank_destination(provider: str, backend: str) -> Path:
    """A new bank file beside the saved aer run. Never the canonical path."""

    path = CANONICAL_BANK.parent / f"atlas_bank_{_slug(provider)}_{_slug(backend)}.json"
    if path.resolve() == CANONICAL_BANK.resolve():
        raise RuntimeError("Refusing to overwrite src/atlas_bank.json.")
    return path


def length_label(bank: dict) -> str:
    """How to name |r|, from the bank's recorded execution. Coherence is hardware only."""

    execution = str(bank.get("execution", "")).lower()
    provider = str(bank.get("provider", "")).lower()
    backend = str(bank.get("backend", ""))
    if execution == "hardware":
        return "coherence |r|"
    if execution == "emulated" or backend.startswith("fake_"):
        device = backend.removeprefix("fake_") or backend
        return f"emulated device noise ({device} noise model)"
    if provider == "aer" or execution == "simulator":
        return "shot noise (simulator)"
    return f"|r| · {bank.get('provider', 'unknown')}/{backend or 'unknown'}"


def _execution_for(provider: str, backend: str, payload: dict | None) -> str:
    """Label a new run from the schema defaults and any hardware marker in the result."""

    if provider == "aer":
        return "simulator"
    if backend.startswith("fake_"):
        return "emulated"
    blob = json.dumps(payload or {})
    if '"qpu_job_id":' in blob and '"qpu_job_id": null' not in blob and '"qpu_job_id":null' not in blob:
        return "hardware"
    return "unspecified"


def load_tomography_banks(folder: Path | None = None) -> list[dict]:
    """The canonical aer bank, then any ``atlas_bank_<provider>_<backend>.json`` beside it."""

    folder = folder or CANONICAL_BANK.parent
    banks: list[dict] = []
    seen: set[tuple[str, str]] = set()
    paths = []
    if (folder / "atlas_bank.json").exists():
        paths.append(folder / "atlas_bank.json")
    paths.extend(sorted(folder.glob("atlas_bank_*.json")))
    for path in paths:
        bank = json.loads(path.read_text(encoding="utf-8"))
        bank["_path"] = str(path.name)
        key = (str(bank.get("provider", "")), str(bank.get("backend", "")))
        if key in seen:
            continue
        seen.add(key)
        banks.append(bank)
    return banks


def hardware_pending(banks: list[dict] | None = None) -> bool:
    banks = load_tomography_banks() if banks is None else banks
    return not any(str(bank.get("execution", "")).lower() == "hardware" for bank in banks)


def load_qpixl_readback(folder: Path | None = None) -> dict | None:
    """IBM Fez encode/read of the acceptor curve, if that file is on disk.

    This is ``qpixl-v1`` on ``ibm_fez``, not tomography. Callers must not
    label it coherence.
    """

    folder = folder or CANONICAL_BANK.parent
    path = folder / "atlas_qpixl_ibm_fez.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("backend") != "ibm_fez" or payload.get("engine") != "qpixl-v1":
        raise RuntimeError(f"{path.name} is not the Fez qpixl read-back.")
    return payload


def level_ratios() -> dict[str, float]:
    """Return ``(E2 - E1) / (E1 - E0)`` for H, D, and T at the default well."""

    ratios = {}
    for mass in ("H", "D", "T"):
        _model, evolution = evolve_exact(
            barrier_ev=BARRIER_EV,
            asymmetry_ev=ASYMMETRY,
            well_half_width_angstrom=HALF_WIDTH_A,
            mass=mass,
            n_periods=1.0,
            n_steps=1,
            keep_states=False,
        )
        energies = evolution.energies_j
        splitting = float(energies[1] - energies[0])
        gap = float(energies[2] - energies[1])
        if splitting <= 0:
            raise RuntimeError(f"{mass} splitting was not positive.")
        ratios[mass] = gap / splitting
    return ratios


def population_errors() -> tuple[float, list[float], list[float]]:
    """Max |P_acceptor(rx) - P_acceptor(exact)| over one hydrogen period.

    ``rx(2π t / T) |0⟩`` has acceptor population ``sin²(π t / T)``.
    """

    _model, evolution = evolve_exact(
        barrier_ev=BARRIER_EV,
        asymmetry_ev=ASYMMETRY,
        well_half_width_angstrom=HALF_WIDTH_A,
        mass=ISOTOPE,
        n_periods=1.0,
        n_steps=N_TIMES,
        keep_states=False,
    )
    exact = [float(value) for value in evolution.acceptor_population[:N_TIMES]]
    predicted = []
    worst = 0.0
    for index in range(N_TIMES):
        fraction = index / N_TIMES
        predicted.append(float(math.sin(math.pi * fraction) ** 2))
        worst = max(worst, abs(predicted[-1] - exact[index]))
    return worst, exact, predicted


def circuit_qasm(t_over_period: float) -> str:
    """OpenQASM 2 for ``rx(2π t / T)`` on one qubit."""

    circuit = QuantumCircuit(1, name=f"t{t_over_period:.4f}")
    circuit.rx(2.0 * math.pi * float(t_over_period), 0)
    return qasm2.dumps(circuit)


def tomography_params(
    qasm: str,
    shots: int = SHOTS,
    provider: str = PROVIDER,
    backend: str = BACKEND,
) -> dict:
    """Params object for ``tomography-api-v2``."""

    return {
        "circuit_qasm": qasm,
        "single_tomography": True,
        "double_tomography": False,
        "mutual_information": False,
        "classical_mutual_information": False,
        "shots": int(shots),
        "provider_name": provider,
        "backend_name": backend,
        "qubit_list": [0],
    }


def bloch_from_result(payload: dict) -> dict:
    """Read ``observables['0']`` as the Bloch vector ``(X, Y, Z)``."""

    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Tomography result had no inline result object.")
    observables = result.get("observables")
    if not isinstance(observables, dict) or "0" not in observables:
        raise RuntimeError("Tomography result had no observables['0'] Bloch components.")
    qubit = observables["0"]
    x = float(qubit["X"])
    y = float(qubit["Y"])
    z = float(qubit["Z"])
    length = math.sqrt(x * x + y * y + z * z)
    return {
        "x": x,
        "y": y,
        "z": z,
        "acceptor_population": (1.0 - z) / 2.0,
        "bloch_length": length,
    }


def bloch_length_shot_noise(t_over_period: float, shots: int) -> float:
    """One-sigma sampling width of ``|r|`` for ``rx(2π t/T)`` on a noiseless simulator.

    ``tomography-api-v2`` counts ``shots`` per measurement circuit. One qubit
    is measured in X, Y, and Z separately. For a pure state the radial
    fluctuation is ``|sin(4π t/T)| / sqrt(2 shots)``. It vanishes when the
    state is a Pauli eigenstate, because that axis is then deterministic.
    """

    if shots < 1:
        raise ValueError("shots must be positive.")
    theta = 2.0 * math.pi * float(t_over_period)
    return abs(math.sin(2.0 * theta)) / math.sqrt(2.0 * shots)


def _complete_job(
    client: MothClient,
    job_id: str,
    index: int,
    shots: int,
    provider: str,
    backend: str,
) -> str:
    """Wait for one tomography job, resubmitting when the engine asks for a retry."""

    fraction = index / N_TIMES
    for attempt in range(4):
        if not job_id:
            job_id = client.submit(
                ENGINE,
                tomography_params(circuit_qasm(fraction), shots, provider, backend),
            )
            print(f"submitted t={fraction:.4f} job={job_id}")
        status = client.wait(job_id)
        if status.get("status") == "completed":
            return job_id
        error = status.get("error") if isinstance(status.get("error"), dict) else {}
        retryable = bool(error.get("retryable"))
        print(f"t={fraction:.4f} job={job_id} ended {status.get('status')}: {error}")
        if not retryable or attempt == 3:
            raise RuntimeError(f"{job_id} ended {status.get('status')}: {error}")
        job_id = ""
    raise RuntimeError(f"t={fraction:.4f} did not complete.")


def _collect_points(
    client: MothClient,
    jobs: list[str],
    exact: list[float],
    predicted: list[float],
    shots: int,
    provider: str,
    backend: str,
) -> list[dict]:
    """Finish each time point, one job at a time, and keep the Bloch sample."""

    points = []
    for index in range(N_TIMES):
        job_id = _complete_job(client, jobs[index], index, shots, provider, backend)
        jobs[index] = job_id
        bloch = bloch_from_result(client.result(job_id))
        fraction = index / N_TIMES
        if index == 0 and bloch["z"] < 0.5:
            raise RuntimeError(
                "t=0 Bloch z is not near +1, so (1-z)/2 is the wrong acceptor sign."
            )
        points.append(
            {
                "index": index,
                "t_over_period": round(fraction, 6),
                "job_id": job_id,
                "x": bloch["x"],
                "y": bloch["y"],
                "z": bloch["z"],
                "acceptor_population": bloch["acceptor_population"],
                "bloch_length": bloch["bloch_length"],
                "ideal_acceptor": round(exact[index], 6),
                "rx_acceptor": round(predicted[index], 6),
            }
        )
        print(
            f"t={fraction:.4f} P={bloch['acceptor_population']:.3f} "
            f"|r|={bloch['bloch_length']:.3f}"
        )
    return points


def write_atlas_bank(
    path: Path | None = None,
    shots: int = SHOTS,
    existing_jobs: list[str] | None = None,
    provider: str = PROVIDER,
    backend: str = BACKEND,
) -> Path:
    """Check the RX reduction locally, then sample 24 tomography jobs.

    The file is always ``atlas_bank_<provider>_<backend>.json``. The saved
    aer bank at ``src/atlas_bank.json`` is left untouched.
    """

    path = path or bank_destination(provider, backend)
    if path.resolve() == CANONICAL_BANK.resolve():
        raise RuntimeError("Refusing to overwrite src/atlas_bank.json.")

    ratios = level_ratios()
    for mass, ratio in ratios.items():
        print(f"(E2-E1)/ΔE {mass} = {ratio:.3f}")
        if ratio <= 5.0:
            raise RuntimeError(f"{mass} two-level ratio {ratio:.3f} is not above 5.")
    worst, exact, predicted = population_errors()
    print(f"max |P_rx - P_exact| = {worst:.4f}")
    if worst >= MAX_POPULATION_ERROR:
        raise RuntimeError(
            f"RX circuits miss the exact acceptor population by {worst:.4f}."
        )

    client = MothClient()
    if existing_jobs is None:
        jobs = [""] * N_TIMES
    else:
        jobs = list(existing_jobs)
        if len(jobs) != N_TIMES:
            raise RuntimeError(f"Expected {N_TIMES} existing jobs, got {len(jobs)}.")
    points = _collect_points(client, jobs, exact, predicted, shots, provider, backend)
    sample_payload = client.result(jobs[0]) if jobs and jobs[0] else None
    execution = _execution_for(provider, backend, sample_payload)

    payload = {
        "provider": provider,
        "backend": backend,
        "shots": int(shots),
        "date": datetime.now(timezone.utc).isoformat(),
        "engine": ENGINE,
        "execution": execution,
        "length_label": length_label(
            {"provider": provider, "backend": backend, "execution": execution}
        ),
        "job_ids": jobs,
        "bloch_source": "result.observables['0'] keys X, Y, Z",
        "acceptor_formula": "(1 - z) / 2",
        "points": points,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=(
            "Run tomography-api-v2. The schema defaults are provider_name=aer "
            "and backend_name=automatic. Both are free strings; the schema "
            "enumerates no hardware backend."
        )
    )
    parser.add_argument(
        "--backend",
        default=PROVIDER,
        help="aer (schema defaults aer / automatic), or provider/backend",
    )
    parser.add_argument(
        "existing_jobs",
        nargs="*",
        help="Optional 24 job ids to resume. A new run still writes a new file.",
    )
    args = parser.parse_args(argv)
    provider, backend = parse_backend_flag(args.backend)
    destination = write_atlas_bank(
        path=bank_destination(provider, backend),
        existing_jobs=args.existing_jobs or None,
        provider=provider,
        backend=backend,
    )
    print(f"wrote {destination}")
    return destination


if __name__ == "__main__":
    main()
