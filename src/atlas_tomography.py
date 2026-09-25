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

import json
import math
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
PROVIDER = "aer"
BACKEND = "automatic"
OUTPUT_PATH = Path(__file__).resolve().parent / "atlas_bank.json"
MAX_POPULATION_ERROR = 0.02


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


def tomography_params(qasm: str, shots: int = SHOTS) -> dict:
    """Params object for ``tomography-api-v2``."""

    return {
        "circuit_qasm": qasm,
        "single_tomography": True,
        "double_tomography": False,
        "mutual_information": False,
        "classical_mutual_information": False,
        "shots": int(shots),
        "provider_name": PROVIDER,
        "backend_name": BACKEND,
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


def _complete_job(client: MothClient, job_id: str, index: int, shots: int) -> str:
    """Wait for one tomography job, resubmitting when the engine asks for a retry."""

    fraction = index / N_TIMES
    for attempt in range(4):
        if not job_id:
            job_id = client.submit(ENGINE, tomography_params(circuit_qasm(fraction), shots))
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
) -> list[dict]:
    """Finish each time point, one job at a time, and keep the Bloch sample."""

    points = []
    for index in range(N_TIMES):
        job_id = _complete_job(client, jobs[index], index, shots)
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
    path: Path = OUTPUT_PATH,
    shots: int = SHOTS,
    existing_jobs: list[str] | None = None,
) -> Path:
    """Check the RX reduction locally, then sample 24 tomography jobs."""

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
    points = _collect_points(client, jobs, exact, predicted, shots)

    payload = {
        "provider": PROVIDER,
        "backend": BACKEND,
        "shots": int(shots),
        "date": datetime.now(timezone.utc).isoformat(),
        "engine": ENGINE,
        "execution": "simulator",
        "job_ids": jobs,
        "bloch_source": "result.observables['0'] keys X, Y, Z",
        "acceptor_formula": "(1 - z) / 2",
        "points": points,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


if __name__ == "__main__":
    prior = sys.argv[1:]
    destination = write_atlas_bank(existing_jobs=prior or None)
    print(f"wrote {destination}")
