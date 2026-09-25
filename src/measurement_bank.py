"""Two-level measurement bank for one tunneling period.

Twenty-four times, evenly spaced across one hydrogen period at the default
well, are reduced to a single qubit. |0⟩ is the donor well and |1⟩ is the
acceptor well. The amplitudes are the square roots of those populations,
with the relative phase of the summed well amplitudes.

``run_bank(circuits, shots)`` samples every circuit. ``LocalSimulator`` uses
Qiskit's statevector sampler and is labelled ``local simulator``.
``AtlasBackend`` is a stub for the Moth platform and does not call any
Atlas Cloud SDK.

Run from the repository root:

    python src/measurement_bank.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_sim import evolve_exact

# Same opening well as the Experience movie and the Lab defaults.
BARRIER_EV = 0.15
HALF_WIDTH_A = 0.35
ASYMMETRY = 0.0
ISOTOPE = "H"
N_TIMES = 24
SHOTS = 4096
LOCAL_SOURCE = "local simulator"
OUTPUT_PATH = Path(__file__).resolve().parent / "bank.json"


def run_bank(circuits, shots: int, backend: str = "local") -> list[dict]:
    """Sample each circuit and return one counts dict per circuit.

    ``backend="local"`` uses :class:`LocalSimulator`. ``backend="atlas"``
    uses :class:`AtlasBackend`, which is not implemented yet.
    """

    if backend == "local":
        return LocalSimulator().run_bank(circuits, shots)
    if backend == "atlas":
        return AtlasBackend().run_bank(circuits, shots)
    raise ValueError(f"Unknown measurement backend: {backend}")


class AtlasBackend:
    """Stub for a Moth Platform backend.

    Fill this in from the Moth platform docs. It does not use atlascloud.ai
    or any Atlas Cloud SDK.
    """

    def run_bank(self, circuits, shots: int) -> list[dict]:
        del circuits, shots
        raise NotImplementedError("fill in from Moth Platform docs")


class LocalSimulator:
    """Qiskit statevector sampler.

    Counts returned here are samples from that local simulator.
    """

    def run_bank(self, circuits, shots: int) -> list[dict]:
        if int(shots) < 1:
            raise ValueError("shots must be positive")
        sampler = StatevectorSampler()
        result = sampler.run(list(circuits), shots=int(shots)).result()
        counts_list: list[dict] = []
        for pub in result:
            raw = pub.data.c.get_counts()
            counts_list.append(
                {
                    "0": int(raw.get("0", 0)),
                    "1": int(raw.get("1", 0)),
                }
            )
        return counts_list


def build_circuits(
    barrier_ev: float = BARRIER_EV,
    asymmetry_ev: float = ASYMMETRY,
    well_half_width_angstrom: float = HALF_WIDTH_A,
    mass: str = ISOTOPE,
    n_times: int = N_TIMES,
) -> tuple[list[QuantumCircuit], list[dict]]:
    """Prepare one measured qubit at each of ``n_times`` instants.

    Times are ``k / n_times`` tunneling periods for ``k = 0 … n_times - 1``.
    """

    if n_times < 2:
        raise ValueError("At least two time points are required.")
    _model, evolution = evolve_exact(
        barrier_ev=barrier_ev,
        asymmetry_ev=asymmetry_ev,
        well_half_width_angstrom=well_half_width_angstrom,
        mass=mass,
        n_periods=1.0,
        n_steps=n_times,
        keep_states=True,
    )
    if evolution.states is None:
        raise RuntimeError("The exact evolution did not keep its states.")
    donor_mask = _model.x_m < 0.0
    circuits: list[QuantumCircuit] = []
    records: list[dict] = []
    for index in range(n_times):
        psi = evolution.states[index]
        donor = float(evolution.donor_population[index])
        acceptor = float(evolution.acceptor_population[index])
        phase = _relative_phase(psi, donor_mask)
        circuits.append(_qubit_circuit(donor, acceptor, phase, index))
        records.append(
            {
                "index": index,
                "t_over_period": round(index / n_times, 6),
                "ideal_donor": round(donor, 6),
                "ideal_acceptor": round(acceptor, 6),
                "relative_phase": round(phase, 6),
            }
        )
    return circuits, records


def write_bank(path: Path = OUTPUT_PATH, shots: int = SHOTS) -> Path:
    """Sample the hydrogen bank with the local simulator and write JSON."""

    circuits, records = build_circuits()
    counts = run_bank(circuits, shots, backend="local")
    if len(counts) != len(records):
        raise RuntimeError("The sampler returned a different number of results.")
    for record, outcome in zip(records, counts, strict=True):
        record["counts"] = outcome
    payload = {
        "source": LOCAL_SOURCE,
        "shots": int(shots),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "isotope": ISOTOPE,
        "barrier_ev": BARRIER_EV,
        "half_width_a": HALF_WIDTH_A,
        "asymmetry": ASYMMETRY,
        "n_times": N_TIMES,
        "points": records,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _qubit_circuit(donor: float, acceptor: float, relative_phase: float, index: int) -> QuantumCircuit:
    total = donor + acceptor
    if total <= 0:
        raise ValueError("The well populations were empty.")
    amplitude = np.array(
        [
            np.sqrt(donor / total),
            np.sqrt(acceptor / total) * np.exp(1j * relative_phase),
        ],
        dtype=np.complex128,
    )
    circuit = QuantumCircuit(1, 1, name=f"t{index}")
    circuit.initialize(amplitude, 0)
    circuit.measure(0, 0)
    return circuit


def _relative_phase(psi: np.ndarray, donor_mask: np.ndarray) -> float:
    donor = complex(np.sum(psi[donor_mask]))
    acceptor = complex(np.sum(psi[~donor_mask]))
    if abs(donor) < 1e-15 or abs(acceptor) < 1e-15:
        return 0.0
    return float(np.angle(acceptor * np.conj(donor)))


def _summarize(path: Path) -> None:
    payload = json.loads(path.read_text())
    print(f"wrote {path}")
    print(f"source={payload['source']!r} shots={payload['shots']} points={len(payload['points'])}")
    worst = 0.0
    for point in payload["points"]:
        measured = point["counts"].get("1", 0) / payload["shots"]
        worst = max(worst, abs(measured - point["ideal_acceptor"]))
        total = point["counts"].get("0", 0) + point["counts"].get("1", 0)
        if total != payload["shots"]:
            raise SystemExit(f"Counts at t={point['t_over_period']} do not sum to the shot budget.")
    print(f"max |measured acceptor − ideal| = {worst:.4f}")
    if payload["source"] != LOCAL_SOURCE:
        raise SystemExit("Local results were not labelled 'local simulator'.")


if __name__ == "__main__":
    destination = write_bank()
    _summarize(destination)
