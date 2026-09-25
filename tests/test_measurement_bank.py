"""Checks for the two-level measurement bank.

Run from the repository root:

    python tests/test_measurement_bank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.measurement_bank import LOCAL_SOURCE, AtlasBackend, LocalSimulator, run_bank

BANK_PATH = ROOT / "src" / "bank.json"


def test_atlas_backend_is_a_stub() -> None:
    try:
        AtlasBackend().run_bank([], 16)
    except NotImplementedError as exc:
        if str(exc) != "fill in from Moth Platform docs":
            raise SystemExit(f"Unexpected Atlas stub message: {exc}")
    else:
        raise SystemExit("AtlasBackend ran without raising.")
    try:
        run_bank([], 16, backend="atlas")
    except NotImplementedError as exc:
        if str(exc) != "fill in from Moth Platform docs":
            raise SystemExit(f"Unexpected run_bank atlas message: {exc}")
    else:
        raise SystemExit("run_bank(backend='atlas') ran without raising.")


def test_local_simulator_prepares_basis_states() -> None:
    ground = QuantumCircuit(1, 1)
    ground.initialize([1, 0], 0)
    ground.measure(0, 0)
    excited = QuantumCircuit(1, 1)
    excited.initialize([0, 1], 0)
    excited.measure(0, 0)
    counts = LocalSimulator().run_bank([ground, excited], shots=256)
    if counts[0] != {"0": 256, "1": 0}:
        raise SystemExit(f"Local simulator did not prepare |0>: {counts[0]}")
    if counts[1] != {"0": 0, "1": 256}:
        raise SystemExit(f"Local simulator did not prepare |1>: {counts[1]}")


def test_saved_bank_is_local() -> None:
    if not BANK_PATH.exists():
        raise SystemExit("src/bank.json is missing. Run python src/measurement_bank.py.")
    bank = json.loads(BANK_PATH.read_text())
    source = str(bank["source"])
    if source != LOCAL_SOURCE:
        raise SystemExit(f"Bank source is {source!r}, expected {LOCAL_SOURCE!r}.")
    lowered = source.lower()
    if "atlas" in lowered or "hardware" in lowered:
        raise SystemExit(f"Local bank is mislabelled: {source!r}.")
    shots = int(bank["shots"])
    points = bank["points"]
    if len(points) != 24:
        raise SystemExit(f"Expected 24 time points, found {len(points)}.")
    worst = 0.0
    for point in points:
        counts = point["counts"]
        total = int(counts.get("0", 0)) + int(counts.get("1", 0))
        if total != shots:
            raise SystemExit(f"Counts at index {point['index']} sum to {total}, not {shots}.")
        measured = int(counts.get("1", 0)) / shots
        worst = max(worst, abs(measured - float(point["ideal_acceptor"])))
    if worst > 0.05:
        raise SystemExit(f"Measured acceptor left the exact curve by {worst:.3f}.")
    print(f"bank source={source!r} shots={shots} max deviation={worst:.4f}")


if __name__ == "__main__":
    test_atlas_backend_is_a_stub()
    test_local_simulator_prepares_basis_states()
    test_saved_bank_is_local()
    print("measurement bank checks passed")
