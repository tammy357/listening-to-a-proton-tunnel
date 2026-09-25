"""Checks for the symmetric tunneling doublet and the H/D gap.

Run from the repository root:

    python tests/test_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hamiltonian import ANGSTROM_M, JOULE_PER_EV
from src.quantum_sim import (
    evolve_exact,
    evolve_split_operator,
    split_operator_circuit,
    split_operator_numpy,
)
from src.rates import kie

BARRIER_EV = 0.3
HALF_WIDTH_ANGSTROM = 0.35
# 200 symmetric Trotter steps of this barrier already overlap the exact
# doublet above 0.99. The expectation value of H stays below E_b, so the
# packet is still tunneling. The 0.3 eV checks above exercise the same
# Hamiltonian at a taller barrier.
CIRCUIT_BARRIER_EV = 0.15
CIRCUIT_STEPS = (10, 50, 200)


def _symmetric_run():
    return evolve_exact(
        BARRIER_EV,
        0.0,
        well_half_width_angstrom=HALF_WIDTH_ANGSTROM,
        mass="H",
        n_periods=3.0,
        n_steps=200,
    )


def test_norm_is_preserved() -> None:
    _model, evolution = _symmetric_run()
    drift = np.max(np.abs(evolution.norms - 1.0))
    assert drift < 1e-10, f"statevector norm drifted by {drift}"


def test_symmetric_population_oscillates() -> None:
    _model, evolution = _symmetric_run()
    period = evolution.tunneling_period_s
    half = int(np.argmin(np.abs(evolution.times_s - 0.5 * period)))
    full = int(np.argmin(np.abs(evolution.times_s - period)))
    donor_half = float(evolution.donor_population[half])
    donor_full = float(evolution.donor_population[full])
    assert donor_half < 0.1, f"donor population at T_tun/2 is {donor_half}"
    assert donor_full > 0.9, f"donor population at T_tun is {donor_full}"


def test_deuterium_splitting_is_smaller() -> None:
    ratio = kie(BARRIER_EV, HALF_WIDTH_ANGSTROM)
    assert ratio > 1.0, f"H/D splitting ratio is {ratio}"


def _fidelity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(np.vdot(left, right)) ** 2)


def test_split_operator_circuit_matches_exact() -> None:
    """Print |⟨ψ_circuit|ψ_exact⟩|² and require > 0.99 at the finest step.

    The NumPy split-operator is checked against ``evolve_exact`` first. The
    circuit is then checked against that NumPy product, so a Fourier-order
    mistake cannot hide inside the Trotter error.
    """

    from qiskit import transpile

    model, probe = evolve_exact(
        CIRCUIT_BARRIER_EV,
        0.0,
        well_half_width_angstrom=HALF_WIDTH_ANGSTROM,
        mass="H",
        n_periods=1.0,
        n_steps=1,
        keep_states=True,
    )
    assert probe.states is not None
    psi0 = probe.initial_statevector
    period = probe.tunneling_period_s
    energy = float(np.real(np.vdot(psi0, model.matrix @ psi0))) / JOULE_PER_EV

    one_step = split_operator_circuit(
        model.v_joule, model.mass_kg, model.spacing_m, period / 200.0, n_steps=1
    )
    basis = transpile(
        one_step,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=2,
    )
    print(
        "one Trotter step transpiled to ['rz', 'sx', 'x', 'cx'], "
        "optimization_level=2"
    )
    print(f"depth = {basis.depth()}")
    print(f"gate count = {basis.size()}")
    print(f"ops = {dict(basis.count_ops())}")

    print(
        "split-operator fidelity |<ψ_circuit|ψ_exact>|^2\n"
        f"E_b = {CIRCUIT_BARRIER_EV:.2f} eV, a = {HALF_WIDTH_ANGSTROM:.2f} Å, "
        f"λ = 0, mass = H, <H> = {energy:.6f} eV"
    )
    print(f"{'n_steps':>7}  {'time':<8}  {'fidelity':>10}")

    prepared = evolve_split_operator(
        psi0,
        model.v_joule,
        model.mass_kg,
        model.spacing_m,
        period,
        n_steps=1,
        sample_at=(0,),
    )
    prep_overlap = _fidelity(prepared[0], psi0)
    print(f"state preparation fidelity = {prep_overlap:.12f}")
    assert prep_overlap > 1.0 - 1e-8, f"StatePreparation fidelity is {prep_overlap}"

    finest: dict[str, float] = {}
    for n_steps in CIRCUIT_STEPS:
        dt = period / n_steps
        _model, exact = evolve_exact(
            CIRCUIT_BARRIER_EV,
            0.0,
            well_half_width_angstrom=HALF_WIDTH_ANGSTROM,
            mass="H",
            n_periods=1.0,
            n_steps=n_steps,
            keep_states=True,
        )
        assert exact.states is not None
        circuit_states = evolve_split_operator(
            exact.initial_statevector,
            model.v_joule,
            model.mass_kg,
            model.spacing_m,
            dt,
            n_steps=n_steps,
            sample_at=(n_steps // 2, n_steps),
        )
        for label, frame in (("T_tun/2", n_steps // 2), ("T_tun", n_steps)):
            numpy_state = split_operator_numpy(
                exact.initial_statevector,
                model.v_joule,
                model.mass_kg,
                model.spacing_m,
                dt,
                frame,
            )
            exact_state = exact.states[frame]
            circuit_state = circuit_states[frame]
            overlap = _fidelity(circuit_state, exact_state)
            numpy_overlap = _fidelity(numpy_state, exact_state)
            circuit_numpy = _fidelity(circuit_state, numpy_state)
            print(f"{n_steps:7d}  {label:<8}  {overlap:10.6f}")
            assert circuit_numpy > 1.0 - 1e-6, (
                f"circuit vs numpy at n_steps={n_steps}, {label} is {circuit_numpy}"
            )
            if n_steps == CIRCUIT_STEPS[-1]:
                finest[label] = overlap
                assert numpy_overlap > 0.99, (
                    f"numpy vs exact at {label} is {numpy_overlap}"
                )

    for label, overlap in finest.items():
        assert overlap > 0.99, f"fidelity at n_steps={CIRCUIT_STEPS[-1]}, {label} is {overlap}"


def report_reference_numbers() -> None:
    _model, evolution = _symmetric_run()
    ratio = kie(BARRIER_EV, HALF_WIDTH_ANGSTROM)
    print(f"ΔE = {evolution.splitting_ev:.6e} eV")
    print(f"T_tun = {evolution.tunneling_period_s:.6e} s")
    print(f"KIE = {ratio:.6f}")
    half = int(np.argmin(np.abs(evolution.times_in_periods - 0.5)))
    full = int(np.argmin(np.abs(evolution.times_in_periods - 1.0)))
    print(
        f"donor(T_tun/2) = {evolution.donor_population[half]:.6f}, "
        f"donor(T_tun) = {evolution.donor_population[full]:.6f}"
    )


def _inner_wall_width(x_angstrom: np.ndarray, v_ev: np.ndarray, e0_ev: float) -> float:
    """Distance between the inner classical turning points where V(x) = E0."""

    barrier = int(np.argmin(np.abs(x_angstrom)))
    left = None
    right = None
    for index in range(barrier, 0, -1):
        v0 = float(v_ev[index])
        v1 = float(v_ev[index - 1])
        if (v0 - e0_ev) * (v1 - e0_ev) <= 0.0:
            span = v0 - v1
            fraction = 0.0 if abs(span) < 1e-15 else (e0_ev - v1) / span
            left = float(x_angstrom[index - 1] + fraction * (x_angstrom[index] - x_angstrom[index - 1]))
            break
    for index in range(barrier, len(x_angstrom) - 1):
        v0 = float(v_ev[index])
        v1 = float(v_ev[index + 1])
        if (v0 - e0_ev) * (v1 - e0_ev) <= 0.0:
            span = v1 - v0
            fraction = 0.0 if abs(span) < 1e-15 else (e0_ev - v0) / span
            right = float(x_angstrom[index] + fraction * (x_angstrom[index + 1] - x_angstrom[index]))
            break
    if left is None or right is None or right <= left:
        raise AssertionError("Inner turning points were not found.")
    return right - left


def test_two_level_gap_and_rx_populations() -> None:
    """The first gap above the doublet is wide, and rx(2π t/T) tracks the wells."""

    from src.atlas_tomography import level_ratios, population_errors

    ratios = level_ratios()
    for mass, ratio in ratios.items():
        print(f"(E2 − E1) / ΔE {mass} = {ratio:.3f}")
        assert ratio > 5.0, f"{mass} two-level ratio is {ratio}"
    worst, _exact, _predicted = population_errors()
    print(f"max |P_rx − P_exact| = {worst:.4f}")
    assert worst < 0.02, f"RX acceptor error is {worst}"


def test_ground_energy_and_wall_thickness_follow_mass() -> None:
    """Heavier isotopes sit lower in the well, so the barrier is thicker at E0."""

    found = {}
    for mass in ("H", "D", "T"):
        model, evolution = evolve_exact(
            barrier_ev=CIRCUIT_BARRIER_EV,
            asymmetry_ev=0.0,
            well_half_width_angstrom=HALF_WIDTH_ANGSTROM,
            mass=mass,
            n_periods=1.0,
            n_steps=1,
            keep_states=False,
        )
        e0_ev = float(evolution.energies_j[0] / JOULE_PER_EV)
        width = _inner_wall_width(
            model.x_m / ANGSTROM_M,
            model.v_joule / JOULE_PER_EV,
            e0_ev,
        )
        found[mass] = (e0_ev, width)
        print(f"E0({mass}) = {e0_ev * 1e3:.3f} meV, width = {width:.2f} Å")
    assert found["T"][0] < found["D"][0] < found["H"][0]
    assert found["T"][1] > found["H"][1]


def test_atlas_bank_is_an_aer_simulator() -> None:
    """The saved tomography file is Aer, with the donor at z ≈ +1."""

    import json

    path = ROOT / "src" / "atlas_bank.json"
    bank = json.loads(path.read_text(encoding="utf-8"))
    assert bank["provider"] == "aer"
    assert bank["backend"] == "automatic"
    assert bank["execution"] == "simulator"
    assert bank["shots"] == 4096
    assert len(bank["points"]) == 24
    assert len(bank["job_ids"]) == 24
    first = bank["points"][0]
    assert first["z"] > 0.5
    assert abs(first["acceptor_population"] - (1.0 - first["z"]) / 2.0) < 1e-9


if __name__ == "__main__":
    test_norm_is_preserved()
    test_symmetric_population_oscillates()
    test_deuterium_splitting_is_smaller()
    report_reference_numbers()
    test_split_operator_circuit_matches_exact()
    test_ground_energy_and_wall_thickness_follow_mass()
    test_two_level_gap_and_rx_populations()
    test_atlas_bank_is_an_aer_simulator()
    print("all physics checks passed")
