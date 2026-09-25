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

from src.quantum_sim import evolve_exact
from src.rates import kie

BARRIER_EV = 0.3
HALF_WIDTH_ANGSTROM = 0.35


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


if __name__ == "__main__":
    test_norm_is_preserved()
    test_symmetric_population_oscillates()
    test_deuterium_splitting_is_smaller()
    report_reference_numbers()
    print("all physics checks passed")
