"""Qualitative rate comparison for the model double well.

Nothing in this module is a rate for carbonic anhydrase, and nothing here is
fitted to the PDB structure. The two expressions answer a narrower question:
in *this* one-dimensional Hamiltonian, how does an Arrhenius hop over E_b
compare with the temperature-independent two-level tunneling frequency, and
how does that tunneling gap change from H to D.

Assumptions, stated so they are not mistaken for measurements:

* The classical attempt frequency is ν = 1×10¹³ s⁻¹, a round number of order
  a molecular vibration. It is not computed from the curvature of V(x).
* The classical barrier is the model parameter E_b, with no zero-point
  correction and no recrossing.
* k_tunnel ≈ ΔE / h uses the lowest tunneling splitting of this Hamiltonian
  and does not depend on temperature.
* The kinetic isotope effect reported here is the ratio of those splittings,
  ΔE(H) / ΔE(D), not k_H / k_D from a kinetics experiment.
"""

from __future__ import annotations

import math

from .hamiltonian import (
    JOULE_PER_EV,
    PLANCK_H_J_S,
    build_hamiltonian,
    lowest_eigenstates,
    tunneling_splitting,
)

# Assumed attempt frequency. Not derived from the double-well curvature.
ATTEMPT_FREQUENCY_HZ = 1.0e13
BOLTZMANN_J_PER_K = 1.380649e-23


def k_classical(
    t_kelvin: float,
    barrier_ev: float,
    attempt_hz: float = ATTEMPT_FREQUENCY_HZ,
) -> float:
    """Qualitative Arrhenius estimate k = ν exp(−E_b / k_B T).

    ν defaults to 1×10¹³ s⁻¹. This is an assumption, not a property of
    carbonic anhydrase. The result is in s⁻¹.

    Args:
        t_kelvin: Temperature in kelvin.
        barrier_ev: Model barrier E_b in electron-volts.
        attempt_hz: Attempt frequency ν in s⁻¹.
    """

    if t_kelvin <= 0:
        raise ValueError("Temperature must be positive.")
    if attempt_hz <= 0:
        raise ValueError("Attempt frequency must be positive.")
    barrier_j = float(barrier_ev) * JOULE_PER_EV
    exponent = -barrier_j / (BOLTZMANN_J_PER_K * float(t_kelvin))
    return float(attempt_hz) * math.exp(exponent)


def k_tunnel(splitting_j: float) -> float:
    """Qualitative tunneling frequency k ≈ ΔE / h, in s⁻¹.

    In this two-level model the gap ΔE does not depend on temperature, so
    neither does this estimate. It is 1/T_tun, the number of full donor–
    acceptor round trips per second, not a Golden-Rule rate constant.
    """

    if splitting_j <= 0:
        raise ValueError("Tunneling splitting must be positive.")
    return float(splitting_j) / PLANCK_H_J_S


def kie(barrier_ev: float, well_half_width_angstrom: float) -> float:
    """Ratio of tunneling splittings, ΔE(H) / ΔE(D), for this model only.

    Both gaps are computed on the same V(x) with the H and D mass presets.
    A ratio above 1 means the heavier particle has the smaller gap. This is
    a qualitative comparison of the model Hamiltonian, not an experimental
    kinetic isotope effect.

    Args:
        barrier_ev: Barrier height E_b in electron-volts.
        well_half_width_angstrom: Well half-separation a, in angstroms.
    """

    hydrogen = _splitting_j(barrier_ev, well_half_width_angstrom, "H")
    deuterium = _splitting_j(barrier_ev, well_half_width_angstrom, "D")
    return hydrogen / deuterium


def _splitting_j(barrier_ev: float, well_half_width_angstrom: float, mass: str) -> float:
    model = build_hamiltonian(
        barrier_ev,
        0.0,
        well_half_width_angstrom=well_half_width_angstrom,
        mass=mass,
    )
    energies, _vectors = lowest_eigenstates(model.matrix, k=2)
    gap, _period = tunneling_splitting(energies)
    return gap
