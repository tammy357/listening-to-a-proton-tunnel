"""Exact grid evolution of the localized tunneling doublet.

The initial state is the standard two-level packet

    ψ(0) = (φ0 + σ φ1) / √2,

where φ0 and φ1 are the lowest two eigenstates of H and the relative sign σ
is chosen so that most of the density sits in the donor well (x < 0). In a
symmetric double well this state oscillates between the wells with period
T_tun = h / (E1 − E0).

``evolve_exact`` propagates that state with ``scipy.linalg.expm``. That is a
classical numerical simulation of the Schrödinger equation on the grid, used
as ground truth. It is not a quantum-circuit execution.

The recorded clock is in seconds, and also in units of T_tun. The default
window is 0 to 3 tunneling periods at 200 steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .hamiltonian import (
    HBAR_J_S,
    JOULE_PER_EV,
    ProtonHamiltonian,
    build_hamiltonian,
    lowest_eigenstates,
    tunneling_splitting,
)

DEFAULT_PERIODS = 3.0
DEFAULT_STEPS = 200


@dataclass(frozen=True)
class TunnelingEvolution:
    """Populations and the final 8-qubit state.

    Attributes:
        times_s: Sample times in seconds, including 0 and the final time.
        times_in_periods: The same samples in units of T_tun.
        tunneling_period_s: T_tun = h / ΔE, in seconds.
        splitting_j: ΔE = E1 − E0, in joules.
        splitting_ev: ΔE in electron-volts.
        donor_population: Probability on x < 0 at each sample.
        acceptor_population: Probability on x > 0 at each sample.
        norms: ℓ² norm of the propagated statevector at each sample.
        initial_statevector: Normalised amplitudes at t = 0, length 256.
        final_statevector: Amplitudes at the last sample, length 256.
        final_density: |ψ|² per grid site at the last sample.
        survival_probability: |⟨ψ(0)|ψ(t_final)⟩|².
        energies_j: Lowest eigenenergies used to build ψ(0), in joules.
    """

    times_s: np.ndarray
    times_in_periods: np.ndarray
    tunneling_period_s: float
    splitting_j: float
    splitting_ev: float
    donor_population: np.ndarray
    acceptor_population: np.ndarray
    norms: np.ndarray
    initial_statevector: np.ndarray
    final_statevector: np.ndarray
    final_density: np.ndarray
    survival_probability: float
    energies_j: np.ndarray


def donor_localized_state(
    x_m: np.ndarray, phi0: np.ndarray, phi1: np.ndarray
) -> np.ndarray:
    """Build ψ(0) = (φ0 + σ φ1) / √2 with the donor well occupied.

    σ = ±1. The symmetric and antisymmetric combinations of the lowest
    doublet sit on opposite sides of x = 0. The sign that puts the larger
    share of the probability on x < 0 is kept. Both input vectors must
    already be ℓ²-normalised and orthogonal, as ``eigh`` returns them.
    """

    best: np.ndarray | None = None
    best_donor = -1.0
    donor_mask = np.asarray(x_m) < 0.0
    for sign in (1.0, -1.0):
        psi = (np.asarray(phi0, dtype=np.float64) + sign * np.asarray(phi1, dtype=np.float64))
        psi = psi / np.sqrt(2.0)
        donor = float(np.sum(psi[donor_mask] ** 2))
        if donor > best_donor:
            best_donor = donor
            best = psi
    if best is None:
        raise ValueError("Could not build a donor-localized state.")
    return best.astype(np.complex128)


def evolve_exact(
    barrier_ev: float,
    asymmetry_ev: float = 0.0,
    well_half_width_angstrom: float = 0.35,
    mass: float | str = "H",
    n_periods: float = DEFAULT_PERIODS,
    n_steps: int = DEFAULT_STEPS,
) -> tuple[ProtonHamiltonian, TunnelingEvolution]:
    """Ground-truth evolution by a classical matrix exponential.

    This is a classical numerical simulation of the time-dependent
    Schrödinger equation on the 256-point grid. It is the reference result
    against which any later quantum-circuit propagator should be compared.
    The propagator is

        U(Δt) = exp(−i H Δt / ℏ),

    formed once with ``scipy.linalg.expm`` and applied for ``n_steps``
    steps. The state is not renormalised. The clock runs from 0 to
    ``n_periods * T_tun`` (default 0 to 3 periods, 200 steps).

    Args:
        barrier_ev: Barrier height E_b in electron-volts.
        asymmetry_ev: Slope λ in eV/Å. Use 0 for the symmetric well.
        well_half_width_angstrom: Well half-separation a, in angstroms.
        mass: ``"H"``, ``"D"``, ``"T"``, or a mass in kilograms.
        n_periods: Length of the run in units of T_tun.
        n_steps: Number of exponential steps. The record has ``n_steps + 1``
            samples, including t = 0.

    Returns:
        The Hamiltonian that was evolved, and the population record.
    """

    if n_periods <= 0:
        raise ValueError("The run must cover a positive number of tunneling periods.")
    if n_steps < 1:
        raise ValueError("At least one time step is required.")

    model = build_hamiltonian(
        barrier_ev,
        asymmetry_ev,
        well_half_width_angstrom=well_half_width_angstrom,
        mass=mass,
    )
    energies, vectors = lowest_eigenstates(model.matrix, k=4)
    splitting_j, period_s = tunneling_splitting(energies)
    psi0 = donor_localized_state(model.x_m, vectors[:, 0], vectors[:, 1])

    dt_s = (float(n_periods) * period_s) / int(n_steps)
    # A global energy shift does not change populations. Subtracting E0 keeps
    # the phase of the occupied doublet of order 2π per period, which is the
    # regime where the scaling-and-squaring exponential stays accurate.
    shifted = model.matrix.toarray().astype(np.complex128)
    shifted = shifted - energies[0] * np.eye(shifted.shape[0])
    unitary = expm(-1j * shifted * (dt_s / HBAR_J_S))

    psi = psi0.copy()
    times = np.empty(n_steps + 1, dtype=np.float64)
    donor = np.empty(n_steps + 1, dtype=np.float64)
    acceptor = np.empty(n_steps + 1, dtype=np.float64)
    norms = np.empty(n_steps + 1, dtype=np.float64)
    donor_mask = model.x_m < 0.0

    for frame in range(n_steps + 1):
        norms[frame] = float(np.linalg.norm(psi))
        probability = np.abs(psi) ** 2
        times[frame] = frame * dt_s
        donor[frame] = float(probability[donor_mask].sum())
        acceptor[frame] = float(probability[~donor_mask].sum())
        if frame < n_steps:
            psi = unitary @ psi

    final = np.asarray(psi, dtype=np.complex128)
    overlap = np.vdot(psi0, final)
    evolution = TunnelingEvolution(
        times_s=times,
        times_in_periods=times / period_s,
        tunneling_period_s=period_s,
        splitting_j=splitting_j,
        splitting_ev=splitting_j / JOULE_PER_EV,
        donor_population=donor,
        acceptor_population=acceptor,
        norms=norms,
        initial_statevector=psi0,
        final_statevector=final,
        final_density=np.abs(final) ** 2,
        survival_probability=float(np.abs(overlap) ** 2),
        energies_j=energies,
    )
    return model, evolution


def evolve_tunneling(
    barrier_ev: float,
    asymmetry_ev: float,
    n_periods: float = DEFAULT_PERIODS,
    n_steps: int = DEFAULT_STEPS,
    well_half_width_angstrom: float = 0.35,
    mass: float | str = "H",
) -> tuple[ProtonHamiltonian, TunnelingEvolution]:
    """Run ``evolve_exact`` and return its ground-truth record.

    The time axis is ``n_periods`` multiples of T_tun, not a fixed
    femtosecond window.
    """

    return evolve_exact(
        barrier_ev,
        asymmetry_ev,
        well_half_width_angstrom=well_half_width_angstrom,
        mass=mass,
        n_periods=n_periods,
        n_steps=n_steps,
    )
