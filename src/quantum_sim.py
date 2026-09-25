"""Real-time evolution of the proton on an 8-qubit statevector.

The proton starts as a Gaussian wavepacket sitting in the donor well
(x < 0). Because H is time-independent, the evolution operator

    U(dt) = exp(−i H dt / ħ)

is built once with a matrix exponential, wrapped in a Qiskit ``Operator``,
and applied with ``Statevector.evolve``. Donor population is the probability
on sites with x < 0; acceptor population is the probability on x > 0.

Times inside this module are seconds. The dashboard converts the recorded
clock to femtoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from .hamiltonian import (
    ANGSTROM_M,
    HBAR_J_S,
    ProtonHamiltonian,
    build_hamiltonian,
)

# Packet width. Narrow enough that t = 0 sits in the donor well (population
# ≈ 1 for x < 0). At this width the kinetic energy is about 0.07 eV, below
# a 0.2 eV barrier, so the default run is tunneling rather than classical
# spilling.
PACKET_SIGMA_M = 0.08 * ANGSTROM_M
N_FRAMES = 80


@dataclass(frozen=True)
class TunnelingEvolution:
    """Populations and the final 8-qubit state.

    Attributes:
        times_s: Sample times in seconds, including 0 and the final time.
        donor_population: Probability on x < 0 at each sample.
        acceptor_population: Probability on x > 0 at each sample.
        initial_statevector: Normalised amplitudes at t = 0, length 256.
        final_statevector: Normalised amplitudes at the last sample.
        final_density: |ψ|² per grid site at the last sample. Sums to 1.
        survival_probability: |⟨ψ(0)|ψ(t_final)⟩|².
    """

    times_s: np.ndarray
    donor_population: np.ndarray
    acceptor_population: np.ndarray
    initial_statevector: np.ndarray
    final_statevector: np.ndarray
    final_density: np.ndarray
    survival_probability: float


def gaussian_packet(
    x_m: np.ndarray, center_m: float, sigma_m: float = PACKET_SIGMA_M
) -> np.ndarray:
    """Normalised Gaussian amplitudes on the qubit register.

    The continuous packet exp(−(x−x0)² / 4σ²) is sampled on the grid and
    renormalised in ℓ², which is the norm Qiskit uses for a statevector.
    Grid index i is the computational-basis integer i, little-endian.

    Args:
        x_m: Coordinates in metres.
        center_m: Packet centre in metres. The donor well uses x = −a.
        sigma_m: Position uncertainty σ in metres.

    Returns:
        Complex amplitudes of length 256 with unit ℓ² norm.
    """

    if sigma_m <= 0:
        raise ValueError("Packet width must be positive.")
    raw = np.exp(-((x_m - center_m) ** 2) / (4.0 * sigma_m**2))
    norm = np.linalg.norm(raw)
    if norm == 0:
        raise ValueError("Gaussian packet vanished on this grid.")
    return (raw / norm).astype(np.complex128)


def time_evolution_operator(hamiltonian_j: np.ndarray, dt_s: float) -> Operator:
    """Build U(dt) = exp(−i H dt / ħ) as a Qiskit operator.

    A polar cleanup removes the roundoff that would otherwise accumulate
    when the same unitary is applied for several dozen frames. H is
    Hermitian, so the exact exponential is unitary.

    Args:
        hamiltonian_j: Dense Hamiltonian in joules, shape (256, 256).
        dt_s: Step in seconds.

    Returns:
        Unitary ``Operator`` on the 8-qubit register.
    """

    generator = -1j * np.asarray(hamiltonian_j, dtype=np.complex128) * (dt_s / HBAR_J_S)
    unitary = expm(generator)
    unitary = _nearest_unitary(unitary)
    return Operator(unitary)


def evolve_tunneling(
    barrier_ev: float,
    asymmetry_ev: float,
    duration_s: float,
    n_frames: int = N_FRAMES,
) -> tuple[ProtonHamiltonian, TunnelingEvolution]:
    """Propagate the donor wavepacket and record well populations.

    Args:
        barrier_ev: Barrier height in electron-volts.
        asymmetry_ev: Asymmetry λ in electron-volts.
        duration_s: Total propagation time in seconds. Must be positive.
        n_frames: Number of steps. The result contains ``n_frames + 1``
            samples, including the initial state.

    Returns:
        The Hamiltonian that was evolved, and the population record plus
        the final statevector.
    """

    if duration_s <= 0:
        raise ValueError("Duration must be positive.")
    if n_frames < 1:
        raise ValueError("At least one time step is required.")

    model = build_hamiltonian(barrier_ev, asymmetry_ev)
    dt_s = float(duration_s) / int(n_frames)
    unitary = time_evolution_operator(model.matrix.toarray(), dt_s)

    center_m = -model.well_half_width_m
    psi0 = gaussian_packet(model.x_m, center_m)
    state = Statevector(psi0)

    times = np.empty(n_frames + 1, dtype=np.float64)
    donor = np.empty(n_frames + 1, dtype=np.float64)
    acceptor = np.empty(n_frames + 1, dtype=np.float64)
    donor_mask = model.x_m < 0.0

    for frame in range(n_frames + 1):
        amplitudes = np.asarray(state.data, dtype=np.complex128)
        probability = np.abs(amplitudes) ** 2
        times[frame] = frame * dt_s
        donor[frame] = float(probability[donor_mask].sum())
        acceptor[frame] = float(probability[~donor_mask].sum())
        if frame < n_frames:
            state = state.evolve(unitary)

    final = np.asarray(state.data, dtype=np.complex128)
    final = final / np.linalg.norm(final)
    overlap = np.vdot(psi0, final)
    survival = float(np.abs(overlap) ** 2)
    density = np.abs(final) ** 2

    evolution = TunnelingEvolution(
        times_s=times,
        donor_population=donor,
        acceptor_population=acceptor,
        initial_statevector=psi0,
        final_statevector=final,
        final_density=density,
        survival_probability=survival,
    )
    return model, evolution


def _nearest_unitary(matrix: np.ndarray) -> np.ndarray:
    """Polar factor U (U† U)^(−1/2), so a slightly drifted exponential is unitary."""

    singular_left, _, singular_right = np.linalg.svd(matrix, full_matrices=False)
    return singular_left @ singular_right
