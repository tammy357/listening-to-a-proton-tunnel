"""Statevector time-evolution of the tunneling proton with Qiskit unitaries.

Physics
-------
The proton starts *localised on the donor side* (``x < 0``) as a normalised Gaussian
wavepacket::

    psi(x, 0) ~ exp(-(x - x0)^2 / (4 sigma^2)) * exp(i k0 x)

This is a classic non-stationary state: it is a superposition of the (nearly degenerate)
symmetric/antisymmetric tunneling doublet plus higher vibrational states. Because the
doublet members differ in energy by the **tunneling splitting** ``Delta = E1 - E0``, their
relative phase winds at ``Delta/hbar`` and the probability density sloshes between the wells
with period ``T_tunnel = 2 pi hbar / Delta`` (the textbook coherent double-well oscillation).
A tilted well suppresses the amplitude of that oscillation but speeds up its frequency; a
higher/wider barrier shrinks ``Delta`` exponentially and slows everything down.

Numerics
--------
Time evolution is exact-in-``dt``: we build the propagator

    U(dt) = exp(-i H dt / hbar)

by dense matrix exponentiation (``scipy.linalg.expm``) *once*, wrap it in a Qiskit
:class:`~qiskit.quantum_info.Operator` / :class:`~qiskit.circuit.library.UnitaryGate` acting on
all 8 qubits, and apply it repeatedly to a :class:`~qiskit.quantum_info.Statevector`. Because
``U`` is a genuine unitary (verified with ``Operator.is_unitary()``), the norm is conserved to
machine precision and there is no Trotter/time-step stability constraint to worry about.

Register convention
-------------------
The 256-dimensional statevector index ``k`` is the computational basis state ``|k>`` of the
8-qubit register *and* the spatial grid point ``x[k]``. Amplitudes are therefore directly
interpretable: ``|psi[k]|^2`` is the probability of finding the proton in grid cell ``k``.
Donor population is ``sum_{x[k] < 0} |psi[k]|^2``, acceptor population is the complement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from scipy.sparse import spmatrix

from .hamiltonian import (
    ANGSTROM,
    DEFAULT_WELL_SEPARATION_ANG,
    FEMTOSECOND,
    HBAR,
    M_H,
    SpatialGrid,
    build_grid,
    build_hamiltonian,
    joules_to_ev,
)

DEFAULT_WAVEPACKET_WIDTH_ANG: float = 0.15
"""Gaussian width ``sigma`` of the initial wavepacket (Angstrom).

Chosen close to the zero-point width of the donor well, ``sigma_0 = sqrt(hbar / 2 m omega)``,
so the initial state is a physically sensible "proton sitting on the donor" rather than an
artificially squeezed spike full of high-energy components.
"""


@dataclass(frozen=True)
class TunnelingResult:
    """Everything the UI and the sonifier need from one simulation run.

    Attributes
    ----------
    grid:
        The 256-point spatial grid used (positions in metres).
    potential_j:
        Double-well potential sampled on the grid, in joules.
    times_fs:
        Sample times in femtoseconds, shape ``(n_steps + 1,)``, starting at 0.
    density:
        Probability ``|psi(x, t)|^2`` on the grid for every sample time, shape
        ``(n_steps + 1, 256)``. Rows sum to 1.
    donor_population:
        Probability of finding the proton at ``x < 0`` versus time, shape ``(n_steps + 1,)``.
    acceptor_population:
        Probability of finding the proton at ``x > 0`` versus time — the tunneling signal.
    statevectors:
        The Qiskit :class:`Statevector` at every sample time (8 qubits, dimension 256).
    circuit:
        The 8-qubit circuit holding the single ``U(dt)`` unitary block that is iterated.
    dt_s:
        Time step in seconds.
    norm_fidelity:
        ``|<psi(0)|psi(0)>|`` re-measured on the final state, i.e. how well the propagation
        preserved normalisation. Should be 1 to ~1e-12; reported on the Atlas card.
    energy_ev:
        Expectation value ``<psi(0)|H|psi(0)>`` in eV (conserved by construction).
    tunneling_splitting_ev:
        Energy splitting ``E1 - E0`` of the lowest doublet, in eV, when it was computed.
        ``None`` if the eigensolve was skipped.
    """

    grid: SpatialGrid
    potential_j: np.ndarray
    times_fs: np.ndarray
    density: np.ndarray
    donor_population: np.ndarray
    acceptor_population: np.ndarray
    statevectors: List[Statevector]
    circuit: QuantumCircuit
    dt_s: float
    norm_fidelity: float
    energy_ev: float
    tunneling_splitting_ev: Optional[float] = None

    @property
    def final_statevector(self) -> Statevector:
        """The statevector at the last sample time."""
        return self.statevectors[-1]

    @property
    def max_transfer(self) -> float:
        """Peak acceptor-well population reached during the simulated window."""
        return float(np.max(self.acceptor_population))

    @property
    def num_qubits(self) -> int:
        """Size of the register backing the simulation (8)."""
        return self.grid.num_qubits


def gaussian_wavepacket(
    grid: SpatialGrid,
    center_ang: float = -DEFAULT_WELL_SEPARATION_ANG,
    width_ang: float = DEFAULT_WAVEPACKET_WIDTH_ANG,
    momentum_ev: float = 0.0,
    mass: float = M_H,
) -> Statevector:
    """Prepare a normalised Gaussian wavepacket localised in the donor well.

    Parameters
    ----------
    grid:
        The 256-point grid; the returned statevector has one amplitude per grid point.
    center_ang:
        Centre ``x0`` in Angstroms. Defaults to ``-a`` (the donor minimum), i.e. ``x < 0``.
    width_ang:
        Gaussian standard deviation ``sigma`` of the *amplitude* envelope, in Angstroms.
    momentum_ev:
        Optional initial kinetic energy in eV, imprinted as a plane-wave phase
        ``exp(i k0 x)`` with ``k0 = sqrt(2 m E)/hbar``. Zero gives a state at rest — tunneling
        then happens purely by barrier penetration, not by "throwing" the proton over.
    mass:
        Particle mass in kg, used only to convert ``momentum_ev`` into a wavenumber.

    Returns
    -------
    qiskit.quantum_info.Statevector
        An 8-qubit statevector normalised so ``sum |psi[k]|^2 == 1``.
    """
    x0 = center_ang * ANGSTROM
    sigma = width_ang * ANGSTROM
    envelope = np.exp(-((grid.x - x0) ** 2) / (4.0 * sigma**2))

    if momentum_ev > 0.0:
        from .hamiltonian import ev_to_joules

        k0 = np.sqrt(2.0 * mass * float(ev_to_joules(momentum_ev))) / HBAR
        envelope = envelope.astype(complex) * np.exp(1j * k0 * grid.x)

    psi = np.asarray(envelope, dtype=complex)
    norm = np.linalg.norm(psi)
    if norm == 0.0:  # pragma: no cover - only if sigma underflows
        raise ValueError("Initial wavepacket has zero norm; check center/width arguments.")
    return Statevector(psi / norm)


def build_propagator(hamiltonian: spmatrix, dt_s: float) -> Tuple[Operator, QuantumCircuit]:
    """Build the one-step time-evolution unitary ``U(dt) = exp(-i H dt / hbar)``.

    The dense 256x256 exponential is cheap (milliseconds) and *exact*, so unlike a Trotterised
    or Crank-Nicolson propagator there is no accuracy/stability trade-off in ``dt``.

    Parameters
    ----------
    hamiltonian:
        Sparse Hermitian ``H`` in joules from :func:`src.hamiltonian.build_hamiltonian`.
    dt_s:
        Time step in seconds (SI).

    Returns
    -------
    (operator, circuit)
        ``operator`` is the Qiskit :class:`Operator` wrapping ``U``; ``circuit`` is an 8-qubit
        :class:`QuantumCircuit` containing it as a single ``UnitaryGate`` block, which is what
        would be submitted to a backend (or to Moth Atlas) for hardware execution.

    Raises
    ------
    ValueError
        If the exponentiated matrix fails Qiskit's unitarity check.
    """
    h_dense = np.asarray(hamiltonian.todense(), dtype=complex)
    u_matrix = expm(-1j * h_dense * dt_s / HBAR)

    operator = Operator(u_matrix)
    if not operator.is_unitary(atol=1e-8):  # pragma: no cover - guards numerical blow-ups
        raise ValueError("exp(-iHdt/hbar) is not unitary; H is probably not Hermitian.")

    num_qubits = int(np.log2(u_matrix.shape[0]))
    circuit = QuantumCircuit(num_qubits, name="U_tunneling")
    circuit.append(UnitaryGate(u_matrix, label="U(dt)"), range(num_qubits))
    return operator, circuit


def simulate_tunneling(
    barrier_height_ev: float = 0.15,
    asymmetry_ev: float = 0.10,
    duration_fs: float = 50.0,
    num_steps: int = 120,
    well_separation_ang: float = DEFAULT_WELL_SEPARATION_ANG,
    width_ang: float = DEFAULT_WAVEPACKET_WIDTH_ANG,
    mass: float = M_H,
    compute_splitting: bool = True,
) -> TunnelingResult:
    """Run the full proton-tunneling simulation over ``t`` in ``[0, duration_fs]``.

    Pipeline
    --------
    1. Build the 256-point grid (8 qubits) and the Hamiltonian ``H = T + V``.
    2. Prepare ``|psi(0)>`` as a Gaussian localised in the donor well (``x < 0``).
    3. Exponentiate ``H`` into the Qiskit unitary ``U(dt)`` and wrap it in a circuit.
    4. Apply ``U(dt)`` step by step with ``Statevector.evolve``, recording the density and the
       donor/acceptor populations at every step.

    Parameters
    ----------
    barrier_height_ev:
        Barrier height ``E_b`` in eV (see :func:`src.hamiltonian.double_well_potential`).
    asymmetry_ev:
        Asymmetry drive ``lambda`` in eV; positive biases the transfer downhill.
    duration_fs:
        Total simulated time in femtoseconds. 50 fs covers the sub-picosecond window in which
        the coherent proton dynamics of an enzymatic transfer step live.
    num_steps:
        Number of propagation steps; the returned arrays have ``num_steps + 1`` samples.
    well_separation_ang:
        Half-distance ``a`` between the wells, in Angstroms.
    width_ang:
        Initial wavepacket width ``sigma``, in Angstroms.
    mass:
        Tunneling particle mass in kg (``M_H`` for a proton, ``2 * M_H`` for a deuteron).
    compute_splitting:
        When ``True``, also diagonalise the lowest few eigenstates to report the tunneling
        splitting ``E1 - E0``. Cheap for a 256x256 matrix, and it is the number that explains
        the oscillation period you see in the population plot.

    Returns
    -------
    TunnelingResult
        Time series, densities, statevectors and diagnostics for the run.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")

    grid = build_grid()
    hamiltonian, potential = build_hamiltonian(
        grid,
        barrier_height_ev=barrier_height_ev,
        asymmetry_ev=asymmetry_ev,
        well_separation_ang=well_separation_ang,
        mass=mass,
    )

    psi0 = gaussian_wavepacket(
        grid,
        center_ang=-well_separation_ang,
        width_ang=width_ang,
        mass=mass,
    )

    total_time_s = duration_fs * FEMTOSECOND
    dt_s = total_time_s / num_steps
    _, circuit = build_propagator(hamiltonian, dt_s)

    donor_mask = grid.donor_mask
    acceptor_mask = grid.acceptor_mask

    statevectors: List[Statevector] = [psi0]
    densities: List[np.ndarray] = [np.abs(psi0.data) ** 2]
    state = psi0
    for _ in range(num_steps):
        state = state.evolve(circuit)
        statevectors.append(state)
        densities.append(np.abs(state.data) ** 2)

    density = np.vstack(densities)
    times_fs = np.linspace(0.0, duration_fs, num_steps + 1)
    donor_population = density[:, donor_mask].sum(axis=1)
    acceptor_population = density[:, acceptor_mask].sum(axis=1)

    energy_j = float(
        np.real(np.conjugate(psi0.data) @ (hamiltonian @ psi0.data))
    )
    norm_fidelity = float(np.abs(np.vdot(state.data, state.data)))

    splitting_ev: Optional[float] = None
    if compute_splitting:
        try:
            from scipy.sparse.linalg import eigsh

            eigenvalues = np.sort(
                eigsh(hamiltonian, k=2, which="SA", return_eigenvectors=False)
            )
            splitting_ev = float(joules_to_ev(eigenvalues[1] - eigenvalues[0]))
        except Exception:  # pragma: no cover - ARPACK can fail to converge
            splitting_ev = None

    return TunnelingResult(
        grid=grid,
        potential_j=potential,
        times_fs=times_fs,
        density=density,
        donor_population=donor_population,
        acceptor_population=acceptor_population,
        statevectors=statevectors,
        circuit=circuit,
        dt_s=dt_s,
        norm_fidelity=norm_fidelity,
        energy_ev=float(joules_to_ev(energy_j)),
        tunneling_splitting_ev=splitting_ev,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    result = simulate_tunneling()
    print(f"qubits: {result.num_qubits}, dim: {result.final_statevector.dim}")
    print(f"<H> = {result.energy_ev:.4f} eV, norm = {result.norm_fidelity:.12f}")
    if result.tunneling_splitting_ev is not None:
        print(f"tunneling splitting = {result.tunneling_splitting_ev * 1e3:.3f} meV")
    print(f"donor    P(0) = {result.donor_population[0]:.4f} -> {result.donor_population[-1]:.4f}")
    print(
        f"acceptor P(0) = {result.acceptor_population[0]:.4f} -> "
        f"{result.acceptor_population[-1]:.4f} (peak {result.max_transfer:.4f})"
    )
