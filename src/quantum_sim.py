"""Grid evolution of the localized tunneling doublet.

The initial state is the standard two-level packet

    ψ(0) = (φ0 + σ φ1) / √2,

where φ0 and φ1 are the lowest two eigenstates of H and the relative sign σ
is chosen so that most of the density sits in the donor well (x < 0). In a
symmetric double well this state oscillates between the wells with period
T_tun = h / (E1 − E0).

``evolve_exact`` propagates that state with ``scipy.linalg.expm``. That is a
classical numerical simulation of the Schrödinger equation on the grid, used
as ground truth.

``split_operator_circuit`` is the standard quantum algorithm for the same
equation: a Trotter product of diagonal potential phases and a kinetic phase
in the Fourier basis. The quantum Fourier transform supplies that basis. The
local Trotter error of one symmetric step is O(dt³), so over a fixed time
the error shrinks as n_steps grows (globally O(1/n_steps²)).

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
    QUBITS,
    ProtonHamiltonian,
    build_hamiltonian,
    lowest_eigenstates,
    particle_mass,
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
        states: State at every sample when ``evolve_exact(..., keep_states=True)``.
            ``states[k]`` is ψ(k Δt). ``None`` when the states were not stored.
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
    states: np.ndarray | None = None


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
    keep_states: bool = False,
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
        keep_states: When True, store ψ at every sample on ``states``.

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
    states = (
        np.empty((n_steps + 1, psi0.size), dtype=np.complex128) if keep_states else None
    )
    donor_mask = model.x_m < 0.0

    for frame in range(n_steps + 1):
        if states is not None:
            states[frame] = psi
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
        states=states,
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


def _strang_diagonals(
    potential, mass: float | str, dx: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Position half-step and kinetic full-step phases, both on the grid.

    The kinetic phases use the wave numbers that Qiskit's QFT diagonalizes.
    On an N-point grid the QFT is

        QFT|j⟩ = N^(−1/2) Σ_k exp(+2πi j k / N) |k⟩,

    which is ``sqrt(N) * numpy.fft.ifft`` in little-endian index order. The
    matching wave numbers are ``k = 2π * fftfreq(N, dx)`` in rad/m, including
    the negative aliases. The kinetic symbol is then exp(−i ℏ k² dt / 2m).
    """

    samples = np.asarray(potential, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("Potential must be a one-dimensional grid of length at least 2.")
    spacing = float(dx)
    step = float(dt)
    if spacing <= 0:
        raise ValueError("dx must be positive.")
    if step <= 0:
        raise ValueError("dt must be positive.")
    mass_kg = particle_mass(mass)
    half = np.exp(-1j * samples * (step / (2.0 * HBAR_J_S)))
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(samples.size, d=spacing)
    kinetic = np.exp(-1j * HBAR_J_S * (wave_numbers**2) * step / (2.0 * mass_kg))
    # Diagonal gates require entries on the unit circle.
    half = half / np.abs(half)
    kinetic = kinetic / np.abs(kinetic)
    return half, kinetic


def split_operator_numpy(
    psi: np.ndarray,
    potential,
    mass: float | str,
    dx: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """One symmetric Trotter product, applied with NumPy FFTs.

    Each step is the Strang splitting of exp(−i(T+V)dt/ℏ):

        exp(−i V dt / 2ℏ) · F† · exp(−i ℏ k² dt / 2m) · F · exp(−i V dt / 2ℏ),

    where F is the QFT. In NumPy that Fourier transform is
    ``sqrt(N) * ifft`` and its inverse is ``fft / sqrt(N)``, which is the
    same convention as ``qiskit.circuit.library.QFTGate`` in little-endian
    order. This is the array-level reference for ``split_operator_circuit``.

    The splitting is exact only as dt → 0. Over a fixed interval the error
    falls as n_steps grows.
    """

    if int(n_steps) < 1:
        raise ValueError("At least one Trotter step is required.")
    state = np.array(psi, dtype=np.complex128, copy=True)
    half, kinetic = _strang_diagonals(potential, mass, dx, dt)
    if state.shape != half.shape:
        raise ValueError("State and potential must share the grid length.")
    scale = np.sqrt(state.size)
    for _ in range(int(n_steps)):
        state *= half
        state = np.fft.ifft(state) * scale
        state *= kinetic
        state = np.fft.fft(state) / scale
        state *= half
    return state


def _append_diagonal(circuit, phases: np.ndarray, qubits) -> None:
    """Append a diagonal phase gate, using whichever class this Qiskit ships.

    Qiskit 2.1 deprecated the ``Diagonal`` circuit in favour of
    ``DiagonalGate``. Both apply ``diag[i]`` to the little-endian basis
    state ``|i⟩``.
    """

    import qiskit.circuit.library as library

    if hasattr(library, "DiagonalGate"):
        circuit.append(library.DiagonalGate(phases), qubits)
        return
    if hasattr(library, "Diagonal"):
        circuit.compose(library.Diagonal(phases), qubits, inplace=True)
        return
    raise ImportError("This Qiskit install has neither DiagonalGate nor Diagonal.")


def _append_qft(circuit, n_qubits: int, qubits, inverse: bool) -> None:
    """Append the library QFT, or its inverse, in little-endian order.

    ``QFTGate`` is the current class. The older ``QFT`` circuit implements
    the same transform and is used only when the gate class is absent.
    """

    import qiskit.circuit.library as library

    if hasattr(library, "QFTGate"):
        gate = library.QFTGate(n_qubits)
        circuit.append(gate.inverse() if inverse else gate, qubits)
        return
    qft = library.QFT(n_qubits, do_swaps=True)
    block = qft.inverse() if inverse else qft
    circuit.compose(block, qubits, inplace=True)


def split_operator_circuit(potential, mass: float | str, dx: float, dt: float, n_steps: int):
    """Trotter circuit for the Schrödinger equation on the 8-qubit grid.

    This is the standard quantum algorithm for a Hamiltonian H = T + V on a
    uniform grid. One symmetric Trotter step is

        Diagonal(exp(−i V dt / 2ℏ))
        → QFT
        → Diagonal(exp(−i ℏ k² dt / 2m))
        → QFT†
        → Diagonal(exp(−i V dt / 2ℏ)).

    The kinetic diagonal is written in the momentum ordering of Qiskit's QFT
    (the same order as ``numpy.fft.fftfreq``). Qubits are little-endian, so
    basis index i is grid site i.

    Repeating the step ``n_steps`` times approximates exp(−i H t / ℏ) with
    t = n_steps · dt. The local error of the symmetric product is O(dt³),
    and the error on a fixed interval shrinks as n_steps grows, globally as
    O(1/n_steps²).

    Args:
        potential: V(x) samples in joules, length 256.
        mass: ``"H"``, ``"D"``, ``"T"``, or a mass in kilograms.
        dx: Grid spacing in metres.
        dt: Duration of one Trotter step, in seconds.
        n_steps: How many times to repeat the step.

    Returns:
        A ``QuantumCircuit`` on 8 qubits. It does not include state preparation.
    """

    from qiskit import QuantumCircuit

    if int(n_steps) < 1:
        raise ValueError("At least one Trotter step is required.")
    half, kinetic = _strang_diagonals(potential, mass, dx, dt)
    if half.size != 2**QUBITS:
        raise ValueError(
            f"split_operator_circuit expects the {QUBITS}-qubit grid "
            f"({2**QUBITS} samples), got {half.size}."
        )
    qubits = range(QUBITS)
    step = QuantumCircuit(QUBITS, name="trotter_step")
    _append_diagonal(step, half, qubits)
    _append_qft(step, QUBITS, qubits, inverse=False)
    _append_diagonal(step, kinetic, qubits)
    _append_qft(step, QUBITS, qubits, inverse=True)
    _append_diagonal(step, half, qubits)
    if int(n_steps) == 1:
        return step
    circuit = QuantumCircuit(QUBITS, name="split_operator")
    for _ in range(int(n_steps)):
        circuit.compose(step, inplace=True)
    return circuit


def evolve_split_operator(
    psi0: np.ndarray,
    potential,
    mass: float | str,
    dx: float,
    dt: float,
    n_steps: int,
    sample_at: list[int] | tuple[int, ...] | None = None,
) -> dict[int, np.ndarray]:
    """Prepare ψ(0) and apply the split-operator circuit.

    The initial amplitudes are loaded with ``StatePreparation``. Evolution
    uses ``Statevector.from_instruction`` for that preparation and
    ``Statevector.evolve`` for each Trotter step. Returned arrays are copies
    of the statevector data at the requested step counts (0 is the prepared
    state, before any Trotter step).
    """

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import Statevector

    steps = int(n_steps)
    if steps < 1:
        raise ValueError("At least one Trotter step is required.")
    wanted = {steps} if sample_at is None else {int(frame) for frame in sample_at}
    if any(frame < 0 or frame > steps for frame in wanted):
        raise ValueError("sample_at must lie between 0 and n_steps.")

    amplitudes = np.asarray(psi0, dtype=np.complex128)
    prepared = QuantumCircuit(QUBITS)
    prepared.append(StatePreparation(Statevector(amplitudes)), range(QUBITS))
    state = Statevector.from_instruction(prepared)
    samples: dict[int, np.ndarray] = {}
    if 0 in wanted:
        samples[0] = np.array(state.data, dtype=np.complex128, copy=True)

    step = split_operator_circuit(potential, mass, dx, dt, n_steps=1)
    for frame in range(1, steps + 1):
        state = state.evolve(step)
        if frame in wanted:
            samples[frame] = np.array(state.data, dtype=np.complex128, copy=True)
    return samples
