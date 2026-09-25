"""Spatial grid and the proton Hamiltonian in SI units.

The reaction coordinate x is the proton's position between a donor well
(x < 0, zinc-bound water) and an acceptor well (x > 0). An 8-qubit register
holds the wavefunction on 256 equally spaced sites,

    x ∈ [-1.5 Å, +1.5 Å],    2^8 = 256.

The potential is an asymmetric double well

    V(x) = E_b (1 − (x/a)^2)^2 + λ x.

E_b is the barrier at x = 0. The well half-separation a is a parameter
(default 0.35 Å), not a number taken from the PDB file. λ is an energy per
length: the value passed in as electron-volts is λ in eV/Å, so λx is an
energy when x is in angstroms. Positive λ lowers the donor side (x < 0)
and raises the acceptor side.

The same grid supports three isotope masses. H, D, and T change only the
kinetic term. They do not change V(x).

The kinetic energy is the second-order finite-difference form of
−(ħ²/2m) d²/dx² with Dirichlet walls outside the grid. H = T + V is stored
as a sparse matrix. Callers densify it only to build the short-time unitary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

# CODATA-scale constants fixed by the hackathon brief.
HBAR_J_S = 1.0545718e-34
PLANCK_H_J_S = 6.62607015e-34
PROTON_MASS_KG = 1.6735575e-27
DEUTERON_MASS_KG = 3.3443e-27
TRITON_MASS_KG = 5.0082e-27
JOULE_PER_EV = 1.60218e-19

# Isotope presets. Values are kilograms. "H" is the proton mass above.
MASS_PRESETS = {
    "H": PROTON_MASS_KG,
    "D": DEUTERON_MASS_KG,
    "T": TRITON_MASS_KG,
}

ANGSTROM_M = 1.0e-10
GRID_POINTS = 256
QUBITS = 8
X_MIN_A = -1.5
X_MAX_A = 1.5

# Minima of (1 − (x/a)^2)^2 sit at x = ±a. This default is a model choice.
DEFAULT_WELL_HALF_ANGSTROM = 0.35
WELL_HALF_WIDTH_M = DEFAULT_WELL_HALF_ANGSTROM * ANGSTROM_M


@dataclass(frozen=True)
class ProtonHamiltonian:
    """Sparse H = T + V on the 256-point grid, with the arrays plots need.

    Attributes:
        x_m: Grid coordinates in metres, shape (256,).
        v_joule: Diagonal potential samples in joules, shape (256,).
        matrix: Sparse Hamiltonian in joules, shape (256, 256).
        barrier_joule: Barrier height E_b.
        asymmetry_joule: Slope λ converted as eV/Å → J/Å. The potential term is λx.
        well_half_width_m: Well-minimum location a, in metres.
        spacing_m: Grid spacing dx.
        mass_kg: Particle mass used in the kinetic term.
    """

    x_m: np.ndarray
    v_joule: np.ndarray
    matrix: sparse.csr_matrix
    barrier_joule: float
    asymmetry_joule: float
    well_half_width_m: float
    spacing_m: float
    mass_kg: float


def spatial_grid() -> tuple[np.ndarray, float]:
    """Return the 256-point coordinate array and its spacing, both in metres.

    Endpoints are exactly ±1.5 Å, so the eight qubits label a closed interval
    rather than a periodic ring.
    """

    if GRID_POINTS != 2**QUBITS:
        raise ValueError("The spatial grid must be exactly 2^8 = 256 sites.")
    x_angstrom = np.linspace(X_MIN_A, X_MAX_A, GRID_POINTS, dtype=np.float64)
    x_m = x_angstrom * ANGSTROM_M
    spacing_m = float(x_m[1] - x_m[0])
    return x_m, spacing_m


def double_well_potential(
    x_m: np.ndarray,
    barrier_ev: float,
    asymmetry_ev: float,
    well_half_width_m: float = WELL_HALF_WIDTH_M,
) -> np.ndarray:
    """Evaluate V(x) in joules.

    Args:
        x_m: Coordinates in metres.
        barrier_ev: Barrier height E_b in electron-volts.
        asymmetry_ev: Slope λ in eV/Å. The term added to V is λx, with x in
            angstroms, then converted to joules. This is not λ(x/a).
        well_half_width_m: Distance a from the barrier to each minimum.

    Returns:
        Potential samples in joules, same shape as ``x_m``.
    """

    if well_half_width_m <= 0:
        raise ValueError("Well half-width must be positive.")
    barrier_j = float(barrier_ev) * JOULE_PER_EV
    # λ in J/m, so that λx with x in metres is an energy.
    # asymmetry_ev is λ expressed in eV per angstrom.
    slope_per_m = float(asymmetry_ev) * JOULE_PER_EV / ANGSTROM_M
    xi = np.asarray(x_m, dtype=np.float64) / well_half_width_m
    shape = (1.0 - xi**2) ** 2
    return barrier_j * shape + slope_per_m * np.asarray(x_m, dtype=np.float64)


def particle_mass(mass: float | str = "H") -> float:
    """Resolve an isotope preset or a mass in kilograms.

    Args:
        mass: ``"H"``, ``"D"``, ``"T"``, or a positive mass in kilograms.
    """

    if isinstance(mass, str):
        key = mass.strip().upper()
        if key not in MASS_PRESETS:
            known = ", ".join(MASS_PRESETS)
            raise ValueError(f"Unknown isotope {mass!r}. Presets are {known}.")
        return MASS_PRESETS[key]
    value = float(mass)
    if value <= 0:
        raise ValueError("Mass must be positive.")
    return value


def kinetic_operator(
    spacing_m: float,
    n: int = GRID_POINTS,
    mass: float | str = "H",
) -> sparse.csr_matrix:
    """Second-order finite-difference kinetic energy, in joules.

    T = −(ħ² / 2m) d²/dx² becomes the tridiagonal stencil

        T_ii     =  ħ² / (m dx²)
        T_i,i±1  = −ħ² / (2 m dx²)

    with no corner wrap-around, i.e. ψ = 0 outside the grid.
    """

    if spacing_m <= 0:
        raise ValueError("Grid spacing must be positive.")
    mass_kg = particle_mass(mass)
    hbar2 = HBAR_J_S**2
    diag = hbar2 / (mass_kg * spacing_m**2)
    off = -hbar2 / (2.0 * mass_kg * spacing_m**2)
    return sparse.diags(
        [np.full(n - 1, off), np.full(n, diag), np.full(n - 1, off)],
        offsets=(-1, 0, 1),
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )


def build_hamiltonian(
    barrier_ev: float,
    asymmetry_ev: float,
    well_half_width_angstrom: float = DEFAULT_WELL_HALF_ANGSTROM,
    mass: float | str = "H",
) -> ProtonHamiltonian:
    """Assemble the sparse Hamiltonian H = T + V.

    Args:
        barrier_ev: Barrier height in electron-volts, typically 0.1–1.0.
        asymmetry_ev: Slope λ in eV/Å. Zero is the symmetric double well.
        well_half_width_angstrom: Well half-separation a, in angstroms.
        mass: ``"H"``, ``"D"``, ``"T"``, or a mass in kilograms.

    Returns:
        Grid, potential samples, and sparse H in SI units.
    """

    if well_half_width_angstrom <= 0:
        raise ValueError("Well half-separation a must be positive.")
    mass_kg = particle_mass(mass)
    half_width_m = float(well_half_width_angstrom) * ANGSTROM_M
    x_m, spacing_m = spatial_grid()
    potential = double_well_potential(
        x_m, barrier_ev, asymmetry_ev, well_half_width_m=half_width_m
    )
    kinetic = kinetic_operator(spacing_m, n=x_m.size, mass=mass_kg)
    potential_matrix = sparse.diags(potential, offsets=0, format="csr", dtype=np.float64)
    hamiltonian = (kinetic + potential_matrix).tocsr()
    return ProtonHamiltonian(
        x_m=x_m,
        v_joule=potential,
        matrix=hamiltonian,
        barrier_joule=float(barrier_ev) * JOULE_PER_EV,
        asymmetry_joule=float(asymmetry_ev) * JOULE_PER_EV,
        well_half_width_m=half_width_m,
        spacing_m=spacing_m,
        mass_kg=mass_kg,
    )


def lowest_eigenstates(
    hamiltonian: sparse.spmatrix | np.ndarray, k: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest k eigenpairs of H from a dense symmetric diagonalisation.

    256×256 is small enough that ``numpy.linalg.eigh`` is the whole solver.
    A sparse matrix is densified here. Eigenvalues are sorted ascending and
    are in joules when H is in joules. Each column of the vector array is
    one normalised eigenvector.

    Args:
        hamiltonian: Sparse or dense H, shape (256, 256).
        k: How many lowest states to return. Clipped to the matrix size.

    Returns:
        ``(energies_j, vectors)`` with shapes ``(k,)`` and ``(n, k)``.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")
    dense = hamiltonian.toarray() if sparse.issparse(hamiltonian) else np.asarray(hamiltonian)
    energies, vectors = np.linalg.eigh(np.asarray(dense, dtype=np.float64))
    count = min(int(k), energies.size)
    return energies[:count].copy(), np.real(vectors[:, :count]).copy()


def tunneling_splitting(energies_j: np.ndarray) -> tuple[float, float]:
    """Gap and tunneling period of the lowest doublet.

    ΔE = E1 − E0. The population of a state built from this doublet returns
    after one period

        T_tun = h / ΔE,

    with h = 6.62607015×10⁻³⁴ J·s (Planck's constant, not ℏ). This is the
    two-level result. It is the oscillation period of (φ0 ± φ1)/√2, not a
    rate measured on the enzyme.

    Args:
        energies_j: Eigenvalues in joules, lowest first. At least two.

    Returns:
        ``(delta_e_joule, tunneling_period_s)``.
    """

    if len(energies_j) < 2:
        raise ValueError("Need at least the lowest two eigenvalues.")
    delta = float(energies_j[1] - energies_j[0])
    if delta <= 0:
        raise ValueError("The lowest two eigenvalues are not split.")
    return delta, PLANCK_H_J_S / delta
