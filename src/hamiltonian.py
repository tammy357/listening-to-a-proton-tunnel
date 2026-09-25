"""Spatial grid and the proton Hamiltonian in SI units.

The reaction coordinate x is the proton's position between a donor well
(x < 0, zinc-bound water) and an acceptor well (x > 0). An 8-qubit register
holds the wavefunction on 256 equally spaced sites,

    x ∈ [-1.5 Å, +1.5 Å],    2^8 = 256.

The potential is an asymmetric double well

    V(x) = E_b (1 − (x/a)^2)^2 + λ (x/a).

E_b is the barrier at x = 0. The second term is the brief's λx written in
the dimensionless coordinate ξ = x/a, so λ stays an energy and matches the
electron-volt slider. Positive λ lowers the donor side and raises the
acceptor side, which biases the proton back toward the zinc.

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
PROTON_MASS_KG = 1.6735575e-27
JOULE_PER_EV = 1.60218e-19

ANGSTROM_M = 1.0e-10
GRID_POINTS = 256
QUBITS = 8
X_MIN_A = -1.5
X_MAX_A = 1.5

# Minima of (1 − (x/a)^2)^2 sit at x = ±a. 0.32 Å keeps both wells inside
# the 3 Å window and makes the barrier thin enough that a real proton mass
# moves visible amplitude into the acceptor well inside 50 fs. Wider wells
# push the tunneling period far past the slider.
WELL_HALF_WIDTH_M = 0.32 * ANGSTROM_M


@dataclass(frozen=True)
class ProtonHamiltonian:
    """Sparse H = T + V on the 256-point grid, with the arrays plots need.

    Attributes:
        x_m: Grid coordinates in metres, shape (256,).
        v_joule: Diagonal potential samples in joules, shape (256,).
        matrix: Sparse Hamiltonian in joules, shape (256, 256).
        barrier_joule: Barrier height E_b.
        asymmetry_joule: Asymmetry energy λ.
        well_half_width_m: Well-minimum location a, in metres.
        spacing_m: Grid spacing dx.
    """

    x_m: np.ndarray
    v_joule: np.ndarray
    matrix: sparse.csr_matrix
    barrier_joule: float
    asymmetry_joule: float
    well_half_width_m: float
    spacing_m: float


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
        asymmetry_ev: Asymmetry λ in electron-volts. The term added to V is
            λ (x/a), the energy-dimensioned form of λx.
        well_half_width_m: Distance a from the barrier to each minimum.

    Returns:
        Potential samples in joules, same shape as ``x_m``.
    """

    if well_half_width_m <= 0:
        raise ValueError("Well half-width must be positive.")
    barrier_j = float(barrier_ev) * JOULE_PER_EV
    asymmetry_j = float(asymmetry_ev) * JOULE_PER_EV
    xi = np.asarray(x_m, dtype=np.float64) / well_half_width_m
    shape = (1.0 - xi**2) ** 2
    return barrier_j * shape + asymmetry_j * xi


def kinetic_operator(spacing_m: float, n: int = GRID_POINTS) -> sparse.csr_matrix:
    """Second-order finite-difference kinetic energy, in joules.

    T = −(ħ² / 2m) d²/dx² becomes the tridiagonal stencil

        T_ii     =  ħ² / (m dx²)
        T_i,i±1  = −ħ² / (2 m dx²)

    with no corner wrap-around, i.e. ψ = 0 outside the grid.
    """

    if spacing_m <= 0:
        raise ValueError("Grid spacing must be positive.")
    hbar2 = HBAR_J_S**2
    diag = hbar2 / (PROTON_MASS_KG * spacing_m**2)
    off = -hbar2 / (2.0 * PROTON_MASS_KG * spacing_m**2)
    return sparse.diags(
        [np.full(n - 1, off), np.full(n, diag), np.full(n - 1, off)],
        offsets=(-1, 0, 1),
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )


def build_hamiltonian(barrier_ev: float, asymmetry_ev: float) -> ProtonHamiltonian:
    """Assemble the sparse proton Hamiltonian H = T + V.

    Args:
        barrier_ev: Barrier height in electron-volts, typically 0.1–1.0.
        asymmetry_ev: Asymmetry drive in electron-volts, typically 0–0.2.

    Returns:
        Grid, potential samples, and sparse H in SI units.
    """

    x_m, spacing_m = spatial_grid()
    potential = double_well_potential(x_m, barrier_ev, asymmetry_ev)
    kinetic = kinetic_operator(spacing_m, n=x_m.size)
    potential_matrix = sparse.diags(potential, offsets=0, format="csr", dtype=np.float64)
    hamiltonian = (kinetic + potential_matrix).tocsr()
    return ProtonHamiltonian(
        x_m=x_m,
        v_joule=potential,
        matrix=hamiltonian,
        barrier_joule=float(barrier_ev) * JOULE_PER_EV,
        asymmetry_joule=float(asymmetry_ev) * JOULE_PER_EV,
        well_half_width_m=WELL_HALF_WIDTH_M,
        spacing_m=spacing_m,
    )
