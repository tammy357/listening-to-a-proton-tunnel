"""Physical constants, spatial grid, double-well potential and the Hamiltonian ``H = T + V``.

Physics
-------
We reduce the proton-transfer coordinate of carbonic anhydrase II to a single effective 1-D
degree of freedom ``x``: the position of the transferring hydrogen along the
donor(Zn-OH2) -> acceptor(ordered water / His64) axis measured from the midpoint of the two
heavy atoms.

Along that coordinate the proton feels a **double-well** potential — one minimum next to the
donor heavy atom, one next to the acceptor, separated by a barrier created by the partially
broken/partially formed O-H bonds::

    V(x) = E_b * (1 - (x/a)^2)^2  +  lambda * (x/a)
           \\_________________/      \\____________/
             symmetric barrier         asymmetry drive

* ``E_b`` (eV) is the barrier height at ``x = 0``; the two minima sit at ``x = +/- a``.
* ``lambda`` (eV) tilts the landscape, lowering the acceptor well by ``lambda`` relative to
  the donor well (``+lambda`` at ``x = -a``, ``-lambda`` at ``x = +a``). Physically this is the
  thermodynamic driving force ``-dG`` of the transfer, modulated in a real enzyme by
  electrostatics and heavy-atom "gating" motions. A symmetric well (``lambda = 0``) gives
  coherent back-and-forth Rabi-like tunneling oscillations; a tilted well biases the proton
  towards the acceptor.

The quantum Hamiltonian is::

    H = T + V,     T = -(hbar^2 / 2 m_H) d^2/dx^2

with the Laplacian discretised by the standard 2nd-order central finite difference on a
uniform grid::

    d^2 psi / dx^2  ~  (psi_{k-1} - 2 psi_k + psi_{k+1}) / dx^2

so ``T`` is a symmetric tridiagonal matrix with ``-hbar^2/(2 m dx^2)`` off the diagonal and
``+hbar^2/(m dx^2)`` on it, and ``V`` is diagonal. Dirichlet ("hard wall") boundaries are
implied by truncating the stencil at the grid edges — fine here because the wavefunction is
bound and the walls of the quartic sit far outside the classically allowed region.

Discretisation
--------------
The grid has **exactly 256 points**, which is ``2**8``: one basis state per computational
basis state of an **8-qubit register**. Grid index ``k`` <-> qubit bitstring ``k`` <-> position
``x[k]``. That mapping is what makes the whole simulation natively "quantum-shaped" and is
also what the QPAM sonification consumes.

Units
-----
Everything returned by this module is **SI** (metres, joules, kg, seconds). Function
arguments carrying convenience units are suffixed ``_ang`` (Angstrom) or ``_ev``
(electron-volt) and converted at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.sparse import dia_matrix, diags, spmatrix

# --------------------------------------------------------------------------------------
# Physical constants (SI)
# --------------------------------------------------------------------------------------
HBAR: float = 1.0545718e-34  #: Reduced Planck constant, J*s
M_H: float = 1.6735575e-27  #: Hydrogen (proton) mass, kg
EV: float = 1.60218e-19  #: 1 electron-volt, in J
ANGSTROM: float = 1.0e-10  #: 1 Angstrom, in m
FEMTOSECOND: float = 1.0e-15  #: 1 femtosecond, in s

# --------------------------------------------------------------------------------------
# Quantum register / grid geometry
# --------------------------------------------------------------------------------------
NUM_QUBITS: int = 8  #: 8 qubits ...
NUM_GRID_POINTS: int = 2**NUM_QUBITS  #: ... = 256 spatial grid points.
X_MIN_ANG: float = -1.5  #: Left edge of the proton coordinate, Angstrom.
X_MAX_ANG: float = +1.5  #: Right edge of the proton coordinate, Angstrom.
DEFAULT_WELL_SEPARATION_ANG: float = 0.6
"""Half-separation ``a`` of the two wells (Angstrom).

The minima sit at ``x = +/- a``, i.e. ~1.2 A apart, the canonical proton-transfer geometry for a
short, strong hydrogen bond between a donor and acceptor ~2.6-2.8 A apart (which is exactly the
Zn-water / relay-water arrangement in the CA II active site).
"""


@dataclass(frozen=True)
class SpatialGrid:
    """The 256-point position grid backing the 8-qubit register.

    Attributes
    ----------
    x:
        Grid positions in **metres** (SI), shape ``(256,)``.
    dx:
        Uniform spacing in metres.
    num_qubits:
        ``8`` — ``len(x) == 2**num_qubits`` is guaranteed by construction.
    """

    x: np.ndarray
    dx: float
    num_qubits: int = NUM_QUBITS

    @property
    def size(self) -> int:
        """Number of grid points (== dimension of the statevector == 256)."""
        return int(self.x.size)

    @property
    def x_angstrom(self) -> np.ndarray:
        """Grid positions converted to Angstroms, for plotting only."""
        return self.x / ANGSTROM

    @property
    def donor_mask(self) -> np.ndarray:
        """Boolean mask selecting the donor side of the barrier (``x < 0``)."""
        return self.x < 0.0

    @property
    def acceptor_mask(self) -> np.ndarray:
        """Boolean mask selecting the acceptor side of the barrier (``x > 0``)."""
        return self.x > 0.0


def build_grid(
    x_min_ang: float = X_MIN_ANG,
    x_max_ang: float = X_MAX_ANG,
    num_points: int = NUM_GRID_POINTS,
) -> SpatialGrid:
    """Build the uniform 256-point spatial grid mapped onto 8 qubits.

    Parameters
    ----------
    x_min_ang, x_max_ang:
        Domain edges in Angstroms (inclusive). Defaults span +/-1.5 A, comfortably wider
        than the +/-0.6 A well minima so the bound wavefunction never touches the walls.
    num_points:
        Must be a power of two — the grid *is* the computational basis of the register.

    Raises
    ------
    ValueError
        If ``num_points`` is not a power of two.
    """
    if num_points < 2 or (num_points & (num_points - 1)) != 0:
        raise ValueError(
            f"num_points must be a power of two so it maps onto a qubit register; got {num_points}"
        )
    x = np.linspace(x_min_ang * ANGSTROM, x_max_ang * ANGSTROM, num_points)
    dx = float(x[1] - x[0])
    return SpatialGrid(x=x, dx=dx, num_qubits=int(np.log2(num_points)))


def double_well_potential(
    grid: SpatialGrid,
    barrier_height_ev: float = 0.15,
    asymmetry_ev: float = 0.10,
    well_separation_ang: float = DEFAULT_WELL_SEPARATION_ANG,
) -> np.ndarray:
    """Asymmetric quartic double-well potential ``V(x)``, returned in **joules**.

    ``V(x) = E_b * (1 - (x/a)^2)^2 + lambda * (x/a)``

    Parameters
    ----------
    grid:
        The 256-point spatial grid (positions in metres).
    barrier_height_ev:
        ``E_b``, the height of the central barrier above the *symmetric* well minima, in eV.
        Realistic enzymatic proton-transfer barriers are a few tenths of an eV; a lower
        barrier means a thinner classically forbidden region and exponentially faster
        tunneling.
    asymmetry_ev:
        ``lambda``, the energy bias in eV applied at the well minima. Positive values raise
        the donor well (``x = -a``) by ``lambda`` and lower the acceptor well (``x = +a``) by
        ``lambda``, i.e. a downhill (exergonic) proton transfer of ``2*lambda``.
    well_separation_ang:
        ``a``, the half-distance between the two minima in Angstroms.

    Returns
    -------
    numpy.ndarray
        ``V`` sampled on the grid, shape ``(256,)``, in joules.
    """
    a = well_separation_ang * ANGSTROM
    reduced = grid.x / a  # dimensionless x/a
    symmetric = barrier_height_ev * EV * (1.0 - reduced**2) ** 2
    tilt = -asymmetry_ev * EV * reduced  # lower on the acceptor side (x > 0)
    return symmetric + tilt


def kinetic_energy_matrix(grid: SpatialGrid, mass: float = M_H) -> spmatrix:
    """2nd-order central finite-difference kinetic energy operator ``T``, in joules.

    ``T = -(hbar^2 / 2 m) d^2/dx^2`` becomes the symmetric tridiagonal matrix::

        T[k, k]   = + hbar^2 / (m * dx^2)
        T[k, k+1] = T[k+1, k] = - hbar^2 / (2 m * dx^2)

    Parameters
    ----------
    grid:
        Spatial grid providing ``dx`` in metres.
    mass:
        Particle mass in kg. Defaults to the proton mass; pass ``2 * M_H`` to model a
        deuteron and watch the tunneling rate collapse — that ratio *is* the kinetic isotope
        effect that flags tunneling experimentally.

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse ``(256, 256)`` operator in joules.
    """
    coeff = HBAR**2 / (2.0 * mass * grid.dx**2)
    main = 2.0 * coeff * np.ones(grid.size)
    off = -coeff * np.ones(grid.size - 1)
    return diags([off, main, off], offsets=[-1, 0, 1], format="csr")


def build_hamiltonian(
    grid: SpatialGrid,
    barrier_height_ev: float = 0.15,
    asymmetry_ev: float = 0.10,
    well_separation_ang: float = DEFAULT_WELL_SEPARATION_ANG,
    mass: float = M_H,
) -> Tuple[spmatrix, np.ndarray]:
    """Assemble the sparse Hamiltonian ``H = T + V`` for the proton coordinate.

    Returns
    -------
    (H, V)
        ``H`` is a sparse ``(256, 256)`` Hermitian matrix in joules; ``V`` is the diagonal
        potential (shape ``(256,)``, joules) returned alongside because the UI plots it.
    """
    potential = double_well_potential(
        grid,
        barrier_height_ev=barrier_height_ev,
        asymmetry_ev=asymmetry_ev,
        well_separation_ang=well_separation_ang,
    )
    kinetic = kinetic_energy_matrix(grid, mass=mass)
    v_diag: dia_matrix = diags(potential, offsets=0, format="csr")
    return (kinetic + v_diag).tocsr(), potential


def joules_to_ev(energy_j: np.ndarray | float) -> np.ndarray | float:
    """Convert an SI energy (J) to electron-volts, for display only."""
    return energy_j / EV


def ev_to_joules(energy_ev: np.ndarray | float) -> np.ndarray | float:
    """Convert electron-volts to SI joules."""
    return energy_ev * EV


def femtoseconds_to_seconds(time_fs: np.ndarray | float) -> np.ndarray | float:
    """Convert femtoseconds to SI seconds."""
    return time_fs * FEMTOSECOND


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from scipy.sparse.linalg import eigsh

    g = build_grid()
    H, V = build_hamiltonian(g, barrier_height_ev=0.15, asymmetry_ev=0.10)
    energies = np.sort(eigsh(H, k=6, which="SA", return_eigenvectors=False))
    print(f"grid: {g.size} points, dx = {g.dx / ANGSTROM:.4f} A ({g.num_qubits} qubits)")
    print(f"barrier at x=0: {joules_to_ev(V[g.size // 2]):.3f} eV, well floor: {joules_to_ev(V.min()):.3f} eV")
    print("lowest eigenvalues (eV):", np.round(joules_to_ev(energies), 4))
    print(f"tunneling splitting E1-E0 = {joules_to_ev(energies[1] - energies[0]) * 1e3:.3f} meV")
