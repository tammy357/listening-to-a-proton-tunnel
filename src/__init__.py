"""Quantum proton tunneling & Moth Quantum sonification engine (Moth Hacks 2026).

Modules
-------
``pdb_parser``
    Pulls the Human Carbonic Anhydrase II structure (RCSB PDB ``1HED``) and measures the
    catalytic Zn(II) -> water donor/acceptor geometry that sets the tunneling distance.
``hamiltonian``
    SI-unit physical constants, the 256-point spatial grid (8 qubits), the asymmetric
    double-well potential ``V(x)`` and the sparse Hamiltonian ``H = T + V``.
``quantum_sim``
    Gaussian wavepacket preparation and Qiskit ``Statevector`` propagation under
    ``U(dt) = exp(-i H dt / hbar)``.
``moth_sonification``
    Moth Quantum ``quantumaudio`` QPAM encoding/decoding, WAV rendering and the Atlas
    platform job payload.
"""

__all__ = ["pdb_parser", "hamiltonian", "quantum_sim", "moth_sonification"]
__version__ = "1.0.0"
