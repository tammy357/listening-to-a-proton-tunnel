"""Active-site geometry for human carbonic anhydrase II, PDB 1HED.

The catalytic zinc (ZN A 262) polarises a bound water (HOH A 342). The
oxygen–zinc separation is the structural scale of the proton-transfer
problem: the proton on that water faces a donor well on the zinc side and
an acceptor well toward the proton wire. This module reads those two
centres from the RCSB coordinate file and returns their Euclidean distance
in angstroms.

PDB coordinates are stored in angstroms. The distance is therefore reported
in angstroms as well. The Hamiltonian never consumes this number directly;
the reaction coordinate is a local 1-D path of fixed length, and the
measured distance is the experimental context shown on the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

PDB_ID = "1HED"
RCSB_URL = "https://files.rcsb.org/download/1HED.pdb"

# Offline stand-in supplied for the hackathon brief. These are not the
# deposited 1HED coordinates; they are used only when RCSB cannot be reached
# or the residue records are missing.
FALLBACK_ZINC_A = (12.5, -8.7, 11.7)
FALLBACK_WATER_A = (13.9, -7.4, 15.9)

ZINC_RESIDUE = ("ZN", "A", 262)
WATER_RESIDUE = ("HOH", "A", 342)


@dataclass(frozen=True)
class ActiveSite:
    """Zinc and catalytic-water centres, plus the donor–acceptor distance.

    Attributes:
        pdb_id: Four-character PDB identifier.
        zinc_xyz: Cartesian coordinates of ZN A 262, in angstroms.
        water_xyz: Cartesian coordinates of HOH A 342, in angstroms.
        distance_angstrom: Euclidean Zn–O distance, in angstroms.
        source: ``"rcsb"`` when the file was downloaded, ``"fallback"`` otherwise.
        note: Short explanation of which coordinates were used.
    """

    pdb_id: str
    zinc_xyz: tuple[float, float, float]
    water_xyz: tuple[float, float, float]
    distance_angstrom: float
    source: str
    note: str


def fetch_active_site(timeout: float = 8.0) -> ActiveSite:
    """Download 1HED and measure the Zn–HOH distance.

    The RCSB file endpoint is tried first. Any network error, non-200
    response, or missing residue falls back to the fixed coordinates in the
    hackathon brief so the dashboard still opens offline.

    Args:
        timeout: Seconds to wait for the RCSB response.

    Returns:
        Active-site geometry and the distance in angstroms.
    """

    try:
        text = _download_pdb(timeout=timeout)
        zinc = _find_residue(text, *ZINC_RESIDUE)
        water = _find_residue(text, *WATER_RESIDUE)
    except (requests.RequestException, OSError, ValueError):
        zinc, water = None, None

    if zinc is None or water is None:
        zinc = FALLBACK_ZINC_A
        water = FALLBACK_WATER_A
        return ActiveSite(
            pdb_id=PDB_ID,
            zinc_xyz=zinc,
            water_xyz=water,
            distance_angstrom=_distance(zinc, water),
            source="fallback",
            note=(
                "RCSB was unreachable or ZN A 262 / HOH A 342 was missing. "
                "Using the offline zinc and water coordinates."
            ),
        )

    return ActiveSite(
        pdb_id=PDB_ID,
        zinc_xyz=zinc,
        water_xyz=water,
        distance_angstrom=_distance(zinc, water),
        source="rcsb",
        note="Coordinates read from the RCSB entry 1HED.",
    )


def _download_pdb(timeout: float) -> str:
    response = requests.get(
        RCSB_URL,
        timeout=timeout,
        headers={"User-Agent": "moth-hacks-2026-tunneling/1.0"},
    )
    response.raise_for_status()
    return response.text


def _find_residue(
    pdb_text: str, resname: str, chain: str, resseq: int
) -> tuple[float, float, float] | None:
    """Return the first matching heavy-atom coordinate, preferring altLoc blank or A."""

    preferred: tuple[float, float, float] | None = None
    backup: tuple[float, float, float] | None = None
    for line in pdb_text.splitlines():
        parsed = _parse_hetatm(line)
        if parsed is None:
            continue
        got_name, got_chain, got_seq, alt_loc, xyz = parsed
        if (got_name, got_chain, got_seq) != (resname, chain, resseq):
            continue
        if alt_loc in ("", "A"):
            preferred = xyz
            break
        if backup is None:
            backup = xyz
    return preferred if preferred is not None else backup


def _parse_hetatm(
    line: str,
) -> tuple[str, str, int, str, tuple[float, float, float]] | None:
    """Parse one ATOM/HETATM record using the fixed PDB columns.

    Columns follow the PDB v3 specification: residue name 18–20, chain 22,
    residue sequence 23–26, orthogonal coordinates 31–54 (1-based).
    """

    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    if len(line) < 54:
        return None
    try:
        resseq = int(line[22:26])
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None
    resname = line[17:20].strip()
    chain = line[21].strip()
    alt_loc = line[16].strip()
    return resname, chain, resseq, alt_loc, xyz


def _distance(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))
