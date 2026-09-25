"""Illustrative active-site context from PDB 1HED.

Human carbonic anhydrase II has a catalytic zinc (ZN A 262) and a bound
water (HOH A 342). This module can read those two centres from the RCSB
file and report their Euclidean distance in angstroms.

That distance is context for the dashboard only. It is not a parameter of
the double-well Hamiltonian, and it does not set the well separation, the
barrier, or the mass. If the download or the residue lookup fails, the
caller receives ``None`` rather than stand-in coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

PDB_ID = "1HED"
RCSB_URL = "https://files.rcsb.org/download/1HED.pdb"

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
            Illustrative only: the tunneling model does not read this value.
        source: ``"rcsb"`` when the file was downloaded.
        note: Short explanation of which coordinates were used.
    """

    pdb_id: str
    zinc_xyz: tuple[float, float, float]
    water_xyz: tuple[float, float, float]
    distance_angstrom: float
    source: str
    note: str


def fetch_active_site(timeout: float = 8.0) -> ActiveSite | None:
    """Download 1HED and measure the Zn–HOH distance.

    The distance is illustrative active-site context. Nothing in the
    Hamiltonian is set from it.

    Args:
        timeout: Seconds to wait for the RCSB response.

    Returns:
        The parsed geometry, or ``None`` when the file or either residue
        is unavailable.
    """

    try:
        text = _download_pdb(timeout=timeout)
        zinc = _find_residue(text, *ZINC_RESIDUE)
        water = _find_residue(text, *WATER_RESIDUE)
    except (requests.RequestException, OSError, ValueError):
        return None

    if zinc is None or water is None:
        return None

    return ActiveSite(
        pdb_id=PDB_ID,
        zinc_xyz=zinc,
        water_xyz=water,
        distance_angstrom=_distance(zinc, water),
        source="rcsb",
        note=(
            "Coordinates read from RCSB entry 1HED. "
            "Illustrative active-site context; not an input to the Hamiltonian."
        ),
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
