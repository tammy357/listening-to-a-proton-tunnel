"""RCSB PDB ``1HED`` active-site parser for Human Carbonic Anhydrase II.

Why this structure?
-------------------
Carbonic anhydrase II (CA II) is one of the fastest enzymes known: it hydrates CO2 at up to
~10^6 turnovers per second. The rate-limiting step of the catalytic cycle is **not** the CO2
chemistry itself but a *proton transfer* — the zinc-bound water molecule must hand a proton
off to a nearby acceptor (His64 / an ordered water wire) to regenerate the nucleophilic
Zn-OH(-) species.

That step is the textbook example of **quantum hydrogen tunneling in biology**: the proton is
light enough (m_H ~ 1836 m_e) that its de Broglie wavelength is comparable to the barrier
width, so it leaks *through* the activation barrier instead of climbing over it. The
experimental fingerprints are a large kinetic isotope effect (k_H/k_D ~ 3-8) and a weak
temperature dependence of that ratio.

Geometry drives everything. The donor-acceptor distance R extracted here sets the width of
the double-well potential built in :mod:`src.hamiltonian`, and the tunneling probability is
exponentially sensitive to it (roughly ``exp(-2 * kappa * R)``): shaving 0.1 A off the
donor-acceptor separation can change the transfer rate by an order of magnitude.

What this module does
---------------------
1. Downloads ``1HED.pdb`` from the RCSB web API (with an on-disk cache).
2. Locates the catalytic zinc ``ZN A 262`` and the ordered active-site water ``HOH A 342``.
3. Returns their Cartesian coordinates and the interatomic distance in Angstroms.

Everything here is in **Angstroms**, matching the PDB file format. Conversion to SI happens
in :mod:`src.hamiltonian`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

try:  # ``requests`` is a hard dependency, but the module must stay importable without it.
    import requests
except ImportError:  # pragma: no cover - defensive
    requests = None  # type: ignore[assignment]

PDB_ID = "1HED"
RCSB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_PATH = os.path.join(CACHE_DIR, f"{PDB_ID}.pdb")

#: Catalytic Zn(II) ion of the CA II active site (proton *donor* side anchor).
DONOR_SELECTOR: Tuple[str, str, int] = ("ZN", "A", 262)
#: Ordered catalytic water in the active-site cavity (proton *acceptor* / relay water).
ACCEPTOR_SELECTOR: Tuple[str, str, int] = ("HOH", "A", 342)

#: Offline fallbacks (Angstroms) taken from the deposited 1HED coordinates, so the whole
#: pipeline still runs end-to-end on a hackathon Wi-Fi network that hates you.
FALLBACK_ZN_COORD: Tuple[float, float, float] = (12.5, -8.7, 11.7)
FALLBACK_HOH_COORD: Tuple[float, float, float] = (13.9, -7.4, 15.9)


@dataclass(frozen=True)
class ActiveSite:
    """Geometry of the CA II proton-transfer active site.

    Attributes
    ----------
    pdb_id:
        RCSB identifier the coordinates came from (``1HED``).
    zn_coord:
        Cartesian position of the catalytic Zn(II) ion, in Angstroms.
    water_coord:
        Cartesian position of the ordered catalytic water oxygen, in Angstroms.
    distance_ang:
        Donor-acceptor interatomic distance ``|r_HOH - r_Zn|`` in Angstroms. This is the
        physical scale that the double-well potential is stretched to match.
    source:
        Provenance of the coordinates: ``"rcsb"`` (freshly downloaded), ``"cache"``
        (previously downloaded copy on disk) or ``"fallback"`` (hardcoded offline values).
    note:
        Human-readable provenance detail, surfaced in the Streamlit header.
    """

    pdb_id: str
    zn_coord: Tuple[float, float, float]
    water_coord: Tuple[float, float, float]
    distance_ang: float
    source: str
    note: str = field(default="")

    @property
    def is_offline_fallback(self) -> bool:
        """``True`` when the coordinates came from the hardcoded offline fallback."""
        return self.source == "fallback"


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean distance between two 3-vectors (Angstroms in, Angstroms out)."""
    return math.sqrt(sum((float(ai) - float(bi)) ** 2 for ai, bi in zip(a, b)))


def fetch_pdb_text(timeout: float = 10.0, use_cache: bool = True) -> Tuple[Optional[str], str]:
    """Fetch the raw ``1HED.pdb`` text from RCSB.

    Parameters
    ----------
    timeout:
        Per-request timeout in seconds. Kept short so the UI never hangs on a bad network.
    use_cache:
        When ``True`` a successful download is written to ``data/1HED.pdb`` and reused on
        subsequent runs (and on later runs without connectivity).

    Returns
    -------
    (text, source)
        ``text`` is the PDB file contents or ``None`` if unavailable; ``source`` is one of
        ``"cache"``, ``"rcsb"`` or ``"unavailable"``.
    """
    if use_cache and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as handle:
                return handle.read(), "cache"
        except OSError:
            pass

    if requests is None:
        return None, "unavailable"

    try:
        response = requests.get(RCSB_URL, timeout=timeout)
        response.raise_for_status()
    except Exception:  # network down, DNS blocked, RCSB rate limiting, ...
        return None, "unavailable"

    text = response.text
    if use_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            pass
    return text, "rcsb"


def _parse_atom_record(line: str) -> Optional[dict]:
    """Parse one fixed-column ``ATOM``/``HETATM`` record into a dict.

    The PDB format is column-addressed (not whitespace-delimited), which matters for the
    zinc: ``ZN`` atom names are right-justified differently from carbon/nitrogen names.
    """
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    try:
        return {
            "atom_name": line[12:16].strip(),
            "res_name": line[17:20].strip(),
            "chain_id": line[21:22].strip(),
            "res_seq": int(line[22:26]),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "element": line[76:78].strip(),
        }
    except (ValueError, IndexError):
        return None


def extract_atom(
    pdb_text: str,
    res_name: str,
    chain_id: str,
    res_seq: int,
    atom_name: Optional[str] = None,
) -> Optional[Tuple[float, float, float]]:
    """Return the coordinates of a selected residue/atom, or ``None`` if absent.

    Parameters
    ----------
    pdb_text:
        Raw PDB file contents.
    res_name, chain_id, res_seq:
        Residue selector, e.g. ``("ZN", "A", 262)`` or ``("HOH", "A", 342)``.
    atom_name:
        Optional atom-name filter. Omitted for single-atom residues (``ZN``) and waters,
        where the first (oxygen) record is the one we want.
    """
    for line in pdb_text.splitlines():
        record = _parse_atom_record(line)
        if record is None:
            continue
        if record["res_name"] != res_name:
            continue
        if chain_id and record["chain_id"] != chain_id:
            continue
        if record["res_seq"] != res_seq:
            continue
        if atom_name is not None and record["atom_name"] != atom_name:
            continue
        return (record["x"], record["y"], record["z"])
    return None


def _fallback_site(note: str) -> ActiveSite:
    """Build the offline :class:`ActiveSite` from the hardcoded 1HED coordinates."""
    return ActiveSite(
        pdb_id=PDB_ID,
        zn_coord=FALLBACK_ZN_COORD,
        water_coord=FALLBACK_HOH_COORD,
        distance_ang=euclidean_distance(FALLBACK_ZN_COORD, FALLBACK_HOH_COORD),
        source="fallback",
        note=note,
    )


def load_active_site(timeout: float = 10.0, use_cache: bool = True) -> ActiveSite:
    """Load the CA II proton donor/acceptor geometry from PDB ``1HED``.

    Resolution order: on-disk cache -> RCSB download -> hardcoded offline fallback. The
    function never raises; a hackathon demo that dies because a firewall blocked
    ``files.rcsb.org`` is a demo that does not get judged.

    Returns
    -------
    ActiveSite
        Zn(II) and catalytic-water coordinates plus their separation in Angstroms.
    """
    pdb_text, source = fetch_pdb_text(timeout=timeout, use_cache=use_cache)
    if pdb_text is None:
        return _fallback_site("RCSB unreachable - using deposited 1HED reference coordinates.")

    zn = extract_atom(pdb_text, *DONOR_SELECTOR)
    hoh = extract_atom(pdb_text, *ACCEPTOR_SELECTOR)

    if zn is None or hoh is None:
        missing = " and ".join(
            name for name, coord in (("ZN A 262", zn), ("HOH A 342", hoh)) if coord is None
        )
        return _fallback_site(f"{missing} not found in the downloaded file - using fallback.")

    note = (
        "Coordinates read from the cached 1HED.pdb file."
        if source == "cache"
        else "Coordinates downloaded live from the RCSB PDB web API."
    )
    return ActiveSite(
        pdb_id=PDB_ID,
        zn_coord=zn,
        water_coord=hoh,
        distance_ang=euclidean_distance(zn, hoh),
        source=source,
        note=note,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    site = load_active_site()
    print(f"PDB {site.pdb_id} ({site.source}): {site.note}")
    print(f"  ZN  A 262 : {site.zn_coord}")
    print(f"  HOH A 342 : {site.water_coord}")
    print(f"  donor-acceptor distance = {site.distance_ang:.3f} A")
