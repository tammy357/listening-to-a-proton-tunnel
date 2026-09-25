"""Precompute the Experience movie from the exact grid evolution.

This does not change the propagator. It calls ``evolve_exact`` at the same
default well the Lab opens on (E_b, a, λ = 0) for H, D, and T, over two
tunneling periods at 240 frames per period.

Screen time keeps the physical period ratios. One hydrogen period lasts
1.6 s on screen. Deuterium and tritium last 1.6 s times T_tun(isotope) /
T_tun(H). The scale factor is screen seconds per physical second.

Run from the repository root:

    python src/export_frames.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hamiltonian import ANGSTROM_M, JOULE_PER_EV
from src.moth_sonification import DEFAULT_PITCH_HZ, beat_frequency_hz
from src.quantum_sim import evolve_exact

# Same opening well as app.py. λ = 0.
BARRIER_EV = 0.15
HALF_WIDTH_A = 0.35
ASYMMETRY = 0.0
ISOTOPES = ("H", "D", "T")
PERIODS = 2
FRAMES_PER_PERIOD = 240
H_SCREEN_PERIOD_S = 1.6
PITCH_HZ = float(DEFAULT_PITCH_HZ)

OUTPUT_PATH = Path(__file__).resolve().parent / "experience_data.json"


def build_payload() -> dict:
    """Return the JSON-ready movie for H, D, and T.

    ``|ψ|²`` is rounded to 4 decimal places. Phase is ``arg(ψ)`` in radians,
    rounded to 2 decimal places, and cleared where the rounded density is 0
    so empty sites do not glitter. The grid is symmetric about x = 0, and
    λ = 0, so the acceptor-localized packet is the spatial mirror of frame 0.
    """

    movies = [_isotope_movie(name) for name in ISOTOPES]
    period_h = movies[0]["period_s"]
    if period_h <= 0:
        raise RuntimeError("Hydrogen tunneling period was not positive.")
    time_scale = H_SCREEN_PERIOD_S / period_h
    exponent = int(round(math.log10(time_scale)))
    for movie in movies:
        movie["screen_period_s"] = round(movie["period_s"] * time_scale, 6)
        movie["period_ratio_to_h"] = round(movie["period_s"] / period_h, 6)
    x_angstrom, v_ev = movies[0]["x_angstrom"], movies[0]["v_ev"]
    isotopes = {}
    for movie in movies:
        isotopes[movie["isotope"]] = {
            "period_s": movie["period_s"],
            "period_fs": round(movie["period_s"] * 1e15, 2),
            "screen_period_s": movie["screen_period_s"],
            "period_ratio_to_h": movie["period_ratio_to_h"],
            "deltaE_meV": movie["deltaE_meV"],
            "e0_ev": movie["e0_ev"],
            "f0_hz": PITCH_HZ,
            "f_beat_hz": movie["f_beat_hz"],
            "donor": movie["donor"],
            "acceptor": movie["acceptor"],
            "density": movie["density"],
            "phase": movie["phase"],
        }
    return {
        "barrier_ev": BARRIER_EV,
        "half_width_a": HALF_WIDTH_A,
        "asymmetry": ASYMMETRY,
        "periods": PERIODS,
        "frames_per_period": FRAMES_PER_PERIOD,
        "h_screen_period_s": H_SCREEN_PERIOD_S,
        "time_scale": time_scale,
        "time_scale_label": f"time slowed ~10^{exponent}×",
        "x": x_angstrom,
        "v": v_ev,
        "isotopes": isotopes,
    }


def write_experience_data(path: Path = OUTPUT_PATH) -> Path:
    """Write the movie JSON. Returns the path."""

    payload = build_payload()
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    path.write_text(text)
    return path


def _isotope_movie(isotope: str) -> dict:
    n_steps = FRAMES_PER_PERIOD * PERIODS
    model, evolution = evolve_exact(
        barrier_ev=BARRIER_EV,
        asymmetry_ev=ASYMMETRY,
        well_half_width_angstrom=HALF_WIDTH_A,
        mass=isotope,
        n_periods=float(PERIODS),
        n_steps=n_steps,
        keep_states=True,
    )
    if evolution.states is None:
        raise RuntimeError(f"{isotope} evolution did not keep its states.")
    # Drop the sample at t = 2 T_tun. It matches the t = 0 packet closely
    # enough that the animation can loop from the last frame back to the first.
    states = evolution.states[:-1]
    density = np.abs(states) ** 2
    phase = np.angle(states)
    density_q = np.round(density, 4)
    phase_q = np.round(phase, 2)
    phase_q = np.where(density_q == 0.0, 0.0, phase_q)
    return {
        "isotope": isotope,
        "period_s": float(evolution.tunneling_period_s),
        "deltaE_meV": round(float(evolution.splitting_ev) * 1e3, 4),
        "e0_ev": round(float(evolution.energies_j[0]) / JOULE_PER_EV, 6),
        "f_beat_hz": round(beat_frequency_hz(float(evolution.splitting_ev)), 4),
        "donor": [float(f"{value:.4f}") for value in evolution.donor_population[:-1]],
        "acceptor": [float(f"{value:.4f}") for value in evolution.acceptor_population[:-1]],
        "density": _rows(density_q, 4),
        "phase": _rows(phase_q, 2),
        "x_angstrom": [float(f"{value:.4f}") for value in model.x_m / ANGSTROM_M],
        "v_ev": [float(f"{value:.4f}") for value in model.v_joule / JOULE_PER_EV],
    }


def _rows(matrix: np.ndarray, decimals: int) -> list[list[float]]:
    fmt = f"{{:.{decimals}f}}"
    rows: list[list[float]] = []
    for row in matrix:
        rows.append([float(fmt.format(float(value))) for value in row])
    return rows


def _summarize(payload: dict, path: Path) -> None:
    hydrogen = payload["isotopes"]["H"]
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(payload["time_scale_label"], f"scale={payload['time_scale']:.3e}")
    for name, movie in payload["isotopes"].items():
        frames = len(movie["density"])
        print(
            f"{name}: ΔE={movie['deltaE_meV']:.3f} meV, "
            f"E0={movie['e0_ev']:.4f} eV, "
            f"T={movie['period_fs']:.2f} fs, "
            f"screen={movie['screen_period_s']:.3f} s, "
            f"ratio={movie['period_ratio_to_h']:.3f}, "
            f"beat={movie['f_beat_hz']:.2f} Hz, frames={frames}, "
            f"donor0={movie['donor'][0]:.3f}, "
            f"donor_half={movie['donor'][FRAMES_PER_PERIOD // 2]:.3f}"
        )
    if abs(hydrogen["screen_period_s"] - H_SCREEN_PERIOD_S) > 1e-6:
        raise SystemExit("Hydrogen screen period is not 1.6 s.")
    for name in ("D", "T"):
        movie = payload["isotopes"][name]
        expected = hydrogen["screen_period_s"] * movie["period_ratio_to_h"]
        if abs(movie["screen_period_s"] - expected) > 1e-4:
            raise SystemExit(f"{name} screen period does not keep the physical ratio.")


if __name__ == "__main__":
    destination = write_experience_data()
    _summarize(json.loads(destination.read_text()), destination)
