"""Equal-power stereo of one hydrogen crossing, matching StereoPannerNode.

``StereoPannerNode`` uses gains ``cos((pan+1) π/4)`` and ``sin((pan+1) π/4)``.
The saved populations never sit at exactly 0 and 1, so the Experience
normalises ``acceptor − donor`` by its peak. That sends the panner to −1 on
the donor side and +1 on the acceptor side.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "src" / "experience_data.json"


def panner_gains(pan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Web Audio StereoPannerNode equal-power gains for pan in [-1, 1]."""

    angle = (np.clip(pan, -1.0, 1.0) + 1.0) * (math.pi / 4.0)
    return np.cos(angle), np.sin(angle)


def crossing_pan(movie: dict) -> np.ndarray:
    """One period of pan, scaled so the extremes are exactly −1 and +1."""

    frames = int(json.loads(EXPERIENCE.read_text(encoding="utf-8"))["frames_per_period"])
    donor = np.asarray(movie["donor"][:frames], dtype=np.float64)
    acceptor = np.asarray(movie["acceptor"][:frames], dtype=np.float64)
    raw = acceptor - donor
    extent = float(np.max(np.abs(raw))) or 1.0
    return np.clip(raw / extent, -1.0, 1.0)


def crossing_stereo(sample_rate: int = 48000) -> dict:
    """A steady tone panned across one hydrogen period. Returns RMS figures."""

    data = json.loads(EXPERIENCE.read_text(encoding="utf-8"))
    movie = data["isotopes"]["H"]
    pan = crossing_pan(movie)
    seconds = float(movie["screen_period_s"])
    total = int(round(seconds * sample_rate))
    times = np.arange(total, dtype=np.float64) / sample_rate
    frame = np.clip((times / seconds * (len(pan) - 1)).astype(int), 0, len(pan) - 1)
    pan_s = pan[frame]
    left_gain, right_gain = panner_gains(pan_s)
    tone = 0.2 * np.sin(2.0 * math.pi * 220.0 * times)
    left = tone * left_gain
    right = tone * right_gain
    donor = pan_s < 0
    acceptor = ~donor

    def rms(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples * samples)))

    donor_l = rms(left[donor])
    donor_r = rms(right[donor])
    acceptor_l = rms(left[acceptor])
    acceptor_r = rms(right[acceptor])
    return {
        "pan_min": float(pan.min()),
        "pan_max": float(pan.max()),
        "donor_l_rms": donor_l,
        "donor_r_rms": donor_r,
        "donor_lr_difference": donor_l - donor_r,
        "acceptor_l_rms": acceptor_l,
        "acceptor_r_rms": acceptor_r,
        "acceptor_lr_difference": acceptor_l - acceptor_r,
        "full_l_rms": rms(left),
        "full_r_rms": rms(right),
    }


def print_crossing_stereo() -> dict:
    report = crossing_stereo()
    print(
        f"H crossing StereoPannerNode pan {report['pan_min']:+.3f} → {report['pan_max']:+.3f}"
    )
    print(
        "donor side  L RMS "
        f"{report['donor_l_rms']:.5f}  R RMS {report['donor_r_rms']:.5f}  "
        f"L−R {report['donor_lr_difference']:.5f}"
    )
    print(
        "acceptor side  L RMS "
        f"{report['acceptor_l_rms']:.5f}  R RMS {report['acceptor_r_rms']:.5f}  "
        f"L−R {report['acceptor_lr_difference']:.5f}"
    )
    return report


if __name__ == "__main__":
    print_crossing_stereo()
