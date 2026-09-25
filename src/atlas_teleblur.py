"""Donor-to-acceptor frames from Moth ``telablur-v1``.

The third input slot is an optional ``mask``, not a third picture. It is
omitted here. With no mask, an RGB image is teleblurred across the whole
frame (the schema uses image1's alpha only when that channel exists).

``strength`` is documented as "Strength of the teleportation effect, between
0.0 and 1.0". The engine rotates a selector qubit so the two images
interpolate at the amplitude level. Output alpha, when present, is
``alpha1·(1−strength) + alpha2·strength``. That is not a statement that RGB
is a linear morph fraction, so the saved note says so instead of relabeling
``strength``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.moth_client import MothClient, MothError

ENGINE = "telablur-v1"
SIZE = 128
STRENGTHS = [index / 10 for index in range(11)]
INPUT_DIR = ROOT / "atlas_inputs"
OUTPUT_DIR = ROOT / "atlas_outputs"
STRENGTH_NOTE = (
    "strength is the teleportation-effect parameter (0 to 1): a selector-qubit "
    "rotation interpolates the images at the amplitude level. Output alpha is "
    "alpha1*(1-strength)+alpha2*strength. The schema does not define strength "
    "as a linear RGB morph fraction. The optional mask slot was not sent."
)


def _paint(path: Path, side: str) -> None:
    """RGB double-well sketch. ``side`` is ``donor`` (left) or ``acceptor`` (right)."""

    image = Image.new("RGB", (SIZE, SIZE), (14, 14, 16))
    pixels = image.load()
    center = (SIZE * 0.28) if side == "donor" else (SIZE * 0.72)
    tint = (228, 162, 58) if side == "donor" else (60, 186, 166)
    for y in range(SIZE):
        for x in range(SIZE):
            barrier = math.exp(-((x - SIZE * 0.5) / (SIZE * 0.06)) ** 2) * math.exp(
                -((y - SIZE * 0.62) / (SIZE * 0.22)) ** 2
            )
            dx = (x - center) / SIZE
            dy = (y - SIZE * 0.58) / SIZE
            glow = math.exp(-(dx * dx * 18 + dy * dy * 28))
            well = math.exp(-((y - SIZE * 0.72) / (SIZE * 0.08)) ** 2) * (
                0.25 + 0.75 * math.exp(-((x - center) / (SIZE * 0.12)) ** 2)
            )
            red = int(min(255, 14 + barrier * 70 + glow * tint[0] + well * 40))
            green = int(min(255, 14 + barrier * 80 + glow * tint[1] + well * 36))
            blue = int(min(255, 18 + barrier * 120 + glow * tint[2] + well * 70))
            pixels[x, y] = (red, green, blue)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def write_inputs() -> tuple[Path, Path]:
    donor = INPUT_DIR / "donor.png"
    acceptor = INPUT_DIR / "acceptor.png"
    _paint(donor, "donor")
    _paint(acceptor, "acceptor")
    return donor, acceptor


def _download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=120)
    if response.status_code != 200 or not response.content:
        raise MothError(f"Frame download failed with HTTP {response.status_code}.")
    path.write_bytes(response.content)


def _run_strength(client: MothClient, image1: str, image2: str, strength: float) -> tuple[str, dict]:
    params = {
        "strength": strength,
        "size": SIZE,
        "downscale": True,
        "direction": "full",
    }
    files = {"image1": image1, "image2": image2}
    job_id = ""
    for attempt in range(4):
        if not job_id:
            job_id = client.submit(ENGINE, params, input_files=files)
            print(f"submitted strength={strength:.1f} job={job_id}")
        status = client.wait(job_id, timeout_s=900)
        if status.get("status") == "completed":
            return job_id, params
        error = status.get("error") if isinstance(status.get("error"), dict) else {}
        print(f"strength={strength:.1f} job={job_id} ended {status.get('status')}: {error}")
        if not error.get("retryable") or attempt == 3:
            raise MothError(f"{job_id} ended {status.get('status')}: {error}")
        job_id = ""
    raise MothError(f"strength {strength} did not complete.")


def write_frames() -> Path:
    """Upload the two PNGs and save one frame per strength in ``atlas_outputs``."""

    donor, acceptor = write_inputs()
    client = MothClient()
    image1 = client.upload_asset(donor, "image/png")
    image2 = client.upload_asset(acceptor, "image/png")
    print(f"uploaded image1={image1}")
    print(f"uploaded image2={image2}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, strength in enumerate(STRENGTHS):
        job_id, params = _run_strength(client, image1, image2, strength)
        payload = client.result(job_id)
        outputs = payload.get("outputs") or []
        if not outputs or not outputs[0].get("url"):
            raise MothError(f"{job_id} returned no download URL.")
        frame = OUTPUT_DIR / f"frame_{index:02d}.png"
        _download(outputs[0]["url"], frame)
        sidecar = {
            "engine": ENGINE,
            "job_id": job_id,
            "params": params,
            "input_files": {"image1": "donor.png", "image2": "acceptor.png"},
            "mask": "omitted",
            "strength_note": STRENGTH_NOTE,
            "output_slot": outputs[0].get("slot"),
            "content_type": outputs[0].get("content_type"),
        }
        (OUTPUT_DIR / f"frame_{index:02d}.json").write_text(
            json.dumps(sidecar, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved {frame.name} job={job_id}")
    return OUTPUT_DIR


if __name__ == "__main__":
    print(f"wrote {write_frames()}")
