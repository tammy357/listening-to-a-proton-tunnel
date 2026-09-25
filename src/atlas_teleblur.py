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
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.moth_client import MothClient, MothError

ENGINE = "telablur-v1"
SIZE = 512
STRENGTHS = [0.25, 0.4, 0.5, 0.6, 0.75]
INPUT_DIR = ROOT / "atlas_inputs"
OUTPUT_DIR = ROOT / "atlas_outputs"
RUN_DIR = OUTPUT_DIR / "runs"
STRENGTH_NOTE = (
    "strength is the teleportation-effect parameter (0 to 1): a selector-qubit "
    "rotation interpolates the images at the amplitude level. Output alpha is "
    "alpha1*(1-strength)+alpha2*strength. The schema does not define strength "
    "as a linear RGB morph fraction. The optional mask slot was not sent."
)


def _paint(path: Path, side: str) -> None:
    """Black 512 canvas with one large soft white-violet blob on one half."""

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cx = SIZE * (0.28 if side == "donor" else 0.72)
    cy = SIZE * 0.50
    sigma = SIZE * 0.22
    glow = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    # White at the core, violet in the soft shoulder, black outside.
    red = glow * (255.0 * glow + 196.0 * (1.0 - glow))
    green = glow * (246.0 * glow + 150.0 * (1.0 - glow))
    blue = glow * (255.0 * glow + 255.0 * (1.0 - glow))
    rgb = np.clip(np.stack([red, green, blue], axis=-1), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def _params(strength: float) -> dict:
    return {
        "strength": strength,
        "size": SIZE,
        "downscale": False,
        "direction": "full",
    }


def both_shapes_score(image: np.ndarray) -> float:
    """Higher when both halves carry a bright shape, and neither half dominates."""

    gray = np.asarray(image, dtype=np.float64).mean(axis=2)
    width = gray.shape[1]
    left = gray[:, : width // 2]
    right = gray[:, width // 2 :]
    left_mass = float(left[left > 28].sum())
    right_mass = float(right[right > 28].sum())
    if left_mass <= 0.0 or right_mass <= 0.0:
        return 0.0
    balance = min(left_mass, right_mass) / max(left_mass, right_mass)
    return balance * min(left_mass, right_mass)


def _submit_strength(client: MothClient, image1: str, image2: str, strength: float) -> str:
    params = _params(strength)
    files = {"image1": image1, "image2": image2}
    job_id = client.submit(ENGINE, params, input_files=files)
    print(f"submitted strength={strength:.2f} job={job_id} downscale=false size={SIZE}", flush=True)
    return job_id


def _collect(
    client: MothClient,
    pending: dict[str, float],
    image1: str,
    image2: str,
) -> list[tuple[float, str, dict]]:
    """Poll until every job completes. Retry a retryable failure once."""

    finished: list[tuple[float, str, dict]] = []
    retries: dict[str, int] = {job_id: 0 for job_id in pending}
    deadline = time.monotonic() + 900.0
    while pending:
        if time.monotonic() > deadline:
            raise MothError(f"Still waiting on {sorted(pending)}.")
        for job_id in list(pending):
            status = client.status(job_id)
            state = str(status.get("status", ""))
            if state not in {"completed", "failed", "cancelled"}:
                continue
            strength = pending.pop(job_id)
            if state == "completed":
                payload = client.result(job_id)
                finished.append((strength, job_id, payload))
                print(f"completed strength={strength:.2f} job={job_id}", flush=True)
                continue
            error = status.get("error") if isinstance(status.get("error"), dict) else {}
            print(f"strength={strength:.2f} job={job_id} ended {state}: {error}", flush=True)
            if error.get("retryable") and retries.get(job_id, 0) < 2:
                replacement = _submit_strength(client, image1, image2, strength)
                pending[replacement] = strength
                retries[replacement] = retries.get(job_id, 0) + 1
                deadline = max(deadline, time.monotonic() + 900.0)
                continue
            raise MothError(f"{job_id} ended {state}: {error}")
        if pending:
            time.sleep(2.0)
    return finished


def write_best() -> Path:
    """Upload the blobs, try several strengths, and keep the clearest both-shape frame."""

    donor, acceptor = write_inputs()
    client = MothClient()
    image1 = client.upload_asset(donor, "image/png")
    image2 = client.upload_asset(acceptor, "image/png")
    print(f"uploaded image1={image1}", flush=True)
    print(f"uploaded image2={image2}", flush=True)
    pending = {
        _submit_strength(client, image1, image2, strength): strength for strength in STRENGTHS
    }
    finished = _collect(client, pending, image1, image2)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ranked: list[tuple[float, float, str, Path]] = []
    for strength, job_id, payload in finished:
        outputs = payload.get("outputs") or []
        if not outputs or not outputs[0].get("url"):
            raise MothError(f"{job_id} returned no download URL.")
        frame = RUN_DIR / f"strength_{strength:.2f}.png"
        _download(outputs[0]["url"], frame)
        image = np.asarray(Image.open(frame).convert("RGB"))
        score = both_shapes_score(image)
        ranked.append((score, strength, job_id, frame))
        sidecar = {
            "engine": ENGINE,
            "job_id": job_id,
            "params": _params(strength),
            "input_files": {"image1": "donor.png", "image2": "acceptor.png"},
            "mask": "omitted",
            "strength_note": STRENGTH_NOTE,
            "both_shapes_score": score,
            "output_slot": outputs[0].get("slot"),
            "content_type": outputs[0].get("content_type"),
        }
        frame.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        print(f"saved {frame.name} score={score:.1f} job={job_id}", flush=True)
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_strength, best_job, best_frame = ranked[0]
    chosen = OUTPUT_DIR / "teleblur.png"
    chosen.write_bytes(best_frame.read_bytes())
    summary = {
        "engine": ENGINE,
        "job_id": best_job,
        "params": _params(best_strength),
        "input_files": {"image1": "donor.png", "image2": "acceptor.png"},
        "mask": "omitted",
        "strength_note": STRENGTH_NOTE,
        "both_shapes_score": best_score,
        "candidates": [
            {"strength": strength, "job_id": job_id, "both_shapes_score": score}
            for score, strength, job_id, _path in ranked
        ],
    }
    (OUTPUT_DIR / "teleblur.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"chose strength={best_strength:.2f} score={best_score:.1f} job={best_job} -> {chosen}",
        flush=True,
    )
    return chosen


def write_frames() -> Path:
    """Notebook entry point. Keeps the clearest both-shape Teleblur frame."""

    return write_best()


if __name__ == "__main__":
    print(f"wrote {write_best()}", flush=True)
