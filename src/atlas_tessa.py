"""Colour the proton on Tessa's sphere, then read the pictures back.

``tessa-image-v1`` encodes each pixel as a point on a colour sphere with the
same coordinates as a qubit state. The live schema (moth-api v0.41.0) names
the machine in ``params.machine``:

* ``aer`` — noiseless local simulator
* ``fake_fez`` — local simulator calibrated to IBM Fez's noise
* ``ibm_fez`` — the Fez processor itself

This module sends ``aer`` and ``fake_fez`` only, with ``shots`` 4096 and
``distortion`` 0. ``fake_fez`` is the emulated device. The schema rejects a
positive distortion on every machine except real IBM hardware, and it rejects
images larger than the 64×64 simulated lattice.

Results are cached under ``atlas_tessa/`` with a sidecar JSON
(engine, job_id, device, shots). With no ``MOTH_API_KEY`` the loader reads
that cache and does not call the API.
"""

from __future__ import annotations

import colorsys
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hamiltonian import ANGSTROM_M
from src.moth_client import MothClient, MothError
from src.quantum_sim import evolve_exact

ENGINE = "tessa-image-v1"
TESSA_DIR = ROOT / "atlas_tessa"
BANK_PATH = ROOT / "src" / "atlas_bank.json"
SHOTS = 4096
DISTORTION = 0
# Schema enum. aer is noiseless. fake_fez is Fez's noise model on a simulator.
DEVICES = ("aer", "fake_fez")
DEVICE_LABELS = {
    "aer": "Aer · noiseless simulator",
    "fake_fez": "Emulated IBM Fez · calibrated noise model on a simulator",
}
MAX_LATTICE = 64
CARPET_N = 64
X_CROP_A = 0.7
BARRIER_EV = 0.15
HALF_WIDTH_A = 0.35
# 24 samples do not divide 64. Two pixels per swatch stays inside the lattice.
SWATCH_W = 2
SWATCH_H = 32


def bloch_to_rgb(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Map one Bloch vector to Tessa's colour sphere.

    The engine's colour sphere is the Bloch sphere. Grey sits at the centre.
    The white pole is ``theta = 0`` and the black pole is ``theta = pi``.
    Hue is the azimuth, and the saturated colours lie around the equator.

    For a vector ``(x, y, z)`` that convention is:

    * lightness follows ``z``, from the black pole (``z = -1``) to the white
      pole (``z = +1``)
    * hue is ``atan2(y, x)``
    * saturation is the in-plane radius ``sqrt(x² + y²)``
    * the origin is 50% grey

    Values are clipped into the unit interval so a shot-noisy vector that
    pokes slightly outside the ball still becomes a colour. Components are
    floats in ``[0, 1]``.
    """

    lightness = float(np.clip((float(z) + 1.0) / 2.0, 0.0, 1.0))
    saturation = float(np.clip(np.hypot(float(x), float(y)), 0.0, 1.0))
    hue = float(np.atan2(float(y), float(x)) / (2.0 * np.pi)) % 1.0
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return red, green, blue


def colour_strip(bank: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(colours, t_over_period)`` for every Atlas Bloch sample.

    ``colours`` has shape ``(n, 3)`` with floats in ``[0, 1]``, in the bank's
    time order.
    """

    points = sorted(bank["points"], key=lambda point: float(point["t_over_period"]))
    colours = np.array(
        [bloch_to_rgb(point["x"], point["y"], point["z"]) for point in points],
        dtype=np.float64,
    )
    times = np.array([float(point["t_over_period"]) for point in points], dtype=np.float64)
    return colours, times


def pole_check(colours: np.ndarray, times: np.ndarray) -> str:
    """Confirm ``t = 0`` is near white and ``t = T/2`` is near black."""

    start = int(np.argmin(np.abs(times)))
    half = int(np.argmin(np.abs(times - 0.5)))
    white = colours[start]
    black = colours[half]
    if float(white.min()) < 0.85:
        raise ValueError(f"t = 0 should be near white, got RGB {white}.")
    if float(black.max()) > 0.15:
        raise ValueError(f"t = T/2 should be near black, got RGB {black}.")
    return (
        f"t = 0 → RGB ({white[0]:.3f}, {white[1]:.3f}, {white[2]:.3f}), near white. "
        f"t = T/2 → RGB ({black[0]:.3f}, {black[1]:.3f}, {black[2]:.3f}), near black."
    )


def upscale_strip(colours: np.ndarray) -> np.ndarray:
    """Nearest-neighbour blocks, small enough for the 64×64 lattice."""

    row = (np.clip(colours, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    image = np.repeat(row, SWATCH_W, axis=0)
    image = np.repeat(image[np.newaxis, :, :], SWATCH_H, axis=0)
    _require_lattice(image.shape[1], image.shape[0])
    return image


def quantum_carpet() -> np.ndarray:
    """64×64 RGB image of ψ(x, t) for hydrogen over one tunneling period.

    Rows are time, from ``t = 0`` at the top to ``t = T_tun`` at the bottom
    (64 samples, including both ends). Columns are position, from ``-0.7 Å``
    to ``+0.7 Å``. Hue is ``arg ψ``, the same azimuth rule as the colour
    sphere. Lightness is ``|ψ|²`` divided by the brightest cell in this
    window, used as an HSV value with full saturation, so a bright cell stays
    a saturated hue instead of washing out to the white pole.
    """

    _model, evolution = evolve_exact(
        barrier_ev=BARRIER_EV,
        asymmetry_ev=0.0,
        well_half_width_angstrom=HALF_WIDTH_A,
        mass="H",
        n_periods=1.0,
        n_steps=CARPET_N - 1,
        keep_states=True,
    )
    if evolution.states is None or evolution.states.shape[0] != CARPET_N:
        raise RuntimeError("The carpet needs 64 wavefunction samples.")
    x_a = _model.x_m / ANGSTROM_M
    x_out = np.linspace(-X_CROP_A, X_CROP_A, CARPET_N)
    field = np.empty((CARPET_N, CARPET_N), dtype=np.complex128)
    for row, psi in enumerate(evolution.states):
        field[row] = np.interp(x_out, x_a, np.real(psi)) + 1j * np.interp(
            x_out, x_a, np.imag(psi)
        )
    density = np.abs(field) ** 2
    peak = float(density.max())
    if peak <= 0.0:
        raise RuntimeError("The cropped wavefunction has no probability.")
    value = density / peak
    hue = np.angle(field) / (2.0 * np.pi)
    hue = np.mod(hue, 1.0)
    rgb = np.empty((CARPET_N, CARPET_N, 3), dtype=np.float64)
    for i in range(CARPET_N):
        for j in range(CARPET_N):
            rgb[i, j] = colorsys.hsv_to_rgb(float(hue[i, j]), 1.0, float(value[i, j]))
    image = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    _require_lattice(image.shape[1], image.shape[0])
    return image


def section09_grid() -> Path | None:
    """The 24×8 Bloch grid, if section 09 has saved one."""

    notebook = ROOT / "notebooks" / "moth_tunneling_workflow.ipynb"
    text = notebook.read_text(encoding="utf-8") if notebook.exists() else ""
    if "## 09" not in text:
        return None
    for candidate in (
        TESSA_DIR / "section09_grid.png",
        ROOT / "atlas_outputs" / "section09_grid.png",
    ):
        if candidate.exists():
            return candidate
    return None


def load_tessa() -> dict:
    """Build the local pictures, then load or submit the Tessa jobs."""

    bundle = _build_local()
    if os.environ.get("MOTH_API_KEY", "").strip():
        print("path: live — MOTH_API_KEY is set, submitting tessa-image-v1")
        bundle["jobs"] = write_tessa(bundle)
    else:
        print("path: cache — MOTH_API_KEY is not set, loading atlas_tessa/")
        bundle["jobs"] = read_tessa()
    return bundle


def read_tessa() -> list[dict]:
    jobs = []
    for source in ("colour_strip", "carpet"):
        for device in DEVICES:
            jobs.append(_read_job(source, device))
    grid = section09_grid()
    if grid is not None:
        for device in DEVICES:
            jobs.append(_read_job("section09", device))
    return jobs


def write_tessa(bundle: dict | None = None) -> list[dict]:
    pictures = bundle if bundle is not None else _build_local()
    TESSA_DIR.mkdir(parents=True, exist_ok=True)
    client = MothClient()
    jobs = [
        _encode(client, pictures["strip_path"], "colour_strip", device) for device in DEVICES
    ]
    jobs.extend(_encode(client, pictures["carpet_path"], "carpet", device) for device in DEVICES)
    grid = section09_grid()
    if grid is None:
        print("section 09 is not in this notebook, so its grid is not sent", flush=True)
    else:
        print(f"section 09 grid {grid.name}", flush=True)
        jobs.extend(_encode(client, grid, "section09", device) for device in DEVICES)
    return jobs


def _build_local() -> dict:
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    colours, times = colour_strip(bank)
    report = pole_check(colours, times)
    print(report, flush=True)
    TESSA_DIR.mkdir(parents=True, exist_ok=True)
    strip = upscale_strip(colours)
    carpet = quantum_carpet()
    strip_path = TESSA_DIR / "colour_strip.png"
    carpet_path = TESSA_DIR / "carpet.png"
    _save_png(strip, strip_path)
    _save_png(carpet, carpet_path)
    return {
        "colours": colours,
        "times": times,
        "pole_report": report,
        "strip": strip,
        "carpet": carpet,
        "strip_path": strip_path,
        "carpet_path": carpet_path,
    }


def _encode(client: MothClient, image_path: Path, source: str, device: str) -> dict:
    if device not in DEVICES:
        raise MothError(f"{device} is not one of the simulator machines this section uses.")
    params = {"machine": device, "shots": SHOTS, "distortion": DISTORTION}
    asset = client.upload_asset(image_path, "image/png")
    job_id, payload = _submit_and_wait(client, params, {"image": asset})
    outputs = {
        str(item.get("slot") or "result"): item.get("url")
        for item in payload.get("outputs") or []
        if isinstance(item, dict) and item.get("url")
    }
    url = outputs.get("result")
    if not url:
        raise MothError(f"{job_id} returned no result image. Slots: {list(outputs)}")
    destination = TESSA_DIR / f"{source}_{device}.png"
    _download(url, destination)
    sidecar = {
        "engine": ENGINE,
        "job_id": job_id,
        "device": device,
        "shots": SHOTS,
        "params": params,
        "source": image_path.name,
        "device_label": DEVICE_LABELS[device],
    }
    (TESSA_DIR / f"{source}_{device}.json").write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {destination.name} job={job_id} device={device}", flush=True)
    sidecar["image"] = destination
    return sidecar


def _read_job(source: str, device: str) -> dict:
    image = TESSA_DIR / f"{source}_{device}.png"
    sidecar_path = TESSA_DIR / f"{source}_{device}.json"
    if not image.exists() or not sidecar_path.exists():
        raise FileNotFoundError(
            f"atlas_tessa cache is missing {image.name}. Set MOTH_API_KEY to build it."
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for key in ("engine", "job_id", "device", "shots"):
        if key not in sidecar:
            raise FileNotFoundError(f"{sidecar_path.name} has no {key}.")
    sidecar["image"] = image
    return sidecar


def _submit_and_wait(
    client: MothClient,
    params: dict,
    files: dict[str, str],
) -> tuple[str, dict]:
    job_id = ""
    for attempt in range(4):
        if not job_id:
            job_id = client.submit(ENGINE, params, input_files=files)
            print(f"submitted {ENGINE} device={params['machine']} job={job_id}", flush=True)
        status = client.wait(job_id, timeout_s=1800)
        if status.get("status") == "completed":
            return job_id, client.result(job_id)
        error = status.get("error") if isinstance(status.get("error"), dict) else {}
        print(f"{ENGINE} job={job_id} ended {status.get('status')}: {error}", flush=True)
        if not error.get("retryable") or attempt == 3:
            raise MothError(f"{job_id} ended {status.get('status')}: {error}")
        job_id = ""
    raise MothError(f"{ENGINE} did not complete.")


def _require_lattice(width: int, height: int) -> None:
    if width > MAX_LATTICE or height > MAX_LATTICE:
        raise ValueError(
            f"tessa-image-v1 rejects images above the {MAX_LATTICE}×{MAX_LATTICE} lattice "
            f"(got {width}×{height})."
        )


def _save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path, format="PNG")


def _download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=180)
    if response.status_code != 200 or not response.content:
        raise MothError(f"Download failed with HTTP {response.status_code}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


if __name__ == "__main__":
    result = load_tessa()
    for job in result["jobs"]:
        print(job["engine"], job["device"], job["shots"], job["job_id"])
