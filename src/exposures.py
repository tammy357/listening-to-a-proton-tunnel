"""Develop the proton one position measurement at a time.

Each grain is one draw from the exact ``|ψ(x, t)|²`` at ``t = T/4``.
Section 09 would be an 8-qubit Atlas shot record over the 256 grid sites.
That section is not in the notebook, and the tomography bank is 24 one-qubit
clocks, not 256 position counts. The sheets therefore say they were
developed from samples of the exact ``|ψ|²``.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atlas_tessa import section09_grid

EXPERIENCE = ROOT / "src" / "experience_data.json"
OUT = ROOT / "exposures"
ISOTOPES = ("H", "D", "T")
COUNTS = (64, 256, 1024, 4096)
VIDEO_N = 4096
VIDEO_SECONDS = 10.0
VIDEO_FPS = 24
FRAME_W = 960
FRAME_H = 540
SAMPLE_RATE = 48000
CAPTION = "developed from samples of the exact |ψ|²"
ATLAS_CAPTION = "developed from Moth Atlas shots"
WARM_LINE = "Every grain is one time the proton was looked at. Look enough times, and the wave appears."
NAMES = {"H": "hydrogen", "D": "deuterium", "T": "tritium"}


def exposure_caption() -> str:
    """Say which record the grains come from. Atlas position shots are not on disk."""

    if section09_grid() is not None:
        return ATLAS_CAPTION
    return CAPTION


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _quarter_density(movie: dict, frames_per_period: int) -> np.ndarray:
    index = frames_per_period // 4
    return np.asarray(movie["density"][index], dtype=np.float64)


def _sample_indices(density: np.ndarray, count: int, seed: int) -> np.ndarray:
    pdf = np.clip(density, 0.0, None)
    total = float(pdf.sum())
    if total <= 0:
        pdf = np.ones_like(pdf)
        total = float(pdf.sum())
    cdf = np.cumsum(pdf / total)
    rng = np.random.default_rng(seed)
    picks = rng.random(count)
    return np.clip(np.searchsorted(cdf, picks, side="right"), 0, len(density) - 1)


def _layout(indices: np.ndarray, seed: int, width: int, height: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    xs = margin + (indices.astype(np.float64) / 255.0) * (width - 2 * margin)
    ys = margin + rng.random(len(indices)) * (height - 2 * margin)
    return xs, ys


def _splat(width: int, height: int, xs: np.ndarray, ys: np.ndarray, limit: int) -> np.ndarray:
    image = np.zeros((height, width), dtype=np.float32)
    radius = 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(xx * xx + yy * yy) / 2.2).astype(np.float32)
    use = min(limit, len(xs))
    for i in range(use):
        x = int(xs[i])
        y = int(ys[i])
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        kx0 = x0 - (x - radius)
        ky0 = y0 - (y - radius)
        image[y0:y1, x0:x1] += kernel[ky0 : ky0 + (y1 - y0), kx0 : kx0 + (x1 - x0)]
    peak = float(image.max()) or 1.0
    # Keep single grains visible and a full sheet from blowing out.
    image = np.clip(image / max(peak * 0.35, 1.0), 0.0, 1.0)
    return image


def _to_rgb(field: np.ndarray) -> np.ndarray:
    silver = np.array([236, 228, 214], dtype=np.float32)
    rgb = field[..., None] * silver
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _grains_for(isotope: str, count: int, frames_per_period: int, movie: dict, x: np.ndarray) -> dict:
    density = _quarter_density(movie, frames_per_period)
    seed = {"H": 1, "D": 2, "T": 3}[isotope] * 10007 + count
    indices = _sample_indices(density, count, seed)
    return {"indices": indices, "x": x[indices]}


def contact_sheet(path: Path | None = None) -> Path:
    data = json.loads(EXPERIENCE.read_text(encoding="utf-8"))
    x = np.asarray(data["x"], dtype=np.float64)
    frames = int(data["frames_per_period"])
    cell_w, cell_h = 280, 168
    label_h = 36
    rebate = 28
    cols, rows = 4, 3
    width = rebate * 2 + cols * cell_w
    height = rebate * 2 + rows * (cell_h + label_h) + 28
    canvas = Image.new("RGB", (width, height), (8, 8, 10))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    small = _font(13)
    number = 1
    for row, isotope in enumerate(ISOTOPES):
        movie = data["isotopes"][isotope]
        for col, count in enumerate(COUNTS):
            grains = _grains_for(isotope, count, frames, movie, x)
            xs, ys = _layout(grains["indices"], 50 + row * 10 + col, cell_w, cell_h, 16)
            field = _splat(cell_w, cell_h, xs, ys, count)
            tile = Image.fromarray(_to_rgb(field), "RGB")
            left = rebate + col * cell_w
            top = rebate + row * (cell_h + label_h)
            canvas.paste(tile, (left, top))
            caption = f"{number:02d}  {isotope} · {NAMES[isotope]} · N = {count}"
            draw.text((left + 8, top + cell_h + 8), caption, fill=(236, 228, 214), font=font)
            number += 1
    # Thin rebate: inner rule and a few sprocket notches.
    draw.rectangle((8, 8, width - 9, height - 9), outline=(186, 176, 150), width=1)
    draw.rectangle((rebate - 6, rebate - 6, width - rebate + 5, height - rebate + 5), outline=(90, 86, 74), width=1)
    for y in range(36, height - 20, 28):
        draw.rectangle((12, y, 20, y + 10), outline=(186, 176, 150))
        draw.rectangle((width - 21, y, width - 13, y + 10), outline=(186, 176, 150))
    draw.text((rebate, height - 24), exposure_caption(), fill=(168, 162, 148), font=small)
    destination = path or (OUT / "contact_sheet.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def _schedule(count: int = VIDEO_N) -> np.ndarray:
    """Time of each grain. Early grains are separated; the end becomes a hiss."""

    index = np.arange(1, count + 1, dtype=np.float64)
    return VIDEO_SECONDS * (index / count) ** 2.0


def _click_audio(xs: np.ndarray, x_axis: np.ndarray, times: np.ndarray) -> np.ndarray:
    total = int(VIDEO_SECONDS * SAMPLE_RATE)
    mix = np.zeros((total, 2), dtype=np.float64)
    length = int(0.0045 * SAMPLE_RATE)
    ramp = np.arange(length, dtype=np.float64)
    envelope = np.exp(-ramp / (SAMPLE_RATE * 0.0011))
    carrier = np.sin(2.0 * math.pi * 2400.0 * ramp / SAMPLE_RATE) * envelope
    x_min = float(x_axis[0])
    x_max = float(x_axis[-1])
    span = max(1e-9, x_max - x_min)
    for grain, when in enumerate(times):
        start = int(when * SAMPLE_RATE)
        if start >= total:
            continue
        stop = min(total, start + length)
        used = stop - start
        pan = float(np.clip((xs[grain] - x_min) / span * 2.0 - 1.0, -1.0, 1.0))
        angle = (pan + 1.0) * (math.pi / 4.0)
        left = math.cos(angle)
        right = math.sin(angle)
        # Later grains are a little softer so the hiss does not clip before loudnorm.
        level = 0.22 * (0.55 + 0.45 * (1.0 - grain / len(times)))
        mix[start:stop, 0] += carrier[:used] * left * level
        mix[start:stop, 1] += carrier[:used] * right * level
    peak = float(np.max(np.abs(mix))) or 1.0
    return mix * (0.9 / peak)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    clipped = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(clipped.tobytes())


def developing_video(isotope: str, path: Path | None = None) -> Path:
    data = json.loads(EXPERIENCE.read_text(encoding="utf-8"))
    x = np.asarray(data["x"], dtype=np.float64)
    frames = int(data["frames_per_period"])
    movie = data["isotopes"][isotope]
    grains = _grains_for(isotope, VIDEO_N, frames, movie, x)
    xs, ys = _layout(grains["indices"], 90 + ord(isotope), FRAME_W, FRAME_H, 36)
    times = _schedule(VIDEO_N)
    audio = _click_audio(grains["x"], x, times)
    destination = path or (OUT / f"developing_{isotope}.mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    wav_path = destination.with_suffix(".wav")
    _write_wav(wav_path, audio)
    font = _font(22)
    n_frames = int(VIDEO_SECONDS * VIDEO_FPS)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{FRAME_W}x{FRAME_H}",
        "-r",
        str(VIDEO_FPS),
        "-i",
        "pipe:0",
        "-i",
        str(wav_path),
        "-af",
        # Sparse clicks sit under the loudnorm gate. The extra 2 dB is what
        # brought the shipped films from about −18 LUFS to −16 LUFS integrated.
        "loudnorm=I=-16:TP=-1.5:LRA=11,volume=2.0dB",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    field = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    radius = 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(xx * xx + yy * yy) / 2.2).astype(np.float32)
    placed = 0
    try:
        for frame_index in range(n_frames):
            t = frame_index / VIDEO_FPS
            target = int(np.searchsorted(times, t, side="right"))
            while placed < target:
                x_px = int(xs[placed])
                y_px = int(ys[placed])
                x0 = max(0, x_px - radius)
                x1 = min(FRAME_W, x_px + radius + 1)
                y0 = max(0, y_px - radius)
                y1 = min(FRAME_H, y_px + radius + 1)
                kx0 = x0 - (x_px - radius)
                ky0 = y0 - (y_px - radius)
                field[y0:y1, x0:x1] += kernel[ky0 : ky0 + (y1 - y0), kx0 : kx0 + (x1 - x0)]
                placed += 1
            shown = np.clip(field / 6.0, 0.0, 1.0)
            rgb = _to_rgb(shown)
            image = Image.fromarray(rgb, "RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, FRAME_W - 11, FRAME_H - 11), outline=(186, 176, 150), width=1)
            draw.text((24, 24), f"{isotope} · {NAMES[isotope]}", fill=(236, 228, 214), font=font)
            draw.text((24, FRAME_H - 48), f"measurements: {placed}", fill=(236, 228, 214), font=font)
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed for {isotope}: {stderr[-500:]}")
    finally:
        if process.poll() is None:
            process.kill()
    wav_path.unlink(missing_ok=True)
    return destination


def write_exposures(force: bool = False) -> dict:
    """Write the contact sheet and the three films.

    A finished set with the current caption is kept. The notebook can run
    from that cache without encoding the films again.
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write the developing films.")
    caption = exposure_caption()
    note = OUT / "caption.txt"
    sheet = OUT / "contact_sheet.png"
    videos = {isotope: OUT / f"developing_{isotope}.mp4" for isotope in ISOTOPES}
    ready = (
        not force
        and sheet.exists()
        and note.exists()
        and caption in note.read_text(encoding="utf-8")
        and all(path.exists() and path.stat().st_size > 1000 for path in videos.values())
    )
    if ready:
        return {
            "contact_sheet": str(sheet),
            "videos": {isotope: str(path) for isotope, path in videos.items()},
            "caption": caption,
        }
    sheet = contact_sheet()
    videos = {isotope: str(developing_video(isotope)) for isotope in ISOTOPES}
    note.write_text(caption + "\n" + WARM_LINE + "\n", encoding="utf-8")
    return {"contact_sheet": str(sheet), "videos": videos, "caption": caption}


if __name__ == "__main__":
    written = write_exposures()
    print(written["caption"])
    print(written["contact_sheet"])
    for isotope, path in written["videos"].items():
        print(isotope, path)
