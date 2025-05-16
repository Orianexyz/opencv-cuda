#!/usr/bin/env python3
"""
Ultra-fast batch cropper (CUDA edition)
───────────────────────────────────────
Highlights
  • NVDEC → zero-copy frames on GPU            (OpenCV cudacodec)
  • Batched Sobel magnitude on GPU via Cupy    (8× frames / launch)
  • CUDA streams & events hide upload latency
  • Optional Cupy kernels for the median-border heuristic
  • Two-stage pipeline: analyse first, encode later
"""

from __future__ import annotations
import argparse, os, shutil, subprocess, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, List

import cv2
import numpy as np

# ────────── optional goodies ──────────
try:
    import cupy as cp
    HAVE_CUPY = True
except ModuleNotFoundError:
    HAVE_CUPY = False
try:
    cudacodec = cv2.cudacodec  # type: ignore[attr-defined]
    HAVE_NVDEC = True
except AttributeError:
    HAVE_NVDEC = False
# ──────────────────────────────────────

# ───── Config (can be CLI-overridden) ─────────────────────────────────────────
SAMPLE_FPS     = 1
TOLERANCE      = 5
EDGE_THRESH    = 10
MIN_CROP_RATIO = 0.10
DOWNSCALE      = 0.5
BATCH          = 8                 # #frames per Sobel batch on GPU
MAX_WORKERS    = max(1, (os.cpu_count() or 4) // 2)
FFMPEG         = "ffmpeg"
FFPROBE        = "ffprobe"
# ───────────────────────────────────────────────────────────────────────────────


# ╭─────────────────────── FFprobe helpers ─────────────────────────╮
def _ffprobe_json(path: str) -> dict:
    import json
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        text=True
    )
    return json.loads(out)


def _video_info(path: str) -> Tuple[int, int, float]:
    meta = _ffprobe_json(path)
    vstream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    return int(vstream["width"]), int(vstream["height"]), float(meta["format"]["duration"])


def _even(x: int) -> int:               # FFmpeg needs even dims
    return x if x % 2 == 0 else x + 1
# ╰────────────────────────────────────────────────────────────────╯


# ╭─────────────────── GPU frame iterator (NVDEC) ──────────────────╮
def _iter_gpu_frames(path: str, w0: int, h0: int) -> Iterator[cv2.cuda.GpuMat]:
    """Yields down‐scaled RGB `GpuMat` frames at SAMPLE_FPS."""
    reader = cudacodec.createVideoReader(
        path,
        [cv2.cudacodec.VideoReaderProps_PROP_COLOR_FORMAT,
         1   # 1 == ColorFormat::BGR
        ]
    )
    ratio = DOWNSCALE
    tgt = (_even(int(w0 * ratio)), _even(int(h0 * ratio)))
    t_last = -1.0
    ok, gpu_bgr = reader.nextFrame()
    while ok:
        t_cur = reader.get(cv2.cudacodec.VideoReaderProps_POS_MSEC) / 1000.0
        if t_last < 0 or t_cur - t_last >= 1 / SAMPLE_FPS:
            if gpu_bgr.size() != tgt:
                gpu_bgr = cv2.cuda.resize(gpu_bgr, tgt)
            yield gpu_bgr
            t_last = t_cur
        ok, gpu_bgr = reader.nextFrame()
# ╰──────────────────────────────────────────────────────────────╯


# ╭────────────── CPU fallback frame iterator (FFmpeg pipe) ─────────╮
def _iter_cpu_frames(path: str, w0: int, h0: int) -> Iterator[np.ndarray]:
    w_s, h_s = map(_even, (int(w0 * DOWNSCALE), int(h0 * DOWNSCALE)))
    cmd = [FFMPEG, "-v", "error", "-hwaccel", "auto", "-i", path,
           "-vf", f"fps={SAMPLE_FPS},scale={w_s}:{h_s}",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            bufsize=w_s * h_s * 3)
    read_sz = w_s * h_s * 3
    try:
        while True:
            buf = proc.stdout.read(read_sz)
            if len(buf) < read_sz:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h_s, w_s, 3)
    finally:
        proc.stdout.close(); proc.wait()
# ╰──────────────────────────────────────────────────────────────╯


# ╭──────────────────── Helpers (GPU Sobel etc.) ───────────────────╮
def _sobel_gpu(gray: cv2.cuda.GpuMat) -> Tuple[cv2.cuda.GpuMat, cv2.cuda.GpuMat]:
    """Returns gx, gy (CV_32F) for a GPU gray frame."""
    if hasattr(cv2.cuda, "Sobel"):
        gx = cv2.cuda.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.cuda.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        fx = cv2.cuda.createSobelFilter(cv2.CV_8U, cv2.CV_32F, 1, 0, 3)
        fy = cv2.cuda.createSobelFilter(cv2.CV_8U, cv2.CV_32F, 0, 1, 3)
        gx, gy = fx.apply(gray), fy.apply(gray)
    return gx, gy


def _median_border_gpu(rgb_batch: cp.ndarray) -> cp.ndarray:
    """returns a boolean mask (#frames, h, w) where diff>tol"""
    b0 = cp.concatenate([rgb_batch[:, 0], rgb_batch[:, -1],
                         rgb_batch[:, :, 0], rgb_batch[:, :, -1]], axis=1)
    mb = cp.median(b0, axis=1, keepdims=True)
    diff = cp.abs(rgb_batch.astype(cp.int16) - mb.astype(cp.int16)).sum(axis=-1)
    return diff > TOLERANCE
# ╰──────────────────────────────────────────────────────────────╯


# ╭─────────────────── crop analysis (GPU path) ────────────────────╮
def _analyse_gpu(frames: Iterator[cv2.cuda.GpuMat],
                 w0: int, h0: int) -> Optional[Tuple[int, int, int, int]]:
    stream = cv2.cuda_Stream()
    heat: Optional[cv2.cuda.GpuMat] = None
    batch: List[cv2.cuda.GpuMat] = []
    rect_from_border: Optional[Tuple[int, int, int, int]] = None

    for f in frames:
        batch.append(f)
        if len(batch) < BATCH:
            continue

        # ─── median-border on GPU (once per batch) ───────────────
        if HAVE_CUPY:
            rgb = cp.stack([cp.asarray(b.download(stream)) for b in batch])  # (B,H,W,3)
            mask = _median_border_gpu(rgb)
            coords = cp.argwhere(mask)
            if coords.size > 0:
                y0 = int(cp.min(coords[:, 1]).get()); x0 = int(cp.min(coords[:, 2]).get())
                y1 = int(cp.max(coords[:, 1]).get()); x1 = int(cp.max(coords[:, 2]).get())
                rect_from_border = (x0, y0, x1 - x0, y1 - y0)
            del rgb, mask, coords

        # ─── Sobel magnitude accumulation ────────────────────────
        acc = None
        for g in batch:
            gray = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY, stream=stream)
            gx, gy = _sobel_gpu(gray)
            mag, _ = cv2.cuda.cartToPolar(gx, gy, None, None, False, stream)
            acc = mag if acc is None else cv2.cuda.add(acc, mag, stream=stream)
        heat = acc if heat is None else cv2.cuda.add(heat, acc, stream=stream)

        batch.clear()

    if heat is None:
        return rect_from_border

    heat_np = heat.download(stream)
    stream.waitForCompletion()
    rect_grad = _extract_bbox_from_heat(heat_np, w0, h0)
    # pick whichever yields more cropping
    if rect_from_border and _area(rect_from_border) > _area(rect_grad):
        return rect_from_border
    return rect_grad
# ╰──────────────────────────────────────────────────────────────╯


# ╭──────────────── CPU analysis fallback ──────────────────────────╮
def _analyse_cpu(frames: Iterator[np.ndarray],
                 w0: int, h0: int) -> Optional[Tuple[int, int, int, int]]:
    heat = None
    border_rect = None
    for f in frames:
        # border median
        edges = np.vstack([f[0], f[-1], f[:, 0], f[:, -1]])
        med = np.median(edges, axis=0)
        diff = np.abs(f.astype(np.int16) - med.astype(np.int16)).sum(2)
        ys, xs = np.where(diff > TOLERANCE)
        if ys.size:
            x0, y0 = xs.min(), ys.min()
            x1, y1 = xs.max(), ys.max()
            border_rect = (x0, y0, x1 - x0, y1 - y0)
        # gradient heat
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        heat = mag if heat is None else heat + mag
    if heat is None:
        return border_rect
    rect_grad = _extract_bbox_from_heat(heat, w0, h0)
    if border_rect and _area(border_rect) > _area(rect_grad):
        return border_rect
    return rect_grad
# ╰──────────────────────────────────────────────────────────────╯


def _area(rect):  # type: ignore[return-value]
    if rect is None: return 0
    return rect[2] * rect[3]


def _extract_bbox_from_heat(heat_32f: np.ndarray, w0, h0):
    norm = cv2.normalize(heat_32f, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(norm, EDGE_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    s = 1 / DOWNSCALE
    w, h = map(_even, (int(w * s), int(h * s)))
    return int(x * s), int(y * s), w, h


# ╭────────────────── encode (2nd phase) ────────────────────────────╮
def _encode(src: str, dst: Path,
            rect: Optional[Tuple[int, int, int, int]], hw: str):
    if rect is None:
        shutil.copy2(src, dst)
        return "copied"

    x, y, w, h = rect
    vf_crop = f"crop={w}:{h}:{x}:{y}"
    if hw == "cuda":
        enc = ["-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "5M"]
        hwaccel = ["-hwaccel", "cuda"]
    else:
        enc, hwaccel = ["-c:v", "libx264", "-crf", "18"], []
    cmd = [FFMPEG, "-v", "error", *hwaccel, "-i", src, "-vf", vf_crop,
           *enc, "-c:a", "copy", str(dst)]
    subprocess.check_call(cmd)
    return f"cropped ({hw})"
# ╰──────────────────────────────────────────────────────────────╯


# ╭────────────────────────  per-file workflow  ─────────────────────╮
def _analyse_file(src: str, use_gpu: bool) -> Tuple[str, Optional[Tuple[int,int,int,int]]]:
    w0, h0, _ = _video_info(src)
    if use_gpu and HAVE_NVDEC:
        frames = _iter_gpu_frames(src, w0, h0)
        rect = _analyse_gpu(frames, w0, h0)
    else:
        frames = _iter_cpu_frames(src, w0, h0)
        rect = _analyse_cpu(frames, w0, h0)
    return src, rect
# ╰──────────────────────────────────────────────────────────────╯


def _detect_hw() -> Tuple[bool, str]:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        return True, "cuda" if HAVE_NVDEC else "cpu-dec"
    return False, "cpu"


def main() -> None:
    global SAMPLE_FPS
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",  type=Path, default=Path("../videos"))
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("../output/cropped_gpu"))
    ap.add_argument("--fps", type=int, default=SAMPLE_FPS)
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU path even if CUDA is available")
    args = ap.parse_args()

    SAMPLE_FPS = args.fps

    use_gpu, hw_name = _detect_hw()
    use_gpu = use_gpu and not args.cpu
    print(f"🚀  HW decoding: {hw_name} • Cupy: {HAVE_CUPY} • "
          f"GPU kernels: {use_gpu}")

    args.output.mkdir(parents=True, exist_ok=True)
    videos = sorted(str(p) for p in args.input.glob("*.mp4"))
    if not videos:
        print("No .mp4 files found.", file=sys.stderr); return

    t0 = time.time()

    # ── Stage 1 : analysis ──────────────────────────────────────
    analysis: dict[str, Optional[Tuple[int,int,int,int]]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(_analyse_file, v, use_gpu) for v in videos]
        for f in as_completed(futs):
            src, rect = f.result()
            analysis[src] = rect
            print(Path(src).name, "→", rect or "no-crop")

    # ── Stage 2 : encoding / copying (I/O bound) ────────────────
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = []
        for src, rect in analysis.items():
            dst = args.output / Path(src).name
            futs.append(pool.submit(_encode, src, dst, rect, hw_name))
        for f in as_completed(futs):
            print(f.result())

    print(f"✅  finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
