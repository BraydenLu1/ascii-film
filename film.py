#!/usr/bin/env python3
"""film.py — THE GREAT GIG: one continuous lateral camera across the element strips.

The model is a physical film strip, not a timeline of dissolves:

    strip 0        strip 1              strip 2
    [=========]
          [==================]
                        [===============]
          ^ overlap 0.8s (the next strip's head lies under the previous tail)

Every element clip is a strip of length L_i = d_i + OVERLAP (the first strip has
no head to overlap).  A single camera coordinate c(t) walks left -> right over
the strips.  The local playback time of strip i is exactly

    local_i(t) = c(t) - X_i

so the camera does not "trigger" a clip, it *is* the playhead.  A splice is not
a cut: for the last 0.8 strip-seconds of strip i the next strip is already under
the lens, and a vertical edge sweeps left -> right, revealing the incoming
material over the outgoing one.  The edge position is

    edge_x(t) = W * (c(t) - X_{i+1}) / OVERLAP

i.e. the edge is locked to the camera, not to wall time.  No dissolve, no black.

Camera speed is 1.0 strip-second per second with a gentle Gaussian dip at every
splice (18%, sigma 0.45s), integrated and normalised so c(0)=0 and c(26)=26.
It never stops and never goes backwards (min speed ~0.82).

Usage:
    python film.py --mode draft          # 960x540 @ 8fps review cut + stills
    python film.py --mode final          # 1920x1080 @ 30fps -> film_26s.mp4
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from render import FFMPEG  # noqa: E402  (same ffmpeg the rest of the project uses)

TOTAL = 32.0
OVERLAP = 0.8
SRC_FPS = 30

# durations are the visible slot lengths from the approved shot table
# 2026-10-18 rev 2: the whole piece is 32 s (was 26) and the music starts at
# 12.0 s, on the end of the piano intro, so the band arrives two seconds in
# instead of the film opening on a wall of voice.  The wave is out, the phoenix
# is the penultimate beat, and the prism sits before it.
NAMES = ["01_evolution", "02_creation", "03_expulsion", "04_pact",
         "06_dispersion", "05_phoenix", "07_name"]
DUR = [10.0, 3.5, 4.0, 3.5, 3.5, 3.5, 4.0]
# source in-points; each strip then consumes DUR[i] + OVERLAP seconds
SRC_IN = [1.0, 0.8, 0.2, 0.6, 0.5, 0.3, 0.6]
AUDIO_IN = 20.5
CLIPS = [os.path.join(HERE, "elements", n + ".mp4") for n in NAMES]

# strip geometry -------------------------------------------------------------
# X[0] = 0, X[i] = start of slot i minus OVERLAP.  The first strip gets no head
# overlap; every later strip starts 0.8s before its slot, under the previous
# strip's tail, and is 0.8s longer than the slot.
X = np.zeros(len(DUR), np.float64)
L = np.zeros(len(DUR), np.float64)
_acc = 0.0
for i, d in enumerate(DUR):
    X[i] = 0.0 if i == 0 else _acc - OVERLAP
    L[i] = d if i == 0 else d + OVERLAP
    _acc += d
assert abs(_acc - TOTAL) < 1e-9, _acc
SPLICE_X = X[1:]  # camera positions where each incoming strip starts

CAM_TABLE_HZ = 1000.0


def build_camera():
    """c(t): monotone, ~unit speed, gentle dip at each splice, c(26)=26."""
    n = int(TOTAL * CAM_TABLE_HZ) + 1
    t = np.linspace(0.0, TOTAL, n)
    speed = np.ones_like(t)
    for sx in SPLICE_X:
        # slowest in the middle of the wipe, back to full speed outside it
        centre = sx + OVERLAP * 0.5
        speed -= 0.18 * np.exp(-((t - centre) / 0.45) ** 2)
    assert speed.min() > 0.5, "camera must never stop or reverse"
    c = np.empty_like(t)
    c[0] = 0.0
    c[1:] = np.cumsum((speed[1:] + speed[:-1]) * 0.5 * np.diff(t))
    c *= TOTAL / c[-1]          # absorb the distance lost to the dips
    assert np.all(np.diff(c) > 0)
    return t, c


CAM_T, CAM_C = build_camera()


def cam(t):
    return float(np.interp(t, CAM_T, CAM_C))


def cam_inverse(c):
    """wall time at which the camera reaches strip coordinate c."""
    return float(np.interp(c, CAM_C, CAM_T))


# frame readers --------------------------------------------------------------
class ClipReader:
    """Sequential rawvideo reader for one strip, opened at its source in-point."""

    def __init__(self, path, src_in, length, w, h, lead=2.0 / SRC_FPS):
        self.path = path
        self.src_in = src_in
        self.in_start = max(0.0, src_in - lead)
        self.w, self.h = w, h
        self.nbytes = w * h * 3
        self.next_idx = 0
        self.last = None
        self.log = open(path + ".film_reader.log", "w")
        cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", path, "-an",
               "-ss", f"{self.in_start:.5f}", "-t", f"{length + 1.2:.3f}"]
        vf = []
        if (w, h) != (1920, 1080):
            vf.append(f"scale={w}:{h}:flags=lanczos")
        # clone the last frame for a second: local time may land exactly on the
        # clip's end and we never want a short read at the tail
        vf.append("tpad=stop_mode=clone:stop_duration=1.0")
        cmd += ["-vf", ",".join(vf), "-r", str(SRC_FPS),
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=self.log)

    def _read_exact(self):
        buf = bytearray(self.nbytes)
        view = memoryview(buf)
        got = 0
        while got < self.nbytes:
            k = self.p.stdout.readinto(view[got:])
            if not k:
                raise RuntimeError(f"short read from {self.path} "
                                   f"(frame {self.next_idx})")
            got += k
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def frame(self, local):
        idx = int(round((self.src_in + local - self.in_start) * SRC_FPS))
        idx = max(0, idx)
        if idx < self.next_idx - 1:
            raise RuntimeError(f"non-monotone read on {self.path}: "
                               f"{idx} after {self.next_idx}")
        while self.next_idx <= idx:
            self.last = self._read_exact()
            self.next_idx += 1
        return self.last

    def close(self):
        try:
            self.p.stdout.close()
        except Exception:
            pass
        try:
            self.p.terminate()
            self.p.wait(timeout=5)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass
        self.log.close()


def active(c):
    """All strips under the lens at camera coordinate c (1 or 2, adjacent)."""
    out = []
    for i in range(len(DUR)):
        if X[i] - 1e-9 <= c <= X[i] + L[i] + 1e-9:
            out.append(i)
    if not out:
        raise RuntimeError(f"camera at {c} is not over any strip")
    return out


def wipe(a, b, p, feather=2.5):
    """Reveal b over a from the left; edge at p*W. No dissolve."""
    if p <= 0.0:
        return a
    if p >= 1.0:
        return b
    w = a.shape[1]
    e = p * (w + 2.0 * feather) - feather
    k = int(np.floor(e - feather))
    k = max(0, min(w, k))
    out = a.copy()
    if k > 0:
        out[:, :k] = b[:, :k]
    hi = min(w, int(np.ceil(e + feather)))
    if hi > k:
        xs = np.arange(k, hi, dtype=np.float32)
        alpha = np.clip((e - xs) / (2.0 * feather) + 0.5, 0.0, 1.0)
        alpha = alpha[None, :, None]
        out[:, k:hi] = (b[:, k:hi].astype(np.float32) * alpha +
                        a[:, k:hi].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return out


def _scaled(img, s):
    """Crop-zoom an image (s >= 1) about its centre, no empty borders."""
    if abs(s - 1.0) < 1e-4:
        return img
    h, w = img.shape[:2]
    im = Image.fromarray(img)
    cw, ch = w / s, h / s
    x0, y0 = (w - cw) * 0.5, (h - ch) * 0.5
    return np.asarray(im.resize((w, h), Image.BILINEAR,
                                box=(x0, y0, x0 + cw, y0 + ch)))


def push(a, b, p, feather=0.30):
    """Fly through, for the last splice.

    A plain vertical wipe into the closing card read as a hard cut — the note
    was 「没有运镜和技术，单纯就是一个黑屏」.  Here the outgoing frame keeps
    accelerating toward the viewer while the incoming one arrives from further
    in, resolved through a soft round aperture: one continuous move into the
    light, then the names are what is left behind.
    """
    if p <= 0.0:
        return a
    if p >= 1.0:
        return b
    A = _scaled(a, 1.0 + 0.30 * p * p)
    B = _scaled(b, 1.36 - 0.36 * p)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w * 0.5) / (w * 0.5), (yy - h * 0.5) / (h * 0.5))
    al = np.clip((p * 1.42 - r) / feather, 0.0, 1.0)[:, :, None]
    return (B.astype(np.float32) * al +
            A.astype(np.float32) * (1.0 - al)).astype(np.uint8)


def frame_at(readers, c, w, h):
    acts = active(c)
    i0 = acts[0]
    if readers[i0] is None:
        readers[i0] = ClipReader(CLIPS[i0], SRC_IN[i0], L[i0], w, h)
    img = readers[i0].frame(c - X[i0])
    if len(acts) == 2:
        i1 = acts[1]
        if readers[i1] is None:
            readers[i1] = ClipReader(CLIPS[i1], SRC_IN[i1], L[i1], w, h)
        p = (c - X[i1]) / OVERLAP
        inc = readers[i1].frame(c - X[i1])
        img = push(img, inc, p) if i1 == len(DUR) - 1 else wipe(img, inc, p)
    return img


def describe():
    print(f"[film] {TOTAL:.3f}s, strips={len(DUR)}, overlap={OVERLAP:.2f}s, "
          f"source {SRC_FPS}fps", flush=True)
    print("[strip]  #  name            strip_x   src_in   src_out  len   slot", flush=True)
    slot = 0.0
    for i, n in enumerate(NAMES):
        src_out = SRC_IN[i] + L[i]
        print(f"[strip] {i:2d}  {n:<14s} {X[i]:7.3f}  {SRC_IN[i]:7.2f}  "
              f"{src_out:7.2f}  {L[i]:4.2f}  {slot:5.2f}-{slot + DUR[i]:5.2f}",
              flush=True)
        slot += DUR[i]
    c = cam(0.0)
    speeds = np.diff(CAM_C) * CAM_TABLE_HZ
    print(f"[camera] c(0)={c:.4f} c(26)={cam(TOTAL):.4f} "
          f"speed min={speeds.min():.4f} max={speeds.max():.4f}", flush=True)
    for i, sx in enumerate(SPLICE_X):
        t0, t1 = cam_inverse(sx), cam_inverse(sx + OVERLAP)
        print(f"[wipe] {NAMES[i]} -> {NAMES[i + 1]}: "
              f"wall {t0:6.3f}-{t1:6.3f}s (camera {sx:.2f}-{sx + OVERLAP:.2f})",
              flush=True)


def render(mode, out, w, h, fps, crf, preset, stills_dir):
    n = int(round(TOTAL * fps))
    print(f"[render] {mode}: {w}x{h} @ {fps}fps, {n} frames, crf={crf}, "
          f"preset={preset}", flush=True)
    readers = [None] * len(DUR)
    if stills_dir:
        os.makedirs(stills_dir, exist_ok=True)
    still_times = [round(2.0 * k, 3) for k in range(int(TOTAL // 2))] + [TOTAL - 0.5]
    still_at = {int(round(t * fps)): t for t in still_times}

    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "pipe:0",
           "-ss", f"{AUDIO_IN:.3f}", "-t", f"{TOTAL:.3f}", "-i",
           r"D:/Work/music/GreatGig.mp3",
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-tune", "film",
           "-c:a", "aac", "-b:a", "256k",
           "-movflags", "+faststart", "-shortest", out]
    err = open(out + ".ffmpeg.log", "w")
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                           stdout=subprocess.DEVNULL, stderr=err)
    t_start = time.time()
    try:
        for fi in range(n):
            t = fi / fps
            c = cam(t)
            img = frame_at(readers, c, w, h)
            # release a strip as soon as the camera has left it: at most two
            # decoders are alive at any moment
            for i in range(len(DUR)):
                if readers[i] is not None and c > X[i] + L[i] + 1e-6:
                    readers[i].close()
                    readers[i] = None
            enc.stdin.write(img.tobytes())
            if fi in still_at:
                tt = still_at[fi]
                p = os.path.join(stills_dir, f"{mode}_t{tt:05.2f}.png")
                Image.fromarray(img).save(p)
                print(f"[still] {p}", flush=True)
            if fi % 30 == 0 or fi == n - 1:
                dt = time.time() - t_start
                print(f"[render] {fi + 1}/{n}  {dt:6.1f}s "
                      f"({(fi + 1) / max(dt, 1e-6):5.1f} fps)", flush=True)
    finally:
        enc.stdin.close()
        enc.wait()
        err.close()
        for r in readers:
            if r is not None:
                r.close()
    if enc.returncode != 0:
        raise SystemExit(f"ffmpeg failed, see {out}.ffmpeg.log")
    print(f"[done] {out}  ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"{time.time() - t_start:.1f}s)", flush=True)


def wait_stable(paths, quiet=4.0, timeout=240.0):
    """Do not start while an element clip is still being written.

    Every element is re-rendered by its own process, and a reader opened on a
    half-written mp4 dies with "moov atom not found" or a short read. Waiting
    for the mtimes to sit still for `quiet` seconds is cheap and has already
    saved two failed runs.
    """
    t0 = time.time()
    last = {p: os.path.getmtime(p) for p in paths}
    calm = time.time()
    while time.time() - t0 < timeout:
        time.sleep(1.0)
        cur = {p: os.path.getmtime(p) for p in paths}
        if cur != last:
            last, calm = cur, time.time()
        elif time.time() - calm >= quiet:
            print(f"[film] sources stable ({time.time() - t0:.1f}s wait)",
                  flush=True)
            return
    raise SystemExit("[film] element clips are still being written; "
                     "re-run when the renders finish")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["draft", "final"], default="final")
    ap.add_argument("--out", default="")
    ap.add_argument("--stills", default="")
    ap.add_argument("--crf", type=int, default=-1)
    ap.add_argument("--preset", default="")
    args = ap.parse_args()

    wait_stable(CLIPS)
    describe()
    if args.mode == "draft":
        w, h, fps = 960, 540, 8
        out = args.out or os.path.join(HERE, "film_26s_draft_960x540_8fps.mp4")
        crf = args.crf if args.crf >= 0 else 23
        preset = args.preset or "veryfast"
        stills = args.stills or os.path.join(HERE, "film_stills_draft")
    else:
        w, h, fps = 1920, 1080, 30
        out = args.out or os.path.join(HERE, f"film_{int(round(TOTAL))}s.mp4")
        crf = args.crf if args.crf >= 0 else 18
        preset = args.preset or "slow"
        stills = args.stills or os.path.join(HERE, "film_stills")
    render(args.mode, out, w, h, fps, crf, preset, stills)


if __name__ == "__main__":
    main()
