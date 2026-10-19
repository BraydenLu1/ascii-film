#!/usr/bin/env python3
"""Shared scaffolding for the standalone element clips.

Every element is the same film rig: a Grid of characters, a Buf that keeps the
brightest point per cell, a camera, and shade(). Only the world changes.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Grid, analyze, shade, FFMPEG  # noqa: E402

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
DUST = " .,:;'"
FIRE = " .'^*"
GLASS = " /\\|[]+=."
POOLS = {0: DUST, 1: ALNUM, 2: FIRE, 3: GLASS, 4: None}  # 4 keeps the ramp
CHARSET = "".join(sorted(set("".join(p for p in POOLS.values() if p) + " .:-=+*#%@")))
# the ramp glyphs must exist in the grid font table, or class 4 draws blanks

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUDS = os.path.join(HERE, "clouds")
REFS = os.path.join(HERE, "refs")
IMGS = r"./refs"


class Buf:
    def __init__(self, rows, cols):
        n = rows * cols
        self.rows, self.cols = rows, cols
        self.v = np.zeros(n, np.float32)
        self.r = np.zeros(n, np.float32)
        self.g = np.zeros(n, np.float32)
        self.b = np.zeros(n, np.float32)
        self.c = np.zeros(n, np.int8)

    def add(self, g, sx, sy, inten, rgb, cls):
        col = ((sx - g.ox) / g.cw).astype(np.int32)
        row = ((sy - g.oy) / g.ch).astype(np.int32)
        ok = ((col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows) &
              (inten > 0.008) & np.isfinite(inten))
        if not ok.any():
            return
        idx = row[ok] * self.cols + col[ok]
        val = inten[ok]
        win = val > self.v[idx]
        if not win.any():
            return
        idx, val = idx[win], val[win]
        self.v[idx] = val
        self.r[idx] = np.broadcast_to(rgb[0], ok.shape)[ok][win]
        self.g[idx] = np.broadcast_to(rgb[1], ok.shape)[ok][win]
        self.b[idx] = np.broadcast_to(rgb[2], ok.shape)[ok][win]
        self.c[idx] = cls


def project(pts, cam, f, W, H):
    X = pts[0] - cam[0]
    Y = pts[1] - cam[1]
    Z = pts[2] - cam[2]
    Zr = np.maximum(Z, 0.05)
    return W * 0.5 + f * X / Zr, H * 0.5 - f * Y / Zr, Zr


def compose(buf, g, t):
    rows, cols = buf.rows, buf.cols
    v = buf.v.reshape(rows, cols)
    cls = buf.c.reshape(rows, cols)
    R = buf.r.reshape(rows, cols)
    G = buf.g.reshape(rows, cols)
    B = buf.b.reshape(rows, cols)

    rng = np.random.default_rng(int(t * 6) * 7919 % 99991)
    ramp = np.array(list(" .:-=+*#%@"), dtype="U1")
    li = np.clip((v ** 0.6) * (len(ramp) - 1), 0, len(ramp) - 1).astype(int)
    chars = ramp[li]
    lit = v > 0.010
    for k, pool in POOLS.items():
        if not pool:
            continue
        m = (cls == k) & lit
        if m.any():
            chars[m] = rng.choice(list(pool), int(m.sum()))
    chars = np.where(lit, chars, " ")
    colors = np.stack([np.clip(R, 0, 255), np.clip(G, 0, 255),
                       np.clip(B, 0, 255)], axis=2)
    gain = 0.12 + 0.92 * np.clip(v, 0, 1.05)
    # class 4 keeps the ramp: its brightness is already in the glyph, so it
    # must not be dimmed again — that turned the engraving into black paper
    m = (cls != 0) & (cls != 4) & lit
    if m.any():
        colors[m] = colors[m] * gain[m][:, None]
    return chars, colors


def run(args, frame_fn, preview_times=None):
    """Analyse the audio, then either dump stills or pipe frames to ffmpeg."""
    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    g = Grid(args.w, args.h, args.font, CHARSET)
    print(f"[grid] {g.cols}x{g.rows} frames={n}", flush=True)
    cw, chh = g.cols * g.cw, g.rows * g.ch

    def feats_at(fi):
        return {k: float(feats[k][fi]) for k in feats
                if isinstance(feats[k], np.ndarray)}

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        times = preview_times or [0.02, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.97]
        for tt in times:
            fi = min(n - 1, int(tt * n))
            f = feats_at(fi)
            ch, co = frame_fn(f, fi / args.fps, g)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas, f, fi / args.fps, cw, chh,
                           np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview,
                                                   f"t{tt:.2f}.png"))
            print(f"[preview] {tt:.2f}", flush=True)
        return

    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.w}x{args.h}",
           "-r", str(args.fps), "-i", "pipe:0",
           "-ss", f"{args.start:.3f}", "-t", f"{args.dur:.3f}",
           "-i", args.audio,
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-tune", "film",
           "-c:a", "aac", "-b:a", "256k", "-shortest", args.out]
    errlog = open(args.out + ".ffmpeg.log", "w")
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=errlog)

    srng = np.random.default_rng(9)
    for fi in range(n):
        t = fi / args.fps
        f = feats_at(fi)
        ch, co = frame_fn(f, t, g)
        canvas = g.render(g.idx(ch), co)
        canvas = shade(canvas, f, t, cw, chh, srng)
        out = np.zeros((args.h, args.w, 3), np.uint8)
        out[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
        pipe.stdin.write(out.tobytes())
        if fi % 30 == 0:
            print(f"[render] {fi}/{n}", flush=True)

    pipe.stdin.close()
    pipe.wait()
    errlog.close()
    print("[done]", args.out, flush=True)


def add_args(ap, out_default):
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=58.5)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--out", default=out_default)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=12)
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--crf", type=int, default=19)
    return ap


def img_to_world(px, py, shape, scale=1.0):
    """image pixels -> world: height 1.0 across the image, x centred."""
    h, w = shape[:2]
    X = (px - w * 0.5) / h * scale
    Y = (h - py) / h * scale
    return X.astype(np.float32), Y.astype(np.float32)


def fill_points(mask, shape, n, rng, scale=1.0, jitter=0.0015):
    """every pixel of the mask becomes a point (or a uniform subset)."""
    ys, xs = np.nonzero(mask)
    if n is None or n >= len(xs):
        pick = np.arange(len(xs))
    else:
        pick = rng.choice(len(xs), n, replace=False)
    X, Y = img_to_world(xs[pick], ys[pick], shape, scale)
    return np.stack([X + rng.uniform(-0.5, 0.5, len(pick)) * jitter,
                     Y + rng.uniform(-0.5, 0.5, len(pick)) * jitter]), pick


def cloud_edge(p, nx=220, ny=320):
    x0, x1 = p[0].min(), p[0].max()
    y0, y1 = p[1].min(), p[1].max()
    ix = np.clip(((p[0] - x0) / max(x1 - x0, 1e-6) * (nx - 1)).astype(int),
                 0, nx - 1)
    iy = np.clip(((p[1] - y0) / max(y1 - y0, 1e-6) * (ny - 1)).astype(int),
                 0, ny - 1)
    m = np.zeros((ny, nx), bool)
    m[iy, ix] = True
    m = ndimage.binary_closing(m, np.ones((4, 4)))
    m = ndimage.binary_opening(m, np.ones((2, 2)))
    dist = ndimage.distance_transform_edt(m)
    e = 1.0 - dist[iy, ix] / (max(float(dist.max()), 1.0) * 0.55)
    return np.clip(e, 0, 1).astype(np.float32)
