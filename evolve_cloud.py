#!/usr/bin/env python3
"""
MARCH — five bodies of letters, one continuous take.

The March of Progress lithograph, pulled into point clouds, then played as a
single shot: the ape stands, its letters break away and flow forward, and the
body reassembles one step further up the line. Four metamorphoses, no cuts,
one camera that never stops.

Techniques borrowed from the motion-craft case law:
  - anticipation before each release (a short lean-back, then the flow)
  - staggered departure so the silhouette emerges from a swarm, not a dissolve
  - one hero technique (the morph); trails are secondary and never lead
  - camera is one function of k; slower is better than faster
  - hollow body: only the silhouette carries ink, the belly stays dark
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Grid, analyze, shade, FFMPEG  # noqa: E402

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
DUST = " .,:;'"
CLOUDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clouds")

# the four metamorphoses: start time of each, seconds
T0 = [1.10, 2.35, 3.60, 4.85]
D = 1.20            # one metamorphosis, seconds
DRIFT = 0.14        # how far the body advances per stage, world units
TRAVEL = 0.70       # the one move: world units per second, left to right

# ape -> human: the light cools as the body stands up
TINT = np.array([
    [236, 168, 112],
    [238, 190, 146],
    [232, 210, 176],
    [224, 224, 206],
    [206, 230, 236],
], np.float32)


def load_cloud(i, n=7000, rng=None):
    d = np.load(os.path.join(CLOUDS, f"march{i}.npz"))
    p = d["p"].astype(np.float32).copy()
    if p.shape[0] == 2:
        p = np.vstack([p, np.zeros((1, p.shape[1]), np.float32)])
    if rng is not None:
        p[2] += rng.normal(0, 0.010, p.shape[1])
    return p


def cloud_edge(p, nx=200, ny=300):
    """Distance-to-silhouette, computed from the cloud itself.

    The hand clouds carry an `edge` channel because they were traced with an
    edt pass; these bodies never were, so we rebuild the silhouette from the
    points and walk the distance transform back out to each point.
    """
    x0, x1 = p[0].min(), p[0].max()
    y0, y1 = p[1].min(), p[1].max()
    ix = np.clip(((p[0] - x0) / max(x1 - x0, 1e-6) * (nx - 1)).astype(int),
                 0, nx - 1)
    iy = np.clip(((p[1] - y0) / max(y1 - y0, 1e-6) * (ny - 1)).astype(int),
                 0, ny - 1)
    m = np.zeros((ny, nx), bool)
    m[iy, ix] = True
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    m = ndimage.binary_opening(m, np.ones((2, 2)))
    dist = ndimage.distance_transform_edt(m)
    dmax = max(float(dist.max()), 1.0)
    e = np.clip(1.0 - dist[iy, ix] / (dmax * 0.55), 0.0, 1.0)
    return e.astype(np.float32)


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


def compose(buf, g, t, gain_lo=0.12):
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
    m = (cls == 1) & lit
    if m.any():
        chars[m] = rng.choice(list(ALNUM), int(m.sum()))
    m = (cls == 0) & lit
    if m.any():
        chars[m] = rng.choice(list(DUST), int(m.sum()))
    m = (cls == 2) & lit
    if m.any():
        chars[m] = rng.choice(list(" .:*#@+"), int(m.sum()))
    chars = np.where(lit, chars, " ")
    colors = np.stack([np.clip(R, 0, 255), np.clip(G, 0, 255),
                       np.clip(B, 0, 255)], axis=2)
    # brightness follows the field; without this every lit cell is a white
    # letter and the hollow body fills back in
    gain = gain_lo + 0.92 * np.clip(v, 0, 1.05)
    m = (cls == 1) | (cls == 2)
    m &= lit
    if m.any():
        colors[m] = colors[m] * gain[m][:, None]
    return chars, colors


def smoothstep(a):
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=61.0)
    ap.add_argument("--dur", type=float, default=7.5)
    ap.add_argument("--out", default="march_evolve.mp4")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=12)
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--crf", type=int, default=19)
    args = ap.parse_args()

    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    g = Grid(args.w, args.h, args.font, "".join(sorted(set(ALNUM + DUST + " .,:*#@+"))))
    print(f"[grid] {g.cols}x{g.rows} frames={n}", file=sys.stderr, flush=True)

    rng = np.random.default_rng(11)
    clouds = [load_cloud(i, rng=rng) for i in range(5)]
    edges = [cloud_edge(c) for c in clouds]

    # match every point of stage i to its nearest neighbour in stage i+1
    match = []
    for i in range(4):
        tr = cKDTree(clouds[i + 1][:2].T)
        _, j = tr.query(clouds[i][:2].T)
        match.append(j.astype(np.int32))

    # stable per-point phase, keyed to the point's index (deterministic)
    ph = np.random.default_rng(23).random(7000).astype(np.float32)

    dust = []
    for zlo, zhi, cnt, gain, streak, cls in (
            (-1.05, -0.80, 12000, 1.20, 10, 2),
            (0.00, 0.90, 3000, 0.90, 4, 0),
            (1.20, 4.00, 5000, 0.95, 0, 0)):
        dp = np.stack([rng.uniform(-3.0, 10.0, cnt),
                       rng.uniform(-0.10, 1.30, cnt),
                       rng.uniform(zlo, zhi, cnt)])
        dust.append((dp, gain, streak, cls))
    # a floor of dust: the eye needs a line to measure the move against
    gnd = np.stack([rng.uniform(-3.0, 10.0, 3200),
                    rng.uniform(-0.030, 0.004, 3200),
                    rng.uniform(0.00, 1.70, 3200)])

    def state(t):
        """-> (stage, raw progress 0..1). raw 0 means a rest pose."""
        if t < T0[0]:
            return 0, 0.0
        for i, t0 in enumerate(T0):
            if t0 <= t < t0 + D:
                return i, (t - t0) / D
        stage = 0
        for i, t0 in enumerate(T0):
            if t >= t0 + D:
                stage = i + 1
        return stage, 0.0

    def frame(f, t):
        V = f["voice"]
        k = t / max(1e-6, args.dur)
        si, raw = state(t)
        if raw > 0.0:
            j = match[si]
            p0, p1 = clouds[si], clouds[si + 1][:, j]
            e0, e1 = edges[si], edges[si + 1][j]
            yn = (p0[1] - p0[1].min()) / max(p0[1].max() - p0[1].min(), 1e-6)
            # the change sweeps up the body: feet first, the head last, so
            # the old face is the last thing to go — a wipe, not a pop
            delay = 0.55 * (1.0 - yn) + 0.18 * ph
            pr = smoothstep((raw - delay) / np.maximum(1.0 - delay, 1e-3))
            pts = p0 + (p1 - p0) * pr
            ed = e0 + (e1 - e0) * pr
            tint = TINT[si] + (TINT[si + 1] - TINT[si]) * pr[:, None]
            # anticipation: a short lean back before the swarm commits
            # (cosine pulse: starts from zero speed, so the body does not jolt)
            if raw < 0.16:
                wind = 0.5 * (1.0 - math.cos(2.0 * math.pi * raw / 0.16))
            else:
                wind = 0.0
            pts[0] -= 0.015 * wind * (1.0 - pr)
            trail = (pr > 0.02) & (pr < 0.98)
        else:
            pts = clouds[si].copy()
            ed = edges[si]
            tint = np.broadcast_to(TINT[si], (pts.shape[1], 3)).copy()
            pr = np.zeros(pts.shape[1], np.float32)
            trail = np.zeros(pts.shape[1], bool)

        # the body advances a little with every stage, so the camera has
        # something to chase and the dust has something to slide against
        ax = DRIFT * (si + (float(pr.mean()) if raw > 0 else 0.0))
        travel = TRAVEL * t
        pts[0] += ax + travel

        # the camera reveals: tight on the hunched ape, then it draws back
        # and lifts as the body stands, so the last figure has air above it
        sp = (si + float(pr.mean())) / 4.0
        zoom = 0.80 - 0.20 * float(smoothstep(np.array(sp)))
        cx = travel + ax + 0.10 + 0.10 * k
        cy = 0.48 + 0.04 * float(smoothstep(np.array(k)))
        cz = -1.30
        cam = np.array([cx, cy, cz])
        flen = args.w * 0.92 * zoom
        buf = Buf(g.rows, g.cols)

        for dp, gain, streak, cls in dust:
            sx, sy, z = project(dp, cam, flen, args.w, args.h)
            fog = np.exp(-z * 0.9)
            inten = (0.30 + 0.35 * V) * fog * gain
            col = (50 + 50 * fog, 68 + 62 * fog, 100 + 72 * fog)
            buf.add(g, sx, sy, inten, col, cls)
            if streak:
                # the near layer is dragged by the move: a letter, then its
                # own tail — this is where the speed is read from
                step = flen * (TRAVEL / args.fps) / np.maximum(z, 0.05)
                for q in range(1, streak + 1):
                    frac = q / streak
                    buf.add(g, sx + frac * step, sy,
                            inten * (1.0 - 0.78 * frac), col, cls)

        sx, sy, z = project(gnd, cam, flen, args.w, args.h)
        fog = np.exp(-z * 1.1)
        gint = 0.30 * (0.7 + 0.3 * V) * fog
        gcol = (52 + 40 * fog, 70 + 50 * fog, 98 + 60 * fog)
        buf.add(g, sx, sy, gint, gcol, 0)
        gstep = flen * (TRAVEL / args.fps) / np.maximum(z, 0.05)
        for q in range(1, 6):
            frac = q / 5
            buf.add(g, sx + frac * gstep, sy, gint * (1.0 - 0.78 * frac),
                    gcol, 0)

        # the ancestor lingers as a ghost while the new body takes the
        # light: the eye never loses the old silhouette
        if raw > 0.0:
            gsx, gsy, _ = project(p0 + ax + travel, cam, flen, args.w, args.h)
            # fade the ghost in and out with the change: it must never
            # appear in a single frame, or the cut reads as a cut
            ghost = 0.22 * math.sin(math.pi * raw)
            gint = np.full(gsx.shape, ghost * (0.4 + 0.6 * V), np.float32)
            buf.add(g, gsx, gsy, gint, (110.0, 132.0, 152.0), 0)

        # the ghost trail: where the letter was a beat ago
        if trail.any():
            pa = np.maximum(pr - 0.13, 0.0)
            gp = p0 + (p1 - p0) * pa
            gp[0] += ax
            sx, sy, z = project(gp[:, trail], cam, flen, args.w, args.h)
            buf.add(g, sx, sy, 0.22 * (1.0 - pa[trail]) * (0.5 + 0.5 * V),
                    (150.0, 165.0, 180.0), 0)

        sx, sy, z = project(pts, cam, flen, args.w, args.h)
        rim = np.clip((ed - 0.42) / 0.38, 0, 1) ** 1.5
        inten = (0.02 + 1.45 * rim) * (0.84 + 0.28 * (1.0 - ed)) * \
            (0.80 + 0.30 * V)
        fog = np.exp(-(z - 0.0) * 0.30)
        buf.add(g, sx, sy, inten * fog, (tint[:, 0], tint[:, 1], tint[:, 2]), 1)

        return compose(buf, g, t)

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        for tt in [0.04, 0.12, 0.19, 0.26, 0.34, 0.42, 0.50, 0.58,
                   0.66, 0.74, 0.82, 0.90, 0.97]:
            fi = min(n - 1, int(tt * n))
            f = {k: float(feats[k][fi]) for k in feats
                 if isinstance(feats[k], np.ndarray)}
            ch, co = frame(f, fi / args.fps)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas, f, fi / args.fps, g.cols * g.cw,
                           g.rows * g.ch, np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + g.rows * g.ch, g.ox:g.ox + g.cols * g.cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview, f"t{tt:.2f}.png"))
            print(f"[preview] {tt:.2f}", file=sys.stderr, flush=True)
        return

    cw, chh = g.cols * g.cw, g.rows * g.ch
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
        f = {k: float(feats[k][fi]) for k in feats
             if isinstance(feats[k], np.ndarray)}
        ch, co = frame(f, t)
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


if __name__ == "__main__":
    main()
