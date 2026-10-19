#!/usr/bin/env python3
"""
MARCH LINE — the Darwin lineup, five bodies in one row.

The reference is a single picture: five figures walking right, ape to human.
So they are all built into one world, in a row, and the camera does the only
thing that picture asks for — it tracks left to right, stops being wide to
take a close look at each body, and lets that body come apart into dust as it
moves on to the next.

Camera:
  x(t) is the integral of a speed that dips at every body (a dolly that
  lingers), zoom is a function of the same x, so the close-up lands exactly
  when the camera is on the body. One curve, five breaths, never a stop.

Techniques: depth-layer-moves (three parallax strata, near layer smeared by
the actual local speed), crane-rise-reveal (one manoeuvre, one progress),
disney anticipation/follow-through (the dissolve starts behind the move).
"""
import argparse
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
CLOUDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clouds")

XS = np.array([0.0, 0.95, 1.90, 2.85, 3.80])   # the five bodies, world x
X_START, X_END = -0.70, 3.88
Z_CAM = -1.35
ZOOM_LO = 0.40
# close-up framing is per body: fill ~0.75 of the frame height whatever the
# body's real height is, so the ape gets a real close-up (zoom 1.5) and the
# modern human a wide one (zoom 0.75) instead of one fixed 0.78 for all five
FILL = 0.75
CY_FILL = 0.52          # camera eye sits a touch above the body's mid-line
CY_WIDE0, CY_WIDE1 = 0.30, 0.52   # ...and rises across the row as they grow
# the lineup must climb, small to large. The lithograph's fourth figure is
# drawn shorter than the third, which breaks the reading, so each body is
# scaled about its feet onto a monotonic ladder.
H_TARGET = np.array([0.50, 0.66, 0.78, 0.89, 1.00])

TINT = np.array([
    [255, 202, 150],
    [255, 214, 170],
    [250, 230, 200],
    [238, 236, 220],
    [220, 240, 246],
], np.float32)


def load_cloud(i, rng=None):
    """-> (xyz points, thin flag, ink density) per point.

    `thin` marks strokes under 3.5 px (spears, clubs), which otherwise draw as
    a hard horizontal line. `ink` is how dark the source pixel was, used as a
    gentle relief so the filled body still reads as a drawing, not a sticker.
    """
    d = np.load(os.path.join(CLOUDS, f"march{i}.npz"))
    p = d["p"].astype(np.float32).copy()
    if p.shape[0] == 2:
        p = np.vstack([p, np.zeros((1, p.shape[1]), np.float32)])
    n = p.shape[1]
    thin = (d["thin"].astype(np.float32) if "thin" in d
            else np.zeros(n, np.float32))
    ink = (d["ink"].astype(np.float32) if "ink" in d
           else np.full(n, 0.8, np.float32))
    if rng is not None:
        p[2] += rng.normal(0, 0.010, n)
    return p, thin, ink


def cloud_edge(p, nx=220, ny=320):
    """Distance-to-silhouette, rebuilt from the points themselves."""
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
    gain = 0.12 + 0.92 * np.clip(v, 0, 1.05)
    m = ((cls == 1) | (cls == 2)) & lit
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
    ap.add_argument("--dur", type=float, default=11.0)
    ap.add_argument("--out", default="march_line.mp4")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=12)
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--crf", type=int, default=19)
    args = ap.parse_args()

    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    g = Grid(args.w, args.h, args.font,
             "".join(sorted(set(ALNUM + DUST + " .,:*#@+"))))
    print(f"[grid] {g.cols}x{g.rows} frames={n}", file=sys.stderr, flush=True)

    rng = np.random.default_rng(11)
    loaded = [load_cloud(i, rng=rng) for i in range(5)]
    clouds = [c for c, _, _ in loaded]
    thins = [th for _, th, _ in loaded]
    inks = [k for _, _, k in loaded]
    edges = [cloud_edge(c) for c in clouds]
    # put every body on the climbing ladder, scaled about its feet (y=0)
    measured = np.array([float(c[1].max() - c[1].min()) for c in clouds])
    for c, h, t in zip(clouds, measured, H_TARGET):
        c *= (t / max(h, 0.05))
    # real relative heights measured off the clouds: 0.50 / 0.71 / 0.84 /
    # 0.74 / 1.00. Framing follows them instead of a fixed close-up zoom.
    hrel = np.array([float(c[1].max() - c[1].min()) for c in clouds])
    zoom_st = FILL / np.maximum(hrel, 0.05)
    cy_st = CY_FILL * hrel
    print("[bodies] measured", np.round(measured, 3).tolist(),
          "-> H", np.round(hrel, 3).tolist(),
          "zoom", np.round(zoom_st, 2).tolist(),
          "cy", np.round(cy_st, 2).tolist(), file=sys.stderr, flush=True)
    # per-point scatter directions for the dissolve (one array per body: the
    # clouds no longer share a point count)
    drng = np.random.default_rng(37)
    scat = [np.stack([drng.uniform(-0.55, 1.0, c.shape[1]),
                      drng.uniform(0.25, 1.05, c.shape[1]),
                      drng.normal(0, 1.0, c.shape[1])]).astype(np.float32)
            for c in clouds]

    # ---- the camera as one function of x ------------------------------
    xg = np.linspace(X_START, X_END, 8001)
    slow = np.zeros_like(xg)
    for xc in XS:
        slow += np.exp(-((xg - xc) / 0.30) ** 2)
    # speed dips at every body: the dolly lingers, it never halts
    speed = 1.0 - 0.64 * np.clip(slow, 0, 1)
    tg = np.cumsum(1.0 / speed)
    tg = (tg - tg[0]) / (tg[-1] - tg[0])

    def x_of_t(t):
        return float(np.interp(min(max(t / args.dur, 0.0), 1.0), tg, xg))

    def wide_cy(x):
        """eye level when nobody is being looked at: slowly climbs the row"""
        u = min(max((x - X_START) / (X_END - X_START), 0.0), 1.0)
        return CY_WIDE0 + (CY_WIDE1 - CY_WIDE0) * u

    def cam_of_x(x):
        """zoom and camera height from the same five bumps — still x alone.

        `b` is how close the nearest body is; at b=1 the framing is that
        body's own close-up, at b=0 it is the wide tracking shot.
        """
        w = np.exp(-((x - XS) / 0.19) ** 2)
        tot = float(w.sum())
        b = min(tot, 1.0)
        zc = float((w * zoom_st).sum() / tot)
        cc = float((w * cy_st).sum() / tot)
        return ((1.0 - b) * ZOOM_LO + b * zc,
                (1.0 - b) * wide_cy(x) + b * cc, b)

    # where (in seconds) each body gets its close-up
    station_t = [float(np.interp(xc, xg, tg)) * args.dur for xc in XS]
    print("[stations]", [round(s, 2) for s in station_t], file=sys.stderr,
          flush=True)

    # ---- dust: three parallax strata + a floor -----------------------
    dust = []
    for zlo, zhi, cnt, gain, streak, cls in (
            (-1.05, -0.80, 12000, 1.20, 10, 2),
            (0.00, 0.90, 3000, 0.90, 4, 0),
            (1.20, 4.00, 5000, 0.95, 0, 0)):
        dp = np.stack([rng.uniform(-4.0, 12.0, cnt),
                       rng.uniform(-0.10, 1.30, cnt),
                       rng.uniform(zlo, zhi, cnt)])
        dust.append((dp, gain, streak, cls))
    gnd = np.stack([rng.uniform(-4.0, 12.0, 3600),
                    rng.uniform(-0.030, 0.004, 3600),
                    rng.uniform(0.00, 1.70, 3600)])

    def frame(f, t):
        V = f["voice"]
        cx = x_of_t(t)
        zoom, cy, b = cam_of_x(cx)
        cam = np.array([cx, cy, Z_CAM])
        flen = args.w * 0.92 * zoom
        vloc = max(0.05, (x_of_t(min(t + 0.05, args.dur)) - cx) / 0.05)
        buf = Buf(g.rows, g.cols)

        # dust
        for dp, gain, streak, cls in dust:
            sx, sy, z = project(dp, cam, flen, args.w, args.h)
            fog = np.exp(-z * 0.9)
            inten = (0.30 + 0.35 * V) * fog * gain
            col = (50 + 50 * fog, 68 + 62 * fog, 100 + 72 * fog)
            buf.add(g, sx, sy, inten, col, cls)
            if streak:
                step = flen * vloc / args.fps / np.maximum(z, 0.05)
                for q in range(1, streak + 1):
                    frac = q / streak
                    buf.add(g, sx + frac * step, sy,
                            inten * (1.0 - 0.78 * frac), col, cls)

        sx, sy, z = project(gnd, cam, flen, args.w, args.h)
        fog = np.exp(-z * 1.1)
        gint = 0.30 * (0.7 + 0.3 * V) * fog
        gcol = (52 + 40 * fog, 70 + 50 * fog, 98 + 60 * fog)
        buf.add(g, sx, sy, gint, gcol, 0)
        gstep = flen * vloc / args.fps / np.maximum(z, 0.05)
        for q in range(1, 6):
            frac = q / 5
            buf.add(g, sx + frac * gstep, sy, gint * (1.0 - 0.78 * frac),
                    gcol, 0)

        # the five bodies; each one comes apart after the camera has looked
        for i, (p0, ed, th, ik) in enumerate(zip(clouds, edges, thins, inks)):
            xc = XS[i]
            if i < 4:
                d = min(max((cx - (xc + 0.03)) / 0.55, 0.0), 1.0)
            else:
                d = 0.0
            if d >= 1.0:
                continue
            q = float(smoothstep(np.array(d)))
            pts = p0.copy()
            pts[0] += xc
            if q > 0.0:
                pts[0] += q * q * (0.42 + 0.10 * scat[i][0])
                pts[1] += q * (0.14 + 0.10 * scat[i][1])
                pts[2] += q * 0.07 * scat[i][2]
            sx, sy, z = project(pts, cam, flen, args.w, args.h)
            rim = np.clip((ed - 0.05) / 0.75, 0, 1) ** 1.0
            # solid body: every keyed-out pixel is a point, so the mass reads
            # filled. rim keeps the silhouette crisp, ink keeps the drawing's
            # internal relief — neither of them punches holes in the middle.
            inten = (0.70 + 0.34 * rim) * (0.72 + 0.44 * ik) * \
                (0.86 + 0.22 * V) * ((1.0 - q) ** 1.35)
            # fine strokes (spears, clubs) go ghostly instead of reading as a
            # dead horizontal line across the close-up
            inten = inten * (1.0 - 0.75 * th)
            col = TINT[i]
            # a body half out of frame reads as a column, not a person:
            # fade it out before its centre reaches the edge
            sx0 = args.w * 0.5 + flen * (xc - cx) / 1.35
            u = abs(sx0 - args.w * 0.5) / (args.w * 0.5)
            vis = min(max((1.0 - u) / 0.18, 0.0), 1.0)
            buf.add(g, sx, sy, inten * vis, (col[0], col[1], col[2]), 1)

        return compose(buf, g, t)

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        times = [0.02, 0.10, 0.18] + [s / args.dur for s in station_t] + \
                [0.66, 0.82, 0.94]
        for tt in sorted(set(round(v, 3) for v in times)):
            fi = min(n - 1, int(tt * n))
            f = {k: float(feats[k][fi]) for k in feats
                 if isinstance(feats[k], np.ndarray)}
            ch, co = frame(f, fi / args.fps)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas, f, fi / args.fps, g.cols * g.cw,
                           g.rows * g.ch, np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + g.rows * g.ch, g.ox:g.ox + g.cols * g.cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview, f"t{tt:.3f}.png"))
            print(f"[preview] {tt:.3f}", file=sys.stderr, flush=True)
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
