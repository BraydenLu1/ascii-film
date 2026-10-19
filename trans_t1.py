#!/usr/bin/env python3
"""
T1 · 尘成臂 — 进化 → 造物, 5 s transition demo.

The last walker comes apart and is not allowed to settle. The same wind that
moves the camera carries the dust right, and it lands on two arms — God's
from the left, Adam's from the right — building them out of its own body.
The lens opens on the half-inch between two fingers.

One camera, one curve, no cuts. The transition is a material swap held
together by a carrier: dust leaves the man, dust becomes the arm.
"""
import argparse
import math
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
HERE = os.path.dirname(os.path.abspath(__file__))
CLOUDS = os.path.join(HERE, "clouds")
REFS = os.path.join(HERE, "refs")

# ---- world ---------------------------------------------------------------
X_HUMAN = -0.50          # where the last walker stands
GAP_X, GAP_Y = 0.80, 0.768    # the half-inch between two fingers
X_START, X_END = -1.05, 0.80
Z_CAM = -1.00
ZOOM_LO, ZOOM_HI = 0.55, 1.55
CY_LO, CY_HI = 0.52, 0.78

# ---- beats (seconds) -----------------------------------------------------
T_MORPH = (1.10, 3.55)     # the body is eaten by the wind, the arms grow
T_SPARK = 4.30

TINT_HUMAN = np.array([220, 240, 246], np.float32)
TINT_ARM = np.array([232, 218, 196], np.float32)


# --------------------------------------------------------------------------
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


def load_body(n, rng):
    """The modern human, resampled to n particles (the morph needs a count)."""
    d = np.load(os.path.join(CLOUDS, "march4.npz"))
    p = d["p"].astype(np.float32)
    thin = d["thin"].astype(np.float32)
    ink = d["ink"].astype(np.float32)
    src = p.shape[1]
    idx = rng.choice(src, n, replace=n > src)
    q = np.stack([p[0][idx] + X_HUMAN, p[1][idx],
                  rng.normal(0, 0.008, n)]).astype(np.float32)
    return q, thin[idx], ink[idx]


def load_arms(n, rng):
    """Both arms, sampled from the keyed mask: every pixel is a point."""
    mask = np.array(Image.open(os.path.join(REFS, "hands_mask.png"))
                    .convert("L")) > 128
    lab, nl = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, nl + 1))
    two = sorted(list(np.argsort(sizes)[-2:] + 1),
                 key=lambda k: np.nonzero(lab == k)[1].mean())
    lum = np.array(Image.open(os.path.join(
        REFS, "Creation_of_Adam__Michelangelo__Detail.jpg")).convert("L"),
        np.float32)
    xs_all, ys_all, ink_all = [], [], []
    for k, side in zip(two, ("l", "r")):
        ys, xs = np.nonzero(lab == k)
        ref = np.load(os.path.join(CLOUDS, f"hand_{side}.npz"))["p"]
        x0, x1 = float(ref[0].min()), float(ref[0].max())
        y0, y1 = float(ref[1].min()), float(ref[1].max())
        X = x0 + (xs - xs.min()) / max(xs.max() - xs.min(), 1) * (x1 - x0)
        Y = y0 + (ys.max() - ys) / max(ys.max() - ys.min(), 1) * (y1 - y0)
        # ink relief from the fresco itself, normalised inside each arm
        L = lum[ys, xs]
        ink = (L.max() - L) / max(float(L.max() - L.min()), 1e-6)
        xs_all.append(X)
        ys_all.append(Y)
        ink_all.append(ink)
    X = np.concatenate(xs_all)
    Y = np.concatenate(ys_all)
    ink = np.concatenate(ink_all)
    pick = rng.choice(len(X), n, replace=n > len(X))
    p = np.stack([X[pick] + rng.uniform(-0.5, 0.5, n) * (1.0 / 549.0),
                  Y[pick] + rng.uniform(-0.5, 0.5, n) * (1.0 / 549.0),
                  rng.normal(0, 0.012, n)]).astype(np.float32)
    return p, ink[pick].astype(np.float32)


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


def ease(a):
    a = np.clip(a, 0.0, 1.0)
    return float(a * a * (3.0 - 2.0 * a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=62.0)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--out", default="trans_t1.mp4")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=12)
    ap.add_argument("--pts", type=int, default=46000)
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--crf", type=int, default=19)
    args = ap.parse_args()

    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    g = Grid(args.w, args.h, args.font,
             "".join(sorted(set(ALNUM + DUST + " .,:*#@+"))))
    print(f"[grid] {g.cols}x{g.rows} frames={n}", flush=True)

    rng = np.random.default_rng(21)
    P0, thin0, ink0 = load_body(args.pts, rng)
    P1, ink1 = load_arms(args.pts, rng)
    thin1 = np.zeros(args.pts, np.float32)
    rim0 = cloud_edge(P0)
    rim1 = cloud_edge(P1)
    print(f"[morph] {args.pts} particles  arm x[{P1[0].min():.2f},"
          f"{P1[0].max():.2f}] y[{P1[1].min():.2f},{P1[1].max():.2f}]",
          flush=True)

    # break-up directions: everything is carried right and up
    scat = np.stack([rng.uniform(0.00, 0.55, args.pts),
                     rng.uniform(0.10, 0.55, args.pts),
                     rng.normal(0, 1.0, args.pts)]).astype(np.float32)
    # each particle leaves on its own clock, ordered by where it lands: the
    # left end of God's arm is laid first, the far end of Adam's last. A
    # common clock would put 46k letters in the air at once — a white wall.
    rank = (P1[0] - P1[0].min()) / max(P1[0].max() - P1[0].min(), 1e-6)
    delay = (0.58 * rank + rng.uniform(0, 0.10, args.pts)).astype(np.float32)
    delay = np.clip(delay, 0.0, 0.62).astype(np.float32)

    # ---- camera: one move, hold then push --------------------------------
    # no integral-of-zoom trick here: the shot needs the lens to sit still
    # while the dust crosses, and to spend its speed in the last two seconds
    def ease01(u):
        u = min(max(float(u), 0.0), 1.0)
        return u * u * (3.0 - 2.0 * u)

    def cam_at(k):
        x = X_START + (X_END - X_START) * ease01((k - 0.45) / 0.41)
        z = ZOOM_LO + (ZOOM_HI - ZOOM_LO) * ease01((k - 0.62) / 0.30)
        return x, z

    # ---- dust: world-fixed, three strata --------------------------------
    dust = []
    for zlo, zhi, cnt, gain, streak, cls in (
            (-1.05, -0.80, 9000, 1.10, 9, 2),
            (0.00, 0.90, 2400, 0.85, 4, 0),
            (1.20, 4.00, 3600, 0.90, 0, 0)):
        dp = np.stack([rng.uniform(-4.0, 8.0, cnt),
                       rng.uniform(-0.10, 1.30, cnt),
                       rng.uniform(zlo, zhi, cnt)])
        dust.append((dp, gain, streak, cls))

    # the spark: a fixed constellation, it glows, it does not reshuffle
    SP_N = 4200
    sp_a = rng.random(SP_N) * 2 * math.pi
    sp_r = rng.random(SP_N) ** 2.4 * 0.040
    sp_z = rng.normal(0, 0.004, SP_N)

    def frame(f, t):
        V = f["voice"]
        k = t / max(1e-6, args.dur)
        a = ease((t - T_MORPH[0]) / (T_MORPH[1] - T_MORPH[0]))
        # per-particle progress: leaves on its own clock, lands where it belongs
        pg = smoothstep(np.clip((a - delay) / (1.0 - delay), 0.0, 1.0))

        cx, zoom = cam_at(k)
        cy = CY_LO + (CY_HI - CY_LO) * a
        cam = np.array([cx, cy, Z_CAM])
        flen = args.w * 0.92 * zoom
        vloc = max(0.05, (cam_at(min(k + 0.01, 1.0))[0] - cx) / (0.01 * args.dur))
        buf = Buf(g.rows, g.cols)

        # dust
        for dp, gain, streak, cls in dust:
            sx, sy, z = project(dp, cam, flen, args.w, args.h)
            fog = np.exp(-z * 0.9)
            inten = (0.26 + 0.30 * V) * fog * gain
            col = (44 + 48 * fog, 62 + 58 * fog, 96 + 66 * fog)
            buf.add(g, sx, sy, inten, col, cls)
            if streak:
                step = flen * vloc / args.fps / np.maximum(z, 0.05)
                for q in range(1, streak + 1):
                    frac = q / streak
                    buf.add(g, sx + frac * step, sy,
                            inten * (1.0 - 0.78 * frac), col, cls)

        # ---- the transition itself ---------------------------------------
        # the body is eaten by the wind and re-laid as the two arms; each
        # particle flies its own leg, with a puff that peaks mid-flight
        pos = P0 + (P1 - P0) * pg + scat * (0.12 * np.sin(np.pi * pg))

        rim = (1.0 - pg) * rim0 + pg * rim1
        ink = (1.0 - pg) * ink0 + pg * ink1
        th = (1.0 - pg) * thin0
        tint = ((1.0 - pg)[None, :] * TINT_HUMAN[:, None] +
                pg[None, :] * TINT_ARM[:, None])

        inten = (0.70 + 0.34 * rim) * (0.72 + 0.44 * ink) * \
            (0.84 + 0.24 * V)
        # in flight they are dust, not flesh; on both ends they are solid
        inten = inten * (1.0 - 0.85 * np.clip(np.sin(np.pi * pg), 0, None) ** 0.7)
        inten = inten * (1.0 - 0.75 * th)
        sx, sy, z = project(pos, cam, flen, args.w, args.h)
        if os.environ.get("T1DBG"):
            print(f"[dbg] t={t:.2f} k={k:.2f} cam={cx:.2f},{cy:.2f} zoom={zoom:.2f} "
                  f"pos x[{pos[0].min():.2f},{pos[0].max():.2f}] "
                  f"sx[{sx.min():.0f},{sx.max():.0f}] sy[{sy.min():.0f},{sy.max():.0f}] "
                  f"inten[{inten.min():.2f},{inten.max():.2f}] pg[{pg.min():.2f},{pg.max():.2f}]",
                  file=sys.stderr, flush=True)

        # the paint catches the light where the two fingers meet
        near = np.exp(-(((pos[0] - GAP_X) ** 2 +
                         (pos[1] - GAP_Y) ** 2) / 0.010))
        d = abs(t - T_SPARK)
        spark = math.exp(-(d / 0.38) ** 2)
        inten = inten * (1.0 + 1.4 * spark * near)
        buf.add(g, sx, sy, inten, (tint[0], tint[1], tint[2]), 1)

        # the spark: light in the half-inch between the fingers
        if spark > 0.01:
            ang = sp_a + 0.35 * t
            rr = sp_r * (0.55 + 0.45 * spark)
            sp = np.stack([GAP_X + np.cos(ang) * rr,
                           GAP_Y + np.sin(ang) * rr * 0.9, sp_z])
            sx2, sy2, _ = project(sp, cam, flen, args.w, args.h)
            buf.add(g, sx2, sy2, 2.4 * spark *
                    np.clip(1 - rr / 0.045, 0, 1) ** 1.2,
                    (255.0, 248.0, 232.0), 2)

        return compose(buf, g, t)

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        for tt in [0.02, 0.12, 0.24, 0.38, 0.52, 0.64, 0.74, 0.84, 0.92]:
            fi = min(n - 1, int(tt * n))
            f = {kk: float(feats[kk][fi]) for kk in feats
                 if isinstance(feats[kk], np.ndarray)}
            ch, co = frame(f, fi / args.fps)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas, f, fi / args.fps, g.cols * g.cw,
                           g.rows * g.ch, np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + g.rows * g.ch, g.ox:g.ox + g.cols * g.cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview,
                                                   f"t{tt:.2f}.png"))
            print(f"[preview] {tt:.2f}", flush=True)
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
        f = {kk: float(feats[kk][fi]) for kk in feats
             if isinstance(feats[kk], np.ndarray)}
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
