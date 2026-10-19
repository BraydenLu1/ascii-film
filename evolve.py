#!/usr/bin/env python3
"""
EVOLVE — a single lateral take through the March of Progress.

The camera starts on the shoulder of a knuckle-walking ape, slides to the
next one, and pulls back as that one stands. Then the humans just walk.

Bodies are volumetric point clouds: limbs are cylinders with radius, the head
is an ellipsoid, the torso is a swept ellipse along a curved spine. That is
what gives them weight — a body has a front and a back, and when the camera
moves you feel it turn.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Grid, analyze, shade, FFMPEG  # noqa: E402

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
DUST = " .,:;'"


# --------------------------------------------------------------------------
# sampling primitives — volume, not surface
# --------------------------------------------------------------------------
def cyl(p0, p1, r0, r1, n, rng, sq=1.15):
    """Solid tapered cylinder between two points, elliptical cross-section."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    d = p1 - p0
    L = float(np.hypot(d[0], d[1])) + 1e-9
    nx, ny = -d[1] / L, d[0] / L
    t = rng.random(n)
    ang = rng.random(n) * 2 * math.pi
    rad = np.sqrt(rng.random(n))
    r = (r0 + (r1 - r0) * t) * rad
    # cross-section in x/y, plus depth so the body has a front and a back
    x = p0[0] + d[0] * t + nx * r * np.cos(ang)
    y = p0[1] + d[1] * t + ny * r * np.cos(ang)
    z = r * np.sin(ang) * sq
    return np.stack([x, y, z])


def ell(cx, cy, rx, ry, rz, n, rng):
    v = rng.normal(0, 1, (3, n))
    v /= np.linalg.norm(v, axis=0) + 1e-9
    s = rng.random(n) ** (1 / 3)
    return np.stack([cx + v[0] * s * rx, cy + v[1] * s * ry, v[2] * s * rz])


def bez(pts, n):
    """Quadratic-ish bezier through a list of 3 control points, sampled."""
    p = np.array(pts, np.float64)
    t = np.linspace(0, 1, n)[None, :]
    if len(p) == 3:
        return ((1 - t) ** 2) * p[0, :, None] + 2 * (1 - t) * t * p[1, :, None] + \
            (t ** 2) * p[2, :, None]
    return p.T


# --------------------------------------------------------------------------
# the body
# --------------------------------------------------------------------------
def figure(x, stage, phase, rng, scale=1.0):
    """stage 0 = knuckle-walking ape, 1 = upright human, facing +x."""
    u = float(np.clip(stage, 0, 1))
    hp = (0.56 + 0.48 * u) * scale                 # pelvis height
    th = math.radians(78 - 74 * u)                 # torso lean from vertical
    Lt = (0.58 + 0.14 * u) * scale                 # torso length
    neckw = np.array([x + math.sin(th) * Lt, hp + math.cos(th) * Lt])
    # the muzzle juts forward more the less evolved you are
    jut = (0.11 * (1 - u) + 0.02) * scale

    P, I = [], []

    def add(p, w=1.0):
        P.append(p)
        I.append(np.full(p.shape[1], w))

    # ---- spine: a curve, never a straight line -------------------------
    pelvis = np.array([x, hp])
    mid = (pelvis + neckw) / 2 + np.array([-0.06 * (1 - u) * scale, 0.03 * scale])
    sp = bez([pelvis, mid, neckw], 60)
    # radius profile: waist, chest, neck
    for k in range(sp.shape[1] - 1):
        tt = k / (sp.shape[1] - 1)
        r = (0.128 + 0.034 * math.sin(tt * math.pi * 0.87)) * (1 - 0.40 * tt) * scale
        add(cyl(sp[:, k], sp[:, k + 1], r, r * 0.96, 26, rng, sq=0.62))

    # ---- pelvis + chest masses -----------------------------------------
    add(ell(pelvis[0], pelvis[1], 0.140 * scale, 0.125 * scale, 0.090 * scale,
            260, rng))
    ch = pelvis + (neckw - pelvis) * 0.62
    add(ell(ch[0], ch[1], 0.160 * scale, 0.150 * scale, 0.098 * scale, 300, rng))

    # ---- shoulders -------------------------------------------------------
    sw = (0.175 + 0.035 * u) * scale
    sh_l = neckw + np.array([-sw * 0.35, -0.045 * scale])
    sh_r = neckw + np.array([sw * 0.50, -0.055 * scale])
    add(ell(sh_r[0], sh_r[1], 0.075 * scale, 0.070 * scale, 0.085 * scale, 150, rng))

    # ---- neck + head ------------------------------------------------------
    head = neckw + np.array([math.sin(th) * 0.13 + jut * 0.55,
                             math.cos(th) * 0.13]) * scale
    add(cyl(neckw, head, 0.062 * scale, 0.075 * scale, 90, rng))
    hr = (0.105 + 0.022 * u) * scale
    add(ell(head[0], head[1], hr * 1.00, hr * 1.10, hr * 0.86, 420, rng))
    # muzzle / jaw
    muzzle = head + np.array([hr * (1.05 - 0.45 * u), -hr * 0.32])
    add(ell(muzzle[0], muzzle[1], hr * (0.52 - 0.22 * u), hr * 0.40,
            hr * 0.55, 170, rng))
    # brow ridge
    brow = head + np.array([hr * 0.72, hr * 0.30])
    add(ell(brow[0], brow[1], hr * 0.38, hr * 0.10, hr * 0.42, 60, rng))

    # ---- legs -------------------------------------------------------------
    for side in (-1, 1):
        ph = phase if side > 0 else phase + math.pi
        s = math.sin(ph)
        lift = max(0.0, math.cos(ph))
        hip = pelvis + np.array([side * 0.085 * scale, -0.02 * scale])
        knee = np.array([hip[0] + s * 0.20 * scale,
                         hp * (0.52 + 0.06 * u) - 0.02 * scale])
        ankle = np.array([hip[0] + s * 0.34 * scale,
                          max(0.055, 0.075 * scale - lift * 0.03 * scale)])
        add(cyl(hip, knee, 0.095 * scale, 0.072 * scale, 340, rng))
        add(cyl(knee, ankle, 0.070 * scale, 0.048 * scale, 300, rng))
        # foot
        toe = ankle + np.array([0.11 * scale, -0.045 * scale])
        add(cyl(ankle, toe, 0.048 * scale, 0.035 * scale, 120, rng))

    # ---- arms --------------------------------------------------------------
    knuckle = u < 0.30
    for side in (-1, 1):
        ph = phase + (math.pi if side > 0 else 0.0)
        s = math.sin(ph)
        if knuckle:
            # the arm is a strut: shoulder -> elbow -> knuckles on the ground
            elbow = np.array([sh_r[0] + side * 0.02 * scale + 0.03 * scale,
                              hp * 0.62])
            hand = np.array([sh_r[0] + s * 0.10 * scale + 0.06 * scale, 0.035])
        else:
            elbow = np.array([sh_r[0] + s * 0.18 * scale,
                              sh_r[1] - (0.30 + 0.05 * u) * scale])
            hand = np.array([elbow[0] + s * 0.17 * scale,
                             elbow[1] - (0.28 - 0.06 * u) * scale])
        add(cyl(sh_r, elbow, 0.070 * scale, 0.055 * scale, 280, rng))
        add(cyl(elbow, hand, 0.055 * scale, 0.042 * scale, 280, rng))
        add(ell(hand[0], hand[1], 0.050 * scale, 0.045 * scale, 0.040 * scale,
                90, rng))

    pts = np.concatenate(P, axis=1)
    inten = np.concatenate(I)
    # the body sits just in front of the rail plane, and leans a little in z
    pts[2] += 0.0
    return pts, inten


# --------------------------------------------------------------------------
class Shot:
    def __init__(self, W, H, fps, dur):
        self.W, self.H, self.fps, self.dur = W, H, fps, dur

    def subject(self, t):
        """where the ape is at time t (it walks while we watch it)."""
        return 11.0 + 0.45 * t

    def camera(self, t):
        d = self.dur
        # --- the move, written as a shot list ---------------------------
        # 0.0-2.4  close on the shoulder of ape 1, tracking it
        # 2.4-4.2  slide right to ape 2
        # 4.2-7.0  pull back and rise while ape 2 stands up
        # 7.0-end  let the humans walk, wide
        a = self.subject(t)          # ape 1 walks right
        b = 15.20 + 0.06 * t         # ape 2, the one that stands up
        if t < 2.4:                  # 1. shoulder and head, very close
            cx, cy, cz, zoom = a + 0.04, 0.78, -2.75, 1.24
            tx, ty = a + 0.04, 0.80
        elif t < 4.0:                # 2. slide along the back to the next one
            k = (t - 2.4) / 1.6
            e = k * k * (3 - 2 * k)
            cx = a + 0.04 + e * (b - a - 0.15)
            cz = -2.55 - 0.30 * e
            cy = 0.72 + 0.04 * e
            zoom = 1.42 - 0.12 * e
            tx = a + 0.04 + e * (b - a)
            ty = 0.70 + 0.08 * e
        elif t < 6.6:                # 3. it stands; we rise and pull back
            k = (t - 4.0) / 2.6
            e = k * k * (3 - 2 * k)
            cx = b - 0.10 + e * 0.5
            cz = -2.40 - 3.55 * e
            cy = 0.72 + 0.45 * e
            zoom = 1.82 - 0.85 * e
            tx = b + e * 0.3
            ty = 0.78 + 0.18 * e
        else:                        # 4. the humans walk, wide
            k = min(1.0, (t - 6.6) / max(0.01, d - 6.6))
            cx = 16.1 + k * 8.4
            cz = -5.95 - 1.4 * k
            cy = 1.17 + 0.04 * k
            zoom = 0.97 - 0.16 * k
            tx = 23.5
            ty = 0.88
        # the camera always aims at what it is looking at
        dz = -cz
        yaw = math.atan2(tx - cx, dz)
        pitch = -math.atan2(cy - ty, dz) * 0.6
        return np.array([cx, cy, cz]), zoom, yaw, pitch

    def build(self, f, t, rng):
        W, H = self.W, self.H
        g = self.g
        V, E = f["voice"], f["rms"]
        cam, zoom, yaw, pitch = self.camera(t)
        flen = W * 0.92 * zoom

        buf = Buf(g.rows, g.cols)

        def add(pts, inten, rgb, cls):
            sx, sy, z = project(pts, cam, yaw, pitch, flen, W, H)
            buf.add(g, sx, sy, inten, rgb, cls)

        # ---- dust, three shells ------------------------------------------
        for lo, hi, rad, gain in ((-9.0, -2.8, 1.4, 0.9),
                                  (-2.0, 1.5, 2.0, 0.7),
                                  (2.0, 9.0, 2.6, 0.5)):
            n = 3200
            p = np.stack([rng.uniform(cam[0] - 9, cam[0] + 22, n),
                          rng.normal(1.0, rad * 0.9, n),
                          rng.uniform(lo, hi, n)])
            sx, sy, z = project(p, cam, yaw, pitch, flen, W, H)
            fog = np.exp(-(z - cam[2]) * 0.06)
            add(p, (0.10 + 0.22 * E) * fog * np.clip(2.4 / z, 0, 1) * gain,
                (30 + 26 * fog, 44 + 34 * fog, 72 + 52 * fog), 0)

        # ---- ground rail ---------------------------------------------------
        n = 1500
        gx = np.linspace(cam[0] - 6, cam[0] + 30, n)
        gp = np.stack([gx, np.full(n, 0.01), np.full(n, 0.0)])
        sx, sy, z = project(gp, cam, yaw, pitch, flen, W, H)
        fog = np.exp(-(z - cam[2]) * 0.05)
        add(gp, 0.40 * fog * (0.5 + 0.6 * E), (62, 80, 106), 0)

        # ---- the march ------------------------------------------------------
        # every figure is a body; two of them are in the middle of a change
        specs = [
            (11.0, lambda tt: 0.00, 0.0),          # knuckle-walker, moves
            (15.2, lambda tt: 0.35, 0.28),         # starts low...
            (19.4, lambda tt: 0.72, 0.0),          # hominid
            (23.2, lambda tt: 1.00, 0.0),          # human
            (26.6, lambda tt: 1.00, 0.0),          # human, further on
        ]
        for i, (bx, stf, morph) in enumerate(specs):
            if morph:
                k = float(np.clip((t - 3.4) / 3.2, 0, 1))
                k = k * k * (3 - 2 * k)
                st = stf(t) + morph * k
            else:
                st = stf(t)
            fx = bx + (0.45 * t if i == 0 else 0.06 * t)
            ph = t * 2.6 + i * 1.3
            pts, inten = figure(fx, st, ph, rng, scale=1.05 + 0.22 * st)
            shade_far = 1.0
            add(pts, inten * (0.62 + 0.40 * V) * shade_far,
                (206, 219, 238), 1)

        return buf


# --------------------------------------------------------------------------
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
              (inten > 0.01) & np.isfinite(inten))
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


def project(pts, cam, yaw, pitch, f, W, H):
    X = pts[0] - cam[0]
    Y = pts[1] - cam[1]
    Z = pts[2] - cam[2]
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    Xr = X * cy_ - Z * sy_
    Zr = X * sy_ + Z * cy_
    cp, sp = math.cos(pitch), math.sin(pitch)
    Yr = Y * cp - Zr * sp
    Zr = np.maximum(Y * sp + Zr * cp, 0.20)
    return W * 0.5 + f * Xr / Zr, H * 0.5 - f * Yr / Zr, Zr


def compose(buf, g):
    rows, cols = buf.rows, buf.cols
    v = buf.v.reshape(rows, cols)
    cls = buf.c.reshape(rows, cols)
    R = buf.r.reshape(rows, cols)
    G = buf.g.reshape(rows, cols)
    B = buf.b.reshape(rows, cols)

    rng = np.random.default_rng(77)
    ramp = np.array(list(" .:-=+*#%@"), dtype="U1")
    li = np.clip((v ** 0.6) * (len(ramp) - 1), 0, len(ramp) - 1).astype(int)
    chars = ramp[li]

    lit = v > 0.012
    m = (cls == 1) & lit
    if m.any():
        chars[m] = rng.choice(list(ALNUM), int(m.sum()))
    m = (cls == 0) & lit
    if m.any():
        chars[m] = rng.choice(list(DUST), int(m.sum()))

    chars = np.where(lit, chars, " ")
    colors = np.stack([np.clip(R, 0, 255), np.clip(G, 0, 255),
                       np.clip(B, 0, 255)], axis=2)
    return chars, colors


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=58.5)
    ap.add_argument("--dur", type=float, default=9.0)
    ap.add_argument("--out", default="evolve.mp4")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=13)
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    chars_all = "".join(sorted(set(ALNUM + DUST + " ")))
    g = Grid(args.w, args.h, args.font, chars_all)
    print(f"[grid] {g.cols}x{g.rows} cell {g.cw}x{g.ch} frames={n}", flush=True)

    shot = Shot(args.w, args.h, args.fps, args.dur)
    shot.g = g
    cw, chh = g.cols * g.cw, g.rows * g.ch

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        for tt in [0.02, 0.12, 0.24, 0.36, 0.50, 0.64, 0.78, 0.92]:
            fi = min(n - 1, int(tt * n))
            f = {k: float(feats[k][fi]) for k in feats
                 if isinstance(feats[k], np.ndarray)}
            rng = np.random.default_rng(100 + fi)
            buf = shot.build(f, fi / args.fps, rng)
            ch, co = compose(buf, g)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas, f, fi / args.fps, cw, chh,
                           np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview, f"e{tt:.2f}.png"))
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

    srng = np.random.default_rng(5)
    for fi in range(n):
        t = fi / args.fps
        f = {k: float(feats[k][fi]) for k in feats
             if isinstance(feats[k], np.ndarray)}
        buf = shot.build(f, t, srng)
        ch, co = compose(buf, g)
        canvas = g.render(g.idx(ch), co)
        canvas = shade(canvas, f, t, cw, chh, srng)
        frame = np.zeros((args.h, args.w, 3), np.uint8)
        frame[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
        pipe.stdin.write(frame.tobytes())
        if fi % 30 == 0:
            print(f"[render] {fi}/{n}", flush=True)

    pipe.stdin.close()
    pipe.wait()
    errlog.close()
    print("[done]", args.out, flush=True)


if __name__ == "__main__":
    main()
