#!/usr/bin/env python3
"""
MARCH OF LIGHT — one continuous lateral move, left to right.

A corridor of glyphs. The camera tracks right and passes, in order:

  0  seed        a single character blinking in the dust
  1  helix       a DNA double helix, spinning, built from letters
  2  march       five figures, progressively more upright (Evolution of Man)
  3  creation    two hands reaching, index fingers a breath apart; touch = spark
  4  handshake   two figures shake hands; one of them catches fire (WYWH)
  5  prism       the spark runs down the corridor and breaks on glass

Every pixel is a letter, a digit or a mark. No cuts.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Grid, analyze, shade, FFMPEG  # noqa: E402

ALNUM = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
DUST = " .,:;'"
LIGHT = " .:-=+*#%@"
SOUP = ALNUM + "!?#$%&*/\\|<>[]{}()"
CLASS = {"dust": 0, "fig": 1, "glass": 2, "light": 3, "fire": 4, "soup": 5}

WORLD_DUR = 20.0
HEAT_S = np.array([0.0, 0.30, 0.58, 0.82, 1.0])


def heat(u):
    u = np.clip(u, 0, 1)
    r = np.interp(u, HEAT_S, [42, 122, 208, 255, 255])
    g = np.interp(u, HEAT_S, [7, 26, 78, 176, 244])
    b = np.interp(u, HEAT_S, [3, 8, 18, 84, 232])
    return r, g, b


def spectral_rgb(u):
    xs = [0.0, 0.10, 0.25, 0.40, 0.55, 0.72, 0.88, 1.0]
    r = np.interp(u, xs, [90, 45, 20, 10, 90, 215, 255, 255])
    g = np.interp(u, xs, [20, 70, 150, 220, 255, 220, 120, 40])
    b = np.interp(u, xs, [200, 255, 255, 200, 60, 20, 10, 10])
    return r, g, b


# --------------------------------------------------------------------------
# geometry: everything is sampled points, nothing is drawn as a shape
# --------------------------------------------------------------------------
def capsule(p0, p1, r, n, rng, edge=0.72):
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    d = p1 - p0
    L = float(np.hypot(d[0], d[1])) + 1e-9
    nx, ny = -d[1] / L, d[0] / L
    t = rng.random(n)
    rad = r * (edge + (1 - edge) * rng.random(n))
    sgn = rng.choice([-1.0, 1.0], n)
    x = p0[0] + d[0] * t + nx * rad * sgn
    y = p0[1] + d[1] * t + ny * rad * sgn
    z = rng.normal(0, 0.035, n)
    # brightness: the silhouette edge is where the ink is
    inten = 0.55 + 0.55 * (rad / r)
    return np.stack([x, y, z]), inten


def blob(cx, cy, rx, ry, n, rng):
    a = rng.random(n) * 2 * math.pi
    rad = np.sqrt(rng.random(n))
    x = cx + np.cos(a) * rad * rx
    y = cy + np.sin(a) * rad * ry
    z = rng.normal(0, 0.03, n)
    inten = 0.5 + 0.6 * rad
    return np.stack([x, y, z]), inten


def figure(x, stage, phase, rng, scale=1.0):
    """A walking hominid in profile, facing +x. stage 0..1 = uprightness."""
    u = stage
    hip_h = (0.52 + 0.48 * u) * scale
    torso = (0.46 + 0.34 * u) * scale
    lean = math.radians(58 - 56 * u)
    head_r = (0.150 - 0.02 * u) * scale
    limb_r = (0.115 - 0.018 * u) * scale

    hip = np.array([x, hip_h])
    neck = hip + np.array([math.sin(lean) * torso, math.cos(lean) * torso])
    sh = neck - np.array([0.0, 0.06 * scale])

    P, I = [], []

    # spine + head
    p, i = capsule(hip, neck, limb_r * 1.30, 900, rng)
    P.append(p); I.append(i)
    hx = neck[0] + math.sin(lean) * head_r * 1.3
    hy = neck[1] + math.cos(lean) * head_r * 1.3
    p, i = blob(hx, hy, head_r, head_r * 1.1, 700, rng)
    P.append(p); I.append(i * 1.05)

    stride = math.sin(phase)
    lift = max(0.0, math.cos(phase))

    # legs: hip -> knee -> foot, one forward one back
    for side, ph in ((1, 0.0), (-1, math.pi)):
        s = math.sin(phase + ph)
        kx = hip[0] + s * 0.26 * scale
        ky = hip[1] * 0.52
        fx = hip[0] + s * 0.44 * scale
        fy = max(0.02, hip[1] * 0.10 + max(0.0, -s) * 0.05)
        p, i = capsule((hip[0], hip[1] - 0.02), (kx, ky), limb_r, 420, rng)
        P.append(p); I.append(i * (1.0 if side > 0 else 0.8))
        p, i = capsule((kx, ky), (fx, fy), limb_r * 0.85, 420, rng)
        P.append(p); I.append(i * (1.0 if side > 0 else 0.8))

    # arms: they swing opposite the legs; the longer the lower the stage
    arm_len = (0.52 - 0.16 * u) * scale
    for side, ph in ((1, math.pi), (-1, 0.0)):
        s = math.sin(phase + ph)
        ex = sh[0] + s * arm_len * 0.42
        ey = sh[1] - arm_len * (0.50 - 0.10 * u)
        wx = ex + s * arm_len * 0.40
        wy = ey - arm_len * (0.46 - 0.16 * u)
        p, i = capsule(sh, (ex, ey), limb_r * 0.85, 340, rng)
        P.append(p); I.append(i * (1.0 if side > 0 else 0.78))
        p, i = capsule((ex, ey), (wx, wy), limb_r * 0.7, 340, rng)
        P.append(p); I.append(i * (1.0 if side > 0 else 0.78))

    return np.concatenate(P, axis=1), np.concatenate(I)


def hand(x, y, side, rng, gap=0.22, scale=2.1):
    """A forearm + palm + one reaching index finger. side=+1 reaches right."""
    P, I = [], []
    d = side
    wrist = np.array([x, y])
    elbow = wrist - np.array([d * 1.30, -0.14])
    p, i = capsule(elbow, wrist, 0.125, 1200, rng)
    P.append(p); I.append(i * 0.95)

    palm = wrist + np.array([d * 0.30, 0.0])
    p, i = blob(palm[0], palm[1], 0.30, 0.21, 1800, rng)
    P.append(p); I.append(i)

    # the index finger: long, extended, the whole point of the image
    base = palm + np.array([d * 0.20, 0.06])
    tip = palm + np.array([d * (0.52 + gap * 0.10), 0.075])
    p, i = capsule(base, tip, 0.058, 900, rng)
    P.append(p); I.append(i * 1.25)

    # the other three curl under
    for k in range(3):
        a = -0.30 + k * 0.22
        mid = palm + np.array([d * (0.30 + 0.02 * k), -0.14 + a * 0.06])
        p, i = capsule(palm + np.array([d * 0.10, -0.02]), mid, 0.055, 380, rng)
        P.append(p); I.append(i * 0.85)
    thumb = palm + np.array([d * 0.06, -0.30])
    p, i = capsule(palm + np.array([0, -0.12]), thumb, 0.055, 380, rng)
    P.append(p); I.append(i * 0.85)

    pts = np.concatenate(P, axis=1)
    inten = np.concatenate(I)
    pts[0] = x + (pts[0] - x) * scale
    pts[1] = y + (pts[1] - y) * scale
    pts[2] *= scale
    return pts, inten


def helix(x0, n=260, rng=None):
    t = np.linspace(0, 6.6 * math.pi, n)
    y = 0.55 + t * 0.030
    z = 0.0
    a = 0.30 * np.cos(t)
    b = 0.30 * np.sin(t)
    # two strands, offset in x and a shallow depth
    pa = np.stack([x0 + a, y, z + b * 0.55])
    pb = np.stack([x0 - a, y, z - b * 0.55])
    # rungs
    idx = np.linspace(0, n - 1, 40).astype(int)
    rungs = []
    for i in idx:
        s = np.linspace(0, 1, 7)
        rungs.append(np.stack([pa[0, i] + (pb[0, i] - pa[0, i]) * s,
                               pa[1, i] + (pb[1, i] - pa[1, i]) * s,
                               pa[2, i] + (pb[2, i] - pa[2, i]) * s]))
    rungs = np.concatenate(rungs, axis=1)
    return np.concatenate([pa, pb, rungs], axis=1)


class Buf:
    """depth-aware splat buffer: intensity + colour + which element owns a cell."""

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
        idx = idx[win]
        val = val[win]
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
    Zr2 = Y * sp + Zr * cp
    Zr2 = np.maximum(Zr2, 0.25)
    return W * 0.5 + f * Xr / Zr2, H * 0.5 - f * Yr / Zr2, Zr2


# --------------------------------------------------------------------------
class World:
    def __init__(self, W, H, fps):
        self.W, self.H, self.fps = W, H, fps

    def camera(self, t):
        xs = [0.0, 2.0, 8.0, 11.0, 13.5, 17.0, 20.0]
        xv = [0.0, 6.0, 30.0, 44.0, 56.0, 70.0, 80.0]
        x = float(np.interp(t, xs, xv))
        zs = [0.0, 8.0, 11.0, 13.5, 17.0, 20.0]
        zv = [-9.6, -8.2, -6.6, -5.6, -4.0, -3.4]
        z = float(np.interp(t, zs, zv))
        y = 0.62 + 0.10 * math.sin(t * 0.83)
        yaw = 0.035 * math.sin(t * 0.61) - 0.02
        pitch = 0.012 * math.sin(t * 0.47 + 1.0)
        # focal length = zoom. it never sits still.
        zs = [0.0, 1.5, 3.0, 5.0, 7.0, 8.5, 10.2, 11.5, 13.0, 15.0, 17.0, 18.4, 20.0]
        zv_zoom = [0.92, 1.02, 0.66, 0.70, 0.74, 1.05, 1.55, 1.30, 1.12, 1.30, 1.70, 1.35, 1.05]
        zoom = float(np.interp(t, zs, zv_zoom))
        return np.array([x, y, z]), yaw, pitch, zoom

    def build(self, f, t, rng):
        W, H = self.W, self.H
        g = self.g
        V, E, P = f["voice"], f["rms"], f["pitch"]
        FL = f["flux"]
        cam, yaw, pitch, zoom = self.camera(t)
        flen = W * 0.92 * zoom
        buf = Buf(g.rows, g.cols)

        def add(pts, inten, rgb, cls):
            sx, sy, z = project(pts, cam, yaw, pitch, flen, W, H)
            buf.add(g, sx, sy, inten, rgb, cls)

        # ---- dust: three depth shells, strongest parallax near the lens ----
        for i in range(3):
            n = 3800
            if i == 0:
                p = np.stack([rng.uniform(cam[0] - 10, cam[0] + 26, n),
                              rng.normal(0.5, 1.7, n),
                              rng.uniform(-8.0, -2.4, n)])
            elif i == 1:
                p = np.stack([rng.uniform(cam[0] - 10, cam[0] + 30, n),
                              rng.normal(0.9, 2.4, n),
                              rng.uniform(-1.8, 2.0, n)])
            else:
                p = np.stack([rng.uniform(cam[0] - 6, cam[0] + 34, n),
                              rng.normal(1.0, 3.0, n),
                              rng.uniform(3.0, 9.0, n)])
            sx, sy, z = project(p, cam, yaw, pitch, flen, W, H)
            fog = np.exp(-(z - cam[2]) * 0.07)
            inten = (0.12 + 0.26 * E) * fog * np.clip(2.2 / z, 0, 1)
            rgb = (30 + 24 * fog, 46 + 34 * fog, 76 + 56 * fog)
            buf.add(g, sx, sy, inten, rgb, CLASS["dust"])

        # ---- the ground: a rail of characters, receding into the dark -------
        n = 1600
        gx = np.linspace(-8.0, 98.0, n)
        gp = np.stack([gx, np.full(n, 0.015), np.full(n, 0.0)])
        sx, sy, z = project(gp, cam, yaw, pitch, flen, W, H)
        fog = np.exp(-(z - cam[2]) * 0.045)
        buf.add(g, sx, sy, 0.42 * fog * (0.55 + 0.55 * E),
                (66.0, 84.0, 110.0), CLASS["dust"])

        # ---- 0. the seed: one character blinking ---------------------------
        if t < 2.6:
            k = 1.0 if (t * 3 % 1) < 0.55 else 0.35
            p = np.stack([np.array([1.2 + t * 2.6]), np.array([0.85]),
                          np.array([0.0])])
            sx, sy, z = project(p, cam, yaw, pitch, flen, W, H)
            buf.add(g, sx, sy, np.array([1.4 * k]), (235.0, 245.0, 255.0),
                    CLASS["light"])

        # ---- 1. the helix ---------------------------------------------------
        if 1.2 < t < 7.0:
            hp = helix(6.0, 520, rng)
            # the helix rotates: swap x/z slowly
            ang = t * 0.9
            ca, sa = math.cos(ang), math.sin(ang)
            xr = hp[0] - 6.0
            hp = np.stack([6.0 + xr * ca - hp[2] * sa, hp[1],
                           xr * sa + hp[2] * ca])
            hp[1] += 0.1 * math.sin(t * 2.0)
            inten = np.full(hp.shape[1], 0.85)
            add(hp, inten, (150.0, 220.0, 235.0), CLASS["fig"])

        # ---- 2. the march ---------------------------------------------------
        stages = [0.0, 0.28, 0.55, 0.80, 1.0]
        for i, st in enumerate(stages):
            fx = 12.0 + i * 2.45
            ph = t * 3.2 + i * 1.7
            pts, inten = figure(fx, st, ph, rng, scale=1.02 + 0.12 * st)
            gain = 0.75 + 0.35 * V
            add(pts, inten * gain, (198.0, 214.0, 236.0), CLASS["fig"])

        # ---- 3. creation: two hands, a centimetre apart ---------------------
        # the gap closes with the voice: at the peak they touch
        touch_t = 10.9
        prox = float(np.clip(1.0 - abs(t - touch_t) / 2.2, 0, 1))
        gap = 0.55 * (1 - prox) + 0.02
        hp_l, il = hand(42.10, 1.30, +1, rng, gap, scale=2.1)
        hp_r, ir = hand(44.85, 1.30, -1, rng, gap, scale=2.1)
        for arr in (hp_l, hp_r):
            arr[1] += 0.0
            arr[2] += 0.55
        add(hp_l, il * 1.05, (205.0, 218.0, 238.0), CLASS["fig"])
        add(hp_r, ir * 1.05, (185.0, 200.0, 226.0), CLASS["fig"])

        # the spark at the meeting point
        spark = math.exp(-((t - touch_t) / 0.55) ** 2)
        if spark > 0.05:
            n = 420
            a = rng.random(n) * 2 * math.pi
            rr = rng.random(n) ** 1.6 * 0.55 * spark
            sp = np.stack([43.45 + np.cos(a) * rr,
                           1.30 + np.sin(a) * rr,
                           rng.normal(0, 0.04, n)])
            add(sp, 1.5 * spark * (1 - rr / 0.6),
                (255.0, 250.0, 232.0), CLASS["light"])

        # ---- 4. handshake + fire (Wish You Were Here) -----------------------
        hy = 0.95
        # two figures in profile, facing each other
        for sx_, d, br in ((54.55, +1, 1.0), (59.45, -1, 0.95)):
            body, bi = capsule((sx_, 0.12), (sx_ + d * 0.05, 1.55), 0.19, 900, rng)
            head, hi = blob(sx_ + d * 0.03, 1.78, 0.17, 0.19, 650, rng)
            arm, ai = capsule((sx_ + d * 0.05, 1.40), (56.95, 1.16), 0.075, 520, rng)
            add(np.concatenate([body, head, arm], axis=1),
                np.concatenate([bi, hi, ai]) * br,
                (190.0, 205.0, 228.0), CLASS["fig"])
        # the clasp
        clasp, ci = blob(56.95, 1.16, 0.17, 0.12, 700, rng)
        add(clasp, ci * 1.1, (225.0, 232.0, 245.0), CLASS["fig"])

        # the right-hand figure burns
        burn = float(np.clip((t - 11.6) / 2.2, 0, 1))
        if burn > 0.02:
            n = 700
            px_ = 59.45 + rng.normal(0, 0.22, n)
            py_ = rng.uniform(0.2, 1.95, n) + (t - 11.6) * 0.35
            py_ = np.mod(py_, 2.1)
            pz_ = rng.normal(0, 0.10, n)
            up = np.clip(py_ / 2.1, 0, 1)
            inten = burn * (0.5 + 0.9 * up) * (0.7 + 0.6 * FL)
            add(np.stack([px_, py_, pz_]), inten * 1.1,
                (255.0 * (0.6 + 0.4 * up), 120 * up + 40,
                 20 * (1 - up)), CLASS["fire"])

        # ---- the beam: the spark runs right, all the way to the glass -------
        if t > touch_t - 0.1:
            prog = float(np.clip((t - touch_t) / 2.4, 0, 1))
            x_end = 43.4 + prog * 26.0
            n = 420
            bx = np.linspace(43.4, x_end, n)
            wob = 1.05 + 0.02 * np.sin(bx * 3.0 + t * 6.0)
            bp = np.stack([bx, wob, np.full(n, 0.0)])
            mix = (0.55 + 0.45 * V)
            add(bp, np.full(n, 0.55 + 0.5 * mix), (240.0, 246.0, 255.0),
                CLASS["light"])

        # ---- 5. the prism ---------------------------------------------------
        pz = 0.0
        px0, py0, R = 72.0, 1.15, 0.95
        tri = np.array([[px0 + R * math.cos(math.radians(90 + 120 * k)),
                         py0 + R * math.sin(math.radians(90 + 120 * k))]
                        for k in range(3)])
        xs, ys, zs = [], [], []
        for dep in (0.0, 0.5):
            for k in range(3):
                p0, p1 = tri[k], tri[(k + 1) % 3]
                s = np.linspace(0, 1, 120)
                xs += list(p0[0] + (p1[0] - p0[0]) * s)
                ys += list(p0[1] + (p1[1] - p0[1]) * s)
                zs += [pz + dep] * 120
            for q in np.linspace(0.15, 0.85, 5):
                xs.append(px0 + (-R + 2 * R * q) * 0.55)
                ys.append(py0 + (-R * 0.5 + R * q))
                zs.append(pz + dep)
        for k in range(3):
            s = np.linspace(0, 1, 22)
            xs += [tri[k][0]] * 22
            ys += [tri[k][1]] * 22
            zs += list(pz + 0.5 * s)
        gp = np.stack([np.array(xs, np.float32), np.array(ys, np.float32),
                       np.array(zs, np.float32)])
        add(gp, np.full(len(xs), 0.55 + 0.25 * V), (168.0, 208.0, 246.0),
            CLASS["glass"])

        # dispersion fan
        for k in range(41):
            ang = math.radians(-21 + 42 * k / 40)
            s = np.linspace(0.10, 8.0, 190)
            xx = px0 + R * math.cos(math.radians(30)) + s * math.cos(ang)
            yy = py0 + R * math.sin(math.radians(30)) + s * math.sin(ang) * 0.92
            zz = pz + 0.02 * k + 0.02 * np.sin(s * 0.7 + k)
            u = k / 40.0
            r_, g_, b_ = spectral_rgb(np.full(len(s), u))
            edge = 1.0 - 0.40 * abs(u * 2 - 1)
            inten = np.exp(-s * 0.14) * (0.55 + 0.5 * V) * edge * 1.25
            add(np.stack([xx, yy, zz]), inten, (r_, g_, b_), CLASS["light"])

        return buf


# --------------------------------------------------------------------------
# the end card — a terminal in the dark, typed by a machine
# --------------------------------------------------------------------------
_CARD_CACHE = {}
_FONT_PATH = r"C:\Windows\Fonts\consola.ttf"


def _fit_font(d, text, W, H, wfrac, hfrac):
    """largest size that fits the box."""
    lo, hi = 4, int(H * 1.2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        f = ImageFont.truetype(_FONT_PATH, mid)
        bb = d.textbbox((0, 0), text, font=f)
        if (bb[2] - bb[0]) <= wfrac * W and (bb[3] - bb[1]) <= hfrac * H:
            lo = mid
        else:
            hi = mid - 1
    return ImageFont.truetype(_FONT_PATH, max(6, lo))


def _blk(cols, rows, text, wfrac, yc, hfrac, S=4):
    W, H = cols * S, rows * S
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    font = _fit_font(d, text, W, H, wfrac, hfrac)
    bb = d.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((W - w) / 2 - bb[0], yc * H - h / 2 - bb[1]), text, fill=255, font=font)
    a = np.array(img, np.float32).reshape(rows, S, cols, S).mean(axis=(1, 3))
    return np.clip(a / 255.0, 0, 1)


def _phoenix(cols, rows, yc, hx_frac, hy_frac, S=4):
    """A bird with its wings up, drawn as polygons, sampled per cell."""
    W, H = cols * S, rows * S
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    cx, cy = W * 0.5, yc * H
    hx = hx_frac * W * 0.5
    hy = hy_frac * H * 0.5

    def P(*pts):
        d.polygon([(cx + x * hx, cy + y * hy) for x, y in pts], fill=255)

    P((-0.04, -0.10), (-1.00, -0.62), (-0.92, -0.30), (-0.55, 0.02), (-0.10, 0.10))
    P((0.04, -0.10), (1.00, -0.62), (0.92, -0.30), (0.55, 0.02), (0.10, 0.10))
    P((-0.10, -0.02), (-0.62, -0.30), (-0.54, -0.08), (-0.16, 0.08))
    P((0.10, -0.02), (0.62, -0.30), (0.54, -0.08), (0.16, 0.08))
    d.ellipse([cx - 0.13 * hx, cy - 0.10 * hy, cx + 0.13 * hx, cy + 0.22 * hy], fill=255)
    P((0.02, -0.06), (0.09, -0.34), (0.15, -0.32), (0.10, 0.00))
    d.ellipse([cx + 0.04 * hx, cy - 0.40 * hy, cx + 0.19 * hx, cy - 0.26 * hy], fill=255)
    P((0.18, -0.35), (0.30, -0.31), (0.18, -0.29))
    P((0.07, -0.40), (0.02, -0.60), (0.09, -0.44))
    P((0.12, -0.42), (0.16, -0.62), (0.17, -0.42))
    P((-0.05, 0.18), (-0.24, 0.62), (-0.10, 0.60), (0.02, 0.24))
    P((0.02, 0.18), (0.10, 0.66), (0.22, 0.60), (0.10, 0.20))
    P((0.08, 0.16), (0.34, 0.56), (0.38, 0.44), (0.14, 0.14))

    a = np.array(img, np.float32).reshape(rows, S, cols, S).mean(axis=(1, 3))
    return np.clip(a / 255.0, 0, 1)


def make_card(g, t, dur):
    rows, cols = g.rows, g.cols
    key = (rows, cols)
    if key not in _CARD_CACHE:
        _CARD_CACHE[key] = (
            _phoenix(cols, rows, 0.165, 0.20, 0.165),
            _blk(cols, rows, "ASCII", 0.72, 0.435, 0.285),
            _blk(cols, rows, "\u00d7", 0.03, 0.598, 0.035),
            _blk(cols, rows, "FILM", 0.40, 0.745, 0.20),
        )
    m_bird, m_left, m_x, m_film = _CARD_CACHE[key]

    p = np.clip((t - (dur - 2.35)) / 1.05, 0, 1)
    colidx = np.arange(cols)[None, :]
    reveal = np.clip((p * cols * 1.05 - colidx) / 12.0, 0, 1)

    chars = np.full((rows, cols), " ", dtype="U1")
    colors = np.zeros((rows, cols, 3), np.float32)
    rng = np.random.default_rng(int(t * 60) % 99991)

    def stamp(mask, pool, rgb, gain=1.0, ramp=False):
        use = mask * reveal * gain
        m = use > 0.12
        if not m.any():
            return
        if ramp:
            rv = np.array(list(" .:-=+*#%@"), dtype="U1")
            li = np.clip(use * (len(rv) - 1), 0, len(rv) - 1).astype(int)
            ch = rv[li]
        else:
            ch = np.full((rows, cols), " ", dtype="U1")
            n = int(m.sum())
            ch[m] = rng.choice(list(pool), n)
        new = m & (chars == " ")
        chars[new] = ch[new]
        for k in range(3):
            colors[:, :, k][new] = rgb[k]

    stamp(m_bird, "", (255.0, 150.0, 62.0), gain=0.9, ramp=True)
    stamp(m_left, ALNUM, (234.0, 242.0, 255.0), gain=1.0)
    stamp(m_x, ALNUM, (150.0, 170.0, 195.0), gain=0.9)
    stamp(m_film, ALNUM, (255.0, 196.0, 120.0), gain=1.0)

    # terminal chrome
    prompt = "> demo --with film"
    typed = int(np.clip((t - (dur - 2.30)) / 0.70, 0, 1) * len(prompt))
    r0 = max(1, int(rows * 0.062))
    c0 = max(2, (cols - len(prompt) - 2) // 2)
    for i, c in enumerate(prompt[:typed]):
        if 0 <= r0 < rows and c0 + i < cols:
            chars[r0, c0 + i] = c
            colors[r0, c0 + i] = (120.0, 210.0, 170.0)
    if typed < len(prompt) or (t * 2.2) % 1.0 < 0.55:
        cc = c0 + min(typed, len(prompt))
        if cc < cols:
            chars[r0, cc] = "\u2588" if (t * 2.2) % 1.0 < 0.5 else "_"
            colors[r0, cc] = (120.0, 210.0, 170.0)

    r1 = min(rows - 2, int(rows * 0.865))
    line = "\u2500" * (cols - 2 * c0)
    for i, c in enumerate(line):
        if c0 + i < cols:
            chars[r1, c0 + i] = c
            colors[r1, c0 + i] = (58.0, 82.0, 104.0)

    return chars, colors


# --------------------------------------------------------------------------
def compose(buf, g, t, dur):
    rows, cols = buf.rows, buf.cols
    v = buf.v.reshape(rows, cols)
    cls = buf.c.reshape(rows, cols)
    R = buf.r.reshape(rows, cols)
    G = buf.g.reshape(rows, cols)
    B = buf.b.reshape(rows, cols)

    # ---- the collapse: everything falls back into noise -------------------
    teardown = 0.0
    if t > dur - 2.6:
        teardown = float(np.clip((t - (dur - 2.6)) / 1.9, 0, 1))

    ramp = np.array(list(" .:-=+*#%@"), dtype="U1")
    li = np.clip((v ** 0.65) * (len(ramp) - 1), 0, len(ramp) - 1).astype(int)
    chars = ramp[li]

    rng = np.random.default_rng(int(t * 997) % 100003)

    # class -> alphabet
    is_fig = cls == CLASS["fig"]
    is_fire = cls == CLASS["fire"]
    is_light = cls == CLASS["light"]
    is_dust = cls == CLASS["dust"]
    is_glass = cls == CLASS["glass"]

    def rand_chars(mask, pool):
        n = int(mask.sum())
        if n == 0:
            return
        chars[mask] = rng.choice(list(pool), n)

    lit = v > 0.012
    rand_chars(is_fig & lit, ALNUM)
    rand_chars(is_glass & lit, " /\\|[]+=.")
    rand_chars(is_dust & lit, DUST)
    rand_chars(is_light & lit, LIGHT)
    rand_chars(is_fire & lit, " .'^*")

    colors = np.stack([np.clip(R, 0, 255), np.clip(G, 0, 255),
                       np.clip(B, 0, 255)], axis=2)

    # ---- teardown: rows slip, glyphs turn to soup, brightness dies --------
    if teardown > 0.0:
        shift = (rng.normal(0, 1, rows) * teardown * 18).astype(int)
        for r in range(rows):
            if shift[r]:
                chars[r] = np.roll(chars[r], shift[r])
                colors[r] = np.roll(colors[r], shift[r], axis=0)
        # characters corrupt
        m = (rng.random((rows, cols)) < teardown * 0.55) & (chars != " ")
        chars[m] = rng.choice(list(SOUP), int(m.sum()))
        # ...and die out
        fade = np.clip(1.0 - teardown * 0.95, 0, 1)
        colors *= fade

    # ---- the card: demo x film, in the machines own voice -------------
    if t > dur - 2.35:
        card_ch, card_co = make_card(g, t, dur)
        return card_ch, card_co

    chars = np.where(v > 0.012, chars, " ")
    return chars, colors


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--out", default="march.mp4")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=13)
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    chars_all = "".join(sorted(set(ALNUM + DUST + LIGHT + SOUP + " ")))
    g = Grid(args.w, args.h, args.font, chars_all)
    print(f"[grid] {g.cols}x{g.rows} cell {g.cw}x{g.ch} frames={n}", flush=True)

    world = World(args.w, args.h, args.fps)
    world.g = g
    cw, chh = g.cols * g.cw, g.rows * g.ch

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        for tt in [0.04, 0.14, 0.28, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92, 0.965]:
            fi = min(n - 1, int(tt * n))
            f = {k: float(feats[k][fi]) for k in feats
                 if isinstance(feats[k], np.ndarray)}
            rng = np.random.default_rng(1000 + fi)
            buf = world.build(f, args.t0 + fi / args.fps, rng)
            ch, co = compose(buf, g, args.t0 + fi / args.fps, WORLD_DUR)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas, f, fi / args.fps, cw, chh,
                           np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview, f"f{tt:.2f}.png"))
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

    srng = np.random.default_rng(4)
    for fi in range(n):
        t = args.t0 + fi / args.fps
        f = {k: float(feats[k][fi]) for k in feats
             if isinstance(feats[k], np.ndarray)}
        buf = world.build(f, t, srng)
        ch, co = compose(buf, g, t, WORLD_DUR)
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
