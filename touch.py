#!/usr/bin/env python3
"""
THE TOUCH — a lateral track along two arms.

God's arm comes in from the left, Adam's from the right. The camera runs
along the line of them, left to right, and passes the half-inch between
two index fingers at the moment the music says so.

The arms are traced from Michelangelo's fresco: every glyph is a pixel of
paint that was put there in 1512.
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
CLOUDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clouds")


def load_hand(side, n=52000, rng=None):
    rng = rng or np.random.default_rng(1)
    d = np.load(os.path.join(CLOUDS, f"hand_{side}.npz"))
    p, lum = d["p"], d["lum"]
    # edge = 1 on the silhouette, 0 in the belly of the arm. It is what
    # keeps the body hollow: only the outline gets ink.
    edge = d["edge"] if "edge" in d.files else np.ones(len(lum), np.float32)
    if len(lum) > n:
        idx = rng.choice(len(lum), n, replace=False)
        p, lum, edge = p[:, idx], lum[idx], edge[idx]
    p = p.copy()
    p[2] += rng.normal(0, 0.012, p.shape[1])
    return p, lum, edge


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


def compose(buf, g, t, dur):
    rows, cols = buf.rows, buf.cols
    v = buf.v.reshape(rows, cols)
    cls = buf.c.reshape(rows, cols)
    R = buf.r.reshape(rows, cols)
    G = buf.g.reshape(rows, cols)
    B = buf.b.reshape(rows, cols)

    # the letters turn over six times a second, not thirty: a stable
    # texture is what lets the eye read the camera as movement
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
    # the field value is also the light on the glyph: a rim letter burns,
    # a letter in the belly of the arm is only a ghost of itself
    gain = 0.12 + 0.92 * np.clip(v, 0, 1.05)
    m = ((cls == 1) | (cls == 2)) & lit
    if m.any():
        colors[m] = colors[m] * gain[m][:, None]
    return chars, colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=58.5)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--out", default="touch.mp4")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=12)
    ap.add_argument("--pts", type=int, default=26000,
                    help="points per hand (the cloud is dense; fewer = more air)")
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    g = Grid(args.w, args.h, args.font, "".join(sorted(set(ALNUM + DUST + " .,:*#@+"))))
    print(f"[grid] {g.cols}x{g.rows} frames={n}", flush=True)

    rng = np.random.default_rng(7)
    hl, lum_l, ed_l = load_hand("l", args.pts, rng)
    hr, lum_r, ed_r = load_hand("r", args.pts, rng)
    # the clouds were cut at the edge of the reference image: fade those
    # blunt ends into dust so the forearm never ends in a straight wall
    fade_l = np.clip((hl[0] - hl[0].min()) / 0.15, 0, 1) ** 1.5
    fade_r = np.clip((hr[0].max() - hr[0]) / 0.15, 0, 1) ** 1.5
    # the fingers meet at x = 0.80
    GAP_X = 0.80
    X0, TRAVEL = 0.10, 1.44

    cw, chh = g.cols * g.cw, g.rows * g.ch

    # ---- camera: one continuous move -----------------------------------
    # One path, three channels. The lens opens with a raised cosine over
    # two seconds (no snap at either end), and the pan speed is divided
    # by the same curve, so the world keeps an even pace on screen while
    # the close-up opens. Everything is a function of k only: no cuts,
    # no separate "zoom event" to trip over.
    KT = (GAP_X - X0) / TRAVEL
    BELL_W = 0.40
    ZOOM_LO, ZOOM_HI = 0.56, 1.56
    _KN = np.linspace(0.0, 1.0, 8001)
    _U = (_KN - KT) / BELL_W
    _BN = np.where(np.abs(_U) < 1.0,
                   0.5 * (1.0 + np.cos(np.pi * np.clip(_U, -1.0, 1.0))), 0.0)
    _ZN = ZOOM_LO + (ZOOM_HI - ZOOM_LO) * _BN
    _SN = np.cumsum(_ZN ** -0.9)
    _SN = (_SN - _SN[0]) / (_SN[-1] - _SN[0])

    def S(k):
        return float(np.interp(k, _KN, _SN))

    # dust lives in world space, generated once: a stable field is what
    # makes a lateral move read as a dolly instead of noise
    dust = []
    for zlo, zhi, cnt, gain in ((-0.55, -0.18, 900, 0.55),
                                (0.30, 2.60, 2600, 0.85)):
        dp = np.stack([rng.uniform(-1.8, 3.4, cnt),
                       rng.uniform(0.05, 1.60, cnt),
                       rng.uniform(zlo, zhi, cnt)])
        dust.append((dp, gain))

    # the spark is a fixed constellation too: it glows and swirls, it does
    # not reshuffle itself every frame
    SP_N = 4200
    sp_a = rng.random(SP_N) * 2 * math.pi
    sp_r = rng.random(SP_N) ** 2.4 * 0.040
    sp_z = rng.normal(0, 0.004, SP_N)

    def frame(f, t):
        V = f["voice"]
        k = t / max(1e-6, args.dur)
        touch_t = KT * args.dur
        s = S(k)
        u = (k - KT) / BELL_W
        bell = 0.5 * (1.0 + math.cos(math.pi * u)) if abs(u) < 1.0 else 0.0
        d = abs(t - touch_t)
        spark = math.exp(-(d / 0.38) ** 2)
        cx = X0 + TRAVEL * s
        cy = 0.78 + 0.008 * math.sin(t * 0.9)
        cz = -0.95 + 0.14 * s
        zoom = ZOOM_LO + (ZOOM_HI - ZOOM_LO) * bell
        cam = np.array([cx, cy, cz])
        flen = args.w * 0.92 * zoom

        buf = Buf(g.rows, g.cols)

        # dust in front of and behind the arms
        for dp, gain in dust:
            sx, sy, z = project(dp, cam, flen, args.w, args.h)
            fog = np.exp(-z * 0.9)
            buf.add(g, sx, sy, (0.10 + 0.18 * V) * fog * gain,
                    (30 + 30 * fog, 44 + 38 * fog, 70 + 50 * fog), 0)

        # God's arm, then Adam's
        for pts, lum, ed, fd, cls in ((hl, lum_l, ed_l, fade_l, 1),
                                      (hr, lum_r, ed_r, fade_r, 1)):
            sx, sy, z = project(pts, cam, flen, args.w, args.h)
            ink = np.clip(1.15 - lum, 0, 1)
            # hollow body: only the traced silhouette carries ink, the
            # belly of the arm stays dark and the letters fall away
            rim = np.clip((ed - 0.28) / 0.55, 0, 1) ** 1.4
            inten = (0.02 + 1.20 * rim) * (0.84 + 0.28 * ink) * \
                (0.80 + 0.30 * V) * fd
            # the paint catches the light where the two fingers meet
            near = np.exp(-(((pts[0] - GAP_X) ** 2 +
                             (pts[1] - 0.768) ** 2) / 0.010))
            inten = inten * (1.0 + 1.5 * spark * near)
            fog = np.exp(-(z - 0.0) * 0.35)
            buf.add(g, sx, sy, inten * fog,
                    (232 - 40 * ink, 218 - 40 * ink, 196 - 30 * ink), cls)

        # the spark: light in the half-inch between the fingers
        if spark > 0.01:
            ang = sp_a + 0.35 * t
            rr = sp_r * (0.55 + 0.45 * spark)
            sp = np.stack([GAP_X + np.cos(ang) * rr,
                           0.768 + np.sin(ang) * rr * 0.9,
                           sp_z])
            sx, sy, z = project(sp, cam, flen, args.w, args.h)
            buf.add(g, sx, sy, 2.6 * spark *
                    np.clip(1 - rr / 0.045, 0, 1) ** 1.2,
                    (255.0, 248.0, 232.0), 2)

        return compose(buf, g, t, args.dur)

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        for tt in [0.02, 0.18, 0.36, 0.486, 0.64, 0.80, 0.95]:
            fi = min(n - 1, int(tt * n))
            f = {k: float(feats[k][fi]) for k in feats
                 if isinstance(feats[k], np.ndarray)}
            ch, co = frame(f, fi / args.fps)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas, f, fi / args.fps, cw, chh,
                           np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview, f"t{tt:.2f}.png"))
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
