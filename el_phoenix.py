#!/usr/bin/env python3
"""
05 · 涅槃 — the phoenix rises out of the fire and comes at the lens.

The bird is the user's dotted reference, traced to 9k points and resampled to
40k so it holds up when it is close. The fire is a bed of embers on the
ground; the bird starts inside it, small and low, and climbs while the
perspective does the growing — the camera never dollies, the bird flies.
"""
import argparse

import numpy as np

import elib
from elib import Buf, project, compose, run, add_args, CLOUDS


def main():
    ap = argparse.ArgumentParser()
    add_args(ap, "el_phoenix.mp4")
    ap.add_argument("--pts", type=int, default=42000)
    args = ap.parse_args()

    d = np.load(CLOUDS + r"\bird0.npz")
    p = d["p"].astype(np.float32)
    src = p.shape[1]
    rng = np.random.default_rng(31)
    idx = rng.choice(src, args.pts)
    bird = np.stack([p[0][idx] + rng.normal(0, 0.007, args.pts),
                     p[1][idx] + rng.normal(0, 0.007, args.pts),
                     rng.normal(0, 0.010, args.pts)]).astype(np.float32)
    print(f"[phoenix] {src} -> {args.pts} pts  "
          f"x[{bird[0].min():.2f},{bird[0].max():.2f}] "
          f"y[{bird[1].min():.2f},{bird[1].max():.2f}]", flush=True)

    # fire bed: fixed embers on the ground, rising and dying
    n_fire = 9000
    fx = rng.uniform(-0.9, 0.9, n_fire)
    fy = rng.uniform(-0.02, 0.10, n_fire)
    fz = rng.uniform(0.2, 2.6, n_fire)
    fv = rng.uniform(0.35, 0.95, n_fire)
    fph = rng.uniform(0, 2 * np.pi, n_fire)
    flife = rng.uniform(0.5, 1.0, n_fire)

    def frame(f, t, g):
        V = f["voice"]
        u = t / args.dur
        cx = -0.06 + 0.12 * u
        cy = 0.42 + 0.10 * u
        cam = np.array([cx, cy, -1.0])
        zoom = 0.62 + 0.10 * u
        flen = args.w * 0.92 * zoom
        buf = Buf(g.rows, g.cols)

        # ---- fire ------------------------------------------------------
        life = np.mod(t * fv / np.maximum(flife, 0.2) + fph * 0.11, 1.0)
        em = np.stack([fx + 0.05 * np.sin(fph + t * 2.2) * life,
                       fy + life * 0.55,
                       fz - life * 0.25])
        sx, sy, z = project(em, cam, flen, args.w, args.h)
        heat = np.clip(1.0 - life, 0, 1) ** 1.25
        fade = 1.0 - 0.45 * u          # the bed dies back as the bird climbs
        inten = (0.35 + 1.35 * heat) * (0.80 + 0.40 * V) * fade
        cr = 255 * np.clip(1.05 * heat + 0.30, 0, 1)
        cg = 255 * np.clip(0.80 * heat + 0.10, 0, 1)
        cb = 255 * np.clip(0.32 * heat, 0, 1)
        buf.add(g, sx, sy, inten, (cr, cg, cb), 2)

        # ---- the bird --------------------------------------------------
        grow = 0.62 + 0.95 * (u ** 1.25)
        bz = 3.30 - 2.55 * (u ** 1.10)
        by = 0.30 + 0.26 * (u ** 1.15)
        bp = np.stack([bird[0] * grow,
                       (bird[1] - 0.5) * grow + by,
                       bird[2] * grow + bz + 0.0])
        # ragged edges: wing tips thinner than the body
        sx, sy, z = project(bp, cam, flen, args.w, args.h)
        edge = np.clip((bp[1] - by) / max(grow, 1e-3) + 0.5, 0, 1)
        inten = (0.75 + 1.05 * (1.0 - edge * 0.45)) * (0.82 + 0.30 * V)
        inten = inten * np.clip(0.55 + 1.4 * u, 0, 1.35)
        buf.add(g, sx, sy, inten, (255.0, 178.0, 86.0), 1)

        # the wing beat throws a dimmer second copy a few cells off
        buf.add(g, sx, sy + 5.0 * np.sin(t * 6.0), inten * 0.22,
                (255.0, 120.0, 50.0), 1)
        return compose(buf, g, t)

    run(args, frame, [0.02, 0.16, 0.32, 0.48, 0.64, 0.80, 0.93])


if __name__ == "__main__":
    main()
