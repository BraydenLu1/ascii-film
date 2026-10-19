#!/usr/bin/env python3
"""
00 · 浪 — the woodblock wave, keyed by colour.

The print is Prussian blue on cream paper. Only the blue is the wave: every
one of its pixels becomes a letter; the foam — paper showing through the
crest — is a second, sparse cloud that drifts up and dies. The camera tracks
along the crest, left to right, and never stops.
"""
import argparse

import numpy as np
from PIL import Image
from scipy import ndimage

import elib
from elib import Buf, project, compose, run, add_args, IMGS

WAVE = IMGS + r"\9f960c01f66a0d62e22c09858f1efc88_320a47.jpg"


def main():
    ap = argparse.ArgumentParser()
    add_args(ap, "el_wave.mp4")
    ap.add_argument("--pts", type=int, default=52000)
    args = ap.parse_args()

    img = Image.open(WAVE).convert("RGB")
    rgb = np.array(img, np.float32)
    lum = np.array(img.convert("L"), np.float32)
    # key by "not paper": the print's paper is cream (red-heavy, bright), the
    # wave is everything cool and darker — deep Prussian through pale blue.
    # A blue>red test alone kept only the shadows and lost the curl.
    blue = (lum < 185) & ((rgb[:, :, 0] - rgb[:, :, 2]) < 12)
    blue = ndimage.binary_opening(blue, np.ones((2, 2)))
    blue = ndimage.binary_closing(blue, np.ones((3, 3)))
    # foam = paper showing through the crest: bright, close to the wave, and
    # in a neighbourhood that is more than a quarter wave
    dens = ndimage.uniform_filter(blue.astype(np.float32), 41)
    foam = ((lum > 175) & (dens > 0.28) &
            ndimage.binary_dilation(blue, np.ones((12, 12))))
    foam = ndimage.binary_opening(foam, np.ones((2, 2)))

    rng = np.random.default_rng(4)
    ys, xs = np.nonzero(blue)
    pick = rng.choice(len(xs), min(args.pts, len(xs)), replace=False)
    wx, wy = elib.img_to_world(xs[pick], ys[pick], lum.shape)
    ink_b = np.clip((140.0 - lum[ys, xs][pick]) / 110.0, 0.0, 1.0)
    pw = np.stack([wx + rng.uniform(-0.5, 0.5, len(pick)) * 0.0015,
                   wy + rng.uniform(-0.5, 0.5, len(pick)) * 0.0015,
                   rng.normal(0, 0.02, len(pick))]).astype(np.float32)

    # the sky is where the print leaves paper: it reads as an empty corner,
    # so it gets a very faint field of letters — the same air the rest of the
    # film breathes, and the top-right stops being a hole
    sky = ((lum > 176) & (lum < 240) &
           ~ndimage.binary_dilation(blue, np.ones((9, 9))))
    yss, xss = np.nonzero(sky)
    picks = rng.choice(len(xss), min(17000, len(xss)), replace=False)
    sx_, sy_ = elib.img_to_world(xss[picks], yss[picks], lum.shape)
    ps = np.stack([sx_, sy_, rng.normal(0, 0.03, len(picks))]).astype(np.float32)
    sky_tone = np.clip((238.0 - lum[yss, xss][picks]) / 62.0, 0.10, 1.0)

    ysf, xsf = np.nonzero(foam)
    pickf = rng.choice(len(xsf), min(30000, len(xsf)), replace=False)
    fx, fy = elib.img_to_world(xsf[pickf], ysf[pickf], lum.shape)
    pf = np.stack([fx, fy, rng.normal(0, 0.03, len(pickf))]).astype(np.float32)
    print(f"[wave] blue {blue.mean()*100:.1f}%  foam {foam.mean()*100:.2f}%  "
          f"pts {pw.shape[1]}/{pf.shape[1]}", flush=True)

    Z_CAM = -1.0
    XS = (-0.26, 0.26)
    ZOOM = (0.58, 0.68)

    def frame(f, t, g):
        V = f["voice"]
        u = t / args.dur
        cx = XS[0] + (XS[1] - XS[0]) * (u ** 1.15)
        zoom = ZOOM[0] + (ZOOM[1] - ZOOM[0]) * u
        cy = 0.51 + 0.02 * u
        cam = np.array([cx, cy, Z_CAM])
        flen = args.w * 0.92 * zoom
        buf = Buf(g.rows, g.cols)

        sx, sy, z = project(pw, cam, flen, args.w, args.h)
        inten = (0.50 + 0.78 * ink_b) * (0.86 + 0.20 * V)
        col = (46 + 64 * (1.0 - ink_b), 90 + 76 * (1.0 - ink_b),
               150 + 62 * (1.0 - ink_b))
        buf.add(g, sx, sy, inten, col, 1)

        sx, sy, z = project(ps, cam, flen, args.w, args.h)
        buf.add(g, sx, sy, (0.06 + 0.16 * sky_tone) * (0.85 + 0.25 * V),
                (46.0, 66.0, 96.0), 0)

        drift = pf.copy()
        drift[1] += 0.05 * np.sin(np.pi * np.clip(u, 0, 1))
        sx, sy, z = project(drift, cam, flen, args.w, args.h)
        twinkle = 0.72 + 0.40 * np.sin(t * 3.1 + pf[0] * 37.0)
        buf.add(g, sx, sy, (0.85 + 0.55 * V) * twinkle,
                (238.0, 238.0, 226.0), 1)

        return compose(buf, g, t)

    run(args, frame, [0.02, 0.16, 0.32, 0.48, 0.64, 0.80, 0.93])


if __name__ == "__main__":
    main()
