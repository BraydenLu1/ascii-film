#!/usr/bin/env python3
"""
03 - expulsion, keyed from the SAW 85-92 sleeve.

The sleeve is a white field holding one heavy black ring and a fat A inside
it. The white is thrown away completely: only ink pixels become characters,
so the figure floats in the void instead of arriving as a framed square. The
credit line at the bottom is dropped by keeping only the two largest ink
components, which also gets rid of the grey type.

Two cuts share the same keyed object and the same left-to-right camera:

  A / door   the ring and the A are one lit mass. The camera tracks right
             while pushing from zoom 0.6 to 1.2, until the ring's opening
             nearly fills the frame and the shot reads as passing through it.
  B / expel  the ring holds its shape while the camera still tracks and
             pushes. The A stays whole for two seconds, then tears into
             characters that are thrown out of the right edge, leaving the
             ring empty.

Nothing here is sampled as grey tone: the ink mask is sampled as points, one
per ink pixel (or a uniform subset), so the silhouette keeps its hard edge at
character resolution.
"""
import argparse

import numpy as np
from PIL import Image
from scipy import ndimage

import elib
from elib import Buf, project, compose, run, add_args, IMGS

COVER = IMGS + r"\4bb15bbfbd9cf6ce74f5b8f5adaeda62_32f3c1.jpg"
INK = 90.0          # black ink; the credit line is grey and stays out


def key_ink():
    """Return point clouds for the ring and the A, in world units."""
    lum = np.array(Image.open(COVER).convert("L"), np.float32)
    hh, ww = lum.shape
    m = lum < INK
    lab, n = ndimage.label(m)
    sz = ndimage.sum(m, lab, range(1, n + 1))

    ring_id = glyph_id = None
    for i in range(1, n + 1):
        if sz[i - 1] < 2000:
            continue
        ys, xs = np.nonzero(lab == i)
        if xs.max() - xs.min() > ww * 0.85:
            ring_id = i
        elif glyph_id is None:
            glyph_id = i
    if ring_id is None or glyph_id is None:
        raise RuntimeError("cover key failed: ring or A not found")

    ring = lab == ring_id
    glyph = lab == glyph_id

    rng = np.random.default_rng(7)
    ring_d = ndimage.distance_transform_edt(ring)
    glyph_d = ndimage.distance_transform_edt(glyph)

    def cloud(mask, dist, npts, edge_px):
        ys, xs = np.nonzero(mask)
        if npts < len(xs):
            pick = rng.choice(len(xs), npts, replace=False)
        else:
            pick = np.arange(len(xs))
        X, Y = elib.img_to_world(xs[pick], ys[pick], mask.shape, 1.0)
        pts = np.stack([X, Y, np.zeros(len(pick), np.float32)])
        rim = 1.0 - np.clip(dist[ys[pick], xs[pick]] / edge_px, 0, 1)
        return pts, rim.astype(np.float32)

    ring_pts, ring_rim = cloud(ring, ring_d, 200000, 9.0)
    glyph_pts, glyph_rim = cloud(glyph, glyph_d, 150000, 14.0)
    print(f"[expel] ink ring={int(ring.sum())}px A={int(glyph.sum())}px  "
          f"points {len(ring_pts[0])}+{len(glyph_pts[0])}", flush=True)
    return ring_pts, ring_rim, glyph_pts, glyph_rim


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def main():
    ap = argparse.ArgumentParser()
    add_args(ap, "el_expel.mp4")
    ap.add_argument("--variant", choices=["A", "B"], default="A")
    args = ap.parse_args()

    ring_pts, ring_rim, glyph_pts, glyph_rim = key_ink()

    # version B flies as characters, not as a filled field: the A is resampled
    # coarse enough that each glyph can separate from its neighbours in flight
    rng = np.random.default_rng(23)
    ji = rng.choice(glyph_pts.shape[1], 7000, replace=False)
    jet_pts = glyph_pts[:, ji]
    jet_rim = glyph_rim[ji]

    nB = jet_pts.shape[1]
    b_speed = rng.uniform(0.50, 1.30, nB).astype(np.float32)
    bx0, bx1 = jet_pts[0].min(), jet_pts[0].max()
    bnx = (jet_pts[0] - bx0) / max(bx1 - bx0, 1e-5)
    b_rel = (1.98 + 0.82 * bnx +
             rng.normal(0, 0.090, nB).astype(np.float32))
    bcx, bcy = jet_pts[0].mean(), jet_pts[1].mean()
    ux = jet_pts[0] - bcx
    uy = jet_pts[1] - bcy
    un = np.sqrt(ux * ux + uy * uy) + 1e-6
    b_dirx = 1.0 + 0.50 * ux / un
    b_diry = 0.18 * uy / un
    dn = np.sqrt(b_dirx * b_dirx + b_diry * b_diry)
    b_dirx = (b_dirx / dn).astype(np.float32)
    b_diry = (b_diry / dn).astype(np.float32)
    b_jx = rng.normal(0, 0.030, nB).astype(np.float32)
    b_jy = rng.normal(0, 0.024, nB).astype(np.float32)

    def door(f, t, g):
        V = f["voice"]
        u = t / max(args.dur, 1e-6)
        e = smooth(u)
        zoom = 0.60 + 0.60 * e
        flen = args.h * zoom / 1.14          # visible height 1.9 -> 0.95
        cam = np.array([-0.340 + 0.360 * e,
                        0.500 + 0.007 * np.sin(t * 1.05), -1.0],
                       np.float32)
        buf = Buf(g.rows, g.cols)

        sx, sy, _ = project(ring_pts, cam, flen, args.w, args.h)
        inten = (1.02 + 0.18 * ring_rim) * (0.86 + 0.20 * V)
        col = (194.0 + 40.0 * ring_rim,
               202.0 + 6.0 * ring_rim,
               222.0 - 38.0 * ring_rim)       # cold body, warm rim
        buf.add(g, sx, sy, inten, col, 1)

        sx, sy, _ = project(glyph_pts, cam, flen, args.w, args.h)
        inten = (1.00 + 0.05 * glyph_rim) * (0.86 + 0.20 * V)
        buf.add(g, sx, sy, inten, (207.0, 215.0, 232.0), 1)
        return compose(buf, g, t)

    def expel(f, t, g):
        V = f["voice"]
        u = t / max(args.dur, 1e-6)
        e = smooth(u)
        zoom = 0.72 + 0.28 * e
        flen = args.h * zoom / 1.024         # visible height 1.42 -> 1.02
        cam = np.array([-0.125 + 0.155 * e, 0.500, -1.0], np.float32)
        buf = Buf(g.rows, g.cols)

        sx, sy, _ = project(ring_pts, cam, flen, args.w, args.h)
        inten = (0.96 + 0.18 * ring_rim) * (0.86 + 0.19 * V)
        col = (194.0 + 34.0 * ring_rim,
               203.0 + 2.0 * ring_rim,
               222.0 - 30.0 * ring_rim)
        buf.add(g, sx, sy, inten, col, 1)

        # the A: whole until two seconds, then a jet of characters going right
        sx, sy, _ = project(jet_pts, cam, flen, args.w, args.h)
        gx, gy, _ = project(np.array([[bcx], [bcy], [0.0]], np.float32),
                            cam, flen, args.w, args.h)
        life = t - b_rel
        sp = np.clip(life, 0.0, None)
        shift = 0.60 * sp ** 1.15          # the whole cloud is already moving right
        growx = 1.0 + 0.35 * sp
        growy = 1.0 + 0.30 * sp
        d = 2.00 * sp ** 1.7 * b_speed
        sx = (gx + (sx - gx) * growx +
              flen * (shift + d * b_dirx + b_jx * sp))
        sy = gy + (sy - gy) * growy - flen * (d * b_diry + b_jy * sp)
        heat = np.clip(life * 0.9, 0.0, 1.0)
        inten = ((1.00 + 0.05 * jet_rim) * (0.86 + 0.19 * V) *
                 (1.0 + 0.45 * np.clip(life * 1.3, 0.0, 1.0)))
        col = (207.0 + 30.0 * heat,
               215.0 + 8.0 * heat,
               232.0 - 38.0 * heat)
        buf.add(g, sx, sy, inten, col, 1)
        return compose(buf, g, t)

    frame = door if args.variant == "A" else expel
    # fractions of the 5 s clip: 0.2, 1.0, 1.8, 2.6, 3.4, 4.4, 4.7, 4.93 s
    run(args, frame, [0.04, 0.20, 0.36, 0.52, 0.68, 0.88, 0.94, 0.99])


if __name__ == "__main__":
    main()
