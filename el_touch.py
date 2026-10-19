#!/usr/bin/env python3
"""
02 · 造物 — the two arms, now solid.

Same camera as touch_5s_v2 (one bell, speed divided by the same curve), but
the arms are read from the keyed mask pixel by pixel instead of the old
edge-weighted sample, so they sit in the same visual language as the
evolution line: a filled body of letters, not an outline.
"""
import argparse
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import elib
from elib import Buf, project, compose, run, add_args, CLOUDS, REFS

GAP_X, GAP_Y = 0.80, 0.768
X0, TRAVEL = 0.10, 1.44
ZOOM_LO, ZOOM_HI = 0.56, 1.56
BELL_W = 0.40
ARM_SOLID = 1.20    # world units from the fingertips that stay fully solid
ARM_FADE = 0.14     # world units of smoothstep dissolve after that
ARM_JIT = 0.55      # cut-contour wobble, as a fraction of ARM_FADE
ARM_WAVE = 11       # px: lobe size of the wobble (small = ragged, not a bow)
RIM_TAU = 0.011     # world units: mean reach of the dust just outside the rim
RIM_MAX = 0.040     # world units: no dust further out than this


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def value_noise(shape, cell, rng):
    """Smooth zero-mean unit-std noise over the mask, one lobe per `cell` px."""
    h, w = shape
    gh, gw = max(3, h // cell + 2), max(3, w // cell + 2)
    g = ndimage.gaussian_filter(rng.standard_normal((gh, gw)), 1.3,
                                mode="nearest")
    Y, X = np.meshgrid(np.linspace(0, gh - 1, h), np.linspace(0, gw - 1, w),
                       indexing="ij")
    out = ndimage.map_coordinates(g, [Y, X], order=1, mode="nearest")
    out -= out.mean()
    return out / max(float(out.std()), 1e-6)


def border_fade(xs, ys, shape, band=150.0):
    """1 inside the picture, 0 at its edge — so a crop never shows as a wall."""
    h, w = shape[:2]
    d = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys))
    return np.clip(d / band, 0.0, 1.0).astype(np.float32)


def arm_axis(comp, side):
    """Origin and unit direction to measure a point's distance from the tips.

    The keyed mask follows the skin straight into the sitter's body: the right
    arm drags a slab of Adam's torso along with it, so the arm has to end
    somewhere out on the forearm. Nothing is cut here — the caller dissolves
    the fringe instead, so the shoulder never shows a straight seam.
    """
    ys, xs = np.nonzero(comp)
    x0, x1 = float(xs.min()), float(xs.max())
    # axis slope from the top edge of the limb — the fingers never touch it
    tx, ty = [], []
    for x in range(int(x0 + (x1 - x0) * 0.15), int(x0 + (x1 - x0) * 0.90), 3):
        col = np.nonzero(comp[:, x])[0]
        if len(col) > 6:
            tx.append(x)
            ty.append(float(col.min()))
    slope = float(np.polyfit(tx, ty, 1)[0]) if len(tx) > 6 else 0.0
    slope = float(np.clip(slope, -0.6, 0.6))
    if side == 1:                       # right arm: fingers left, body right
        sgn, tip = 1.0, x0
    else:                               # left arm: fingers right, body left
        sgn, tip = -1.0, x1
    col = np.nonzero(comp[:, int(tip)])[0]
    ty = float(col.mean()) if len(col) else 0.5 * (ys.min() + ys.max())
    ux, uy = sgn, slope * sgn
    nn = math.hypot(ux, uy)
    return tip, ty, ux / nn, uy / nn


def load_arms(n, rng):
    mask = np.array(Image.open(REFS + r"\hands_mask.png").convert("L")) > 128
    lab, nl = ndimage.label(mask)
    sz = ndimage.sum(mask, lab, range(1, nl + 1))
    two = sorted(list(np.argsort(sz)[-2:] + 1),
                 key=lambda k: np.nonzero(lab == k)[1].mean())
    lum = np.array(Image.open(REFS + r"\Creation_of_Adam__Michelangelo__Detail.jpg")
                   .convert("L"), np.float32)
    # skin has to be shaded against *itself*, not against the bright plaster
    # behind it: measured against the picture's global range the whole arm sat
    # inside a 0.10-0.45 band and every letter came out the same grey.
    lo, hi = np.percentile(lum[mask], [6, 96])
    ink_px = np.clip((lum - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    # how close a pixel is to the limb's own outline — the arm must not be a
    # solid slab of type: at this zoom the limb is three times the size it has
    # in the evolution line, and a filled mass of letters at that scale reads
    # as a wall of text, not as skin.
    edt_all = ndimage.distance_transform_edt(mask)
    edge_all = np.exp(-edt_all / 14.0)
    # two independent noise fields: one wobbles the shoulder dissolve, one the
    # dust along the keyed rim, so neither edge can settle into a line
    nz_cut = value_noise(mask.shape, ARM_WAVE, rng)
    nz_big = value_noise(mask.shape, 60, rng)
    nz_rim = value_noise(mask.shape, 22, rng)
    P, INK, SIDE, WGT, DUST, EDGE = [], [], [], [], [], []
    for k, side in zip(two, ("l", "r")):
        comp = ndimage.binary_opening(lab == k, np.ones((3, 3)))
        comp = ndimage.binary_closing(comp, np.ones((3, 3)))
        ys, xs = np.nonzero(comp)
        bx0, bx1 = float(xs.min()), float(xs.max())
        by0, by1 = float(ys.min()), float(ys.max())
        ref = np.load(CLOUDS + os.sep + f"hand_{side}.npz")["p"]
        x0, x1 = float(ref[0].min()), float(ref[0].max())
        y0, y1 = float(ref[1].min()), float(ref[1].max())
        # the world mapping always uses the *keyed* bbox, so the camera framing
        # of the hands is unchanged no matter how the arm dissolves
        ppu = (bx1 - bx0) / max(x1 - x0, 1e-6)
        ox, oy, ux, uy = arm_axis(comp, 0 if side == "l" else 1)

        def to_world(px, py):
            return (x0 + (px - bx0) / max(bx1 - bx0, 1) * (x1 - x0),
                    y0 + (by1 - py) / max(by1 - by0, 1) * (y1 - y0))

        # 1. the shoulder. The mask runs on into the sitter's body, so
        #    everything past ARM_SOLID dissolves over ARM_FADE world units of
        #    smoothstep, its contour pushed around by noise. No plane cut
        #    anywhere, so the torso cannot leave a straight seam.
        axis = ((xs - ox) * ux + (ys - oy) * uy) / ppu
        jit = (ARM_JIT * nz_cut[ys, xs] + 0.9 * nz_big[ys, xs]) * ARM_FADE
        w = 1.0 - smoothstep((axis - ARM_SOLID + jit) / ARM_FADE)
        # the source photograph's own edges must never reach the screen: where
        # the keyed arm runs into the crop, the point cloud would end on a
        # perfectly straight vertical wall of letters.
        w = w * border_fade(xs, ys, mask.shape)
        keep = w > 0.14
        X, Y = to_world(xs[keep].astype(np.float32),
                        ys[keep].astype(np.float32))
        P.append(np.stack([X, Y]))
        INK.append(ink_px[ys[keep], xs[keep]].astype(np.float32))
        SIDE.append(np.zeros(len(X), np.int8) + (0 if side == "l" else 1))
        WGT.append(w[keep].astype(np.float32))
        EDGE.append(edge_all[ys[keep], xs[keep]].astype(np.float32))
        DUST.append(np.zeros(len(X), bool))

        # 2. the keyed rim. The mask's outer edge is a long straight run in
        #    places — the right hand's fingers were keyed off along one flat
        #    row, the underside of the forearm is another. Keep the body at
        #    full size and scatter dust *outside* it: a rim that thins out
        #    with distance can never read as a ruler line, and the hand never
        #    gets shorter.
        din = ndimage.distance_transform_edt(comp)
        dout = ndimage.distance_transform_edt(~comp)
        bys, bxs = np.nonzero(comp & ~ndimage.binary_erosion(comp))
        gy, gx = np.gradient(dout)
        nx, ny = gx[bys, bxs], gy[bys, bxs]
        ln = np.hypot(nx, ny)
        ln[ln < 1e-6] = 1.0
        nx, ny = nx / ln, ny / ln
        # fan the dust out along the normal, with a little sideways wander
        ang = rng.normal(0.0, 0.45, len(bxs))
        ca, sa = np.cos(ang), np.sin(ang)
        r = np.minimum(rng.exponential(RIM_TAU, len(bxs)), RIM_MAX)
        off = r * ppu
        hx = bxs + (nx * ca - ny * sa) * off
        hy = bys + (nx * sa + ny * ca) * off
        good = (hx > 0) & (hx < mask.shape[1] - 1) & (hy > 0) & (hy < mask.shape[0] - 1)
        hx, hy, r = hx[good], hy[good], r[good]
        src = np.nonzero(good)[0]
        wh = np.exp(-r / RIM_TAU) * np.clip(
            0.50 + 0.55 * nz_rim[bys[src], bxs[src]], 0.05, 1.0)
        wh = wh * border_fade(hx, hy, mask.shape)
        X, Y = to_world(hx.astype(np.float32), hy.astype(np.float32))
        P.append(np.stack([X, Y]))
        INK.append(ink_px[bys[src], bxs[src]].astype(np.float32))
        SIDE.append(np.zeros(len(X), np.int8) + (0 if side == "l" else 1))
        WGT.append(wh.astype(np.float32))
        EDGE.append(np.zeros(len(X), np.float32))
        DUST.append(np.ones(len(X), bool))

        print(f"[touch] arm {side}: {int((lab == k).sum())} px keyed,"
              f" {int(keep.sum())} body, {len(hx)} dust"
              f"  bbox x[{bx0:.0f},{bx1:.0f}] y[{by0:.0f},{by1:.0f}]",
              flush=True)
    X = np.concatenate([p[0] for p in P])
    Y = np.concatenate([p[1] for p in P])
    ink = np.concatenate(INK)
    side = np.concatenate(SIDE)
    wgt = np.concatenate(WGT)
    # the body keeps its old density; the rim dust is one point per boundary
    # pixel, so give it its own share of the budget instead of letting the
    # body's area swamp it
    dust = np.concatenate(DUST)
    edge = np.concatenate(EDGE)
    pick = []
    nbody = n if n > 0 else int((~dust).sum())
    # the outline keeps full density; the interior thins out so the mass has air
    prob = np.sqrt(np.clip(wgt, 0.05, 1.0)) * np.where(dust, 1.0,
                                                       0.22 + 0.78 * edge)
    for sel, cnt in ((~dust, nbody),
                     (dust, max(1, int(round(0.30 * nbody))))):
        idx = np.nonzero(sel)[0]
        if cnt >= len(idx):
            pick.append(idx)          # every keyed pixel is a point
            continue
        p = prob[idx] / prob[idx].sum()
        pick.append(rng.choice(idx, cnt, replace=False, p=p))
    pick = np.concatenate(pick)
    m = len(pick)
    q = np.stack([X[pick] + rng.uniform(-0.5, 0.5, m) * 0.0018,
                  Y[pick] + rng.uniform(-0.5, 0.5, m) * 0.0018,
                  rng.normal(0, 0.012, m)]).astype(np.float32)
    return q, ink[pick].astype(np.float32), side[pick], wgt[pick]


def main():
    ap = argparse.ArgumentParser()
    add_args(ap, "el_touch.mp4")
    ap.add_argument("--pts", type=int, default=0)   # 0 = one point per pixel
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    arms, ink, side, fade = load_arms(args.pts, rng)
    rim = elib.cloud_edge(arms)
    print(f"[touch] arms {arms.shape[1]} pts  x[{arms[0].min():.2f},"
          f"{arms[0].max():.2f}] y[{arms[1].min():.2f},{arms[1].max():.2f}]",
          flush=True)

    KT = (GAP_X - X0) / TRAVEL
    _K = np.linspace(0.0, 1.0, 4001)
    _U = (_K - KT) / BELL_W
    _B = np.where(np.abs(_U) < 1.0,
                  0.5 * (1.0 + np.cos(np.pi * np.clip(_U, -1.0, 1.0))), 0.0)
    _Z = ZOOM_LO + (ZOOM_HI - ZOOM_LO) * _B
    _S = np.cumsum(_Z ** -0.9)
    _S = (_S - _S[0]) / (_S[-1] - _S[0])

    def S(k):
        return float(np.interp(k, _K, _S))

    # spark constellation
    SP_N = 4200
    sp_a = rng.random(SP_N) * 2 * math.pi
    sp_r = rng.random(SP_N) ** 2.4 * 0.040
    sp_z = rng.normal(0, 0.004, SP_N)

    def frame(f, t, g):
        V = f["voice"]
        k = t / max(1e-6, args.dur)
        uu = (k - KT) / BELL_W
        bell = 0.5 * (1.0 + math.cos(math.pi * uu)) if abs(uu) < 1.0 else 0.0
        cx = X0 + TRAVEL * S(k)
        cy = 0.78
        cz = -0.95
        zoom = ZOOM_LO + (ZOOM_HI - ZOOM_LO) * bell
        cam = np.array([cx, cy, cz])
        flen = args.w * 0.92 * zoom
        spark = math.exp(-((t - KT * args.dur) / 0.38) ** 2)
        buf = Buf(g.rows, g.cols)

        sx, sy, z = project(arms, cam, flen, args.w, args.h)
        # the arm has to read as a solid lit limb: shading is what gives it a
        # form, a flat intensity is what read as fog
        inten = (0.30 + 0.92 * ink) * (0.74 + 0.40 * rim) * \
            (0.88 + 0.20 * V) * np.power(fade, 0.55)
        near = np.exp(-(((arms[0] - GAP_X) ** 2 +
                         (arms[1] - GAP_Y) ** 2) / 0.010))
        inten = inten * (1.0 + 1.3 * spark * near)
        col = (236 - 40 * ink, 224 - 40 * ink, 202 - 30 * ink)
        buf.add(g, sx, sy, inten, col, 1)

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

    run(args, frame, [0.02, 0.20, 0.50, 0.72, 0.95])


if __name__ == "__main__":
    main()
