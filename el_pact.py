#!/usr/bin/env python3
"""
04 · 契约 — two men shake hands in a sunlit street; the right-hand man burns.

The cover is a clean studio plate: bright street, white walls, blue sky, and
both men complete from hair to shoes.  So the matte is built the honest way,
at the cover's own 489 px, without any downscale:

  body   the suits are far darker than street or wall, so a luminance cut at
         78 catches them; the lit faces and hands are not dark, they are warm
         and saturated, so a second term (R-B > 24, sat > 0.22) catches the
         skin a luminance cut would drop.  Everything is clipped to the
         pair's strip, opened, closed, and the largest component kept — the
         two men are one piece because the clasp joins them.  Only enclosed
         holes under 500 px are filled, so the daylight between the two arms
         stays open instead of turning the pair into one slab.  The cast
         shadow under the shoes is a flat grey tongue no colour cut can
         separate from the trousers, so the bottom band keeps only the shoe
         boxes.
  fire   the cover's own orange (R-B > 42, sat > 0.30), kept only within a
         dilated body matte so no ember can land in the empty street.

The fire runs in two populations: a core locked to the keyed flame that
flickers, and licks that climb out of it and cool from yellow to red as they
go, plus sparks that leave the fire and die.

Camera holds both whole men for the full five seconds and leans in gently;
the framing is checked at the widest and the tightest zoom so no hair, hand
or shoe ever leaves the frame.
"""
import argparse

import numpy as np
from PIL import Image
from scipy import ndimage

import elib
from elib import Buf, project, compose, run, add_args, IMGS

COVER = IMGS + r"\5408811f0df5e8b5074508fd8a0690c0_652763.jpg"

# The cover is 489 px square and the character grid is 274x83, so at native
# resolution barely two source pixels cover a cell: whole stretches of the
# pair fell between samples and the men came out as a pencil dot-field.  Key
# at 2x (bicubic) so every cell has 4-7 source pixels to average and the
# silhouette closes up.
S = 2

X0, X1, Y0, Y1 = 104, 383, 86, 462      # the strip the two men stand in
FOOT_L = (126, 224, 451)                # left shoe box  x0, x1, sole y
FOOT_R = (326, 384, 459)                # right shoe box
CLASP = (264.0, 252.0)                  # the hands, in cover pixels
BACK_BOX = (146, 224, 136, 240)         # his lit upper back


# --------------------------------------------------------------------------
def key_layers(verbose=True):
    im = Image.open(COVER).convert("RGB")
    a = np.array(im.resize((im.width * S, im.height * S), Image.LANCZOS),
                 np.float32)
    h, w, _ = a.shape
    R, B = a[:, :, 0], a[:, :, 2]
    lum = a.mean(2)
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    warm = R - B

    # ---- fire: the cover's own orange ------------------------------------
    fl = (warm > 42) & (R > 110) & (sat > 0.30)
    fl = ndimage.binary_closing(fl, np.ones((3, 3)))
    lab, n = ndimage.label(fl)
    sz = ndimage.sum(fl, lab, range(1, n + 1))
    fl = np.isin(lab, [k + 1 for k in range(n) if sz[k] >= 35 * S * S])

    # ---- body ------------------------------------------------------------
    dark = lum < 78                       # suits, hair, shoes
    skin = (warm > 24) & (sat > 0.22) & (lum > 62) & (lum < 205)
    body = (dark | skin) & ~ndimage.binary_dilation(fl, np.ones((3, 3)))
    body[:, X1 * S:] = False              # loading dock behind his shoulder
    body[:Y0 * S] = False
    body[Y1 * S:] = False
    body[:, :X0 * S] = False
    body = ndimage.binary_opening(body, np.ones((3, 3)))
    body = ndimage.binary_closing(body, np.ones((7, 7)))
    lab, n = ndimage.label(body)
    sz = ndimage.sum(body, lab, range(1, n + 1))
    body = lab == (int(np.argmax(sz)) + 1)

    # his lit upper back: grey coat, brighter than 78 but far below the white
    # wall.  Grow the suit into it, but only inside the box his shoulder owns.
    yy, xx = np.mgrid[0:h, 0:w]
    bx0, bx1, by0, by1 = [v * S for v in BACK_BOX]
    box = (xx >= bx0) & (xx <= bx1) & (yy >= by0) & (yy <= by1)
    body |= ndimage.binary_propagation(body & box, mask=box & (lum < 172))

    # ---- the cast shadow is a flat tongue at the shoes -------------------
    below = yy > 440 * S
    feet = (xx >= FOOT_L[0] * S) & (xx <= FOOT_L[1] * S) & (yy <= FOOT_L[2] * S)
    feet |= (xx >= FOOT_R[0] * S) & (xx <= FOOT_R[1] * S) & (yy <= FOOT_R[2] * S)
    body[below & ~feet] = False

    # small enclosed holes (a lit face, a specular fold) are filled; the big
    # gap between the two arms must stay open or the pair turns into a slab,
    # and so must the daylight between the legs — that one is closed at the
    # bottom by the shoe shadow, so holes are only filled above the coat hem.
    holes = ndimage.binary_fill_holes(body) & ~body & (yy < 340 * S)
    hl, hn = ndimage.label(holes)
    hsz = ndimage.sum(holes, hl, range(1, hn + 1))
    body |= np.isin(hl, [k + 1 for k in range(hn) if hsz[k] < 500 * S * S])
    body = ndimage.binary_closing(body, np.ones((3, 3)))

    # fire only survives on him or just beside him
    body |= fl & ndimage.binary_dilation(body, np.ones((3, 3)))
    fl = fl & ndimage.binary_dilation(body, np.ones((29 * S, 29 * S)))

    if verbose:
        ys, xs = np.nonzero(body)
        print(f"[pact] body {int(body.sum())} px  fire {int(fl.sum())} px  "
              f"bbox x {xs.min()}..{xs.max()} y {ys.min()}..{ys.max()}",
              flush=True)
        for y in (109, 120, 140, 165, 200, 250, 300, 400, 450):
            xr = np.nonzero(body[y])[0]
            print(f"   row {y:4d}: " +
                  (f"{xr.min()}..{xr.max()} n={len(xr)}" if len(xr) else "-"),
                  flush=True)
    return body, fl, lum


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    add_args(ap, "el_pact.mp4")
    args = ap.parse_args()

    body, fl, lum = key_layers()
    h, w = body.shape

    # photo shading: 0 in the black of the suit, 1 on a lit face.  Measured
    # against the whole plate the two suits sat in a 0.1-0.4 band and every
    # cell of the coats came out the same '#'; normalising inside the matte
    # (fire excluded) lets the folds and the two bodies separate.
    bl = lum[body & ~fl]
    ilo, ihi = np.percentile(bl, [4, 97])
    ink = np.clip((lum - ilo) / max(float(ihi - ilo), 1e-6), 0, 1)
    edge = 1.0 - np.clip(ndimage.distance_transform_edt(body) / (4.0 * S), 0, 1)

    rng = np.random.default_rng(12)

    # ---- body point cloud: one point per matte pixel ---------------------
    ys, xs = np.nonzero(body)
    bx, by = elib.img_to_world(xs, ys, body.shape)
    body_p = np.stack([bx + rng.uniform(-0.5, 0.5, len(xs)) / w,
                       by + rng.uniform(-0.5, 0.5, len(xs)) / w,
                       np.zeros(len(xs), np.float32)]).astype(np.float32)
    body_ink = ink[ys, xs].astype(np.float32)
    body_rim = edge[ys, xs].astype(np.float32)

    # ---- fire point cloud ------------------------------------------------
    fys, fxs = np.nonzero(fl)
    rep = 3
    fys = np.tile(fys, rep)
    fxs = np.tile(fxs, rep)
    fx, fy = elib.img_to_world(fxs, fys, body.shape)
    flame = np.stack([fx + rng.uniform(-0.6, 0.6, len(fxs)) / w,
                      fy + rng.uniform(-0.6, 0.6, len(fxs)) / w,
                      rng.normal(0, 0.012, len(fxs))]).astype(np.float32)

    nf = flame.shape[1]
    # core keeps the keyed shape; licks climb out of it
    is_lick = (rng.random(nf) < 0.52).astype(np.float32)
    f_ph = rng.uniform(0, 2 * np.pi, nf).astype(np.float32)
    f_life = rng.uniform(0.0, 1.0, nf).astype(np.float32)
    f_rate = rng.uniform(0.55, 1.25, nf).astype(np.float32)
    f_rise = rng.uniform(0.040, 0.150, nf).astype(np.float32)
    f_jx = rng.normal(0, 0.005, nf).astype(np.float32)

    # sparks that leave the fire and go out
    sp_n = 380
    spi = rng.choice(nf, sp_n, replace=False)
    spark = flame[:, spi].copy()
    spark_t = rng.uniform(0, 1.6, sp_n).astype(np.float32)
    spark_v = rng.uniform(0.22, 0.55, sp_n).astype(np.float32)
    spark_sw = rng.uniform(0.004, 0.014, sp_n).astype(np.float32)

    # the fire warms the cloth next to it
    fdist = ndimage.distance_transform_edt(~fl)
    body_glow = np.exp(-fdist[ys, xs] / (5.0 * S)).astype(np.float32)

    gx, gy = elib.img_to_world(np.array([CLASP[0]]), np.array([CLASP[1]]),
                               body.shape)
    gx, gy = float(gx[0]), float(gy[0])
    gd2 = (body_p[0] - gx) ** 2 + (body_p[1] - gy) ** 2
    clasp = np.exp(-gd2 / (2 * 0.017 ** 2)).astype(np.float32)

    # ---- camera: both men whole at every moment --------------------------
    # tight enough that a head is six character rows, not four: at 83 rows a
    # wide full-body framing turns the faces into blobs and the clasp into a
    # smudge, and the pair stops reading as two men.
    Z0, Z1 = 0.620, 0.800
    CX0, CX1 = -0.004, 0.010
    CY = 0.448

    def smooth(u):
        u = min(max(u, 0.0), 1.0)
        return u * u * (3.0 - 2.0 * u)

    def frame(f, t, g):
        V = f["voice"]
        u = t / max(args.dur, 1e-6)
        e = smooth(u)
        cam = np.array([CX0 + (CX1 - CX0) * e, CY, -1.0])
        zoom = Z0 + (Z1 - Z0) * e
        flen = args.w * 0.92 * zoom
        buf = Buf(g.rows, g.cols)

        # ---- the two men -------------------------------------------------
        # One point per matte pixel blows the silhouette up by a cell on
        # every edge, because the buffer keeps only the brightest point in a
        # cell: the head then merges into the shoulders and the pair reads as
        # two slabs.  Fold the cloud into the character grid first —
        # coverage gives the edge its antialiasing, mean ink keeps the coat's
        # folds, and the neck survives.
        sx, sy, _ = project(body_p, cam, flen, args.w, args.h)
        cc = np.floor((sx - g.ox) / g.cw).astype(np.int32)
        rr = np.floor((sy - g.oy) / g.ch).astype(np.int32)
        ok = (cc >= 0) & (cc < g.cols) & (rr >= 0) & (rr < g.rows)
        idx = rr[ok] * g.cols + cc[ok]
        ncell = g.rows * g.cols
        cnt = np.bincount(idx, minlength=ncell).astype(np.float32)
        ink_c = np.bincount(idx, weights=body_ink[ok], minlength=ncell)
        rim_c = np.bincount(idx, weights=body_rim[ok], minlength=ncell)
        glw_c = np.bincount(idx, weights=body_glow[ok], minlength=ncell)
        clp_c = np.bincount(idx, weights=clasp[ok], minlength=ncell)
        occ = cnt > 0
        ref = max(float(np.percentile(cnt[occ], 70)), 1.0)
        cov = np.clip(cnt / ref, 0, 1.30)
        inv = 1.0 / np.maximum(cnt, 1.0)
        ink_c, rim_c = ink_c * inv, rim_c * inv
        glw_c, clp_c = glw_c * inv, clp_c * inv
        inten = ((0.50 + 0.78 * ink_c + 0.18 * rim_c) *
                 (0.42 + 0.64 * cov) * (0.85 + 0.20 * V) +
                 0.40 * glw_c * cov)
        colr = np.clip(196 + 44 * ink_c + 30 * glw_c, 0, 255)
        colg = np.clip(190 + 36 * ink_c + 10 * glw_c, 0, 255)
        colb = np.clip(180 + 26 * ink_c - 22 * glw_c, 0, 255)
        cy_, cx_ = np.nonzero(occ.reshape(g.rows, g.cols))
        # class 4 keeps the brightness ramp, so the glyph density — not just
        # the colour — carries the photograph: the coat writes in '=' and the
        # lit face in '@'.  Class 1 would scatter letters evenly and read as a
        # wall of type.
        shade = np.clip(0.34 + 0.62 * ink_c + 0.16 * rim_c +
                        0.45 * glw_c, 0.0, 1.10)
        buf.add(g, g.ox + (cx_ + 0.5) * g.cw, g.oy + (cy_ + 0.5) * g.ch,
                inten[occ], (colr[occ] * shade[occ], colg[occ] * shade[occ],
                             colb[occ] * shade[occ]), 4)

        # ---- the fire ----------------------------------------------------
        life = np.mod(f_life + t * f_rate, 1.0)
        heat = 0.55 + 0.45 * np.sin(t * 5.4 + f_ph * 1.7)
        rise = f_rise * np.power(life, 0.85) * is_lick
        wob = (0.0035 * np.sin(t * 4.6 + f_ph) +
               0.5 * f_jx * np.sin(t * 2.3 + f_ph * 3.1)) * \
            (0.30 + is_lick * life)
        sx, sy, _ = project(np.stack([flame[0] + wob, flame[1] + rise,
                                      flame[2]]),
                            cam, flen, args.w, args.h)
        fade = 1.0 - 0.88 * is_lick * np.power(life, 0.75)
        # the core is a solid flame body — ramp glyphs, so the heat shows as
        # #%@ instead of a dusting of sparks; the licks keep the fire pool and
        # cool to red as they climb.  Gold, not white: white reads as a lamp.
        core = is_lick < 0.5
        heat_c = heat[core]
        buf.add(g, sx[core], sy[core],
                (0.93 + 0.10 * heat_c) * (0.94 + 0.12 * V),
                (255.0, 142.0 + 78.0 * heat_c, 34.0 + 66.0 * heat_c), 4)
        buf.add(g, sx[~core], sy[~core],
                (1.05 * fade[~core]) * (0.92 + 0.18 * V),
                (255.0, 196.0 - 128.0 * life[~core],
                 96.0 - 76.0 * life[~core]), 2)

        # sparks climbing out of the fire and going out
        sl = np.mod(spark_t + t, 1.5) / 1.5
        spz = np.stack([spark[0] + spark_sw * np.sin(t * 5.0 + spark_t * 30),
                        spark[1] + spark_v * sl,
                        spark[2]])
        sx, sy, _ = project(spz, cam, flen, args.w, args.h)
        h2 = np.clip(1.0 - sl, 0, 1) ** 1.5
        buf.add(g, sx, sy, (0.5 + 2.1 * h2) * (0.8 + 0.3 * V),
                (255.0, 168.0 + 50 * h2, 70.0 + 60 * h2), 2)

        # ---- the clasp catches the light ---------------------------------
        m2 = (clp_c > 0.02).reshape(g.rows, g.cols)
        if m2.any():
            gy2, gx2 = np.nonzero(m2)
            buf.add(g, g.ox + (gx2 + 0.5) * g.cw, g.oy + (gy2 + 0.5) * g.ch,
                    clp_c.reshape(g.rows, g.cols)[m2] * (1.35 + 0.7 * V),
                    (255.0, 236.0, 202.0), 2)

        return compose(buf, g, t)

    run(args, frame, [0.02, 0.16, 0.32, 0.48, 0.64, 0.80, 0.93])


if __name__ == "__main__":
    main()
