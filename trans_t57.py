#!/usr/bin/env python3
"""
T5→T7 · 红鸟入卡 — the phoenix fills the lens, then becomes the mark on the card.

The bird keeps coming until its body is the whole screen and the screen is
red. That red field is then read as the phoenix glyph of the end card, huge;
it pulls back to its place at the top of the layout, and the card types in
underneath it. One move, no cut: the transition is the same red mass at two
different scales.
"""
import argparse

import numpy as np

import elib
import march
from elib import Buf, project, compose, run, add_args, CLOUDS

RED = np.array([255.0, 86.0, 48.0], np.float32)


def main():
    ap = argparse.ArgumentParser()
    add_args(ap, "trans_t57.mp4")
    ap.add_argument("--pts", type=int, default=42000)
    args = ap.parse_args()

    d = np.load(CLOUDS + r"\bird0.npz")
    p = d["p"].astype(np.float32)
    src = p.shape[1]
    rng = np.random.default_rng(77)
    idx = rng.choice(src, args.pts)
    bird = np.stack([p[0][idx] + rng.normal(0, 0.008, args.pts),
                     p[1][idx] + rng.normal(0, 0.008, args.pts),
                     rng.normal(0, 0.010, args.pts)]).astype(np.float32)
    print(f"[t57] bird {src} -> {args.pts}", flush=True)

    T_FULL = 1.55          # the bird's body has covered the frame
    T_GLYPH = 2.05         # the red field is now the glyph
    T_SETTLE = 3.45        # the glyph is back at its place on the card
    S_TOP = 7.5            # how much bigger than final the glyph starts

    layers = {}

    def card_masks(g):
        if g.rows not in layers:
            rows, cols = g.rows, g.cols
            layers[g.rows] = (
                march._phoenix(cols, rows, 0.125, 0.20, 0.165),
                march._blk(cols, rows, "ASCII", 0.72, 0.385, 0.285),
                march._blk(cols, rows, "\u00d7", 0.03, 0.545, 0.035),
                march._blk(cols, rows, "FILM", 0.40, 0.680, 0.20),
            )
        return layers[g.rows]

    def smooth(u):
        u = min(max(float(u), 0.0), 1.0)
        return u * u * (3.0 - 2.0 * u)

    def glyph_at(g, s, cy_frac):
        """the phoenix mark, scaled by s around a moving centre"""
        m = card_masks(g)[0]
        rows, cols = m.shape
        ys, xs = np.nonzero(m)
        cr, cc = ys.mean(), xs.mean()
        # while it is huge it is centred on the screen, then it slides up
        # to its own position as it shrinks
        c_dst_r = cr + (rows * 0.5 - cr) * cy_frac
        c_dst_c = cols * 0.5
        rr = (np.arange(rows)[:, None] - c_dst_r) / s + cr
        cc_ = (np.arange(cols)[None, :] - c_dst_c) / s + cc
        r0 = np.clip(np.floor(rr).astype(int), 0, rows - 1)
        c0 = np.clip(np.floor(cc_).astype(int), 0, cols - 1)
        r1 = np.clip(r0 + 1, 0, rows - 1)
        c1 = np.clip(c0 + 1, 0, cols - 1)
        fr = np.clip(rr - r0, 0, 1)
        fc = np.clip(cc_ - c0, 0, 1)
        return (m[r0, c0] * (1 - fr) * (1 - fc) + m[r0, c1] * (1 - fr) * fc +
                m[r1, c0] * fr * (1 - fc) + m[r1, c1] * fr * fc)

    def frame(f, t, g):
        V = f["voice"]
        rng_f = np.random.default_rng(int(t * 12) % 99991)

        # ---------- phase A: the bird comes at the lens, red --------------
        if t < T_GLYPH:
            u = min(t / T_FULL, 1.0)
            cam = np.array([0.0, 0.46, -1.0])
            zoom = 0.70 + 0.10 * u
            flen = args.w * 0.92 * zoom
            buf = Buf(g.rows, g.cols)
            grow = 1.05 + 0.85 * u ** 1.4
            bz = 2.40 - 2.28 * u ** 1.25
            bp = np.stack([bird[0] * grow, (bird[1] - 0.5) * grow + 0.46,
                           bird[2] * grow + bz])
            sx, sy, z = project(bp, cam, flen, args.w, args.h)
            inten = np.full(args.pts, (1.05 + 0.75 * u) *
                            (0.85 + 0.30 * V), np.float32)
            # as it closes on the lens the body blows out to a flat red field
            wash = smooth((t - (T_FULL - 0.45)) / 0.55)
            inten = inten * (1.0 + 2.2 * wash)
            col = (RED[0], RED[1] * (1.0 - 0.25 * u), RED[2] * (1.0 - 0.30 * u))
            buf.add(g, sx, sy, inten, col, 9)
            chars, colors = compose(buf, g, t)
            if wash > 0.45:
                # red everywhere: the silhouette stops mattering, the screen
                # becomes one warm field of letters
                w = np.clip((wash - 0.45) / 0.55, 0, 1)
                ramp = np.array(list(" .:-=+*#%@"), dtype="U1")
                noise = rng_f.random((g.rows, g.cols))
                lvl = np.clip(0.35 + 0.55 * noise + 0.25 * w, 0, 1)
                fill = ramp[(lvl * 9).astype(int)]
                m = chars == " "
                chars = np.where(m, fill, chars)
                colors = np.where(m[:, :, None],
                                  RED[None, None, :] * (0.35 + 0.75 * lvl)[:, :, None],
                                  colors)
            return chars, colors

        # ---------- phase B/C: the red mass is the card's phoenix ---------
        u = (t - T_GLYPH) / (T_SETTLE - T_GLYPH)
        s = S_TOP * (1.0 - smooth(u)) + 1.0 * smooth(u)
        cy_frac = 1.0 - smooth(u)          # 1 = screen centre, 0 = its slot
        gm = glyph_at(g, max(s, 1.0), cy_frac)
        chars = np.full((g.rows, g.cols), " ", dtype="U1")
        colors = np.zeros((g.rows, g.cols, 3), np.float32)

        heat = 0.55 + 0.45 * np.clip((t - T_GLYPH) / 0.8, 0, 1)
        m = gm > 0.10
        n = int(m.sum())
        if n:
            chars[m] = rng_f.choice(list(" .:-=+*#%@"), n) if s > 1.6 \
                else rng_f.choice(list(elib.ALNUM), n)
            amp = 0.55 + 0.45 * rng_f.random(n)
            colors[m] = (RED[None, :] * (heat * amp)[:, None])

        # ---------- the card types itself in underneath -------------------
        if u > 0.55:
            pm = (u - 0.55) / 0.45
            _, m_left, m_x, m_film = card_masks(g)
            colidx = np.arange(g.cols)[None, :]
            reveal = np.clip(pm * g.cols * 1.1 - colidx, 0, 1)
            for mask, rgb, at in ((m_left, (234, 242, 255), 0.0),
                                  (m_x, (150, 170, 195), 0.15),
                                  (m_film, (255, 196, 120), 0.25)):
                use = mask * reveal * np.clip((pm - at) / 0.25, 0, 1)
                mm = use > 0.12
                if mm.any():
                    chars[mm] = rng_f.choice(list(elib.ALNUM), int(mm.sum()))
                    colors[mm] = np.array(rgb, np.float32) * 0.9

        # prompt + rule, once the mark has settled
        if u > 0.92:
            prompt = "> demo --with film"
            typed = int(np.clip((t - (T_SETTLE + 0.15)) / 0.55, 0, 1) *
                        len(prompt))
            r0 = max(1, int(g.rows * 0.045))
            c0 = max(2, (g.cols - len(prompt) - 2) // 2)
            for i, ch in enumerate(prompt[:typed]):
                if c0 + i < g.cols:
                    chars[r0, c0 + i] = ch
                    colors[r0, c0 + i] = (120.0, 210.0, 170.0)
            if typed < len(prompt) or (t * 2.2) % 1.0 < 0.55:
                cx = c0 + min(typed, len(prompt))
                if cx < g.cols:
                    chars[r0, cx] = "_" if (t * 2.2) % 1.0 < 0.5 else "\u2588"
                    colors[r0, cx] = (120.0, 210.0, 170.0)
            r1 = min(g.rows - 2, int(g.rows * 0.825))
            for i in range(g.cols - 2 * c0):
                if c0 + i < g.cols:
                    chars[r1, c0 + i] = "\u2500"
                    colors[r1, c0 + i] = (38.0, 55.0, 72.0)
        return chars, colors

    run(args, frame, [0.02, 0.14, 0.26, 0.34, 0.42, 0.50, 0.62, 0.74, 0.88, 0.97])


if __name__ == "__main__":
    main()
