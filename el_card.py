#!/usr/bin/env python3
"""
07 · 名字 — the spectrum is still running when the card condenses out of it.

The old version faded the code field down to almost nothing and *then* started
the card, so on the film's timeline the join read as a hard cut to black.  The
field is now a camera move instead of a fade: the characters are sampled
through a lens that accelerates toward the vanishing point, so they stream out
past the viewer while the card resolves in their place.  Something is always
lit — the field only goes dark where the card has already taken over.

Timing at dur 7 s:
    t 0.00-2.10   the code field, lens starting to push
    t 2.10-4.60   the push accelerates to ~10x; the card resolves left to right
    t 4.60-7.00   the card, prompt typing, cursor blinking
"""
import argparse

import numpy as np

import elib
import march
from elib import run, add_args

SOUP = "!@#$%&*<>[]{}/\\|~^+=?0123456789abcdefABCDEF"
RAMP = np.array(list(" .:-=+*#%@"), dtype="U1")


def spectral(u):
    xs = [0.0, 0.10, 0.25, 0.40, 0.55, 0.72, 0.88, 1.0]
    r = np.interp(u, xs, [90, 45, 20, 10, 90, 215, 255, 255])
    g = np.interp(u, xs, [20, 70, 150, 220, 255, 220, 120, 40])
    b = np.interp(u, xs, [200, 255, 255, 200, 60, 20, 10, 10])
    return r, g, b


def frac(x):
    return x - np.floor(x)


def main():
    ap = argparse.ArgumentParser()
    add_args(ap, "el_card.mp4")
    args = ap.parse_args()

    def frame(f, t, g):
        V = f["voice"]
        dur = args.dur
        # The card starts resolving here, and make_card's clock is shifted so
        # its reveal lands on this moment instead of at dur-2.35.
        T0 = 0.30 * dur
        PUSH = 0.36 * dur      # length of the accelerating push
        card_t = t - T0 + (dur - 2.35)
        q = float(np.clip((t - T0) / PUSH, 0.0, 1.0))
        amount = float(np.clip((card_t - (dur - 2.35)) / 1.05, 0.0, 1.0))

        # ---- the code field, seen through a lens that is pushing in -------
        # Sampling coordinates contract toward the vanishing point; because the
        # field is a hash of those coordinates, the pattern streams outward
        # past the lens instead of simply rescaling.  The old version faded
        # this field to 26% and then switched to the card, which on the film's
        # timeline read as a hard cut to black; now it never empties on its
        # own — it is eaten away only where the card has landed.
        z = 1.0 - 0.90 * q * q
        cx = 0.5 * g.cols
        cy = 0.44 * g.rows
        cc = np.arange(g.cols)[None, :].astype(np.float32)
        rr = np.arange(g.rows)[:, None].astype(np.float32)
        sc = cx + (cc - cx) * z
        sr = cy + (rr - cy) * z

        h = frac(np.sin(sr * 12.9898 + sc * 78.233 + t * 9.0) * 43758.5453)
        h2 = frac(np.sin(sr * 39.346 + sc * 11.135 + t * 3.7) * 24634.6345)
        keep = h < (0.92 - 0.10 * q)

        chars = np.full((g.rows, g.cols), " ", dtype="U1")
        n = int(keep.sum())
        rng = np.random.default_rng(int(t * 24) % 99991)
        if n:
            chars[keep] = rng.choice(list(SOUP), n)

        hue = np.clip((sc / g.cols) * 0.95 + 0.02 + 0.07 * (h2 - 0.5), 0, 1)
        r, gg, b = spectral(hue)
        # the field brightens as it comes at the lens — flying into the light,
        # not fading out of it
        glow = (0.78 + 0.70 * q) * (0.82 + 0.30 * V)
        colors = np.stack([r, gg, b], axis=2) * glow
        colors = np.where(keep[:, :, None], colors, 0.0)

        if amount <= 0.0:
            return chars, colors

        # ---- the card condenses out of the field -------------------------
        cch, ccol = march.make_card(g, card_t, dur)
        on = cch != " "
        chars[on] = cch[on]
        colors[on] = ccol[on]
        rest = (1.0 - amount) ** 1.4
        colors[~on] *= rest
        return chars, colors

    run(args, frame, [0.02, 0.16, 0.30, 0.42, 0.50, 0.62, 0.74, 0.88, 0.97])


if __name__ == "__main__":
    main()
