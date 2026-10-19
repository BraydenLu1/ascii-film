#!/usr/bin/env python3
"""turn reference images into point clouds we can fly a camera through."""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clouds")
os.makedirs(OUT, exist_ok=True)


def silhouette(path, mode, thresh, min_frac=0.02, invert=False):
    im = Image.open(path).convert("L")
    a = np.array(im, np.float32)
    m = (a < thresh) if not invert else (a > thresh)
    # clean up paper grain
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    lab, n = ndimage.label(m)
    if n == 0:
        return []
    areas = ndimage.sum(m, lab, range(1, n + 1))
    big = [i + 1 for i in range(n) if areas[i] > areas.max() * min_frac]
    out = []
    for i in big:
        sub = lab == i
        ys, xs = np.nonzero(sub)
        out.append((xs, ys, a[ys, xs]))
    # left to right
    out.sort(key=lambda t: t[0].mean())
    return out


def to_cloud(xs, ys, lum, n=6000, rng=None, h_ref=None, fill=False):
    """pixel coords -> normalised coords, height 1.0, centred, +-0.5 wide."""
    rng = rng or np.random.default_rng(3)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    cx = (xs.min() + xs.max()) / 2
    # h_ref keeps the figures' true relative heights: the crouched ape must
    # stay half the height of the standing human, or the evolution is lost
    hr = float(h_ref or h)
    ytop = ys.max()
    if fill:
        # the keyed-out body is a solid mass, not a shell: take every pixel
        # (or a uniform subset when n is smaller), not a darkness-weighted
        # sample — weighting by ink leaves the light pencil torso empty
        if n is None or n >= len(xs):
            pick = np.arange(len(xs))
        else:
            pick = rng.choice(len(xs), n, replace=False)
        # sub-pixel jitter so the character grid never locks onto the
        # source pixel grid
        jx = rng.uniform(-0.5, 0.5, len(pick))
        jy = rng.uniform(-0.5, 0.5, len(pick))
        X = (xs[pick] + jx - cx) / hr
        Y = (ytop - ys[pick] - jy) / hr
        return np.stack([X, Y]), (w / hr), pick
    X = (xs - cx) / hr
    Y = (ytop - ys) / hr             # flip: image y down, world y up
    # sample with replacement weighted by darkness (the ink is the body)
    wt = np.clip(1.0 - lum / 255.0, 0.02, 1.0) ** 1.2
    wt = wt / wt.sum()
    pick = rng.choice(len(X), n, p=wt)
    # pick is returned so callers can carry per-pixel channels onto the cloud
    return np.stack([X[pick], Y[pick]]), (w / hr), pick


def save(name, pts2, aspect, extra=None):
    d = {"p": pts2.astype(np.float32), "aspect": np.float32(aspect)}
    if extra:
        d.update(extra)
    np.savez_compressed(os.path.join(OUT, name + ".npz"), **d)
    print(f"  {name}: {pts2.shape[1]} points, aspect {aspect:.2f}")


def blobs(path, thresh, invert=False, opening=3, closing=5):
    im = Image.open(path).convert("L")
    a = np.array(im, np.float32)
    m = (a > thresh) if invert else (a < thresh)
    m = ndimage.binary_opening(m, np.ones((opening, opening)))
    m = ndimage.binary_closing(m, np.ones((closing, closing)))
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 500:
            continue
        out.append((xs, ys, a[ys, xs]))
    return out, a.shape


def main():
    base = r"./refs"
    march = os.path.join(base, "8db6f31c03d5fb0bcd4a044edd44185e_6c8da5.jpg")
    bird = os.path.join(base, "3d9c3af786ab2404208b0c4969c547df_090d9b.jpg")

    # --- the lithograph: keep only the five standing bodies --------------
    # 105 lost the last figure's lighter upper body (only its legs survived);
    # 120 catches every figure while the five stay separate components
    bs, (Hh, Ww) = blobs(march, 120)
    figs = [b for b in bs if b[1].max() > Hh * 0.75 and len(b[0]) > 2000]
    figs.sort(key=lambda t: t[0].mean())
    print(f"march: {len(figs)} figures")
    HREF = max(float(ys.max() - ys.min() + 1) for xs, ys, lum in figs)
    for i, (xs, ys, lum) in enumerate(figs):
        # fill=True: every silhouette pixel becomes a point, so the body is a
        # solid character mass at close-up instead of a stippled shell
        p2, asp, pick = to_cloud(xs, ys, lum, None, np.random.default_rng(10 + i),
                                 h_ref=HREF, fill=True)
        # stroke width: clubs and spears are thin all the way through, limbs
        # are not. The distance transform alone can't tell them apart — every
        # body's own rim is "within 3.5 px of the edge" — so look at the
        # local *maximum* half-width: a 3 px spear stays below 3.5 px even
        # after dilating, a solid torso does not.
        sub = np.zeros((Hh, Ww), bool)
        sub[ys, xs] = True
        dist = ndimage.distance_transform_edt(sub)
        local_w = ndimage.maximum_filter(dist, size=9)
        thin = (local_w[ys, xs] < 3.5).astype(np.float32)
        # ink density per pixel, so the solid body keeps the drawing's internal
        # modelling instead of reading as a flat sticker
        ink = np.clip((120.0 - lum) / 120.0, 0.0, 1.0).astype(np.float32)
        save(f"march{i}", p2, asp, {"thin": thin[pick], "ink": ink[pick]})

    # --- the phoenix: crop away the frame and the UI column, keep the dots
    im = Image.open(bird).convert("L")
    a = np.array(im, np.float32)
    m = a > 110
    m[:150, :] = False
    m[:, :200] = False
    m[:, 1210:] = False
    m[1040:, :] = False
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    ys, xs = np.nonzero(m)
    print(f"phoenix: {len(xs)} lit pixels  bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
    p2, asp, _ = to_cloud(xs, ys, a[ys, xs], 9000, np.random.default_rng(50))
    save("bird0", p2, asp)


if __name__ == "__main__":
    main()
