#!/usr/bin/env python3
"""
THE GREAT GIG IN THE SKY — 20s climax, rendered as a character-field film.

Concept: a wordless voice tears a seam in the sky.
Every pixel that matters is a glyph. The vocal becomes a column of vowel
letters; the sky is dust made of punctuation; at the peak the light
disperses into colour and the column burns out.

Pipeline: INPUT -> ANALYZE -> SCENE -> TONEMAP -> SHADE -> ENCODE
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# --------------------------------------------------------------------------
# palettes
# --------------------------------------------------------------------------
PAL_DUST = " .\u00b7`':;"
PAL_SPARK = " \u00b7'\u02c4^\u02c6*"
# the voice is made of vowels and phonetic marks — a wordless scream is,
# literally, phonemes without words.
PAL_VOICE = "aeiouy\u00e6\u0251\u0259\u025c\u025b\u026a\u0254\u028a\u028f\u0276"
HEAT_S = np.array([0.0, 0.30, 0.58, 0.82, 1.0])
PAL_TEXT = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,'"


def find_font(size):
    cands = [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        r"C:\Windows\Fonts\cour.ttf",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("no mono font")


class Grid:
    def __init__(self, vw, vh, font_size, chars):
        self.vw, self.vh = vw, vh
        self.font = ImageFont.truetype(find_font(font_size), font_size)
        asc, desc = self.font.getmetrics()
        bbox = self.font.getbbox("M")
        self.cw = max(1, bbox[2] - bbox[0])
        self.ch = asc + desc
        self.cols = vw // self.cw
        self.rows = vh // self.ch
        self.ox = (vw - self.cols * self.cw) // 2
        self.oy = (vh - self.rows * self.ch) // 2

        self.chars = list(chars)
        self.index = {c: i for i, c in enumerate(self.chars)}
        self.bm = np.zeros((len(self.chars), self.ch, self.cw), np.float32)
        for i, c in enumerate(self.chars):
            if c == " ":
                continue
            img = Image.new("L", (self.cw, self.ch), 0)
            ImageDraw.Draw(img).text((0, 0), c, fill=255, font=self.font)
            self.bm[i] = np.array(img, np.float32) / 255.0

        # cell-centre coordinates in pixels
        xs = self.ox + (np.arange(self.cols) + 0.5) * self.cw
        ys = self.oy + (np.arange(self.rows) + 0.5) * self.ch
        self.px = xs[None, :].repeat(self.rows, 0)
        self.py = ys[:, None].repeat(self.cols, 1)
        self.rr = np.arange(self.rows, dtype=np.float32)[:, None]
        self.cc = np.arange(self.cols, dtype=np.float32)[None, :]

    def idx(self, chars):
        out = np.zeros(chars.shape, np.int16)
        for c, i in self.index.items():
            if i:
                out[chars == c] = i
        return out

    def render(self, idx_grid, color_grid):
        """(rows,cols) int idx + (rows,cols,3) float colour -> float canvas."""
        bm = self.bm[idx_grid]                     # R,C,ch,cw
        a = bm.transpose(0, 2, 1, 3).reshape(self.rows * self.ch,
                                             self.cols * self.cw)
        col = np.repeat(np.repeat(color_grid, self.ch, axis=0), self.cw, axis=1)
        return a[:, :, None] * col


# --------------------------------------------------------------------------
# audio analysis
# --------------------------------------------------------------------------
BANDS = [
    ("sub", 20, 80), ("bass", 80, 250), ("lomid", 250, 500),
    ("mid", 500, 2000), ("himid", 2000, 6000), ("hi", 6000, 20000),
]


def ema(a, alpha):
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def normalize(a):
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def analyze(path, fps, start=0.0, dur=None):
    # unique per call: renders run several at a time, and a fixed name meant
    # one process read the wav another was still writing (EOFError)
    fd, tmp = tempfile.mkstemp(prefix="gg_analysis_", suffix=".wav")
    os.close(fd)
    cmd = [FFMPEG, "-y", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", path, "-ac", "1", "-ar", "22050", "-sample_fmt", "s16", tmp]
    subprocess.run(cmd, check=True, capture_output=True)

    with wave.open(tmp) as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    try:
        os.remove(tmp)
    except OSError:
        pass
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    total = len(x) / sr
    n = int(math.floor(total * fps))
    hop = sr // fps
    win = hop * 2
    wnd = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)

    lo_lag, hi_lag = int(sr / 1000), int(sr / 180)
    feats = {k: np.zeros(n, np.float32) for k in
             ["rms", "flux", "cent", "pitch", "voice"] + [b[0] for b in BANDS]}
    prev = None
    for i in range(n):
        s = i * hop
        chunk = x[s:s + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        feats["rms"][i] = float(np.sqrt(np.mean(chunk ** 2)))
        mag = np.abs(np.fft.rfft(chunk * wnd))
        tot = float(mag.sum()) + 1e-9
        for name, lo, hi in BANDS:
            m = (freqs >= lo) & (freqs < hi)
            feats[name][i] = float(np.sqrt(np.mean(mag[m] ** 2)))
        feats["cent"][i] = float((freqs * mag).sum() / tot)
        feats["flux"][i] = 0.0 if prev is None else float(
            np.maximum(0, mag - prev).sum() / tot)
        prev = mag
        # vocal band 250-1200 Hz carries the wail
        m = (freqs >= 250) & (freqs <= 1200)
        feats["voice"][i] = float(np.sqrt(np.mean(mag[m] ** 2)))
        # f0 by autocorrelation of a 180-1000 Hz band-passed copy
        bw = chunk * wnd
        S = np.fft.rfft(bw)
        S[(freqs < 180) | (freqs > 1000)] = 0
        y = np.fft.irfft(S)
        r = np.correlate(y, y, "full")[len(y) - 1:]
        r[:lo_lag] = 0
        r = r / (r[0] + 1e-9)
        k = int(np.argmax(r[lo_lag:hi_lag])) + lo_lag
        feats["pitch"][i] = (sr / k) if r[k] > 0.22 else 0.0

    # normalise — percentile stretch so a wall of voice still breathes
    def stretch(a, p_lo=8, p_hi=94):
        lo_, hi_ = np.percentile(a, [p_lo, p_hi])
        if hi_ - lo_ < 1e-9:
            return np.zeros_like(a)
        return np.clip((a - lo_) / (hi_ - lo_), 0, 1)

    out = {}
    pit = feats["pitch"].astype(np.float64)
    bad = pit < 60
    if bad.all():
        pit[:] = 220.0
    else:
        med = np.median(pit[~bad])
        pit[bad] = med
        # median filter kills octave errors, then EMA calms it
        k = 7
        pad = np.pad(pit, k // 2, mode="edge")
        pit = np.array([np.median(pad[i:i + k]) for i in range(len(pit))])
    out["pitch"] = ema(stretch(np.log2(np.maximum(pit, 60.0)), 3, 97), 0.18)
    for k, v in feats.items():
        if k == "pitch":
            continue
        if k == "voice":
            out[k] = ema(stretch(v, 8, 96), 0.22)
        elif k in ("rms", "flux") or k in [b[0] for b in BANDS]:
            out[k] = ema(normalize(v), 0.30 if k in ("rms", "flux") else 0.18)
        else:
            out[k] = ema(normalize(v), 0.12)

    # beat / flux pulses
    pk, _ = _dummy_peaks(out["flux"], fps)
    bdecay = np.zeros(n, np.float32)
    for p in pk:
        for d in range(fps // 2):
            if p + d < n:
                bdecay[p + d] = max(bdecay[p + d], math.exp(-d * 3.0 / (fps // 2)))
    out["pulse"] = bdecay
    out["n"] = n
    out["sr"] = sr
    out["duration"] = total
    return out


def _dummy_peaks(flux, fps):
    """cheap peak picker (no scipy dependency for import order)."""
    from scipy.signal import find_peaks
    smooth = np.convolve(flux, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smooth, height=np.percentile(smooth, 82),
                          distance=max(2, fps // 4), prominence=0.03)
    return peaks, smooth


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------
def hsv2rgb(h, s, v):
    h = np.mod(h, 1.0)
    c = v * s
    x = c * (1 - np.abs(np.mod(h * 6.0, 2.0) - 1))
    m = v - c
    r = np.zeros_like(h); g = np.zeros_like(h); b = np.zeros_like(h)
    for k, (rr, gg, bb) in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1),
                                      (0, 1, 0), (1, 0, 0), (0, 1, 0)]):
        pass
    sec = (h * 6).astype(int) % 6
    r = np.where(sec == 0, c, np.where(sec == 1, x, np.where(sec == 2, 0,
        np.where(sec == 3, 0, np.where(sec == 4, x, c)))))
    g = np.where(sec == 0, x, np.where(sec == 1, c, np.where(sec == 2, c,
        np.where(sec == 3, x, np.where(sec == 4, 0, 0)))))
    b = np.where(sec == 0, 0, np.where(sec == 1, 0, np.where(sec == 2, x,
        np.where(sec == 3, c, np.where(sec == 4, c, x)))))
    return (np.clip(r + m, 0, 1) * 255, np.clip(g + m, 0, 1) * 255,
            np.clip(b + m, 0, 1) * 255)


def mkc(r, g, b, rows, cols):
    o = np.zeros((rows, cols, 3), np.float32)
    o[:, :, 0] = r; o[:, :, 1] = g; o[:, :, 2] = b
    return o


def _hash2(ix, iy, seed=0):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 +
         np.int64(seed) * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h & 0x7FFFFFFF
    return (h % 65536).astype(np.float32) / 65535.0


def vnoise(x, y, seed=0):
    """value noise: hash lattice + smoothstep, vectorised."""
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    xi = x0.astype(np.int64)
    yi = y0.astype(np.int64)
    n00 = _hash2(xi, yi, seed)
    n10 = _hash2(xi + 1, yi, seed)
    n01 = _hash2(xi, yi + 1, seed)
    n11 = _hash2(xi + 1, yi + 1, seed)
    a = n00 + (n10 - n00) * fx
    b = n01 + (n11 - n01) * fx
    return a + (b - a) * fy


def fbm(px, py, t, scale, octaves=4, seed=0.0):
    """fractal value noise, returns -1..1."""
    px, py = np.broadcast_arrays(np.asarray(px, np.float32),
                                 np.asarray(py, np.float32))
    v = np.zeros(px.shape, np.float32)
    amp = 1.0
    tot = 0.0
    f = scale
    for o in range(octaves):
        v += amp * vnoise(px * f, py * f, int(seed * 97) + o * 131)
        tot += amp
        amp *= 0.5
        f *= 2.03
    return (v / tot - 0.5) * 2.0


def tonemap(c, gamma=0.8):
    f = c.astype(np.float32)
    lo, hi = np.percentile(f[::4, ::4], [1, 99.5])
    if hi - lo < 10:
        hi = lo + 10
    f = np.clip((f - lo) / (hi - lo), 0, 1) ** gamma
    return (f * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------
class Scene:
    def __init__(self, grid, W, H, fps):
        self.g = grid
        self.W, self.H, self.fps = W, H, fps
        self.S = {}

    def field(self, f, t):
        g = self.g
        W, H = self.W, self.H
        px, py = g.px, g.py
        R, C = g.rows, g.cols

        E, V, P = f["rms"], f["voice"], f["pitch"]
        FL = f["flux"]

        yh = H * 0.705
        above = py < yh
        y_up = np.clip(yh - py, 0, None)
        rip = np.clip(py - yh, 0, None)
        Yr = yh + rip * 0.55
        X = px + rip * 0.10 * np.sin(py * 0.06 + t * 2.2)
        Ys = np.where(above, py, Yr)

        cx = (W * 0.5 + W * 0.006 * math.sin(t * 0.63)
              + W * 0.003 * math.sin(t * 1.77 + 1.0))

        # ---- per-cell glyph lottery ---------------------------------
        hb = np.int64((int(t * 12) * 2654435761) & 0x7FFFFFFF)
        h = ((g.rr.astype(np.int64) * 73856093) ^
             (g.cc.astype(np.int64) * 19349663) ^ hb)
        h = ((h * 2654435761) & 0xFFFFFF)
        rnd = ((h % 4096) / 4096.0).astype(np.float32)
        r2 = (((h >> 7) % 4096) / 4096.0).astype(np.float32)
        r3 = (((h >> 13) % 4096) / 4096.0).astype(np.float32)

        # ---- sky: three parallax dust layers -------------------------
        push = 0.06 * V * V
        Xd = X + (px - cx) * push
        d1 = fbm(Xd * 0.00055, Ys * 0.00085, t * 0.10, 1.0, 3, 1.1)
        d2 = fbm(Xd * 0.00170, Ys * 0.00260, t * 0.22, 1.0, 3, 4.3)
        d3 = fbm(Xd * 0.00520, Ys * 0.00800, t * 0.45, 1.0, 2, 7.9)
        dens = np.clip(0.52 * d1 + 0.30 * d2 + 0.18 * d3 + 0.5, 0, 1)
        dens = np.clip((dens - 0.40) / 0.55, 0, 1) ** 1.15
        fade = np.exp(-y_up / (H * 0.55))
        dust_p = (dens ** 2.4) * (0.10 + 0.45 * E) * (0.06 + 0.94 * fade)
        dust_p += (r3 > 0.9975) * np.exp(-y_up / (H * 0.85)) * (0.25 + 0.40 * E)
        band = np.exp(-((py - yh) / (H * 0.020)) ** 2)
        dust_p = dust_p + band * (0.30 + 0.50 * E) * (0.4 + 0.6 * V)
        dust_p = np.clip(dust_p * np.where(above, 1.0, 0.26), 0, 1)

        # ---- the tear ------------------------------------------------
        jit = 8.0 * V * fbm(np.float32(0.0), g.rr * 0.07, t * 1.4, 1.0, 2, 2.2)
        lev = np.clip((yh - py) / H, 0, 1)
        sway = (W * 0.012) * np.sin((yh - py) * 0.0032 - t * 0.9) * \
            (lev ** 1.2) * (0.4 + 0.6 * V)
        dx = X - cx - jit - sway

        y_top = yh - H * (0.18 + 0.70 * V + 0.14 * P)
        span = np.clip((yh - py) / max(yh - y_top, 1.0), 0, 1.0)
        en = fbm(np.float32(0.0), (yh - py) * 0.014 + t * 0.7,
                 t * 0.45, 1.0, 2, 5.1)
        half = W * (0.009 + 0.026 * V) * (1.0 + 0.35 * en) * (1.05 - 0.20 * span)
        vmask = (0.5 * (1 + np.tanh((py - y_top) / (H * 0.07)))) * (py < yh)
        body = (np.exp(-(np.abs(dx) / half) ** 2.6) * vmask *
                (0.35 + 0.65 * V) * (0.68 + 0.32 * (1.0 - span)))
        wings = np.exp(-(np.abs(dx) / (half * 2.6)) ** 2) * vmask * 0.10 * V
        core = np.exp(-(dx / (W * 0.0022 + 1.0)) ** 2) * vmask * (0.5 + 0.5 * V)
        glow_base = (np.exp(-(dx / (W * 0.30)) ** 2) *
                     np.exp(-(np.abs(py - yh) / (H * 0.055)) ** 2) * 0.4 * V)

        # ---- shock rings ---------------------------------------------
        rings = self.S.setdefault("rings", [])
        if FL > 0.72 and (not rings or t - rings[-1] > 0.34):
            rings.append(t)
        rings = [r for r in rings if t - r < 2.2]
        self.S["rings"] = rings
        d = np.sqrt((px - cx) ** 2 + (py - yh) ** 2)
        ringf = np.zeros((R, C), np.float32)
        for r0 in rings:
            age = t - r0
            rad = 150 + 720 * age
            ringf += np.exp(-((d - rad) / (26 + 18 * age)) ** 2) * \
                math.exp(-age * 1.7)

        # ---- sparks rising out of the tear ---------------------------
        sp = self.S.get("sp")
        if sp is None:
            n = 220
            rng = np.random.default_rng(7)
            sp = self.S["sp"] = {
                "x": rng.uniform(0, W, n), "y": rng.uniform(0, H, n),
                "vy": rng.uniform(10, 60, n), "vx": rng.uniform(-10, 10, n),
                "life": rng.uniform(0, 1, n),
            }
        dt = 1.0 / self.fps
        sp["y"] -= (sp["vy"] * (0.4 + 2.0 * V)) * dt
        sp["x"] += sp["vx"] * dt * (0.3 + V)
        sp["life"] -= dt * 0.35
        dead = sp["life"] <= 0
        if dead.any():
            k = int(dead.sum())
            rng = np.random.default_rng(int(t * 1000) % 99991)
            sp["x"][dead] = cx + rng.normal(0, float(half.mean()) * 0.8, k)
            sp["y"][dead] = yh - rng.uniform(0, 60, k)
            sp["life"][dead] = rng.uniform(0.5, 1.4, k)
            sp["vy"][dead] = rng.uniform(10, 60, k)
        spx = np.clip((sp["x"] - g.ox) / g.cw, 0, C - 1).astype(int)
        spy = np.clip((sp["y"] - g.oy) / g.ch, 0, R - 1).astype(int)
        spark = np.zeros((R, C), np.float32)
        np.add.at(spark, (spy, spx), np.clip(sp["life"], 0, 1))

        # ---- glyphs ---------------------------------------------------
        beam_p = np.clip(body + wings + core, 0, 1)
        lit_p = np.clip(dust_p + beam_p * 1.1 + ringf * 0.5 +
                        glow_base * 0.6 + spark * 0.7, 0, 1)
        lit = rnd < lit_p

        dust_ch = np.full((R, C), ".", dtype="U1")
        dust_ch = np.where(r2 > 0.90, ":", dust_ch)
        dust_ch = np.where(r2 > 0.98, "'", dust_ch)
        dust_ch = np.where(r2 > 0.9965, "*", dust_ch)

        vi = (r3 * (len(PAL_VOICE) - 1.001)).astype(int)
        voice_ch = np.array(list(PAL_VOICE))[np.clip(vi, 0, len(PAL_VOICE) - 1)]

        in_beam = (beam_p > 0.20) & lit
        chars = np.where(in_beam, voice_ch, dust_ch)
        chars = np.where(core > 0.55, "\u2502", chars)
        chars = np.where((ringf > 0.32) & ~in_beam, "'", chars)
        chars = np.where(lit, chars, " ")

        # ---- colour ----------------------------------------------------
        # the tear is heat, not paint: white core, amber body, ember edge
        u = np.clip(beam_p * 1.25, 0, 1)
        hr = np.interp(u, HEAT_S, [42, 122, 208, 255, 255])
        hg = np.interp(u, HEAT_S, [7, 26, 78, 176, 244])
        hb = np.interp(u, HEAT_S, [3, 8, 18, 84, 232])
        # the sky stays cold and faint so the tear owns the light
        sr = 14.0 + 26.0 * dens
        sg = 26.0 + 46.0 * dens
        sb = 44.0 + 76.0 * dens
        m = np.clip(beam_p * 1.7, 0, 1)
        r_ = sr + (hr - sr) * m
        g_ = sg + (hg - sg) * m
        b_ = sb + (hb - sb) * m
        # ring + sparks pick up a little heat
        k = np.clip(ringf * 0.55 + spark * 0.7 + glow_base * 0.55, 0, 1)
        r_ = r_ + (215 - r_) * k
        g_ = g_ + (205 - g_) * k
        b_ = b_ + (188 - b_) * k
        # per-cell flicker: some letters burn brighter than their neighbours
        fk = (0.72 + 0.45 * r2)
        r_ = np.clip(r_ * fk, 0, 255)
        g_ = np.clip(g_ * fk, 0, 255)
        b_ = np.clip(b_ * fk, 0, 255)
        return chars, mkc(r_, g_, b_, R, C), dens


def shade(canvas, f, t, W, H, rng):
    """bloom + base fog + vignette + grain."""
    from scipy.ndimage import gaussian_filter
    V = f["voice"]
    # colour-preserving bloom: blur each channel, not a white luminance mask
    sig = 8.0 + 34.0 * V
    glow = gaussian_filter(canvas, sigma=(sig, sig, 0))
    canvas = canvas + glow * (0.50 + 0.90 * V)
    canvas = canvas * (1.06 + 0.22 * V)

    # never pure black: a faint cold fog rising off the horizon
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    base = np.zeros((H, W, 3), np.float32)
    base[:, :, 0] = 3.5 + 5.0 * (1 - yy)
    base[:, :, 1] = 5.0 + 7.0 * (1 - yy)
    base[:, :, 2] = 9.0 + 12.0 * (1 - yy)
    canvas = canvas + base

    # vignette
    yy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    vig = np.clip(1.06 - 0.34 * (xx ** 2 + (yy * 1.15) ** 2), 0.25, 1.0)
    canvas = canvas * vig[:, :, None]
    # grain
    canvas = canvas + rng.normal(0, 2.6, (H, W, 1)).astype(np.float32)
    return np.clip(canvas, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=18)
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--whisper", type=int, default=1)
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()

    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    print(f"[analyze] {n} frames  dur={feats['duration']:.2f}s", flush=True)

    chars_all = "".join(sorted(set(PAL_DUST + PAL_SPARK + PAL_VOICE +
                                  PAL_TEXT + "\u2502")))
    g = Grid(args.w, args.h, args.font, chars_all)
    print(f"[grid] {g.cols}x{g.rows} cell {g.cw}x{g.ch}", flush=True)

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        for tt in [0.06, 0.25, 0.45, 0.62, 0.78, 0.92, 0.99]:
            fi = min(n - 1, int(tt * n))
            f = {k: float(feats[k][fi]) for k in feats
                 if isinstance(feats[k], np.ndarray)}
            sc = Scene(g, args.w, args.h, args.fps)
            ch, co, br = sc.field(f, fi / args.fps)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas.astype(np.float32), f, fi / args.fps,
                           g.cols * g.cw, g.rows * g.ch,
                           np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + g.rows * g.ch, g.ox:g.ox + g.cols * g.cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview, f"f{tt:.2f}.png"))
            print(f"[preview] t={tt:.2f} bright={br.mean():.3f}", flush=True)
        return

    cw, chh = g.cols * g.cw, g.rows * g.ch
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.w}x{args.h}",
           "-r", str(args.fps), "-i", "pipe:0",
           "-ss", f"{args.start:.3f}", "-t", f"{args.dur:.3f}",
           "-i", args.audio,
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-tune", "film",
           "-c:a", "aac", "-b:a", "256k", "-shortest", args.out]
    errlog = open(args.out + ".ffmpeg.log", "w")
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=errlog)

    sc = Scene(g, args.w, args.h, args.fps)
    rng = np.random.default_rng(4)
    for fi in range(n):
        t = fi / args.fps
        f = {k: float(feats[k][fi]) for k in feats
             if isinstance(feats[k], np.ndarray)}
        chars, co, _ = sc.field(f, t)

        # whisper, last 12%: the album's quiet line
        if args.whisper and t > args.dur * 0.87:
            k = min(1.0, (t - args.dur * 0.87) / (args.dur * 0.08))
            txt = "i never said i was frightened of dying"
            row = g.rows - 3
            col0 = (g.cols - len(txt)) // 2
            nvis = int(len(txt) * k)
            fade = 0.55 + 0.45 * (1 - k)
            for i, c in enumerate(txt[:nvis]):
                chars[row, col0 + i] = c
                co[row, col0 + i] = (150 * fade, 150 * fade, 165 * fade)

        canvas = g.render(g.idx(chars), co)
        canvas = shade(canvas, f, t, cw, chh, rng)
        frame = np.zeros((args.h, args.w, 3), np.uint8)
        frame[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
        pipe.stdin.write(frame.tobytes())
        if fi % 60 == 0:
            print(f"[render] {fi}/{n}", flush=True)

    pipe.stdin.close()
    pipe.wait()
    errlog.close()
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
