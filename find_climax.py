#!/usr/bin/env python3
"""Find the 20 seconds of a track where the voice is doing the most.

Scores every 20s window on vocal-band energy (200-1500 Hz), loudness and
pitch movement, then prints the best candidates.
"""
import subprocess
import sys
import tempfile
import wave

import numpy as np

FFMPEG = "ffmpeg"


def load(path):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", path, "-ac", "1",
                    "-ar", "22050", "-sample_fmt", "s16", tmp], check=True)
    with wave.open(tmp) as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0, sr


def main():
    path = sys.argv[1]
    win = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    x, sr = load(path)
    dur = len(x) / sr
    hop = int(sr * 0.25)
    n = max(1, (len(x) - 2048) // hop)
    wnd = np.hanning(2048).astype(np.float32)
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    vmask = (freqs >= 200) & (freqs <= 1500)
    pmask = (freqs >= 150) & (freqs <= 1200)

    rms = np.zeros(n)
    voc = np.zeros(n)
    pit = np.zeros(n)
    for i in range(n):
        s = i * hop
        ch = x[s:s + 2048]
        if len(ch) < 2048:
            ch = np.pad(ch, (0, 2048 - len(ch)))
        rms[i] = np.sqrt(np.mean(ch ** 2))
        mag = np.abs(np.fft.rfft(ch * wnd))
        tot = mag.sum() + 1e-9
        voc[i] = np.sqrt(np.mean(mag[vmask] ** 2)) / (tot / len(mag) + 1e-9)
        band = np.where(pmask, mag, 0)
        pit[i] = freqs[int(np.argmax(band))]

    t = np.arange(n) * 0.25
    per = int(win / 0.25)
    best = []
    for i in range(0, n - per, max(1, int(4 / 0.25))):
        sl = slice(i, i + per)
        v = voc[sl].mean()
        r = rms[sl].mean()
        # pitch movement: how much the f0 wanders in the window
        pv = pit[sl]
        pv = pv[pv > 60]
        spread = np.std(np.log2(pv)) if len(pv) > 4 else 0
        score = v * (0.4 + 0.6 * (r / (rms.max() + 1e-9))) * (0.5 + 2.0 * spread)
        best.append((score, t[i], v, r, spread))
    best.sort(reverse=True)

    print(f"{path}  {dur:.2f}s")
    seen = []
    for sc, ts, v, r, sp in best:
        if any(abs(ts - s) < win * 0.6 for s in seen):
            continue
        seen.append(ts)
        print(f"  start {ts:6.2f}s  ->  {ts + win:6.2f}s   score {sc:.3f}   "
              f"voice {v:.2f}  rms {r:.3f}  pitchspread {sp:.3f}")
        if len(seen) >= 5:
            break


if __name__ == "__main__":
    main()
