#!/usr/bin/env python3
"""
THE GREAT GIG IN THE SKY — 5s spatial study.

A camera flies through a cloud of letters. A triangular prism hangs in the
fog. White light (the voice) enters from the left, refracts, and leaves as a
fan of spectral bands — the cover, rebuilt in three dimensions with nothing
but glyphs.

Real perspective projection, per-cell depth buffer, per-point colour.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Grid, analyze, shade, FFMPEG  # noqa: E402

# --------------------------------------------------------------------------
# palettes
# --------------------------------------------------------------------------
DUST = " .....,,::::;;"
GLASS = " .:;|/\\[]"
LIGHT = " =+*#%@"


def spectral_rgb(u):
    """u in 0..1 (400nm..700nm) -> RGB 0..255. violet->blue->green->red."""
    xs = [0.0, 0.10, 0.25, 0.40, 0.55, 0.72, 0.88, 1.0]
    r = np.interp(u, xs, [90, 45, 20, 10, 90, 215, 255, 255])
    g = np.interp(u, xs, [20, 70, 150, 220, 255, 220, 120, 40])
    b = np.interp(u, xs, [200, 255, 255, 200, 60, 20, 10, 10])
    return r, g, b


# --------------------------------------------------------------------------
class Scene3D:
    def __init__(self, grid, W, H, fps, n_points=34000):
        self.g = grid
        self.W, self.H, self.fps = W, H, fps
        rng = np.random.default_rng(11)
        n = n_points
        # a slab of dust around the camera path
        x = rng.normal(0.0, 3.9, n)
        y = rng.normal(-0.15, 2.15, n)
        z = rng.uniform(0.6, 15.0, n)
        self.p = np.stack([x, y, z]).astype(np.float32)
        self.gl = rng.integers(0, 1 << 30, n)

        # A near layer, close enough to the camera path that it whips past and
        # can be streaked.  Without it every object in the shot sits at roughly
        # the prism's depth, so a lateral move reads as a pan across a diagram
        # instead of a camera travelling through a space.
        rng_n = np.random.default_rng(53)
        nn = 2600
        self.near = np.stack([
            rng_n.uniform(-7.4, 1.9, nn),
            rng_n.normal(-0.10, 1.30, nn),
            rng_n.uniform(1.05, 3.40, nn)]).astype(np.float32)

        # prism: equilateral triangle, a real slab of glass (front + depth)
        self.pz = 6.70
        self.R = 1.52
        self.cy = -0.28
        self.thick = 1.10
        a = [math.radians(90 + 120 * k) for k in range(3)]
        self.tri = np.array([[self.R * math.cos(t),
                              self.cy + self.R * math.sin(t)] for t in a])
        # white light leaves the right face just above the centre line
        self.fan_origin = np.array([0.66, 0.10, self.pz], np.float32)
        self.beam_y = 0.10

        self._build_haze()
        self._build_glass()
        self._build_beam()
        self._build_fan()

    # ---- static point clouds ---------------------------------------------
    def _inside(self, x, y):
        A, B, C = self.tri

        def side(p0, p1):
            return ((p1[0] - p0[0]) * (y - p0[1]) -
                    (p1[1] - p0[1]) * (x - p0[0]))

        s1, s2, s3 = side(A, B), side(B, C), side(C, A)
        return (((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) |
                ((s1 <= 0) & (s2 <= 0) & (s3 <= 0)))

    def _tri_sample(self, rng, n):
        u = rng.random(n)
        v = rng.random(n)
        sw = u + v > 1.0
        u[sw] = 1.0 - u[sw]
        v[sw] = 1.0 - v[sw]
        A, B, C = self.tri
        gx = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
        gy = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
        return gx, gy

    def _glass_shade(self, fx, fy):
        """0..1: hot where the white light hits the left face and the top.

        The floor matters: the glyph ramp's low bins ('.', ':') and '-' are
        almost inkless at 12 px, so any patch of the pane that falls below
        ~0.37 in projected intensity vanishes into what reads as a hole.  The
        old 0.30 base plus a 0.10 low-frequency sine carved exactly such a
        bowl in the lower-right interior.  Keep the base high and the ripple
        shallow so the dimmest cell is still a '=' stroke.
        """
        u = np.clip((fx + 1.32) / 2.64, 0, 1)
        h = np.clip((fy - (self.cy - 0.5 * self.R)) / (1.5 * self.R), 0, 1)
        sh = (0.45 + 0.40 * (1.0 - u) ** 0.9 + 0.18 * h
              + 0.04 * np.sin(fy * 6.2 + fx * 5.1)
              + 0.03 * np.sin(fx * 13.0 - fy * 3.0 + 1.2))
        return np.clip(sh, 0.0, 1.0)

    def _build_glass(self):
        rng = np.random.default_rng(23)
        # front face: a jittered lattice (reads as a machined pane) plus
        # random speckle so the surface never looks like graph paper
        d = 0.0335
        gx = np.arange(-1.38, 1.39, d)
        gy = np.arange(self.cy - 0.80, self.cy + 1.56, d)
        GX, GY = np.meshgrid(gx, gy)
        lx = GX.ravel() + rng.normal(0, d * 0.32, GX.size)
        ly = GY.ravel() + rng.normal(0, d * 0.32, GY.size)
        keep = self._inside(lx, ly)
        lx, ly = lx[keep], ly[keep]

        rx, ry = self._tri_sample(rng, 12000)

        fx = np.concatenate([lx, rx])
        fy = np.concatenate([ly, ry])
        fz = np.full(fx.size, self.pz, np.float32) + rng.normal(0, 0.007, fx.size)
        self.face = (np.stack([fx, fy, fz]).astype(np.float32),
                     self._glass_shade(fx, fy).astype(np.float32))

        # back face: same pane one slab-depth behind, dimmer
        bx, by = self._tri_sample(rng, 7000)
        bz = np.full(bx.size, self.pz + self.thick, np.float32)
        self.back = (np.stack([bx, by, bz]).astype(np.float32),
                     (self._glass_shade(bx, by) * 0.52).astype(np.float32))

        # the volume between the faces: the parallax that makes the slab thick
        vx, vy = self._tri_sample(rng, 9000)
        vz = rng.uniform(self.pz + 0.03, self.pz + self.thick - 0.03, vx.size)
        self.vol = (np.stack([vx, vy, vz]).astype(np.float32),
                    (self._glass_shade(vx, vy) * 0.26).astype(np.float32))

        # edges of both faces + the three connectors
        e = []
        for i in range(3):
            p0, p1 = self.tri[i], self.tri[(i + 1) % 3]
            tt = np.linspace(0, 1, 260)
            e.append(np.stack([p0[0] + (p1[0] - p0[0]) * tt,
                               p0[1] + (p1[1] - p0[1]) * tt,
                               np.full(260, self.pz)], axis=0))
        self.edges_f = np.concatenate(e, axis=1).astype(np.float32)

        e = []
        for i in range(3):
            p0, p1 = self.tri[i], self.tri[(i + 1) % 3]
            tt = np.linspace(0, 1, 200)
            e.append(np.stack([p0[0] + (p1[0] - p0[0]) * tt,
                               p0[1] + (p1[1] - p0[1]) * tt,
                               np.full(200, self.pz + self.thick)], axis=0))
        self.edges_b = np.concatenate(e, axis=1).astype(np.float32)

        e = []
        for i in range(3):
            tt = np.linspace(0, 1, 90)
            e.append(np.stack([np.full(90, self.tri[i][0]),
                               np.full(90, self.tri[i][1]),
                               self.pz + self.thick * tt], axis=0))
        self.conn = np.concatenate(e, axis=1).astype(np.float32)

        # the three side walls.  A prism seen off-axis is recognised by its
        # walls, not by its outline: with only two triangular faces and three
        # hairline connectors the slab projected to a flat triangle no matter
        # where the camera stood.  Each wall is a filled quad spanning one edge
        # and the same edge one slab behind, lit by the beam arriving from -x.
        wx, wy, wz, wsh = [], [], [], []
        cen = self.tri.mean(0)
        ld = np.array([-0.86, 0.32])
        ld /= np.hypot(*ld)
        nu, nv = 170, 54
        for i in range(3):
            p0, p1 = self.tri[i], self.tri[(i + 1) % 3]
            eg = p1 - p0
            nrm = np.array([-eg[1], eg[0]])
            nrm /= np.hypot(*nrm)
            if np.dot(nrm, 0.5 * (p0 + p1) - cen) < 0:      # point outward
                nrm = -nrm
            lit = 0.20 + 0.80 * max(0.0, float(np.dot(nrm, ld))) ** 1.25
            TT, SS = np.meshgrid(np.linspace(0.02, 0.98, nu),
                                 np.linspace(0.0, 1.0, nv))
            ttf, ssf = TT.ravel(), SS.ravel()
            wx.append(p0[0] + eg[0] * ttf + rng.normal(0, 0.005, ttf.size))
            wy.append(p0[1] + eg[1] * ttf + rng.normal(0, 0.005, ttf.size))
            wz.append(self.pz + self.thick * ssf)
            # brightest along the front edge, falling away into the slab
            wsh.append((lit * (0.55 + 0.45 * (1.0 - ssf))).astype(np.float32))
        self.walls = (np.stack([np.concatenate(wx), np.concatenate(wy),
                                np.concatenate(wz)]).astype(np.float32),
                      np.concatenate(wsh).astype(np.float32))

    def _build_haze(self):
        """faint lit fog around the prism so it sits in air, not in a void."""
        rng = np.random.default_rng(19)
        n = 14000
        r = 2.9 * rng.random(n) ** 0.65
        th = rng.uniform(0, 2 * np.pi, n)
        ph = np.arccos(rng.uniform(-1, 1, n))
        x = 0.15 + r * np.sin(ph) * np.cos(th)
        y = self.cy + 0.10 + r * np.cos(ph) * 0.75
        z = self.pz + 0.3 + r * np.sin(ph) * np.sin(th) * 0.55
        w = np.clip(1.0 - r / 2.9, 0, 1) ** 1.4
        self.haze = (np.stack([x, y, z]).astype(np.float32),
                     (0.05 + 0.09 * w).astype(np.float32))

    def _build_beam(self):
        rng = np.random.default_rng(31)
        n = 300
        bx = np.linspace(-6.6, -0.58, n)
        q = np.array([-1.75, -1.35, -0.95, -0.60, -0.28, 0.0,
                      0.28, 0.60, 0.95, 1.35, 1.75])
        wq = np.exp(-(q * 0.72) ** 2)
        self.beam = []
        for dz, gz in ((0.0, 1.0), (0.22, 0.55), (0.44, 0.28)):
            xx = np.repeat(bx, len(q))
            qq = np.tile(q, n)
            ww = np.tile(wq, n)
            yy = self.beam_y + qq * 0.041 + 0.010 * np.sin(xx * 2.7 + dz * 9.0)
            zz = np.full(xx.size, self.pz + dz, np.float32)
            self.beam.append((np.stack([xx, yy, zz]).astype(np.float32),
                              (ww * gz).astype(np.float32)))

        # the refracted segment inside the glass — light travelling through
        n2 = 220
        ix = np.linspace(-0.62, 0.64, n2)
        q2 = np.array([-1.4, -1.0, -0.6, -0.25, 0.0, 0.25, 0.6, 1.0, 1.4])
        w2 = np.exp(-(q2 * 0.85) ** 2)
        ix = np.repeat(ix, len(q2))
        qq = np.tile(q2, n2)
        ww = np.tile(w2, n2)
        iy = self.beam_y - 0.05 * (ix + 0.62) + qq * 0.030
        self.inner = (np.stack([ix, iy,
                                np.full(ix.size, self.pz + 0.02,
                                        np.float32)]).astype(np.float32),
                      ww.astype(np.float32))

        # white-hot knot where the light leaves the glass
        n3 = 3200
        rr = 0.19 * rng.random(n3) ** 0.55
        aa = rng.uniform(0, 2 * np.pi, n3)
        kx = self.fan_origin[0] + rr * np.cos(aa) * 1.15
        ky = self.fan_origin[1] + rr * np.sin(aa)
        self.knot = (np.stack([kx, ky,
                               np.full(n3, self.pz + 0.05)]).astype(np.float32),
                     ((1.0 - rr / 0.19) ** 1.3).astype(np.float32))

    def _build_fan(self):
        rng = np.random.default_rng(77)
        nray, ns = 161, 230
        L = 6.8
        s = np.linspace(0.03, L, ns)
        qf = np.array([-1.0, 0.0, 1.0])
        wf = np.array([0.42, 1.0, 0.42])
        qj = np.array([-1.0, 0.0, 1.0])
        wj = np.array([0.36, 1.0, 0.36])

        X, Y, Z, C, W = [], [], [], [], []
        for k, aa in enumerate(np.radians(np.linspace(-23.0, 23.0, nray))):
            u = k / (nray - 1)
            r_, g_, b_ = spectral_rgb(u)
            wfil = 0.016 + 0.074 * (s / L)
            sub = 0.015 + 0.013 * (s / L)
            qq = np.empty((ns, 9), np.float32)
            ww = np.empty((ns, 9), np.float32)
            c = 0
            for fi in range(3):
                for ji in range(3):
                    qq[:, c] = qf[fi] * wfil + qj[ji] * sub
                    ww[:, c] = wf[fi] * wj[ji]
                    c += 1
            qq += rng.normal(0, 0.009, qq.shape).astype(np.float32)
            qq = qq.ravel()
            ww = ww.ravel()
            ss = np.repeat(s, 9)
            ca, sa = math.cos(aa), math.sin(aa)
            px = self.fan_origin[0] + ss * ca - qq * sa
            py = self.fan_origin[1] + ss * sa * 1.015 + qq * ca
            pz = (self.pz + 0.02 + 0.010 * ss +
                  0.02 * rng.standard_normal(ss.size))
            hot = np.exp(-ss / 1.15)
            rr = r_ + (252.0 - r_) * hot
            gg = g_ + (250.0 - g_) * hot
            bb = b_ + (255.0 - b_) * hot
            edge = 1.0 - 0.28 * abs(2 * u - 1.0) ** 2
            wg = (np.exp(-ss * 0.125) * ww * edge *
                  (0.70 + 0.30 * np.exp(-ss / 0.55)))
            X.append(px); Y.append(py); Z.append(pz)
            C.append(np.stack([rr, gg, bb]))
            W.append(wg)

        self.fan = (np.stack([np.concatenate(X), np.concatenate(Y),
                              np.concatenate(Z)]).astype(np.float32),
                    np.concatenate(C, axis=1).astype(np.float32),
                    np.concatenate(W).astype(np.float32))

        # a dim wide wash behind the crisp bands — the air the fan lives in
        nray2, ns2 = 81, 90
        s2 = np.linspace(0.05, L * 1.05, ns2)
        X, Y, Z, C, W = [], [], [], [], []
        for k, aa in enumerate(np.radians(np.linspace(-34.0, 34.0, nray2))):
            u = k / (nray2 - 1)
            r_, g_, b_ = spectral_rgb(u)
            qq = rng.normal(0, 0.16, (ns2, 5)).ravel()
            ss = np.repeat(s2, 5)
            ca, sa = math.cos(aa), math.sin(aa)
            X.append(self.fan_origin[0] + ss * ca - qq * sa)
            Y.append(self.fan_origin[1] + ss * sa * 1.09 + qq * ca)
            Z.append(self.pz - 0.05 + 0.02 * ss)
            C.append(np.stack([np.full(ss.size, r_), np.full(ss.size, g_),
                               np.full(ss.size, b_)]))
            W.append(np.exp(-ss * 0.16) * 0.13)
        self.wash = (np.stack([np.concatenate(X), np.concatenate(Y),
                               np.concatenate(Z)]).astype(np.float32),
                     np.concatenate(C, axis=1).astype(np.float32),
                     np.concatenate(W).astype(np.float32))

    # ---- geometry helpers -------------------------------------------------
    def project(self, pts, cam, yaw, roll, f):
        X = pts[0] - cam[0]
        Y = pts[1] - cam[1]
        Z = pts[2] - cam[2]
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        Xr = X * cy_ - Z * sy_
        Zr = X * sy_ + Z * cy_
        cr, sr = math.cos(roll), math.sin(roll)
        Yr = Y * cr - Xr * sr
        Xr2 = Xr * cr + Y * sr
        Zr = np.maximum(Zr, 0.15)
        sx = self.W * 0.5 + f * Xr2 / Zr
        sy = self.H * 0.5 - f * Yr / Zr
        return sx, sy, Zr

    def splat(self, sx, sy, z, rgb, inten):
        g = self.g
        inten = np.broadcast_to(np.asarray(inten, np.float32), sx.shape)
        rr = np.broadcast_to(np.asarray(rgb[0], np.float32), sx.shape)
        gg = np.broadcast_to(np.asarray(rgb[1], np.float32), sx.shape)
        bb = np.broadcast_to(np.asarray(rgb[2], np.float32), sx.shape)
        col = ((sx - g.ox) / g.cw).astype(np.int32)
        row = ((sy - g.oy) / g.ch).astype(np.int32)
        ok = ((col >= 0) & (col < g.cols) & (row >= 0) & (row < g.rows) &
              (inten > 0.004))
        if not ok.any():
            return None
        idx = row[ok] * g.cols + col[ok]
        val = inten[ok]
        size = g.cols * g.rows
        best = np.zeros(size, np.float32)
        np.maximum.at(best, idx, val)
        # colour: weighted mean over the splats that land in this cell
        w = val.astype(np.float32)
        ar = np.bincount(idx, weights=w * rr[ok], minlength=size)
        ag = np.bincount(idx, weights=w * gg[ok], minlength=size)
        ab = np.bincount(idx, weights=w * bb[ok], minlength=size)
        wt = np.bincount(idx, weights=w, minlength=size)
        return best.reshape(g.rows, g.cols), (ar / (wt + 1e-6)).reshape(g.rows, g.cols), \
            (ag / (wt + 1e-6)).reshape(g.rows, g.cols), (ab / (wt + 1e-6)).reshape(g.rows, g.cols)

    def stamp(self, sx, sy, z, rgb, inten, rad=1.0, halo=0.34, taps=8):
        """soft halo around a stroke: re-splat at a small disc of pixel
        offsets so thin curves gain body and a glow skirt."""
        g = self.g
        layers = [(0.0, 0.0, 1.0)]
        for i in range(taps):
            a = 2.0 * math.pi * i / taps
            layers.append((rad * math.cos(a), rad * math.sin(a), halo))
        out = None
        for dx, dy, w in layers:
            res = self.splat(sx + dx * g.cw, sy + dy * g.ch, z, rgb, inten * w)
            if res is None:
                continue
            v, r_, g_, b_ = res
            if out is None:
                out = [v.copy(), r_.copy(), g_.copy(), b_.copy()]
            else:
                m = v > out[0]
                out[0][m] = v[m]
                out[1][m] = r_[m]
                out[2][m] = g_[m]
                out[3][m] = b_[m]
        return None if out is None else tuple(out)

    # ---- camera -----------------------------------------------------------
    def camera(self, t, dur):
        """Camera state for time t.  A single eased k drives the truck, the
        push-in and the yaw, so every axis is monotonic — no back-and-forth.

        The lateral move is integrated against 1/zoom (focal compensation):
        the lens tightens as the camera travels, so the world covers less
        screen per unit of travel and the push reads even instead of
        accelerating.
        """
        W = self.W
        ZOOM0, ZOOM1 = 0.50, 1.24              # f_len / W  (0.50x -> 1.24x)
        # The truck now swings around the glass instead of sliding past it in
        # the plane of the picture: at the wide end the camera sits 4.8 units
        # to the left of the prism, so the slab is seen at ~37 degrees and its
        # near face, far face and the three side walls all separate.  Staying
        # off-axis for the whole move is what makes it read as an object in
        # space rather than a diagram of a triangle.
        CAM_X0, CAM_X1 = -4.80, -1.10
        CAM_Y0, CAM_Y1 = -0.50, 0.18           # rise: below the beam, then above
        k = min(max(t / max(dur, 1e-6), 0.0), 1.0)
        e = k * k * (3.0 - 2.0 * k)            # soft start, soft settle
        zoom = ZOOM0 + (ZOOM1 - ZOOM0) * e
        lat = math.log(zoom / ZOOM0) / math.log(ZOOM1 / ZOOM0)
        cam = np.array([CAM_X0 + (CAM_X1 - CAM_X0) * lat,
                        CAM_Y0 + (CAM_Y1 - CAM_Y0) * e,
                        0.30 + 1.30 * e], np.float32)
        f_len = W * zoom
        roll = 0.055 * math.sin(t * 0.7 + 0.5)
        # The truck alone drags the prism off to the left as the camera
        # passes it, and re-aiming at the glass every frame cancels the
        # lateral read.  So aim at a point on screen instead: solve the yaw
        # that lands the prism centre on an explicit 30.5% -> 61.0% width
        # path.  The pan therefore lags the truck by exactly the amount the
        # glass has to slide right — one continuous move, no back-and-forth.
        p_screen = 0.305 + (0.610 - 0.305) * e
        off = W * (p_screen - 0.5)
        dx, dz = -float(cam[0]), self.pz - float(cam[2])
        dy = self.cy - float(cam[1])
        cr, sr = math.cos(roll), math.sin(roll)
        a = (off / f_len) / cr
        for _ in range(4):
            zr = math.hypot(dx, dz) / math.sqrt(1.0 + a * a)
            a = (off / f_len - dy * sr / zr) / cr
        yaw = math.atan2(dx - a * dz, a * dx + dz)
        return cam, yaw, roll, f_len

    # ---- frame ------------------------------------------------------------
    def frame(self, f, t, dur):
        g = self.g
        W, H = self.W, self.H
        V, E, P = f["voice"], f["rms"], f["pitch"]

        cam, yaw, roll, f_len = self.camera(t, dur)

        canvases = []

        # ---- 1. dust cloud ------------------------------------------------
        pts = self.p.copy()
        behind = pts[2] < cam[2] + 0.30
        pts[2][behind] += 14.5
        sx, sy, z = self.project(pts, cam, yaw, roll, f_len)
        fog = np.exp(-(z - cam[2]) * 0.075)
        inten = (0.24 + 0.34 * E) * fog * np.clip(1.7 / z, 0.0, 1.5)
        inten = inten * np.where(self.gl % 83 == 0, 2.4, 1.0)
        near = np.clip(1.7 / z, 0, 1)
        rgb = (30 + 34 * near, 46 + 48 * near, 76 + 78 * near)
        canvases.append((self.splat(sx, sy, z, rgb, inten), DUST))

        # ---- 1b. haze around the prism ------------------------------------
        hp, hw = self.haze
        sx, sy, z = self.project(hp, cam, yaw, roll, f_len)
        hz = np.exp(-(z - cam[2]) * 0.10) * (0.75 + 0.45 * V)
        canvases.append((self.splat(sx, sy, z, (44.0, 62.0, 96.0),
                                    hw * hz * 1.5), GLASS))

        # ---- 1b. near dust, streaked --------------------------------------
        # The only layer close enough to sell the camera's travel: each speck
        # is drawn as a short trail whose length follows 1/depth, so the
        # nearest ones smear and the far ones stay points.  This is what turns
        # a lateral slide into a dolly through a space.
        sx_n, sy_n, z_n = self.project(self.near, cam, yaw, roll, f_len)
        vis = ((z_n > cam[2] + 0.20) & (sx_n > -320) & (sx_n < W + 320) &
               (sy_n > -160) & (sy_n < H + 160))
        if vis.any():
            dep = np.maximum(z_n[vis] - cam[2], 0.20)
            ln = np.clip(1500.0 / dep, 3.0, 110.0)
            amp = np.clip(0.60 * (1.0 - dep / 4.2), 0.02, 0.60)
            for kk in range(6):
                fk = kk / 5.0
                canvases.append((self.splat(
                    sx_n[vis] + ln * fk, sy_n[vis], z_n[vis],
                    (122.0, 154.0, 198.0), amp * (1.0 - 0.82 * fk)
                    * (0.92 + 0.18 * V)), DUST))

        # ---- 2. the body of the glass -------------------------------------
        for pts3, wgt, gain, col in (
                (self.walls[0], self.walls[1], 1.45, (86, 146, 198)),
                (self.vol[0], self.vol[1], 0.34, (54, 96, 138)),
                (self.back[0], self.back[1], 0.52, (78, 132, 180)),
                (self.face[0], self.face[1], 1.00, (104, 168, 214))):
            sx, sy, z = self.project(pts3, cam, yaw, roll, f_len)
            d = np.exp(-(z - cam[2]) * 0.055) * (0.92 + 0.16 * V)
            w = wgt * gain * d
            mix = np.clip(w * 1.5, 0, 1)
            rr = col[0] + (252 - col[0]) * mix
            gg = col[1] + (250 - col[1]) * mix
            bb = col[2] + (255 - col[2]) * mix
            canvases.append((self.splat(sx, sy, z, (rr, gg, bb), w), GLASS))

        # ---- 2b. rim: a bright core plus a wide glow skirt ----------------
        sx, sy, z = self.project(self.edges_f, cam, yaw, roll, f_len)
        d = np.exp(-(z - cam[2]) * 0.05)
        canvases.append((self.stamp(sx, sy, z, (232, 244, 255), 1.25 * d,
                                    rad=0.0), GLASS))
        canvases.append((self.stamp(sx, sy, z, (150, 198, 240), 0.62 * d,
                                    rad=1.15, halo=0.42), GLASS))
        canvases.append((self.stamp(sx, sy, z, (86, 140, 200), 0.30 * d,
                                    rad=2.5, halo=0.30), GLASS))

        sx, sy, z = self.project(self.edges_b, cam, yaw, roll, f_len)
        d = np.exp(-(z - cam[2]) * 0.05)
        canvases.append((self.stamp(sx, sy, z, (120, 172, 222), 0.60 * d,
                                    rad=0.9, halo=0.38), GLASS))

        sx, sy, z = self.project(self.conn, cam, yaw, roll, f_len)
        canvases.append((self.stamp(sx, sy, z, (160, 205, 245), 0.75,
                                    rad=0.8, halo=0.30), GLASS))

        # ---- 3. the white beam entering the prism -------------------------
        for pts3, wgt in self.beam:
            sx, sy, z = self.project(pts3, cam, yaw, roll, f_len)
            d = np.exp(-(z - cam[2]) * 0.045)
            inten = wgt * (0.95 + 0.45 * V) * d * 1.35
            canvases.append((self.splat(sx, sy, z, (246, 249, 255), inten),
                             LIGHT))
        # the beam's own bloom, so it burns instead of merely being drawn
        sx, sy, z = self.project(self.beam[0][0], cam, yaw, roll, f_len)
        canvases.append((self.stamp(sx, sy, z, (150, 180, 235),
                                    0.30 * (0.8 + 0.4 * V), rad=1.7,
                                    halo=0.28, taps=6), LIGHT))

        # refracted segment inside the glass
        sx, sy, z = self.project(self.inner[0], cam, yaw, roll, f_len)
        canvases.append((self.splat(sx, sy, z, (236, 252, 255),
                                    self.inner[1] * (1.05 + 0.45 * V)), LIGHT))

        # ---- 3b. exit flare ------------------------------------------------
        sx, sy, z = self.project(self.knot[0], cam, yaw, roll, f_len)
        canvases.append((self.splat(sx, sy, z, (255, 252, 242),
                                    self.knot[1] * (1.1 + 0.9 * V) * 1.5),
                         LIGHT))
        canvases.append((self.stamp(sx, sy, z, (220, 235, 255),
                                    0.45 * (1.0 + V), rad=2.0, halo=0.25,
                                    taps=6), LIGHT))

        # ---- 4. dispersion fan --------------------------------------------
        fp, fc, fw = self.fan
        # the fan breathes: it opens slightly with the voice
        spread = 1.0 + 0.05 * V
        p2 = fp.copy()
        p2[1] = self.fan_origin[1] + (p2[1] - self.fan_origin[1]) * spread
        sx, sy, z = self.project(p2, cam, yaw, roll, f_len)
        inten = fw * (0.80 + 0.55 * V) * np.exp(-(z - cam[2]) * 0.035) * 1.45
        canvases.append((self.splat(sx, sy, z, fc, inten), LIGHT))

        # wide dim wash behind it
        wp, wc, ww = self.wash
        p3 = wp.copy()
        p3[1] = self.fan_origin[1] + (p3[1] - self.fan_origin[1]) * spread
        sx, sy, z = self.project(p3, cam, yaw, roll, f_len)
        canvases.append((self.splat(sx, sy, z, wc,
                                    ww * (0.75 + 0.6 * V) *
                                    np.exp(-(z - cam[2]) * 0.03)), LIGHT))

        # ---- compose -------------------------------------------------------
        size = g.rows * g.cols
        best = np.zeros(size, np.float32)
        R = np.zeros(size, np.float32)
        G = np.zeros(size, np.float32)
        B = np.zeros(size, np.float32)

        for res, _chars in canvases:
            if res is None:
                continue
            v, r_, g_, b_ = res
            v = v.ravel()
            m = v > best
            best[m] = v[m]
            R[m] = r_.ravel()[m]
            G[m] = g_.ravel()[m]
            B[m] = b_.ravel()[m]

        rows, cols = g.rows, g.cols
        best = best.reshape(rows, cols)
        R = R.reshape(rows, cols); G = G.reshape(rows, cols); B = B.reshape(rows, cols)

        # glyph by intensity
        ramp = np.array(list(" .:-=+*#%@"), dtype="U1")
        li = np.clip((best ** 0.7) * (len(ramp) - 1), 0, len(ramp) - 1).astype(int)
        chars = ramp[li]
        chars = np.where(best > 0.004, chars, " ")

        colors = np.stack([np.clip(R, 0, 255), np.clip(G, 0, 255),
                           np.clip(B, 0, 255)], axis=2).astype(np.float32)
        return chars, colors


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--out", default="out3d.mp4")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", type=int, default=14)
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    feats = analyze(args.audio, args.fps, args.start, args.dur)
    n = min(feats["n"], int(args.dur * args.fps))
    chars_all = "".join(sorted(set(DUST + GLASS + LIGHT + " .")))
    g = Grid(args.w, args.h, args.font, chars_all)
    print(f"[grid] {g.cols}x{g.rows} cell {g.cw}x{g.ch}  frames={n}", flush=True)

    sc = Scene3D(g, args.w, args.h, args.fps)
    cw, chh = g.cols * g.cw, g.rows * g.ch

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        for tt in [0.04, 0.22, 0.40, 0.58, 0.76, 0.94]:
            fi = min(n - 1, int(tt * n))
            f = {k: float(feats[k][fi]) for k in feats
                 if isinstance(feats[k], np.ndarray)}
            ch, co = sc.frame(f, fi / args.fps, args.dur)
            canvas = g.render(g.idx(ch), co)
            canvas = shade(canvas, f, fi / args.fps, cw, chh,
                           np.random.default_rng(fi))
            img = np.zeros((args.h, args.w, 3), np.uint8)
            img[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
            Image.fromarray(img).save(os.path.join(args.preview, f"f{tt:.2f}.png"))
            print(f"[preview] {tt:.2f}", flush=True)
        return

    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.w}x{args.h}",
           "-r", str(args.fps), "-i", "pipe:0",
           "-ss", f"{args.start:.3f}", "-t", f"{args.dur:.3f}",
           "-i", args.audio,
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-tune", "film",
           "-c:a", "aac", "-b:a", "256k", "-shortest", args.out]
    errlog = open(args.out + ".ffmpeg.log", "w")
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=errlog)

    rng = np.random.default_rng(4)
    for fi in range(n):
        t = fi / args.fps
        f = {k: float(feats[k][fi]) for k in feats
             if isinstance(feats[k], np.ndarray)}
        ch, co = sc.frame(f, t, args.dur)
        canvas = g.render(g.idx(ch), co)
        canvas = shade(canvas, f, t, cw, chh, rng)
        frame = np.zeros((args.h, args.w, 3), np.uint8)
        frame[g.oy:g.oy + chh, g.ox:g.ox + cw] = canvas
        pipe.stdin.write(frame.tobytes())
        if fi % 30 == 0:
            print(f"[render] {fi}/{n}", flush=True)

    pipe.stdin.close()
    pipe.wait()
    errlog.close()
    print("[done]", args.out, flush=True)


if __name__ == "__main__":
    main()
