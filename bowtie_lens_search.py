#!/usr/bin/env python3
"""
bowtie_lens_search.py -- pick injection lenses from what is on the shelf.   (self-contained)

Searches all ordered 2-lens and 3-lens combinations from LENSES (one piece of each), optimises the
lens positions for the best total (tangential x sagittal) overlap with the cavity eigenmode, and
lists the best ones with the beam radius on every lens and how much the overlap drops if a lens
is displaced by +-1 / +-5 mm.
Lens positions are measured from the INPUT BEAM WAIST (z = 0) along the beam; the M1-M2 midpoint
(where the eigenmode is evaluated, travelling M1 -> M2) is at Z_TARGET.
Edit the INPUTS block to match your bench.      usage:  python3 bowtie_lens_search.py [W0_IN_mm]
"""
import sys, itertools
import numpy as np
from scipy.optimize import minimize

# ============================== INPUTS ==============================
LAM, N_CRYS, L_CRYS, ROC = 1064e-9, 2.157, 10e-3, 100e-3
B      = 225.83e-3          # M1-M2 (flat-flat)
A23    = 191.68e-3          # M2-M3 arm
D      = 117.88e-3          # M3-M4 (curved-curved)
A41    = 191.33e-3          # M4-M1 arm
TH     = np.radians(13.22)  # incidence angle on the curved mirrors
Z_TARGET = 1750e-3          # distance input waist (z = 0) -> M1-M2 midpoint [m]
LENSES = [35e-3, 50e-3, 300e-3, 400e-3, 500e-3, 750e-3, 1000e-3]    # focal lengths on the shelf
W0_IN  = float(sys.argv[1]) * 1e-3 if len(sys.argv) > 1 else 1.35e-3 # input beam waist radius at z = 0 [m]
CLEAR_M1, CLEAR_IN = 50e-3, 50e-3     # last lens >= 50 mm before M1; first lens >= 50 mm after z = 0
MAX_W_LENS, MIN_GAP = 4e-3, 10e-3     # beam radius on a lens <= 4 mm (1" optics); lens spacing >= 10 mm
# ====================================================================

def P(d):    return np.array([[1.0, d], [0.0, 1.0]])
def LENS(f): return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])
def f_curved(plane): return ROC * np.cos(TH) / 2 if plane == "t" else ROC / (2 * np.cos(TH))

def roundtrip(plane):
    """Round trip from the M1-M2 midpoint: -> M2 -> M3 -> crystal -> M4 -> M1 -> back to midpoint."""
    f, g = f_curved(plane), (D - L_CRYS) / 2
    M = P(B / 2)
    M = P(A23) @ M
    M = LENS(f) @ M
    M = P(g) @ M
    M = P(L_CRYS / N_CRYS) @ M
    M = P(g) @ M
    M = LENS(f) @ M
    M = P(A41) @ M
    M = P(B / 2) @ M
    return M

def eigen_q(M):
    a, b, c, d = M.ravel(); s = (a + d) / 2
    if abs(s) >= 1: return None
    q = (a - d) / (2 * c) + 1j * np.sqrt(1 - s * s) / abs(c)
    return q if q.imag > 0 else np.conj(q)

def w_of(q):  return np.sqrt(-LAM / (np.pi * (1.0 / q).imag))
def overlap(q1, q2): return 4 * q1.imag * q2.imag / abs(q1 - np.conj(q2)) ** 2

qt, qs = eigen_q(roundtrip("t")), eigen_q(roundtrip("s"))
if qt is None or qs is None: sys.exit("cavity unstable with these inputs")
ZMAX = Z_TARGET - B / 2 - CLEAR_M1

def run(fl, w0, z):
    q = 1j * np.pi * w0 ** 2 / LAM; zc = 0.0; ws = []
    for zi, f in zip(z, fl):
        q = q + (zi - zc); zc = zi; ws.append(w_of(q)); q = 1 / (1 / q - 1 / f)
    return q + (Z_TARGET - zc), ws

def ov(fl, w0, z):
    qf, _ = run(fl, w0, z); return overlap(qf, qt) * overlap(qf, qs)

def obj(z, fl, w0):
    z = np.asarray(z)
    if np.any(np.diff(np.concatenate([[CLEAR_IN], z, [ZMAX]])) < MIN_GAP):
        return 1e3 + np.sum(np.abs(z))
    qf, ws = run(fl, w0, z)
    return -overlap(qf, qt) * overlap(qf, qs) + sum(max(0, w - MAX_W_LENS) * 50 for w in ws)

def best(fl, w0, nst=20):
    rng = np.random.default_rng(1); bb = None
    for _ in range(nst):
        z0 = np.sort(rng.uniform(CLEAR_IN, ZMAX, len(fl)))
        r = minimize(obj, z0, args=(fl, w0), method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10, maxiter=3000))
        if bb is None or r.fun < bb.fun: bb = r
    return (-bb.fun if bb.fun < 0.5 else None), bb.x, run(fl, w0, bb.x)[1]

def tolerance(fl, w0, z, shift):
    worst = 1.0
    for i in range(len(fl)):
        for s in (-shift, shift):
            zz = np.array(z, float); zz[i] += s; worst = min(worst, ov(fl, w0, zz))
    return worst

if __name__ == "__main__":
    ceil = minimize(lambda x: -(overlap(x[0] + 1j * x[1], qt) * overlap(x[0] + 1j * x[1], qs)),
                    [qt.real, qt.imag], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-15))
    print("eigenmode at M1-M2 midpoint: w_t = %.0f um, w_s = %.0f um" % (w_of(qt) * 1e6, w_of(qs) * 1e6))
    print("overlap ceiling for ANY spherical-lens injection: %.4f   (input waist %.2f mm)\n" % (-ceil.fun, W0_IN * 1e3))
    for n in (2, 3):
        rows = []
        for fl in itertools.permutations(LENSES, n):
            o, z, ws = best(fl, W0_IN, nst=12 if n == 2 else 6)
            if o is not None: rows.append((o, fl, z, ws))
        rows.sort(key=lambda r: -r[0])
        print("=== best %d-lens combinations ===" % n)
        for o, fl, z, ws in rows[:8]:
            print("  " + " + ".join("f=%d @ %.1f mm" % (f * 1e3, zz * 1e3) for f, zz in zip(fl, z)) +
                  " | overlap %.4f | w on lenses %s mm | worst overlap for +-1 / +-5 mm shift: %.3f / %.3f" %
                  (o, "/".join("%.2f" % (w * 1e3) for w in ws), tolerance(fl, W0_IN, z, 1e-3), tolerance(fl, W0_IN, z, 5e-3)))
