#!/usr/bin/env python3
r"""
Bow-tie OPO cavity -- unified solver + lensguide plot (single file).

SINGLE-PLANE MODEL: no off-axis astigmatism, so there is no tangential/sagittal
split.  The curved mirrors act as ordinary thin lenses with f = R/2 and the
cavity has ONE eigenmode / ONE Gaussian beam profile.

As-built cavity (SPC table):
    M1-M2 (flat-flat)            b   = 225.83 mm
    M2-M3 (diagonal arm)         a23 = 191.68 mm
    M3-M4 (curved-curved)        d   = 117.88 mm
    M4-M1 (diagonal arm)         a41 = 191.33 mm

DEFAULT MODEL: symmetric arms (mean of 191.68/191.33 mm), ROC(M3)=ROC(M4),
injection lenses taken from the lab shelf: f = 400, 300, 500 mm (in beam order;
only their positions are fitted).  Any number of lenses is allowed in FIXED_FOCALS.
Flags below switch back to the fuller model.

NOTE: this single-plane model ignores the 13.22 deg tilt of M3/M4.  With the tilt the
tangential/sagittal waists differ and NO spherical-lens set can exceed ~92 % total
overlap; ASTIG_THETA_DEG below prints that check next to the single-plane result.

FITTING (two stages):
  1. ROC of M3/M4 chosen so the crystal-focus Boyd-Kleinman xi is closest to 2.84.
     (With a single beam profile the lenses can always reach overlap = 1 for any
     ROC, so the ROC is set purely by the BK condition.)
  2. Lens positions (and focal lengths, if free) chosen to maximise the overlap of
     the injected beam with the cavity eigenmode.

Direction of propagation (seed): M1 -> M2 -> M3 -> crystal -> M4 -> M1.
"""
import os
import numpy as np
import matplotlib
if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY") and os.name != "nt":
    pass
import matplotlib.pyplot as plt
from matplotlib import ticker, font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import minimize, differential_evolution

# =====================================================================
# 1. INPUTS
# =====================================================================
LAM      = 1064e-9        # vacuum wavelength [m]
N_CRYS   = 2.157          # crystal refractive index at LAM
L_CRYS   = 10e-3          # crystal length [m]
R1, LOSS = 0.75, 0.01     # output-coupler reflectivity, round-trip loss
C        = 299792458.0
XI_OPT   = 2.84           # Boyd-Kleinman optimum

# ---- AS-BUILT CAVITY (SPC table, corrected diagonals) ----
B_FLAT = 225.83e-3        # M1-M2  (flat-flat)
A_23   = 191.68e-3        # M2-M3  diagonal arm
D_CURV = 117.88e-3        # M3-M4  (curved-curved, crystal between)
A_41   = 191.33e-3        # M4-M1  diagonal arm

SYMMETRIC_ARMS = True     # True: both arms = mean(A_23, A_41)
(A_23, A_41) = ((A_23 + A_41) / 2,) * 2 if SYMMETRIC_ARMS else (A_23, A_41)

# ---- input beam / injection ----
W0_IN    = 1.35e-3                # input waist [m] at z = 0
Z_M1     = 1635e-3                # position of M1 along the bench (z = 0 at input waist)
Z_TARGET = Z_M1 + B_FLAT / 2      # M1-M2 midpoint (eigenmode reference plane)

# ---- FREE-PARAMETER SEARCH RANGES ----
ROC_BOUNDS     = (0.030, 0.500)   # ROC of M3 and M4 [m]
F_BOUNDS       = (0.020, 2.000)   # focal length of each mode-matching lens [m]
LENS_Z_MIN_SEP = 0.020            # lenses may not be closer than this [m]
FIXED_FOCALS   = (0.400, 0.300, 0.500)   # lenses in beam order (L1, L2, L3...), any number; None = free focal lengths
N_FREE_LENSES  = 2                # number of lenses when FIXED_FOCALS is None
MAX_W_LENS     = 4e-3             # beam radius on any lens must stay below this [m] (1-inch optics)
ASTIG_THETA_DEG = 13.22           # tilt of M3/M4 for the astigmatism check printed in main(); None = skip
ROC_FIXED      = 0.100            # fixed ROC of M3 = M4 [m]; None = optimise ROC for xi_opt
SYMMETRIC_ROC  = True             # True: force ROC(M3) = ROC(M4)

MODE_MATCH_THRESHOLD = 0.95
N_TRIPS    = 3
OUT_PREFIX = "bowtie_lensguide"
SHOW_PLOT  = True
DE_SEED, DE_POP, DE_ITER = 0, 40, 400

# =====================================================================
# 1b. PLOT STYLE AND TICK-LABEL CONTROL
# =====================================================================
FONT_FAMILY      = "serif"
SERIF_CANDIDATES = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]  # first installed one is used
MATH_FONTSET     = "stix"       # serif-looking math text ("stix", "cm", "dejavuserif")
FONT_SIZES       = dict(base=12, axes_label=12, title=12, legend=11)
MIRROR_LABEL_FS  = 12          # font size of the mirror / lens position annotations (M1..M4, L f=..)

# Defaults applied to the x and y axis of every plot (override per axis below).
TICK_DEFAULTS = dict(
    labelsize=12,       # tick label font size
    direction="in",     # "in" | "out" | "inout"
    length=4, width=0.8,        # major tick size
    minor_length=2,     # minor tick size
    pad=3,              # distance between tick and its label
    rotation=0,         # tick label rotation [deg]
    top_right=False,    # True: also draw ticks on the top / right spines
    minor=None,         # None = matplotlib default, True = show, False = hide
)

# Per-plot, per-axis control. Keys of the plots:
#   "injection", "roundtrips_log", "crystal_zoom", "beam_shape",
#   "roc_scan_fine", "roc_scan_full", "g_scan_fine", "g_scan_full"
# Every field is optional (leave {} for the defaults above):
#   lim        (lo, hi)        axis limits
#   ticks      [..]            explicit major tick positions
#   step       float           major tick spacing                   (ignored if ticks is given)
#   nbins      int             max number of major ticks            (ignored if ticks/step given)
#   labels     [..]            explicit tick label strings (requires `ticks`)
#   fmt        "%.1f"          number format of the tick labels, e.g. "%d", "%.2f", "%.1e"
#   scale      float           multiply displayed numbers (e.g. 1e-3 to show metres instead of mm)
#   minor      True/False      show / hide minor ticks
#   minor_step float           minor tick spacing
#   minor_n    int             minor subdivisions per major interval
#   labelsize, rotation, direction, length, width, pad, top_right     (as in TICK_DEFAULTS)
#   show       False           hide the tick labels of this axis
# Examples:
#   "injection": {"x": dict(step=200, fmt="%d"), "y": dict(ticks=[-1, 0, 1], fmt="%.1f")},
#   "crystal_zoom": {"x": dict(nbins=5, rotation=30, minor=True)},
#   "beam_shape": {"y": dict(ticks=[-1000, 0, 1000], labels=["-1 mm", "0", "1 mm"])},
AXIS_TICKS = {
    "injection":      {"x": {}, "y": {}},
    "roundtrips_log": {"x": {}, "y": {}},
    "crystal_zoom":   {"x": {}, "y": {}},
    "beam_shape":     {"x": {}, "y": {}},
    "roc_scan_fine":  {"x": {}, "y": {}},
    "roc_scan_full":  {"x": {}, "y": {}},
    "g_scan_fine":    {"x": {}, "y": {}},
    "g_scan_full":    {"x": {}, "y": {}},
}


# =====================================================================
# 2. ABCD TOOLS AND CAVITY MODEL
# =====================================================================
def P(d):    return np.array([[1.0, d], [0.0, 1.0]])
def LENS(f): return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def half_crystal_air(d=D_CURV):
    return (d - L_CRYS) / 2.0


def roundtrip_from_crystal_centre(R3, R4):
    """Start at crystal centre, travelling toward M4:
    xtal/2 -> M4 -> M1 arm -> b -> M2 arm -> M3 -> xtal/2."""
    f3, f4 = R3 / 2, R4 / 2
    g = half_crystal_air()
    M = P(L_CRYS / 2 / N_CRYS)
    M = P(g) @ M
    M = LENS(f4) @ M
    M = P(A_41) @ M
    M = P(B_FLAT) @ M
    M = P(A_23) @ M
    M = LENS(f3) @ M
    M = P(g) @ M
    M = P(L_CRYS / 2 / N_CRYS) @ M
    return M


def roundtrip_from_flat_arm(R3, R4, theta=0.0, plane=None):
    """Start/end at M1-M2 midpoint, travelling M1->M2:
    b/2 -> M2 -> a23 -> M3 -> crystal -> M4 -> a41 -> M1 -> b/2.
    plane=None: single-plane model (f = R/2).  plane='t'/'s': tilted mirrors, f_t = R cos(th)/2, f_s = R/(2 cos(th))."""
    if plane == "t":   f3, f4 = R3 * np.cos(theta) / 2, R4 * np.cos(theta) / 2
    elif plane == "s": f3, f4 = R3 / (2 * np.cos(theta)), R4 / (2 * np.cos(theta))
    else:              f3, f4 = R3 / 2, R4 / 2
    g = half_crystal_air()
    M = P(B_FLAT / 2)
    M = P(A_23) @ M
    M = LENS(f3) @ M
    M = P(g) @ M
    M = P(L_CRYS / N_CRYS) @ M
    M = P(g) @ M
    M = LENS(f4) @ M
    M = P(A_41) @ M
    M = P(B_FLAT / 2) @ M
    return M


def eigen_q(M):
    A, B, C_, D = M.ravel()
    s = (A + D) / 2
    if abs(s) >= 1:
        return None, s
    q = (A - D) / (2 * C_) + 1j * np.sqrt(1 - s * s) / abs(C_)
    return q, s


def w_of(q):       return np.sqrt(LAM * q.imag / np.pi)                    # at a waist
def w_general(q):  return np.sqrt(-LAM / (np.pi * (1.0 / q).imag))         # anywhere
def q_step(M, q):  return (M[0, 0] * q + M[0, 1]) / (M[1, 0] * q + M[1, 1])
def overlap(q1, q2): return 4 * q1.imag * q2.imag / abs(q1 - np.conj(q2)) ** 2
def q_in_beam(w0, z_waist, z_plane): return 1j * np.pi * w0 ** 2 / LAM + (z_plane - z_waist)


def crystal_waist(R3, R4):
    q, s = eigen_q(roundtrip_from_crystal_centre(R3, R4))
    return (None, s) if q is None else (w_of(q), s)


def eigenmode_at_flat_arm(R3, R4):
    return eigen_q(roundtrip_from_flat_arm(R3, R4))[0]


# =====================================================================
# 3. BOYD-KLEINMAN
# =====================================================================
def boyd_kleinman(w0, n=N_CRYS, L=L_CRYS, lam=LAM, xi_opt=XI_OPT):
    zR = np.pi * w0 ** 2 * n / lam
    return L / (2 * zR), np.sqrt(lam * L / (2 * np.pi * n * xi_opt)), zR


def bk_h(xi):
    """Boyd-Kleinman h(sigma_opt, xi) for B = 0, normalised so that xi = 2.84 -> 1."""
    tau = np.linspace(-xi, xi, 4001)
    sig = np.linspace(-1.0, 4.0 / max(xi, 0.2) + 2.0, 241)
    f = 1.0 / (1.0 + 1j * tau)
    val = np.array([abs(_trapz(np.exp(1j * sg * tau) * f, tau)) ** 2 / (4 * xi) for sg in sig]).max()
    return val


_trapz = getattr(np, "trapezoid", None) or np.trapz
H_REF = None


def bk_rel(xi):
    global H_REF
    if H_REF is None: H_REF = bk_h(XI_OPT)
    return bk_h(xi) / H_REF


def xi_of(R3, R4):
    w0, _ = crystal_waist(R3, R4)
    return None if w0 is None else boyd_kleinman(w0)[0]


# =====================================================================
# 4. FIT: ROC(M3), ROC(M4), L1(f1,z1), L2(f2,z2)
# =====================================================================
def q_injected(zs, fs, w0_in=W0_IN, z_target=Z_TARGET):
    """q of the injected beam at z_target after thin lenses fs[i] placed at zs[i] (z from the input waist)."""
    q, zc = q_in_beam(w0_in, 0.0, 0.0), 0.0
    for z, f in sorted(zip(zs, fs)):
        q = q + (z - zc); zc = z
        q = 1.0 / (1.0 / q - 1.0 / f)
    return q + (z_target - zc)


def lens_radii(zs, fs, w0_in=W0_IN):
    """Beam radius on every lens (in beam order)."""
    q, zc, out = q_in_beam(w0_in, 0.0, 0.0), 0.0, []
    for z, f in sorted(zip(zs, fs)):
        q = q + (z - zc); zc = z
        out.append(w_general(q)); q = 1.0 / (1.0 / q - 1.0 / f)
    return out


def evaluate(R3, R4, zs, fs):
    out = dict(R3=R3, R4=R4, stable=True)
    q, s = eigen_q(roundtrip_from_flat_arm(R3, R4))
    if q is None:
        out["stable"] = False; return out
    out["w0"] = crystal_waist(R3, R4)[0]
    out["xi"] = boyd_kleinman(out["w0"])[0]
    out["s"] = s
    qf = q_injected(zs, fs)
    out.update(q_final=qf, q_eig=q, overlap=overlap(qf, q))
    out["meets_threshold"] = out["overlap"] >= MODE_MATCH_THRESHOLD
    order = np.argsort(zs)
    out["z_common"] = np.asarray(zs, float)[order]
    out["focal_common"] = np.asarray(fs, float)[order]
    out["w_lens"] = lens_radii(zs, fs)
    return out


def objective_roc(x):
    """Stage 1: ROC only affects the Boyd-Kleinman xi (lenses can always reach overlap = 1)."""
    R3, R4 = (x[0], x[0]) if SYMMETRIC_ROC else x
    q, s = eigen_q(roundtrip_from_flat_arm(R3, R4))
    if q is None:
        return 50.0 + 10 * (abs(s) - 1)
    return np.log(boyd_kleinman(crystal_waist(R3, R4)[0])[0] / XI_OPT) ** 2


def fit_all(verbose=True):
    # ---- stage 1: ROC(M3), ROC(M4) ----
    if ROC_FIXED is not None:
        R3 = R4 = ROC_FIXED
    else:
        bounds = [ROC_BOUNDS] if SYMMETRIC_ROC else [ROC_BOUNDS, ROC_BOUNDS]
        r = differential_evolution(objective_roc, bounds, seed=DE_SEED, popsize=DE_POP,
                                   maxiter=DE_ITER, tol=1e-10, polish=False)
        R3, R4 = (r.x[0], r.x[0]) if SYMMETRIC_ROC else r.x
    q_eig = eigenmode_at_flat_arm(R3, R4)
    # ---- stage 2: lenses ----
    rng = np.random.default_rng(DE_SEED)
    zlo, zhi = 0.05, Z_M1 - 0.05          # lenses must sit BEFORE M1
    best = None
    if FIXED_FOCALS:
        fs = list(FIXED_FOCALS); n = len(fs)      # lens i (focal fs[i]) is the i-th lens along the beam
        def cost(z):
            z = np.asarray(z)
            if z[0] < zlo or z[-1] > zhi or np.any(np.diff(z) < LENS_Z_MIN_SEP):
                return 10.0
            pen = sum(max(0.0, w - MAX_W_LENS) * 50 for w in lens_radii(z, fs))
            return -overlap(q_injected(z, fs), q_eig) + pen
        sols = []
        for _ in range(300):
            y0 = np.sort(rng.uniform(zlo, zhi, n))
            rr = minimize(cost, y0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-14, maxiter=4000))
            sols.append(rr)
            if best is None or rr.fun < best.fun: best = rr
        # many position sets give the same (best) overlap -> keep the one least sensitive to lens placement
        def worst_overlap(z, shift=5e-3):
            w = 1.0
            for i in range(n):
                for sg in (-shift, shift):
                    zz = np.array(z, float); zz[i] += sg
                    w = min(w, overlap(q_injected(zz, fs), q_eig))
            return w
        good = [r for r in sols if r.fun <= best.fun + 1e-6]
        best = max(good, key=lambda r: worst_overlap(r.x))
        zs = np.asarray(best.x)
    else:
        n = N_FREE_LENSES
        def cost_free(y):
            z, f = np.asarray(y[:n]), np.asarray(y[n:])
            pen = 0.0
            for v in z: pen += max(zlo - v, 0) ** 2 + max(v - zhi, 0) ** 2
            for v in f: pen += max(F_BOUNDS[0] - v, 0) ** 2 + max(v - F_BOUNDS[1], 0) ** 2
            zz = np.sort(z)
            if n > 1 and np.min(np.diff(zz)) < LENS_Z_MIN_SEP:
                pen += (LENS_Z_MIN_SEP - np.min(np.diff(zz))) ** 2
            if pen > 0: return 1e3 * (1 + pen)
            reg = 1e-3 * sum((0.3 / v) ** 2 for v in f)       # tie-break: prefer gentle, practical lenses
            return -overlap(q_injected(z, f), q_eig) + reg
        for _ in range(300):
            y0 = np.concatenate([rng.uniform(zlo, zhi, n),
                                 np.exp(rng.uniform(*np.log(F_BOUNDS), size=n))])
            rr = minimize(cost_free, y0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-13, maxiter=4000))
            if best is None or rr.fun < best.fun: best = rr
        zs, fs = best.x[:n], list(best.x[n:])
    return evaluate(R3, R4, zs, fs)


def astigmatic_check(res, theta_deg):
    """Same lenses, but with the TILTED curved mirrors (f_t = R cos/2, f_s = R/(2 cos)).
    Returns overlap_t, overlap_s, total, and the best total any single circular beam could reach."""
    th = np.radians(theta_deg)
    qt, _ = eigen_q(roundtrip_from_flat_arm(res["R3"], res["R4"], th, "t"))
    qs, _ = eigen_q(roundtrip_from_flat_arm(res["R3"], res["R4"], th, "s"))
    qf = res["q_final"]
    ot, os_ = overlap(qf, qt), overlap(qf, qs)
    r = minimize(lambda x: -overlap(x[0] + 1j * x[1], qt) * overlap(x[0] + 1j * x[1], qs),
                 [qf.real, qf.imag], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-15))
    return ot, os_, ot * os_, -r.fun


def verify_self_consistency(res, n_trips=6):
    M = roundtrip_from_flat_arm(res["R3"], res["R4"])
    q = res["q_final"]; hist = []
    for i in range(n_trips + 1):
        hist.append((i, w_general(q), overlap(q, res["q_eig"])))
        q = q_step(M, q)
    return hist


# =====================================================================
# 5. BEAM TRACER
# =====================================================================
class Tracer:
    def __init__(self, q0, z0=0.0):
        self.q, self.z = q0, z0
        self.zs, self.ws = [z0], [w_general(q0)]

    def drift(self, L, n=1.0, npts=300):
        s = np.linspace(0.0, L, npts)[1:]
        self.zs.extend(self.z + s)
        self.ws.extend(w_general(self.q + s / n))
        self.q, self.z = self.q + L / n, self.z + L

    def lens(self, f):
        self.q = 1.0 / (1.0 / self.q - 1.0 / f)


def trace_injection(res):
    T = Tracer(q_in_beam(W0_IN, 0.0, 0.0), 0.0)
    for z, f in zip(res["z_common"], res["focal_common"]):
        T.drift(z - T.z); T.lens(f)
    T.drift(Z_TARGET - T.z)
    return T


def trace_cavity(T, res, n_trips):
    f3, f4, g = res["R3"] / 2, res["R4"] / 2, half_crystal_air()
    mk = {k: [] for k in ("M1", "M2", "M3", "M4", "xtal_in", "xtal_out", "trip_end")}
    for _ in range(n_trips):
        T.drift(B_FLAT / 2);                    mk["M2"].append(T.z)
        T.drift(A_23);                          mk["M3"].append(T.z); T.lens(f3)
        T.drift(g);                             mk["xtal_in"].append(T.z)
        T.drift(L_CRYS, n=N_CRYS, npts=200);    mk["xtal_out"].append(T.z)
        T.drift(g);                             mk["M4"].append(T.z); T.lens(f4)
        T.drift(A_41);                          mk["M1"].append(T.z)
        T.drift(B_FLAT / 2);                    mk["trip_end"].append(T.z)
    return mk


# =====================================================================
# 6. PLOTTING
# =====================================================================
def setup_style():
    """Apply the global font settings (serif family etc.)."""
    avail = {f.name for f in font_manager.fontManager.ttflist}
    serif = [f for f in SERIF_CANDIDATES if f in avail] or ["DejaVu Serif"]
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.serif": serif,
        "mathtext.fontset": MATH_FONTSET,
        "font.size": FONT_SIZES["base"],
        "axes.labelsize": FONT_SIZES["axes_label"],
        "axes.titlesize": FONT_SIZES["title"],
        "legend.fontsize": FONT_SIZES["legend"],
        "xtick.labelsize": TICK_DEFAULTS["labelsize"],
        "ytick.labelsize": TICK_DEFAULTS["labelsize"],
    })


def apply_ticks(ax, key):
    """Apply TICK_DEFAULTS + AXIS_TICKS[key] to the x and y axis of `ax`."""
    cfg_all = AXIS_TICKS.get(key, {})
    for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        c = dict(TICK_DEFAULTS); c.update(cfg_all.get(name, {}))
        if c.get("lim") is not None:
            (ax.set_xlim if name == "x" else ax.set_ylim)(*c["lim"])
        # --- major tick positions ---
        if c.get("ticks") is not None:
            axis.set_major_locator(ticker.FixedLocator(c["ticks"]))
        elif c.get("step") is not None:
            axis.set_major_locator(ticker.MultipleLocator(c["step"]))
        elif c.get("nbins") is not None:
            axis.set_major_locator(ticker.MaxNLocator(c["nbins"]))
        # --- tick labels ---
        if c.get("labels") is not None:
            if c.get("ticks") is None:
                raise ValueError("AXIS_TICKS[%r][%r]: 'labels' needs 'ticks'" % (key, name))
            axis.set_major_formatter(ticker.FixedFormatter(c["labels"]))
        elif c.get("fmt") is not None or c.get("scale") is not None:
            fmt, sc = c.get("fmt") or "%g", c.get("scale", 1.0)
            axis.set_major_formatter(ticker.FuncFormatter(lambda v, _p, fmt=fmt, sc=sc: fmt % (v * sc)))
        # --- minor ticks ---
        if c.get("minor_step") is not None:
            axis.set_minor_locator(ticker.MultipleLocator(c["minor_step"]))
        elif c.get("minor_n") is not None:
            axis.set_minor_locator(ticker.AutoMinorLocator(c["minor_n"]))
        elif c.get("minor") is True:
            axis.set_minor_locator(ticker.AutoMinorLocator())
        elif c.get("minor") is False:
            axis.set_minor_locator(ticker.NullLocator())
        # --- look ---
        ax.tick_params(axis=name, which="major", labelsize=c["labelsize"], direction=c["direction"],
                       length=c["length"], width=c["width"], pad=c["pad"], labelrotation=c["rotation"])
        ax.tick_params(axis=name, which="minor", direction=c["direction"], length=c["minor_length"],
                       labelsize=c["labelsize"])
        if c["top_right"]:
            ax.tick_params(top=True, right=True)
        if c.get("show") is False:
            ax.tick_params(axis=name, **({"labelbottom": False} if name == "x" else {"labelleft": False}))


def laser_beam(ax, zz, ww):
    """Laser-like beam: a single red fill between +w and -w, with a darker red edge."""
    ax.fill_between(zz, -ww, ww, color="#ff3b30", alpha=.45, lw=0, zorder=2)
    ax.plot(zz, ww, color="#b00000", lw=2, zorder=3)
    ax.plot(zz, -ww, color="#b00000", lw=2, zorder=3)


def make_plot(res):
    setup_style()
    T = trace_injection(res)
    n_inj = len(T.zs)
    mk = trace_cavity(T, res, N_TRIPS)
    z, w = np.array(T.zs) * 1e3, np.array(T.ws) * 1e6
    z_start, z_end = Z_TARGET * 1e3, mk["trip_end"][-1] * 1e3
    L_geo = (mk["trip_end"][0] - Z_TARGET) * 1e3
    color = "#2e86ab"
    head = ("Bow-tie OPO | R3=%.1f / R4=%.1f mm | overlap = %.4f | xi = %.2f"
            % (res["R3"] * 1e3, res["R4"] * 1e3, res["overlap"], res["xi"]))

    def new_fig(wd, ht, top=0.86, bottom=0.12):
        fig, ax = plt.subplots(figsize=(wd, ht))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=bottom, top=top)
        fig.suptitle(head, fontsize=11, y=0.985)
        return fig, ax

    # ---- 1: injection ----
    figA, A = new_fig(10, 5)
    A.plot(z[:n_inj], w[:n_inj] * 1e-3, color=color, lw=2)
    A.plot(z[:n_inj], -w[:n_inj] * 1e-3, color=color, lw=2, alpha=.5)
    ymax = 1.15 * w[:n_inj].max() * 1e-3
    for zc, f in zip(res["z_common"], res["focal_common"]):
        A.axvline(zc * 1e3, color="k", lw=1.5)
        A.text(zc * 1e3, ymax * 0.93, "  f=%.1f mm\n  @%.1f mm" % (f * 1e3, zc * 1e3), va="top", fontsize=MIRROR_LABEL_FS)
    A.axvline(Z_M1 * 1e3, color="gray", lw=1.5, ls="--")
    A.text(Z_M1 * 1e3, ymax * 0.93, "M1  ", va="top", ha="right", fontsize=MIRROR_LABEL_FS)
    A.axvline(z_start, color="green", lw=1.5, ls=":")
    A.text(z_start, ymax * 0.93, "  M1-M2\n  midpoint", va="top", ha="left", fontsize=MIRROR_LABEL_FS, color="green")
    A.set_xlim(0, z_start + 150); A.set_ylim(-ymax, ymax)
    A.set_xlabel("z (mm)"); A.set_ylabel("beam radius w (mm)")
    A.set_title("Injection: input beam (w0 = %.2f mm) -> %d spherical lenses -> cavity" % (W0_IN * 1e3, len(res["z_common"])))
    A.grid(alpha=.3)

    # ---- 2: injection + trips, log ----
    figB, B = new_fig(16, 5.8, top=0.78, bottom=0.2)
    # collimated input beam extended to the left of z = 0 (visual only; same Gaussian propagation)
    z_left = np.linspace(-80e-3, 0.0, 60)[:-1]
    w_left = np.array([w_general(q_in_beam(W0_IN, 0.0, zz_)) for zz_ in z_left]) * 1e6
    laser_beam(B, np.concatenate([z_left * 1e3, z]), np.concatenate([w_left, w]))
    # first (injection) pass hits M1 before M2
    B.axvline(Z_M1 * 1e3, color="gray", lw=1.0, ls="--", zorder=4)
    B.text(Z_M1 * 1e3, 1.01, "M1", transform=B.get_xaxis_transform(), ha="center", va="bottom", fontsize=MIRROR_LABEL_FS)
    for zc, f in zip(res["z_common"], res["focal_common"]):
        B.axvline(zc * 1e3, color="k", lw=1.5, zorder=4)
        B.text(zc * 1e3, 1.01, "L f=%.0f" % (f * 1e3), transform=B.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=MIRROR_LABEL_FS)
    for k in range(N_TRIPS):
        B.axvspan(mk["xtal_in"][k] * 1e3, mk["xtal_out"][k] * 1e3, color="gold", alpha=.35, lw=0)
        for name in ("M1", "M2", "M3", "M4"):
            curved = name in ("M3", "M4"); zz = mk[name][k] * 1e3
            B.axvline(zz, color="k" if curved else "gray", lw=1.6 if curved else 1.0, ls="-" if curved else "--")
            B.text(zz, 1.01, name, transform=B.get_xaxis_transform(), ha="center", va="bottom", fontsize=MIRROR_LABEL_FS)
        B.axvline(mk["trip_end"][k] * 1e3, color="#006400", lw=3.2, ls="--", zorder=4)
        B.text(mk["trip_end"][k] * 1e3, 0.03, "round trip %d" % (k + 1), transform=B.get_xaxis_transform(),
               ha="right", va="bottom", rotation=90, fontsize=12, fontweight="bold", color="#006400",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#006400", lw=1.2, alpha=.9), zorder=5)
    wmax_B = 1.15 * w.max()
    B.set_xlim(-80, z_end); B.set_ylim(-wmax_B, wmax_B); B.grid(alpha=.3)
    B.set_xlabel("z (mm, unrolled)"); B.set_ylabel("beam radius w (um)")
    B.set_title("Injection + %d round trips" % N_TRIPS, pad=24)
    handles = [Patch(fc="gold", alpha=.35), Line2D([], [], color="k", lw=1.6), Line2D([], [], color="gray", lw=1.0, ls="--")]
    B.legend(handles, ["crystal", "curved M3/M4", "flat M1/M2"], loc="upper center",
             bbox_to_anchor=(0.5, -0.13), fontsize=9, ncol=3, frameon=False)

    # ---- 3: crystal zoom ----
    figC, Cx = new_fig(10, 5.5)
    z0, z1 = mk["M3"][0] * 1e3 - 6, mk["M4"][0] * 1e3 + 6
    m = (z >= z0) & (z <= z1) & (np.arange(len(z)) >= n_inj)
    Cx.plot(z[m], w[m], color=color, lw=2)
    Cx.plot(z[m], -w[m], color=color, lw=2, alpha=.5)
    i = np.argmin(np.where(m, w, np.inf))
    Cx.annotate("w0 = %.2f um" % w[i], (z[i], w[i]), textcoords="offset points",
                xytext=(8, 14), color=color, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=color))
    Cx.axvspan(mk["xtal_in"][0] * 1e3, mk["xtal_out"][0] * 1e3, color="gold", alpha=.35, lw=0,
               label="crystal (L=%.0f mm, n=%.3f)" % (L_CRYS * 1e3, N_CRYS))
    for name in ("M3", "M4"):
        Cx.axvline(mk[name][0] * 1e3, color="k", lw=1.6)
        Cx.text(mk[name][0] * 1e3, 0.97, " " + name, transform=Cx.get_xaxis_transform(), va="top", fontsize=MIRROR_LABEL_FS)
    Cx.set_xlim(z0, z1); Cx.grid(alpha=.3)
    Cx.set_xlabel("z (mm)"); Cx.set_ylabel("beam radius w (um)")
    Cx.set_title("Zoom: M3-crystal-M4 focus (round trip 1)")
    Cx.legend(loc="upper center", fontsize=9)

    # ---- 4: beam shape ----
    figD, D = new_fig(16, 5.8, top=0.78, bottom=0.2)
    zb, wb = z[n_inj - 1:], w[n_inj - 1:]
    laser_beam(D, zb, wb)
    for k in range(N_TRIPS):
        D.axvspan(mk["xtal_in"][k] * 1e3, mk["xtal_out"][k] * 1e3, color="gold", alpha=.35, lw=0)
        for name in ("M1", "M2", "M3", "M4"):
            curved = name in ("M3", "M4"); zz = mk[name][k] * 1e3
            D.axvline(zz, color="k" if curved else "gray", lw=1.6 if curved else 1.0, ls="-" if curved else "--")
            D.text(zz, 1.01, name, transform=D.get_xaxis_transform(), ha="center", va="bottom", fontsize=MIRROR_LABEL_FS)
        D.axvline(mk["trip_end"][k] * 1e3, color="#006400", lw=3.2, ls="--", zorder=4)
        D.text(mk["trip_end"][k] * 1e3, 0.03, "round trip %d" % (k + 1), transform=D.get_xaxis_transform(),
               ha="right", va="bottom", rotation=90, fontsize=12, fontweight="bold", color="#006400",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#006400", lw=1.2, alpha=.9), zorder=5)
    wmax = 1.08 * w[n_inj - 1:].max()
    D.set_xlim(z_start, z_end); D.set_ylim(-wmax*1.5, wmax*1.5); D.grid(alpha=.3)
    D.set_xlabel("z (mm, unrolled)"); D.set_ylabel("beam radius w (um)")
    D.set_title("Beam shape over %d round trips" % N_TRIPS, pad=24)
    D.legend(handles, ["crystal", "curved M3/M4", "flat M1/M2"], loc="upper center",
             bbox_to_anchor=(0.5, -0.13), fontsize=9, ncol=3, frameon=False)
    for ax_, key_ in ((A, "injection"), (B, "roundtrips_log"), (Cx, "crystal_zoom"), (D, "beam_shape")):
        apply_ticks(ax_, key_)
    return figA, figB, figC, figD


def roc_scan_plot(R_best):
    """xi vs ROC of M3 = M4 around the best value."""
    setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, (lo, hi, ttl) in enumerate(((R_best - 0.8e-3, R_best + 0.2e-3, "Fine scan around best ROC"),
                                       (0.090, 0.116, "Full stability range"))):
        Rs = np.linspace(lo, hi, 4000)
        xi = np.array([xi_of(R, R) if crystal_waist(R, R)[0] else np.nan for R in Rs], dtype=float)
        a = ax[k]
        a.plot(Rs * 1e3, xi, color="#2e86ab", lw=2)
        a.axhline(XI_OPT, color="green", ls="--", label="xi_opt = %.2f" % XI_OPT)
        a.axhspan(2.5, 3.2, color="green", alpha=.12, label="good (2.5-3.2)")
        a.axvline(R_best * 1e3, color="#d1495b", ls=":", label="best R = %.3f mm" % (R_best * 1e3))
        unst = ~np.isfinite(xi)
        if unst.any():
            a.fill_between(Rs * 1e3, 0, 1, where=unst, transform=a.get_xaxis_transform(),
                           color="gray", alpha=.25, label="unstable")
        a.set_ylim(0, 6 if k == 0 else 3.5); a.grid(alpha=.3)
        a.set_xlabel("ROC of M3 = M4 (mm)"); a.set_ylabel("Boyd-Kleinman xi")
        a.set_title(ttl); a.legend(fontsize=9, loc="upper left")
        apply_ticks(a, "roc_scan_fine" if k == 0 else "roc_scan_full")
    fig.suptitle("BK focusing parameter vs ROC (fixed spacings)")
    fig.tight_layout()
    return fig



# =====================================================================
# 6b. CAVITY STABILITY (g-FACTOR) SCAN -- same ROC ranges as roc_scan_plot
# =====================================================================
def stability_g(R3, R4):
    """Round-trip stability from the full bow-tie ABCD matrix.
    s = (A+D)/2 ;  g = (s+1)/2 = g1*g2 of the equivalent two-mirror cavity.
    Stable  <=>  |s| < 1  <=>  0 < g < 1."""
    M = roundtrip_from_flat_arm(R3, R4)
    s = (M[0, 0] + M[1, 1]) / 2.0
    return s, (s + 1.0) / 2.0


def stability_edges(lo=0.030, hi=0.500, n=20000):
    """ROC values (M3 = M4) where g crosses 0 or 1 (edges of the stable window)."""
    Rs = np.linspace(lo, hi, n)
    g = np.array([stability_g(R, R)[1] for R in Rs])
    edges = []
    for i in range(n - 1):
        for lvl in (0.0, 1.0):
            if (g[i] - lvl) * (g[i + 1] - lvl) < 0:
                a, b = Rs[i], Rs[i + 1]
                for _ in range(60):
                    m = 0.5 * (a + b)
                    if (stability_g(a, a)[1] - lvl) * (stability_g(m, m)[1] - lvl) <= 0: b = m
                    else: a = m
                edges.append((0.5 * (a + b), lvl))
    return edges


def g_scan_plot(R_best):
    """g-factor (and s) vs ROC of M3 = M4, over the same two ranges as roc_scan_plot."""
    setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, (lo, hi, ttl) in enumerate(((R_best - 0.8e-3, R_best + 0.2e-3, "Fine scan around best ROC"),
                                       (0.090, 0.116, "Full stability range"))):
        Rs = np.linspace(lo, hi, 4000)
        sg = np.array([stability_g(R, R) for R in Rs])
        s, g = sg[:, 0], sg[:, 1]
        a = ax[k]
        a.plot(Rs * 1e3, g, color="#2e86ab", lw=2, label="g = (A+D+2)/4")
        a.axhspan(0, 1, color="green", alpha=.12, label="stable (0 < g < 1)")
        a.axhline(0, color="k", lw=.8); a.axhline(1, color="k", lw=.8)
        a.axvline(R_best * 1e3, color="#d1495b", ls=":", label="best R = %.3f mm" % (R_best * 1e3))
        unst = (g <= 0) | (g >= 1)
        if unst.any():
            a.fill_between(Rs * 1e3, 0, 1, where=unst, transform=a.get_xaxis_transform(),
                           color="gray", alpha=.25, label="unstable")
        for Re, lvl in stability_edges():
            if lo <= Re <= hi:
                a.axvline(Re * 1e3, color="gray", ls="--", lw=1)
                a.text(Re * 1e3, 0.97, " edge %.2f mm" % (Re * 1e3), transform=a.get_xaxis_transform(),
                       rotation=90, va="top", ha="right", fontsize=9, color="dimgray")
        a.set_ylim(-0.2, 1.4) if k else a.set_ylim(min(-0.05, g.min() - 0.02), max(1.05, g.max() + 0.02))
        a.grid(alpha=.3)
        a.set_xlabel("ROC of M3 = M4 (mm)"); a.set_ylabel("stability factor g")
        a.set_title(ttl); a.legend(fontsize=9, loc="upper right" if k == 0 else "lower left")
        apply_ticks(a, "g_scan_fine" if k == 0 else "g_scan_full")
    fig.suptitle("Cavity stability g-factor vs ROC (fixed spacings)")
    fig.tight_layout()
    return fig

# =====================================================================
# 7. MAIN
# =====================================================================
def main():
    print("=== AS-BUILT GEOMETRY ===")
    print("b (M1-M2) %.2f | a23 (M2-M3) %.2f | d (M3-M4) %.2f | a41 (M4-M1) %.2f mm"
          % (B_FLAT * 1e3, A_23 * 1e3, D_CURV * 1e3, A_41 * 1e3))
    L_geo = B_FLAT + A_23 + D_CURV + A_41
    L_opt = L_geo + (N_CRYS - 1) * L_CRYS
    FSR = C / L_opt
    print("round trip geo/opt: %.2f / %.2f mm | FSR %.1f MHz | FWHM %.1f MHz | escape eff %.3f" %
          (L_geo * 1e3, L_opt * 1e3, FSR / 1e6,
           FSR / (2 * np.pi / (-np.log(R1) + LOSS)) / 1e6, (1 - R1) / (1 - R1 + LOSS)))

    roc_txt = ("fixed to %.1f mm" % (ROC_FIXED * 1e3)) if ROC_FIXED is not None else "free"
    lens_txt = ("fixed to f = " + ", ".join("%.0f" % (1e3 * f) for f in FIXED_FOCALS) + " mm") if FIXED_FOCALS else "free (%d lenses)" % N_FREE_LENSES
    print("\n=== FIT (ROC %s, lenses %s) ===" % (roc_txt, lens_txt))
    res = fit_all()
    if not res["stable"]:
        print("No stable solution found."); raise SystemExit(1)
    print("ROC M3 / M4        : %.2f / %.2f mm" % (res["R3"] * 1e3, res["R4"] * 1e3))
    for i, (f_, z_, w_) in enumerate(zip(res["focal_common"], res["z_common"], res["w_lens"]), 1):
        print("L%d: f=%4.0f mm at z=%7.1f mm (z from input waist) | beam radius on lens %.2f mm" %
              (i, f_ * 1e3, z_ * 1e3, w_ * 1e3))
    print("   (M1 sits at z = %.1f mm; last lens is %.1f mm before M1)" %
          (Z_M1 * 1e3, (Z_M1 - res["z_common"][-1]) * 1e3))
    if FIXED_FOCALS:
        w5 = 1.0
        for i in range(len(res["z_common"])):
            for sg in (-5e-3, 5e-3):
                zz = res["z_common"].copy(); zz[i] += sg
                w5 = min(w5, overlap(q_injected(zz, res["focal_common"]), res["q_eig"]))
        print("   position tolerance: overlap stays >= %.4f if any single lens is displaced by +-5 mm" % w5)
    xi, w0opt, zR = boyd_kleinman(res["w0"])
    flag = "OK" if abs(xi - XI_OPT) < 0.3 else ("close" if abs(xi - XI_OPT) < 1.0 else "SUBOPTIMAL")
    print("crystal w0 = %6.2f um | stability s = %+.3f | xi = %.3f (opt %.2f, w0_opt %.2f um) | h/h_opt = %.2f [%s]"
          % (res["w0"] * 1e6, res["s"], xi, XI_OPT, w0opt * 1e6, bk_rel(xi), flag))
    if SYMMETRIC_ROC:
        Rc = res["R3"]
        Rs = np.arange(Rc - 2e-3, Rc + 2e-3, 1e-6)
        good = [R for R in Rs if crystal_waist(R, R)[0] and 2.5 <= boyd_kleinman(crystal_waist(R, R)[0])[0] <= 3.2]
        if good:
            print("ROC tolerance: xi stays in 2.5..3.2 only for R = %.3f .. %.3f mm (window %.3f mm)" %
                  (min(good) * 1e3, max(good) * 1e3, (max(good) - min(good)) * 1e3))
    print("mode-matching overlap : %.6f  -> %s (threshold %.0f%%)" %
          (res["overlap"], "PASS" if res["meets_threshold"] else "FAIL", MODE_MATCH_THRESHOLD * 100))

    if ASTIG_THETA_DEG:
        ot, os_, tot, ceil = astigmatic_check(res, ASTIG_THETA_DEG)
        print("with the real %.2f deg tilt of M3/M4 (astigmatic): overlap t / s / TOTAL = %.4f / %.4f / %.4f"
              "  (best any spherical-lens set can do: %.4f)" % (ASTIG_THETA_DEG, ot, os_, tot, ceil))

    print("\n=== Round-trip behaviour of the injected beam (M1-M2 midpoint) ===")
    print("pass | w (um) | overlap")
    for i, wv, ov in verify_self_consistency(res):
        print("%4d | %8.3f | %9.6f" % (i, wv * 1e6, ov))
    print("(overlap is invariant under lossless round trips: it measures how well the "
          "injected beam matches the eigenmode, and stays constant by construction.)")

    if SYMMETRIC_ROC:
        roc_scan_plot(res["R3"])
        g_scan_plot(res["R3"])
        edges = [(R, l) for R, l in stability_edges()]
        s_b, g_b = stability_g(res["R3"], res["R4"])
        print("\n=== STABILITY (g-factor, M3 = M4) ===")
        print("g at R = %.2f mm: %.4f (s = %+.4f)" % (res["R3"] * 1e3, g_b, s_b))
        if len(edges) >= 2:
            lo_e, hi_e = edges[0][0], edges[-1][0]
            print("stable window: R = %.3f .. %.3f mm | margin to edges: %.2f mm / %.2f mm"
                  % (lo_e * 1e3, hi_e * 1e3, (res["R3"] - lo_e) * 1e3, (hi_e - res["R3"]) * 1e3))
    make_plot(res)
    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
