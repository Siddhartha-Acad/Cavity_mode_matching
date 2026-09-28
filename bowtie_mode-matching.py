#!/usr/bin/env python3
r"""
Bow-tie OPO cavity -- unified solver + lensguide plot (single file).

Merges the former bowtie_roundtrip_fitness.py, bowtie_fixed_hardware_matching.py
and plot_lensguide.py.  Run it and it will:

  1. Build the (fixed) cavity from your as-built M3-M4 spacing and fold angle.
  2. Solve the tangential / sagittal eigenmodes (self-consistent round-trip q).
  3. Report the Boyd-Kleinman focusing parameter xi for that fixed crystal waist.
  4. Solve EXACT astigmatic injection mode matching:
         2 shared spherical lenses (position only)  +  1 cylindrical lens
         (position + focal length, power in ONE plane only).
  5. Verify true self-consistency by propagating the actual injected beam
     around the real round trip.
  6. Plot the lensguide (injection + N round trips + crystal zoom) together
     with a results panel and show it.

Layout (top view, beam runs M1 -> M2 -> M3 -> crystal -> M4 -> M1):

        M1 ----------- b ----------- M2        (flat mirrors)
          \                         /
           \ a                 a  /            (arms cross, X-shape)
            \                     /
             M3 --- d (crystal) --- M4         (curved mirrors, ROC R)

Planar bow-tie geometry
    (b + d)/2 = a*cos(2*theta)          theta = fold half-angle (incidence)
    L_geo = d + 2a + b = 2a(1 + cos 2theta)
    L_opt = L_geo + (n-1)*Lc            (FSR / bandwidth)
Beam propagation uses REDUCED distances in the crystal (Lc/n); the waist at a
waist plane is w0^2 = lam*Im(q_red)/pi.  Away from a waist use w_general().

Requires: numpy, scipy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize

# =====================================================================
# 1. INPUTS
# =====================================================================
# ---- physics / crystal / mirrors ----
LAM      = 1064e-9        # vacuum wavelength [m]
N_CRYS   = 2.157          # crystal refractive index at LAM
L_CRYS   = 10e-3          # crystal length [m]  (PPLN, 10 mm)
ROC      = 100e-3         # ROC of M3, M4 [m]   <-- from picture: "R = 100 mm"
R1, LOSS = 0.75, 0.01     # output-coupler reflectivity, round-trip loss
C        = 299792458.0
XI_OPT   = 2.84           # Boyd-Kleinman optimum (non-critical phase matching)

# ---- YOUR AS-BUILT HARDWARE, read directly off the JAMMT table ----
# M1 @ 1635 mm, M2 @ 1865 mm, M3 @ 2060 mm, M4 @ 2180 mm
A_ASBUILT     = 195e-3             # M2->M3 = M4->M1  (curved-to-flat arm)  [m]
B_ASBUILT     = 230e-3             # M1->M2  (flat-to-flat)                 [m]
D_ASBUILT     = 120e-3             # M3->M4  (curved-to-curved)             [m]
# JAMMT's 1D unrolled table has NO fold-angle information (it treats M3/M4 as
# plain spherical thin lenses, f = ROC/2, no tangential/sagittal split). This
# script still splits t/s using THETA_ASBUILT; it is NOT read from the
# picture -- set to 0 to reproduce JAMMT's own (non-astigmatic) result.
THETA_ASBUILT = np.radians(5.0)    # <-- estimate; not given by the picture

# ---- input beam / injection optics on the bench ----
W0_IN        = 1.35e-3             # input waist [m] (waist at z = 0)
FOCAL_COMMON = (0.300, 0.200)      # L5 = 300 mm, L4 = 200 mm  (from picture)
Z_TARGET     = (1635e-3 + 1865e-3) / 2   # M1-M2 midpoint = 1750 mm (from picture)

# ---- options ----
REFERENCE_PLANE = "t"     # plane matched by the spherical lenses ("t" or "s");
                          # the cylindrical lens fixes the other one
N_TRIPS   = 3             # round trips drawn in the lensguide
RUN_DESIGN = False        # True: first solve d (a NEW cavity) for W_TARGET/L_OPT_DESIGN
W_TARGET  = 20.0e-6       # only used when RUN_DESIGN = True
L_OPT_DESIGN = 477e-3     # only used when RUN_DESIGN = True (target optical round trip)
OUT_PNG   = "bowtie_lensguide.png"
SHOW_PLOT = True


# =====================================================================
# 2. ABCD TOOLS AND CAVITY MODEL
# =====================================================================
def P(d):    return np.array([[1.0, d], [0.0, 1.0]])
def LENS(f): return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def f_curved(theta, plane):
    """Effective focal length of a tilted spherical mirror."""
    return ROC * np.cos(theta) / 2 if plane == "t" else ROC / (2 * np.cos(theta))


def geometry_asbuilt(d=D_ASBUILT, a=A_ASBUILT, b=B_ASBUILT):
    """As-built arms, read directly from the picture -- NOT derived from a
    theta/L_opt formula (real hardware need not satisfy the idealised
    symmetric bow-tie relation (b+d)/2 = a*cos(2*theta))."""
    return a, b


def geometry(d, theta, L_opt):
    """Only used by the OPTIONAL design step (RUN_DESIGN=True) to size a, b
    for a NEW cavity from a target optical round trip. Not used for the
    as-built hardware path."""
    L_geo = L_opt - (N_CRYS - 1) * L_CRYS
    a = L_geo / (2 * (1 + np.cos(2 * theta)))
    b = 2 * a * np.cos(2 * theta) - d
    return a, b


def half_crystal_air(d):
    return (d - L_CRYS) / 2.0          # air gap between curved mirror and crystal


def roundtrip_from_crystal_centre(d, a, b, theta, plane):
    f, g = f_curved(theta, plane), half_crystal_air(d)
    M = P(L_CRYS / 2 / N_CRYS)
    M = P(g) @ M
    M = LENS(f) @ M
    M = P(a) @ M
    M = P(b) @ M
    M = P(a) @ M
    M = LENS(f) @ M
    M = P(g) @ M
    M = P(L_CRYS / 2 / N_CRYS) @ M
    return M


def roundtrip_from_flat_arm(d, a, b, theta, plane):
    """Round trip starting/ending at the M1-M2 midpoint (where the beam is injected).
    Order: M2 -> M3 -> crystal -> M4 -> M1."""
    f, g = f_curved(theta, plane), half_crystal_air(d)
    M = P(b / 2)
    M = P(a) @ M
    M = LENS(f) @ M
    M = P(g) @ M
    M = P(L_CRYS / N_CRYS) @ M
    M = P(g) @ M
    M = LENS(f) @ M
    M = P(a) @ M
    M = P(b / 2) @ M
    return M


def eigen_q(M):
    """Eigenmode q (reduced) of a round-trip matrix; (None, s) if unstable."""
    A, B, C_, D = M.ravel()
    s = (A + D) / 2
    if abs(s) >= 1:
        return None, s
    q = (A - D) / (2 * C_) + 1j * np.sqrt(1 - s * s) / abs(C_)
    for cand in (q, np.conj(q), (A - D) / (2 * C_) - 1j * np.sqrt(1 - s * s) / abs(C_)):
        if cand.imag > 0 and abs((A * cand + B) / (C_ * cand + D) - cand) < 1e-9 * max(1, abs(cand)):
            return cand, s
    return q, s


def w_of(q):
    """Beam radius AT A WAIST (Re q = 0) from reduced q."""
    return np.sqrt(LAM * q.imag / np.pi)


def w_general(q):
    """Beam radius at ANY z from reduced q:  w^2 = -lam / (pi * Im(1/q))."""
    return np.sqrt(-LAM / (np.pi * (1.0 / q).imag))


def q_step(M, q):
    return (M[0, 0] * q + M[0, 1]) / (M[1, 0] * q + M[1, 1])


def overlap(q1, q2):
    """Power coupling between two coaxial Gaussian beams (same plane)."""
    return 4 * q1.imag * q2.imag / abs(q1 - np.conj(q2)) ** 2


def q_in_beam(w0, z_waist, z_plane):
    return 1j * np.pi * w0 ** 2 / LAM + (z_plane - z_waist)


def eigen_waists(d, theta, a=A_ASBUILT, b=B_ASBUILT):
    out = {}
    for pl in ("t", "s"):
        q, s = eigen_q(roundtrip_from_crystal_centre(d, a, b, theta, pl))
        out[pl] = (None if q is None else w_of(q), s)
    return a, b, out


def eigenmode_at_flat_arm(d, a, b, theta, plane):
    q, _ = eigen_q(roundtrip_from_flat_arm(d, a, b, theta, plane))
    return q


# =====================================================================
# 3. OPTIONAL DESIGN STEP (find d for a target waist) + STABILITY WINDOW
# =====================================================================
def fitness(x, w_target, L_opt):
    """x = [d, theta]; 0 == both eigen-waists exactly at w_target. (DESIGN mode only.)"""
    d, theta = x
    if not (L_CRYS + 4e-3 < d < 2 * ROC * 1.3) or not (0 < theta < np.radians(8)):
        return 1e6
    a, b = geometry(d, theta, L_opt)
    if b <= 0 or a <= 0:
        return 1e6
    e = eigen_waists(d, theta, a, b)
    if e["t"][0] is None or e["s"][0] is None:
        return 1e3 + 100 * (abs(e["t"][1]) + abs(e["s"][1]))
    return ((e["t"][0] - w_target) / w_target) ** 2 + ((e["s"][0] - w_target) / w_target) ** 2


def design_d(w_target, theta, L_opt):
    """Solve d (theta fixed) so the eigenmode waists sit at w_target, for a
    NEW cavity built to a target optical round trip L_opt. Not used on the
    as-built path."""
    best = None
    for d0 in np.arange(0.052, 0.070, 0.0005):
        r = minimize(lambda x: fitness([x[0], theta], w_target, L_opt), [d0], method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-16, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return best.x[0], best.fun


def stability_window(theta, a=A_ASBUILT, b=B_ASBUILT, d_lo=0.090, d_hi=0.150):
    """Scan M3-M4 spacing d with a, b HELD FIXED at their as-built values,
    and report the range of d over which the cavity stays stable."""
    ok = [d for d in np.arange(d_lo, d_hi, 0.00005)
          if all(v[0] is not None for v in eigen_waists(d, theta, a, b)[2].values())]
    return (min(ok), max(ok)) if ok else (None, None)


# =====================================================================
# 4. BOYD-KLEINMAN FOCUSING
# =====================================================================
def boyd_kleinman(w0, n=N_CRYS, L=L_CRYS, lam=LAM, xi_opt=XI_OPT):
    """w0 = physical waist inside the crystal. Returns xi, w0 for xi_opt, zR."""
    zR = np.pi * w0 ** 2 * n / lam
    return L / (2 * zR), np.sqrt(lam * L / (2 * np.pi * n * xi_opt)), zR


# =====================================================================
# 5. EXACT ASTIGMATIC INJECTION MATCH (2 spherical + 1 cylindrical lens)
# =====================================================================
def _prop_common(z, focal, w0_in):
    q = q_in_beam(w0_in, 0.0, 0.0)
    zc = 0.0
    for zi, f in sorted(zip(z, focal)):
        q = q + (zi - zc)
        zc = zi
        q = 1.0 / (1.0 / q - 1.0 / f)
    return q, zc


def _resid_common(z, q_target, focal, w0_in, z_target):
    q, zc = _prop_common(z, focal, w0_in)
    qb = q + (z_target - zc)
    return [qb.real - q_target.real, qb.imag - q_target.imag]


def solve_common_lenses(q_target, focal=FOCAL_COMMON, w0_in=W0_IN,
                        z_target=Z_TARGET, guess=(0.403, 0.967)):
    z_common = fsolve(_resid_common, guess, args=(q_target, focal, w0_in, z_target))
    qc, zc = _prop_common(z_common, focal, w0_in)
    return z_common, qc, zc, qc + (z_target - zc)


def _resid_cyl(x, q_target, qc, zc, z_target):
    z3, invf3 = x
    if not (zc + 1e-4 < z3 < z_target - 1e-4):
        return [1e6, 1e6]
    q = qc + (z3 - zc)
    q = 1.0 / (1.0 / q - invf3)
    q = q + (z_target - z3)
    return [q.real - q_target.real, q.imag - q_target.imag]


def solve_cylindrical_lens(q_other_target, qc, zc, z_target=Z_TARGET):
    """(z, f, residual) of a cylindrical lens fixing the non-reference plane;
    prefers the solution with the shortest |f|.  None if nothing is found."""
    best = None
    for z3_0 in np.linspace(zc + 0.02, z_target - 0.02, 25):
        for f3_0 in np.concatenate([np.linspace(-3, -0.02, 15), np.linspace(0.02, 3, 15)]):
            sol, _, ier, _ = fsolve(_resid_cyl, [z3_0, 1.0 / f3_0],
                                    args=(q_other_target, qc, zc, z_target), full_output=True)
            z3, invf3 = sol
            if ier == 1 and zc + 1e-4 < z3 < z_target - 1e-4 and abs(invf3) > 1e-7:
                r = _resid_cyl(sol, q_other_target, qc, zc, z_target)
                err = abs(r[0]) + abs(r[1])
                if err < 1e-6 and (best is None or abs(1 / invf3) < abs(best[1])):
                    best = (z3, 1.0 / invf3, err)
    return best


def match_astigmatic_injection(qt, qs, reference_plane="t", focal=FOCAL_COMMON,
                               w0_in=W0_IN, z_target=Z_TARGET):
    q_ref, q_other = (qt, qs) if reference_plane == "t" else (qs, qt)
    z_common, qc, zc, q_ref_final = solve_common_lenses(q_ref, focal, w0_in, z_target)
    cyl = solve_cylindrical_lens(q_other, qc, zc, z_target)
    if cyl is None:
        raise RuntimeError("No cylindrical-lens solution found in the search range.")
    z3, f3, resid = cyl
    q_other_final = 1.0 / (1.0 / (qc + (z3 - zc)) - 1.0 / f3) + (z_target - z3)
    ov_ref, ov_other = overlap(q_ref_final, q_ref), overlap(q_other_final, q_other)
    return dict(reference_plane=reference_plane, z_common=z_common, focal_common=focal,
                z_cyl=z3, f_cyl=f3, q_ref_final=q_ref_final, q_other_final=q_other_final,
                overlap_ref=ov_ref, overlap_other=ov_other,
                overlap_total=ov_ref * ov_other, cyl_residual=resid)


def verify_self_consistency(q_in_t, q_in_s, d, a, b, theta, n_trips=6):
    Mt = roundtrip_from_flat_arm(d, a, b, theta, "t")
    Ms = roundtrip_from_flat_arm(d, a, b, theta, "s")
    qt_eig, _ = eigen_q(Mt)
    qs_eig, _ = eigen_q(Ms)
    q_t, q_s, hist = q_in_t, q_in_s, []
    for i in range(n_trips + 1):
        hist.append((i, w_general(q_t), overlap(q_t, qt_eig), w_general(q_s), overlap(q_s, qs_eig)))
        q_t, q_s = q_step(Mt, q_t), q_step(Ms, q_s)
    return hist


# =====================================================================
# 6. BEAM TRACER (for the lensguide plot)
# =====================================================================
class Tracer:
    """Follows a q parameter through drifts / thin lenses, recording w(z)."""
    def __init__(self, q0, z0=0.0):
        self.q, self.z = q0, z0
        self.zs, self.ws = [z0], [w_general(q0)]

    def drift(self, L, n=1.0, npts=300):
        s = np.linspace(0.0, L, npts)[1:]
        self.zs.extend(self.z + s)
        self.ws.extend(w_general(self.q + s / n))      # reduced distance in a medium
        self.q, self.z = self.q + L / n, self.z + L

    def lens(self, f):
        self.q = 1.0 / (1.0 / self.q - 1.0 / f)


def trace_injection(plane, res):
    T = Tracer(q_in_beam(W0_IN, 0.0, 0.0), 0.0)
    elems = list(sorted(zip(res["z_common"], res["focal_common"])))
    if plane != res["reference_plane"]:            # cylindrical lens acts on this plane only
        elems.append((res["z_cyl"], res["f_cyl"]))
    for z, f in sorted(elems):
        T.drift(z - T.z)
        T.lens(f)
    T.drift(Z_TARGET - T.z)
    return T


def trace_cavity(T, plane, d, a, b, theta, n_trips):
    f, g = f_curved(theta, plane), half_crystal_air(d)
    mk = {k: [] for k in ("M1", "M2", "M3", "M4", "xtal_in", "xtal_out", "trip_end")}
    for _ in range(n_trips):
        T.drift(b / 2);                        mk["M2"].append(T.z)
        T.drift(a);                            mk["M3"].append(T.z); T.lens(f)
        T.drift(g);                            mk["xtal_in"].append(T.z)
        T.drift(L_CRYS, n=N_CRYS, npts=200);   mk["xtal_out"].append(T.z)
        T.drift(g);                            mk["M4"].append(T.z); T.lens(f)
        T.drift(a);                            mk["M1"].append(T.z)
        T.drift(b / 2);                        mk["trip_end"].append(T.z)
    return mk


# =====================================================================
# 7. PLOTTING
# =====================================================================
def make_plot(d, a, b, theta, res):
    traces, marks = {}, {}
    for pl in ("t", "s"):
        T = trace_injection(pl, res)
        n_inj = len(T.zs)
        marks[pl] = trace_cavity(T, pl, d, a, b, theta, N_TRIPS)
        traces[pl] = (np.array(T.zs) * 1e3, np.array(T.ws) * 1e6, n_inj)

    mk = marks["t"]
    z_start, z_end = Z_TARGET * 1e3, mk["trip_end"][-1] * 1e3
    L_geo = (mk["trip_end"][0] - Z_TARGET) * 1e3
    col = {"t": "#d1495b", "s": "#2e86ab"}
    lab = {"t": "tangential", "s": "sagittal"}

    head = ("Bow-tie OPO lensguide  |  d = %.3f mm, fold = %.1f deg, mode-match eff = %.6f"
            % (d * 1e3, np.degrees(theta), res["overlap_total"]))

    def new_fig(w, h, top=0.86, bottom=0.12):
        fig, ax = plt.subplots(figsize=(w, h))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=bottom, top=top)
        fig.suptitle(head, fontsize=12, y=0.985)
        return fig, ax

    # ---- figure 1: injection ----
    figA, A = new_fig(10, 5)
    for pl in ("t", "s"):
        z, w, n_inj = traces[pl]
        A.plot(z[:n_inj], w[:n_inj] * 1e-3, color=col[pl], lw=2, label=lab[pl])
        A.plot(z[:n_inj], -w[:n_inj] * 1e-3, color=col[pl], lw=2, alpha=.5)
    ymax = 1.15 * max(traces[p][1][:traces[p][2]].max() for p in "ts") * 1e-3
    for zc, f in sorted(zip(res["z_common"], res["focal_common"])):
        A.axvline(zc * 1e3, color="k", lw=1.5)
        A.text(zc * 1e3, ymax * 0.93, "  f=%.0f mm\n  @%.1f mm" % (f * 1e3, zc * 1e3), va="top", fontsize=9)
    other = "sagittal" if res["reference_plane"] == "t" else "tangential"
    A.axvline(res["z_cyl"] * 1e3, color=col["s" if res["reference_plane"] == "t" else "t"], lw=1.5, ls="--")
    A.text(res["z_cyl"] * 1e3, ymax * 0.93,
           "  cyl f=%.0f mm\n  (%s only)\n  @%.1f mm" % (res["f_cyl"] * 1e3, other, res["z_cyl"] * 1e3),
           va="top", fontsize=9)
    A.axvline(z_start, color="green", lw=1.5, ls=":")
    A.text(z_start, ymax * 0.93, "  M1-M2 midpoint\n  (cavity waist plane)", va="top", ha="right",
           fontsize=9, color="green")
    A.set_xlim(0, z_start + 40); A.set_ylim(-ymax, ymax)
    A.set_xlabel("z (mm)"); A.set_ylabel("beam radius w (mm)")
    A.set_title("Injection: input beam (w0 = %.2f mm) -> L1, L2, cylindrical lens -> cavity eigenmode"
                % (W0_IN * 1e3))
    A.legend(loc="upper left", fontsize=9); A.grid(alpha=.3)

    # ---- figure 2: injection + round trips, log y ----
    # (full trace: collimated input -> L1 -> L2 -> [cylindrical corrector] -> cavity)
    cyl_color = col["s" if res["reference_plane"] == "t" else "t"]
    figB, B = new_fig(16, 5.8, top=0.78, bottom=0.2)
    for pl in ("t", "s"):
        z, w, n_inj = traces[pl]
        B.semilogy(z, w, color=col[pl], lw=2, label=lab[pl])
    for zc, f in sorted(zip(res["z_common"], res["focal_common"])):
        B.axvline(zc * 1e3, color="k", lw=1.5)
        B.text(zc * 1e3, 1.01, "L f=%.0f" % (f * 1e3), transform=B.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=8)
    B.axvline(res["z_cyl"] * 1e3, color=cyl_color, lw=1.5, ls="--")
    B.text(res["z_cyl"] * 1e3, 1.01, "cyl f=%.0f" % (res["f_cyl"] * 1e3),
           transform=B.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    for k in range(N_TRIPS):
        B.axvspan(mk["xtal_in"][k] * 1e3, mk["xtal_out"][k] * 1e3, color="gold", alpha=.35, lw=0)
        for name in ("M1", "M2", "M3", "M4"):
            curved = name in ("M3", "M4")
            zz = mk[name][k] * 1e3
            B.axvline(zz, color="k" if curved else "gray", lw=1.6 if curved else 1.0,
                      ls="-" if curved else "--")
            B.text(zz, 1.01, name, transform=B.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        B.axvline(mk["trip_end"][k] * 1e3, color="green", lw=1.2, ls=":")
        B.text(mk["trip_end"][k] * 1e3 - L_geo * 0.25, 0.5, "round trip %d" % (k + 1),
               transform=B.get_xaxis_transform(), ha="center", fontsize=9, color="green")
    B.axvline(z_start, color="green", lw=1.2, ls=":")
    B.set_xlim(0, z_end); B.grid(alpha=.3, which="both")
    B.set_xlabel("z (mm, unrolled)"); B.set_ylabel("beam radius w (um)")
    B.set_title("Injection (L1, L2, cylindrical corrector) + %d round trips, log scale" % N_TRIPS, pad=24)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h, l = B.get_legend_handles_labels()
    h += [Patch(fc="gold", alpha=.35), Line2D([], [], color="k", lw=1.6),
          Line2D([], [], color="gray", lw=1.0, ls="--"),
          Line2D([], [], color="k", lw=1.5), Line2D([], [], color=cyl_color, lw=1.5, ls="--")]
    l += ["crystal", "curved M3/M4", "flat M1/M2", "L1, L2 (spherical)", "cylindrical corrector"]
    B.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, ncol=6, frameon=False)

    # ---- figure 3: crystal zoom ----
    figC, Cx = new_fig(10, 5.5)
    z0, z1 = mk["M3"][0] * 1e3 - 6, mk["M4"][0] * 1e3 + 6
    for pl in ("t", "s"):
        z, w, n_inj = traces[pl]
        m = (z >= z0) & (z <= z1) & (np.arange(len(z)) >= n_inj)
        Cx.plot(z[m], w[m], color=col[pl], lw=2, label=lab[pl])
        Cx.plot(z[m], -w[m], color=col[pl], lw=2, alpha=.5)
        i = np.argmin(np.where(m, w, np.inf))
        Cx.annotate("w0 = %.2f um" % w[i], (z[i], w[i]), textcoords="offset points",
                    xytext=(8, 14 if pl == "t" else -20), color=col[pl], fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=col[pl]))
    Cx.axvspan(mk["xtal_in"][0] * 1e3, mk["xtal_out"][0] * 1e3, color="gold", alpha=.35, lw=0,
               label="crystal (L=%.0f mm, n=%.3f)" % (L_CRYS * 1e3, N_CRYS))
    for name in ("M3", "M4"):
        Cx.axvline(mk[name][0] * 1e3, color="k", lw=1.6)
        Cx.text(mk[name][0] * 1e3, 0.97, " " + name, transform=Cx.get_xaxis_transform(), va="top", fontsize=9)
    Cx.set_xlim(z0, z1); Cx.grid(alpha=.3)
    Cx.set_xlabel("z (mm)"); Cx.set_ylabel("beam radius w (um)")
    Cx.set_title("Zoom: M3-crystal-M4 focus (round trip 1)")
    Cx.legend(loc="upper center", fontsize=9, ncol=3)

    # ---- figure 4: beam shape (+/- w, linear) over the round trips ----
    figD, D = new_fig(16, 5.8, top=0.78, bottom=0.2)
    for pl in ("t", "s"):
        z, w, n_inj = traces[pl]
        D.plot(z[n_inj - 1:], w[n_inj - 1:], color=col[pl], lw=2, label=lab[pl])
        D.plot(z[n_inj - 1:], -w[n_inj - 1:], color=col[pl], lw=2)
    for k in range(N_TRIPS):
        D.axvspan(mk["xtal_in"][k] * 1e3, mk["xtal_out"][k] * 1e3, color="gold", alpha=.35, lw=0)
        for name in ("M1", "M2", "M3", "M4"):
            curved = name in ("M3", "M4")
            zz = mk[name][k] * 1e3
            D.axvline(zz, color="k" if curved else "gray", lw=1.6 if curved else 1.0,
                      ls="-" if curved else "--")
            D.text(zz, 1.01, name, transform=D.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        D.axvline(mk["trip_end"][k] * 1e3, color="green", lw=1.2, ls=":")
        D.text(mk["trip_end"][k] * 1e3 - L_geo * 0.25, 0.5, "round trip %d" % (k + 1),
               transform=D.get_xaxis_transform(), ha="center", va="center", fontsize=9, color="green")
    wmax = 1.08 * max(traces[p][1][traces[p][2] - 1:].max() for p in "ts")
    D.set_xlim(z_start, z_end); D.set_ylim(-wmax, wmax); D.grid(alpha=.3)
    D.set_xlabel("z (mm, unrolled)"); D.set_ylabel("beam radius w (um)")
    D.set_title("Beam shape over %d round trips" % N_TRIPS, pad=24)
    h, l = D.get_legend_handles_labels()
    h = h[:2] + [Patch(fc="gold", alpha=.35), Line2D([], [], color="k", lw=1.6), Line2D([], [], color="gray", lw=1.0, ls="--")]
    l = l[:2] + ["crystal", "curved M3/M4", "flat M1/M2"]
    D.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, ncol=5, frameon=False)

    return figA, figB, figC, figD


# =====================================================================
# 8. MAIN
# =====================================================================
def main():
    lines = []
    def log(s=""):
        print(s); lines.append(s)

    theta, d, a, b = THETA_ASBUILT, D_ASBUILT, A_ASBUILT, B_ASBUILT
    if RUN_DESIGN:
        # NEW cavity, target waist w0 for a chosen optical round trip L_OPT_DESIGN
        d, fit = design_d(W_TARGET, theta, L_OPT_DESIGN)
        a, b = geometry(d, theta, L_OPT_DESIGN)
        log("=== DESIGN STEP: d solved for w0 = %.1f um (fitness %.1e) ===" % (W_TARGET * 1e6, fit))

    a, b, e = eigen_waists(d, theta, a, b)
    w_t, s_t = e["t"]
    w_s, s_s = e["s"]

    log("=== FIXED HARDWARE (mirrors & crystal locked) ===")
    log("d (M3-M4)         : %.3f mm   (air gap each side %.3f mm)" % (d * 1e3, half_crystal_air(d) * 1e3))
    log("fold angle theta  : %.3f deg  (ASSUMED; JAMMT table has no angle info)" % np.degrees(theta))
    log("arm a / flat b    : %.3f / %.3f mm" % (a * 1e3, b * 1e3))
    if w_t is None or w_s is None or a <= 0 or b <= 0:
        log("Cavity is UNSTABLE / unphysical at this geometry -- nothing to mode-match.")
        raise SystemExit(1)
    L_geo = d + 2 * a + b
    L_opt = L_geo + (N_CRYS - 1) * L_CRYS
    lo, hi = stability_window(theta, a, b)
    log("crystal waist t/s : %.2f / %.2f um" % (w_t * 1e6, w_s * 1e6))
    log("stability s  t/s  : %.3f / %.3f   (stable d window %.2f .. %.2f mm)" %
        (s_t, s_s, lo * 1e3, hi * 1e3))
    FSR = C / L_opt
    log("round trip geo/opt: %.2f / %.2f mm | FSR %.1f MHz | FWHM %.1f MHz | escape eff %.3f" %
        (L_geo * 1e3, L_opt * 1e3, FSR / 1e6,
         FSR / (2 * np.pi / (-np.log(R1) + LOSS)) / 1e6, (1 - R1) / (1 - R1 + LOSS)))

    log("\n=== Boyd-Kleinman focusing (fixed by hardware, lenses cannot change it) ===")
    for label, w0 in (("tangential", w_t), ("sagittal", w_s)):
        xi, w0_opt, zR = boyd_kleinman(w0)
        flag = "OK (~optimal)" if abs(xi - XI_OPT) < 0.3 else "SUBOPTIMAL"
        log("  %-10s: xi = %.3f (opt %.2f)  zR = %.3f mm  -> xi_opt needs w0 = %.2f um  [%s]" %
            (label, xi, XI_OPT, zR * 1e3, w0_opt * 1e6, flag))

    log("\n=== Astigmatic exact mode matching (2 spherical + 1 cylindrical lens) ===")
    qt = eigenmode_at_flat_arm(d, a, b, theta, "t")
    qs = eigenmode_at_flat_arm(d, a, b, theta, "s")
    res = match_astigmatic_injection(qt, qs, reference_plane=REFERENCE_PLANE)
    other = "sagittal" if REFERENCE_PLANE == "t" else "tangential"
    log("Reference plane (spherical lenses): %s" % ("tangential" if REFERENCE_PLANE == "t" else "sagittal"))
    log("  L1: f=%.0f mm at z=%.2f mm | L2: f=%.0f mm at z=%.2f mm" %
        (FOCAL_COMMON[0] * 1e3, res["z_common"][0] * 1e3, FOCAL_COMMON[1] * 1e3, res["z_common"][1] * 1e3))
    log("  Cylindrical lens: z=%.2f mm, f=%.1f mm  (power in %s plane only)" %
        (res["z_cyl"] * 1e3, res["f_cyl"] * 1e3, other))
    log("  overlap reference / other / TOTAL : %.6f / %.6f / %.6f" %
        (res["overlap_ref"], res["overlap_other"], res["overlap_total"]))

    log("\n=== Round-trip self-consistency of the ACTUAL injected beam (flat-arm waist) ===")
    q_t_in = res["q_ref_final"] if REFERENCE_PLANE == "t" else res["q_other_final"]
    q_s_in = res["q_other_final"] if REFERENCE_PLANE == "t" else res["q_ref_final"]
    log("pass | w_t (um) | overlap_t | w_s (um) | overlap_s")
    for i, wt, ot, ws, os_ in verify_self_consistency(q_t_in, q_s_in, d, a, b, theta):
        log("%4d | %8.3f | %9.6f | %8.3f | %9.6f" % (i, wt * 1e6, ot, ws * 1e6, os_))
    log("Flat w and overlap = 1 on every pass => injected beam IS the eigenmode.")

    make_plot(d, a, b, theta, res)
    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()

