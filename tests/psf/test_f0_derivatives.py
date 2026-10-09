"""Derivative checks on the data term F0, both models, with and without the blur.

Run:
  PYTHONPATH=<repo>/src python tests/psf/test_f0_derivatives.py

With J = K|x|^2 (K the one Gaussian on the detector intensity, the identity when
no blur is configured), a = sqrt(J) and s = sqrt(d):

    model=intensity   F0 = 1/N sum W (a^2 - d)^2
    model=amplitude   F0 = 1/N sum W (a   - s)^2

Everything below is checked FOUR times -- both models x {K = identity, K = a
sigma 1.7 px Gaussian} -- against independent float64 references and against
finite differences.

The Hessian is the thing worth testing: it carries two nested convolutions with
the mask between them, and a wrong one still produces a plausible descent
direction that the gradient checks would not catch.  The amplitude model adds a
second way to get it wrong: its curvature weight s/(4a^3) sits BETWEEN the two
convolutions, in the same slot the mask does, and dropping it leaves a Hessian
that is still symmetric and still positive on its first term.

The finite-difference bars are set against the sum of |terms| rather than the
result: for random x,y,z these are heavily cancelling sums (dF0 is 60-150x
smaller than the sum of its absolute terms), so dividing by the result would
measure the cancellation and not the derivative.
"""
import sys
import numpy as np
import cupy as cp
from scipy.special import ive
from scipy.ndimage import convolve1d as cpu_convolve1d

from holotomocupy.rec_mpi import Rec, _psf_taps

# _psf_taps takes the Gaussian sigma in binned detector pixels directly -- the
# config states that width and nothing converts it -- so these checks, which are
# all in pixels, hand it the number as it is.
def taps(sigma_px):
    return _psf_taps(sigma_px)[1]


fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        fails.append(name)


chunk, nz, n, ndist, ntheta = 2, 64, 48, 2, 5
rng = np.random.default_rng(1234)


def cxa(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


x_h = cxa((chunk, nz, n))
y_h = cxa((chunk, nz, n))
z_h = cxa((chunk, nz, n))
w_h = cxa((chunk, nz, n))
d_h = np.abs(rng.standard_normal((chunk, nz, n))).astype('float32')
my_h = (rng.random((chunk, nz, 1)) > 0.15).astype('float32')
mx_h = (rng.random((chunk, 1, n)) > 0.15).astype('float32')

x, y, z, w = (cp.asarray(a) for a in (x_h, y_h, z_h, w_h))
d = cp.asarray(d_h)
my, mx = cp.asarray(my_h), cp.asarray(mx_h)

DATA_SIZE = ntheta * ndist * nz * n
W64 = (my_h * mx_h).astype('float64')
D64 = d_h.astype('float64')   # d is the measured INTENSITY; nothing squares it


def harness(sigma, model='intensity'):
    r = object.__new__(Rec)
    r.data_size = DATA_SIZE
    r._mask_y, r._mask_x = my, mx
    r._dist_idx = 0
    r.psf_w = None if sigma is None else taps(sigma)
    r.model = model
    # gF0 starts by running the cascade from level 1; these checks hand it the
    # detector-plane field directly, so that step is the identity here.
    r.apply_F_from = lambda v, i: v
    return r


def rp64(a, b):
    """Re(conj(a) b) in float64 -- what reprod computes on the GPU."""
    return (a.real.astype('float64') * b.real.astype('float64')
            + a.imag.astype('float64') * b.imag.astype('float64'))


def cpu_blur(a, sigma):
    """K in float64.  'wrap' is the periodic boundary _psf_blur uses; with a
    symmetric kernel that makes K circulant, hence exactly self-adjoint."""
    if sigma is None:
        return np.asarray(a, dtype='float64')
    rr = max(2, int(np.ceil(4.0 * sigma)))
    k = ive(np.abs(np.arange(-rr, rr + 1)), sigma * sigma)
    k = k / k.sum()
    out = np.asarray(a, dtype='float64')
    for ax in (-2, -1):
        out = cpu_convolve1d(out, k, axis=ax, mode='wrap')
    return out


# The two models as float64 pointwise rules, written straight off the table in
# rec_mpi's F0 comment.  g is the per-pixel misfit, gp is g'/2 (what goes inside
# the outer K to make p) and gc is g''/2 (the weight between the two Hessian
# convolutions).  A floor matching Rec._AMP_FLOOR, so the reference is the same
# function the GPU evaluates and not a subtly different one.
AMP_FLOOR = 1e-3


def rules(model):
    if model == 'intensity':
        return (lambda J: (J - D64) ** 2,
                lambda J: J - D64,
                lambda J: np.ones_like(J))
    S64 = np.sqrt(D64)
    af = lambda J: np.maximum(np.sqrt(J), AMP_FLOOR)      # noqa: E731
    return (lambda J: (np.sqrt(J) - S64) ** 2,
            lambda J: 0.5 * (1.0 - S64 / af(J)),
            lambda J: 0.25 * S64 / af(J) ** 3)


# THE FIELD, and why the two models do not get the same one.
#
# The intensity model is a polynomial in x and is perfectly conditioned on the
# raw complex-normal field, near-zero pixels and all -- that is the point of it,
# so it keeps exactly the field it has always been checked on.
#
# The amplitude model divides by a^3, and a complex normal has |x| down to ~0.01
# here, where a^-3 reaches 8e5.  The ANALYTIC derivative is fine there and is
# checked against float64 on that same raw field below ("near a field zero"),
# but a CENTRAL DIFFERENCE is not: its truncation error goes as h^2 times the
# FOURTH derivative, which at such a pixel is ~1e10, so the difference measures
# that rather than the second derivative at every usable h.  The finite-difference
# checks therefore run on a bright field, x + 2, where |x| >~ 1 -- which is also
# the only regime the amplitude model is ever used in: the data are flat-field
# normalised, so a real propagated hologram has |psi| ~ 1 everywhere.
FIELD = {'intensity': x_h,
         'amplitude': (x_h + 2.0).astype('complex64')}

for MODEL in ('intensity', 'amplitude'):
  X_H = FIELD[MODEL]
  X = cp.asarray(X_H)
  for tag, SIGMA in ((f"{MODEL[:3]} no blur   ", None),
                     (f"{MODEL[:3]} sigma=1.7 ", 1.7)):
    print(f"\n--- {tag.strip()} " + "-" * 50)
    r = harness(SIGMA, MODEL)
    gfun, gpfun, gcfun = rules(MODEL)

    # ---- float64 references -------------------------------------------
    J64 = cpu_blur(rp64(X_H, X_H), SIGMA)
    p64 = cpu_blur(W64 * gpfun(J64), SIGMA)
    c64 = gcfun(J64)

    ref_F0 = (W64 * gfun(J64)).sum() / DATA_SIZE
    ref_dF0 = 4.0 / DATA_SIZE * (p64 * rp64(X_H, y_h)).sum()
    ref_w = 4.0 / DATA_SIZE * (p64 * rp64(X_H, w_h)).sum()
    Kz = cpu_blur(rp64(X_H, z_h), SIGMA)
    Ky = cpu_blur(rp64(X_H, y_h), SIGMA)
    ref_d2 = 4.0 / DATA_SIZE * (2.0 * (W64 * c64 * Kz * Ky).sum()
                                + (p64 * rp64(z_h, y_h)).sum())
    # Both are near-cancelling sums for random x,y,z: dF0 is 60-154x smaller
    # than the sum of |its terms|, so a finite difference divided by the RESULT
    # measures that cancellation rather than the derivative.  The bars below are
    # set against the term scale instead, which makes them ~50x tighter.
    sc_dF0 = 4.0 / DATA_SIZE * np.abs(p64 * rp64(X_H, y_h)).sum()
    sc_d2 = 4.0 / DATA_SIZE * (2.0 * np.abs(W64 * c64 * Kz * Ky).sum()
                               + np.abs(p64 * rp64(z_h, y_h)).sum())

    got_F0 = float(r.F0(X, d).get())
    got_dF0 = float(r.dF0(X, y, d).get())
    got_d2 = float(r.d2F_dF0(X, y, z, None, d).get())
    got_d2w = float(r.d2F_dF0(X, y, z, w, d).get())

    check(f"{tag}F0 vs float64", abs(got_F0 - ref_F0) / abs(ref_F0) < 1e-5,
          f"{got_F0:.10g} vs {ref_F0:.10g}")
    check(f"{tag}dF0 vs float64", abs(got_dF0 - ref_dF0) / abs(ref_dF0) < 1e-5,
          f"{got_dF0:.10g} vs {ref_dF0:.10g}")
    check(f"{tag}d2F_dF0 (w=None) vs float64",
          abs(got_d2 - ref_d2) / abs(ref_d2) < 1e-5, f"{got_d2:.10g} vs {ref_d2:.10g}")
    check(f"{tag}d2F_dF0 w-term vs float64",
          abs((got_d2w - got_d2) - ref_w) / abs(ref_w) < 1e-5,
          f"{got_d2w - got_d2:.10g} vs {ref_w:.10g}")

    # gF0's array contracted with y must reproduce dF0
    ga = r.gF0(X, d)
    gy = float(cp.sum(cp.real(cp.conj(ga) * y)).get())
    check(f"{tag}gF0 array contracted with y == dF0",
          abs(gy - ref_dF0) / abs(ref_dF0) < 1e-5, f"{gy:.10g} vs {ref_dF0:.10g}")

    # ---- finite differences -------------------------------------------
    best = min((abs((float(r.F0(X + h * y, d).get())
                     - float(r.F0(X - h * y, d).get())) / (2 * h) - got_dF0)
                / sc_dF0, h) for h in (1e-3, 3e-3, 1e-2, 3e-2))
    check(f"{tag}dF0 vs central difference", best[0] < 1e-5,
          f"best |d|/sum|terms|={best[0]:.2e} at h={best[1]:g}")

    best = min((abs((float(r.dF0(X + h * z, y, d).get())
                     - float(r.dF0(X - h * z, y, d).get())) / (2 * h) - got_d2)
                / sc_d2, h) for h in (1e-3, 3e-3, 1e-2, 3e-2))
    check(f"{tag}d2F_dF0 vs difference of dF0", best[0] < 1e-5,
          f"best |d|/sum|terms|={best[0]:.2e} at h={best[1]:g}")

    # ---- structure -----------------------------------------------------
    d2_zy = float(r.d2F_dF0(X, z, y, None, d).get())
    check(f"{tag}Hessian is symmetric in y,z",
          abs(d2_zy - got_d2) / abs(got_d2) < 1e-5, f"{d2_zy:.10g} vs {got_d2:.10g}")
    check(f"{tag}d2F_dF0[z,z] >= 0 on the curvature term",
          float(r.d2F_dF0(X, z, z, None, d).get()) > 0)

# --- the mask really is inside the outer convolution ----------------------
# Masking the blurred p afterwards instead is a different number; if it were
# not, the placement would be untested.  Intensity only: the placement of W is
# the same in both models, so checking it once is enough.
rp = harness(1.7)
J64 = cpu_blur(rp64(x_h, x_h), 1.7)
inside = 4.0 / DATA_SIZE * (cpu_blur(W64 * (J64 - D64), 1.7) * rp64(x_h, y_h)).sum()
outside = 4.0 / DATA_SIZE * (W64 * cpu_blur(J64 - D64, 1.7) * rp64(x_h, y_h)).sum()
check("mask-inside and mask-outside are distinguishable",
      abs(inside - outside) / abs(inside) > 0.05,
      f"inside={inside:.8g} outside={outside:.8g}")
check("dF0 matches the mask-INSIDE form",
      abs(float(rp.dF0(x, y, d).get()) - inside) / abs(inside) < 1e-5)

# --- K -> identity as sigma -> 0 ------------------------------------------
# The discrete Gaussian is I + sigma^2 * (second difference) + O(sigma^4): its
# side taps carry ~sigma^2 of the weight.  So the right statement is not "tiny
# sigma equals no blur" -- dF0 is a heavily cancelling sum, and a deviation that
# is negligible against the terms is not negligible against the result -- but
# that the deviation is O(sigma^2) measured against the term scale.  Halving
# sigma must quarter it; anything that is really an identity check dressed up as
# a tolerance would show a ratio of 1.
flat = float(harness(None).dF0(x, y, d).get())
dev = [abs(float(harness(sg).dF0(x, y, d).get()) - flat) for sg in (0.04, 0.02)]
scale = 4.0 / DATA_SIZE * np.abs(
    (W64 * (rp64(x_h, x_h) - D64)) * rp64(x_h, y_h)).sum()
check("tiny sigma reproduces the no-blur path, to O(sigma^2)",
      dev[0] / scale < 1e-3,
      f"|dF0(0.04)-dF0(0)|/sum|terms|={dev[0]/scale:.2e} "
      f"(vs result: {dev[0]/abs(flat):.2e})")
check("... and the deviation falls as sigma^2 when sigma halves",
      3.0 < dev[0] / dev[1] < 5.0, f"ratio={dev[0]/dev[1]:.2f} (expect 4)")
check("sigma=0 gives no kernel at all, i.e. the identity exactly",
      harness(0.0).psf_w is None
      and float(harness(0.0).F0(x, d).get()) == float(harness(None).F0(x, d).get()))

# --- amplitude near a field zero -------------------------------------------
# The bright field above never exercises the floor.  Here the raw complex-normal
# field is used, whose smallest |x| is ~0.011 (a^-3 ~ 8e5), plus a column forced
# to EXACTLY zero so the floor is certainly active.  Only the analytic values
# are checked -- against float64 running the identical floored rule -- because a
# finite difference is meaningless at such a pixel (see FIELD above).  What this
# actually tests is that nothing returns inf or nan there, which the unfloored
# 1/a and 1/a^3 would.
xz_h = x_h.copy()
xz_h[:, :, 3] = 0
xz = cp.asarray(xz_h)
for SIGMA, tg in ((None, "amp zeros no blur "), (1.7, "amp zeros sigma=1.7 ")):
    rz = harness(SIGMA, 'amplitude')
    gfun, gpfun, gcfun = rules('amplitude')
    J64 = cpu_blur(rp64(xz_h, xz_h), SIGMA)
    p64 = cpu_blur(W64 * gpfun(J64), SIGMA)
    Kz = cpu_blur(rp64(xz_h, z_h), SIGMA)
    Ky = cpu_blur(rp64(xz_h, y_h), SIGMA)
    e_F0 = (W64 * gfun(J64)).sum() / DATA_SIZE
    e_dF0 = 4.0 / DATA_SIZE * (p64 * rp64(xz_h, y_h)).sum()
    e_d2 = 4.0 / DATA_SIZE * (2.0 * (W64 * gcfun(J64) * Kz * Ky).sum()
                              + (p64 * rp64(z_h, y_h)).sum())
    vals = [float(rz.F0(xz, d).get()), float(rz.dF0(xz, y, d).get()),
            float(rz.d2F_dF0(xz, y, z, None, d).get())]
    check(f"{tg}all finite (no inf/nan from 1/a, 1/a^3)",
          all(np.isfinite(v) for v in vals), f"{vals}")
    for nm, got, ref in zip(("F0", "dF0", "d2F_dF0"), vals, (e_F0, e_dF0, e_d2)):
        check(f"{tg}{nm} vs floored float64",
              abs(got - ref) / abs(ref) < 1e-4, f"{got:.10g} vs {ref:.10g}")

# --- amplitude-specific: the curvature weight is really in the sandwich ----
# s/(4a^3) sits between the two convolutions.  Dropping it gives a Hessian that
# is still symmetric and whose first term is still >= 0, so nothing above would
# catch it -- only a direct comparison does.
ra = harness(1.7, 'amplitude')
J64 = cpu_blur(rp64(x_h, x_h), 1.7)
Kz = cpu_blur(rp64(x_h, z_h), 1.7)
Ky = cpu_blur(rp64(x_h, y_h), 1.7)
S64 = np.sqrt(D64)
c64 = 0.25 * S64 / np.maximum(np.sqrt(J64), AMP_FLOOR) ** 3
pa64 = cpu_blur(W64 * 0.5 * (1.0 - S64 / np.maximum(np.sqrt(J64), AMP_FLOOR)), 1.7)
with_c = 4.0 / DATA_SIZE * (2.0 * (W64 * c64 * Kz * Ky).sum()
                            + (pa64 * rp64(z_h, y_h)).sum())
without_c = 4.0 / DATA_SIZE * (2.0 * (W64 * Kz * Ky).sum()
                               + (pa64 * rp64(z_h, y_h)).sum())
check("amp: with and without the curvature weight are distinguishable",
      abs(with_c - without_c) / abs(with_c) > 0.05,
      f"with={with_c:.8g} without={without_c:.8g}")
check("amp: d2F_dF0 matches the WITH-weight form",
      abs(float(ra.d2F_dF0(x, y, z, None, d).get()) - with_c) / abs(with_c) < 1e-5)

# --- amplitude is not intensity -------------------------------------------
# Guards against a dispatch that silently falls through to the intensity path.
check("amp and int give different F0",
      abs(float(harness(1.7, 'amplitude').F0(x, d).get())
          - float(harness(1.7, 'intensity').F0(x, d).get())) > 1e-9)
check("amp and int give different dF0",
      abs(float(harness(1.7, 'amplitude').dF0(x, y, d).get())
          - float(harness(1.7, 'intensity').dF0(x, y, d).get())) > 1e-12)

# --- amplitude at psf_sigma=0 IS the historical ||x| - sqrt(d)|^2 misfit ----
# The whole point of putting the blur on intensity and the comparison on
# amplitude: switching the PSF off must land exactly on the old functional.
plain = (W64 * (np.abs(x_h.astype('complex128')) - np.sqrt(D64)) ** 2).sum() / DATA_SIZE
got = float(harness(None, 'amplitude').F0(x, d).get())
check("amp at sigma=0 is the plain ||x| - sqrt(d)|^2 misfit",
      abs(got - plain) / abs(plain) < 1e-6, f"{got:.10g} vs {plain:.10g}")

# --- the 4x relation between the two models --------------------------------
# (a^2 - d) = (a - s)(a + s), so F_int = sum W (a+s)^2 (a-s)^2 exactly.  Not an
# approximation here -- an identity, and a check that both kernels agree on a.
a64 = np.sqrt(cpu_blur(rp64(x_h, x_h), 1.7))
ident = (W64 * (a64 + np.sqrt(D64)) ** 2 * (a64 - np.sqrt(D64)) ** 2).sum() / DATA_SIZE
check("F_int == sum W (a+s)^2 (a-s)^2 exactly",
      abs(float(harness(1.7, 'intensity').F0(x, d).get()) - ident) / ident < 1e-6,
      f"{float(harness(1.7,'intensity').F0(x, d).get()):.10g} vs {ident:.10g}")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
