#!/usr/bin/env python
"""End-to-end and cascade-level checks for the PSF, inside the real Rec.

    PYTHONPATH=<repo>/src python tests/psf/test_psf_e2e.py [--scratch DIR]

test_psf.py checks the blur operator and test_f0_derivatives.py checks the F0
formulas on a hand-built harness.  This file is the level above both: a real
Rec, built from real config keys, forward-modelling and reconstructing a real
(if small) synthetic dataset.  It is what catches a wiring mistake -- a sigma
that never reaches the solver, a distance index off by one, a blur applied to
the model but not to the generated data -- that the unit tests cannot see.

  1. the config key reaches Rec unconverted: psf_sigma is stated in binned
     detector pixels and that is exactly the width Rec convolves with.
  2. F0's Taylor expansion still converges at orders 1/2/3 with the PSF ON, in
     the real Rec rather than the harness.  This is the cascade-level version
     of the derivative tests, and the log-log slopes are the sharp instrument.
  3. check_approximation: the real functional along the BH descent direction
     tracks the quadratic model the line search is built from.  A wrong Hessian
     shows up here as the two curves separating.
  4. the misfit at the TRUE obj/prb/pos is exactly zero for the model that
     generated the data and nonzero for the other one -- the sharpest wiring
     test here, and it needs no iterations at all.
  5. recovery: data generated WITH a blur is reconstructed better by a solver
     that models it than by one that does not, with the sharp-data run as the
     control that says the effect is the blur and not blurring per se.

Twenty-four iterations is not a convergence study; it is enough to separate the
two models under a common referee.  See the comment at section 5 for why the
obvious scores -- each run's own final err, and object-vs-truth correlation --
are both measurably the wrong instrument here.

MISFIT MODEL.  Everything above is a statement about the PSF, and the PSF lives
on the intensity in both misfit models, so every section here has to hold under
either.  MODEL=amplitude re-runs the whole file against
sum (sqrt(K|psi|^2) - sqrt(d))^2 instead of sum (K|psi|^2 - d)^2:

    MODEL=amplitude PYTHONPATH=<repo>/src python tests/psf/test_psf_e2e.py

d is the measured INTENSITY in both, so the generated data is untouched and
section 4 stays the sharp wiring test it was -- a model that matches the data
hits zero whichever way it compares.  lam_prbfit is NOT rescaled between the
arms: PrbfitTerm follows the same knob, so its residual shrinks by the same ~4
as F0 does (the intensity residual is the amplitude one times (a+s) ~ 2) and the
ratio lam multiplies is already invariant.  lam_laplacian, which would have to
be rescaled, is 0 here.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import cupy as cp
from mpi4py import MPI

import holotomocupy.rec_mpi as rec_mpi
from holotomocupy.rec_mpi import Rec, _psf_taps
from holotomocupy.logger_config import set_log_level

cp.cuda.set_pinned_memory_allocator(None)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# ---- geometry (as tests/tomo/test_upsample_e2e.py, known to run) ------------
n = nz = 128
nzobj = nobj = 160
ndist, ntheta, nchunk = 2, 16, 4

energy = 17.1
detector_pixelsize = 1.4760147601476e-6 * 2 * 8
focustodetectordistance = 1.217
z1 = np.array([5.110, 6.879]) * 1e-3
theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')

# The real configs sit at sigma ~ 0.2-1.5 binned px, but this toy geometry is
# binned 16x, so the physical blur would land at ~0.1 px -- too small to change
# a reconstruction.  The width is therefore EXAGGERATED here on purpose: the
# object under test is the machinery, and only a blur the solver can actually
# feel exercises it.
PSF_SIGMA = 1.80

# 'intensity' or 'amplitude'; see the module docstring.
MODEL = os.environ.get('MODEL', 'intensity')
LAM_PRBFIT = 2e-3      # NOT rescaled; PrbfitTerm follows MODEL, see the docstring

comm = MPI.COMM_WORLD


def make_args(psf_sigma, niter=8, **kw):
    a = SimpleNamespace(
        nz=nz, n=n, nzobj=nzobj, nobj=nobj,
        ntheta=ntheta, ndist=ndist, nchunk=nchunk, niter=niter, start_iter=0,
        tomo_upsample=1,
        rho=[1, 0.05, 0.02, 0], lam_prbfit=LAM_PRBFIT, lam_laplacian=0,
        checkpoint_step=-1, error_step=1, vis_step=-1, check_approx=False,
        energy=energy, focustodetectordistance=focustodetectordistance,
        z1=z1, detector_pixelsize=detector_pixelsize, theta=theta,
        mask=0.9, shift_type='cubic', comm=comm,
        psf_sigma=psf_sigma, model=MODEL,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# Hard-edged spheres, not smooth blobs: the blur is a low-pass filter, so a
# phantom with no high spatial frequencies gives it almost nothing to act on.
#
# The amplitude matters as much as the shape.  The obvious phantom to copy is
# the one in tests/tomo/test_upsample_e2e.py, but its 1e-6 is a deliberately
# negligible object -- it gives a peak projected phase of 5e-7 rad, so
# exp(i*proj) is 1 to six digits and the data is pure probe.  Measured: the
# object then moves the data by 1.9e-07 relative, which is the float32 noise
# floor, and every "does the object come back" question is unanswerable.
# OBJ_SCALE = 1.0 gives ~0.45 rad peak phase, and the response is linear in it.
OBJ_SCALE = 1.0


def phantom():
    rng = np.random.default_rng(7)
    ax = (np.arange(nobj) - nobj / 2) / nobj
    Z, Y, X = np.meshgrid((np.arange(nzobj) - nzobj / 2) / nzobj, ax, ax, indexing='ij')
    v = np.zeros((nzobj, nobj, nobj), 'float32')
    for _ in range(10):
        c = rng.uniform(-0.22, 0.22, 3)
        rad = rng.uniform(0.05, 0.13)
        v[((Z - c[0])**2 + (Y - c[1])**2 + (X - c[2])**2) < rad * rad] = 1.0
    return (OBJ_SCALE * (-1.0 + 1j * 0.01) * v).astype('complex64')


def probe():
    x = np.fft.fftfreq(n) * n
    XX, YY = np.meshgrid(x, x, indexing='ij')
    p = np.exp(-(XX**2 + YY**2) / (2 * (n / 3.0)**2)).astype('complex64')
    p = np.tile(p, (ndist, 1, 1))
    return p / np.mean(np.abs(p), axis=(1, 2))[:, None, None]


def free(cl):
    del cl
    cp.get_default_memory_pool().free_all_blocks()


ap = argparse.ArgumentParser()
ap.add_argument('--scratch', default='/local/ssd/vnikitin/psf_e2e')
opt = ap.parse_args()
set_log_level('WARNING')
os.makedirs(opt.scratch, exist_ok=True)

rng = np.random.default_rng(10)
POS = 12 * (rng.random((ndist, ntheta, 2), dtype='float32') - 0.5)
PRB = probe()
OBJ = phantom()

# ---------------------------------------------------------------------------
print("1. config keys reach Rec")
# ---------------------------------------------------------------------------
cl = Rec(make_args(PSF_SIGMA))
want, _ = _psf_taps(PSF_SIGMA)
check("sigma matches _psf_taps", abs(cl.psf_sigma - want) < 1e-12,
      f"sigma={cl.psf_sigma:.3f} det.px")
check("PSF is active", cl.psf_w is not None)
# The config states the sigma in binned detector pixels and Rec convolves with
# exactly that -- no pixel size, no magnification, no per-distance scaling.  The
# geometry that produces the number is arithmetic in the config comment.
check("sigma reaches Rec unconverted",
      cl.psf_sigma == PSF_SIGMA, f"{cl.psf_sigma:.4f} px")

# ---------------------------------------------------------------------------
print("2. F0 Taylor orders with the PSF on, in the real Rec")
# ---------------------------------------------------------------------------
# Two things about the probe direction, both learned by measuring:
#
#  * +0.5 keeps |x0| away from zero, exactly as tests/holotomo3d/
#    test_approximation.py does: F0 differentiates |x|, whose curvature is
#    1/|x|, so a few near-zero pixels would dominate the residual.
#  * the direction is CORRELATED with x0 rather than independent of it.  With a
#    random complex dx every odd-order Taylor coefficient is a random-sign sum
#    and cancels, so the measured rates come out 2 / 2 / 4 instead of 1 / 2 / 3
#    -- e1 never sees the linear term and the O(h^2) check says nothing about
#    dF0.  A direction along x0 with a positive random weight leaves all three
#    orders present, and the rates come out 1 / 2 / 3 as they should.
#
# h stops at 0.05: F0 ~ 0.37 here, so a float32 sum resolves ~3e-8, and below
# h = 0.05 the third-order residual (6e-7 there) reaches that floor and the
# slope stops meaning anything.
r2 = np.random.default_rng(42)


def rc(shape):
    return (r2.standard_normal(shape) + 1j * r2.standard_normal(shape)).astype('complex64')


L = np.array([0.4, 0.283, 0.2, 0.141, 0.1, 0.071, 0.05], dtype='float32')
for j in range(ndist):
    cl._dist_idx = j
    cl._set_mask_chunk(cp.asarray(cl.mask_1d[j][:ntheta]))  # BH hands it a GPU chunk
    x0 = cp.asarray(rc((ntheta, nz, n))) + 0.5
    rf = cp.asarray(np.abs(r2.standard_normal((ntheta, nz, n))).astype('float32'))
    dx0 = x0 * rf * np.float32(0.3)
    dg = cp.asarray((np.abs(r2.standard_normal((ntheta, nz, n))) + 0.1).astype('float32'))
    e1, e2, e3 = (np.zeros(len(L)) for _ in range(3))
    f_w = float(cl.F0(x0, dg))
    for k, l in enumerate(L):
        dx = np.float32(l) * dx0
        a_ = float(cl.F0(x0 + dx, dg))
        df = float(cl.dF0(x0, dx, dg))
        d2f = float(cl.d2F_dF0(x0, dx, dx, None, dg))
        e1[k], e2[k] = abs(f_w - a_), abs(f_w + df - a_)
        e3[k] = abs(f_w + df + 0.5 * d2f - a_)
    slope = lambda e: float(np.polyfit(np.log(L), np.log(e), 1)[0])
    s1, s2, s3 = slope(e1), slope(e2), slope(e3)
    check(f"dist {j}: F0 Taylor slopes 1/2/3 with PSF",
          s1 > 0.85 and s2 > 1.85 and s3 > 2.7,
          f"O(h) {s1:.2f}  O(h^2) {s2:.2f}  O(h^3) {s3:.2f}")
free(cl)

# ---------------------------------------------------------------------------
print("3. forward-model the data (blurred), then check_approximation")
# ---------------------------------------------------------------------------
gen = Rec(make_args(PSF_SIGMA))
gen.vars['obj'][:] = OBJ
gen.vars['prb'][:] = PRB
gen.vars['pos'][:] = POS
gen.gen_data(gen.vars, gen.data)
gen.cl_prb_term.gen_sqrt_ref(gen.vars['prb'], gen.ref)
DATA_BLUR = np.array(gen.data)
free(gen)

gen = Rec(make_args(0.0))
gen.vars['obj'][:] = OBJ
gen.vars['prb'][:] = PRB
gen.vars['pos'][:] = POS
gen.gen_data(gen.vars, gen.data)
DATA_SHARP = np.array(gen.data)
free(gen)

# the blur must actually have reached the generated data, or every comparison
# below is between two identical datasets
rel = np.abs(DATA_BLUR - DATA_SHARP).max() / np.abs(DATA_SHARP).max()
check("gen_data carries the blur", rel > 1e-3, f"max|d|/max|d_sharp|={rel:.3e}")


def run(psf_sigma, data, niter=8, capture=False):
    a = make_args(psf_sigma, niter=niter)
    if capture:
        a.checkpoint_step, a.check_approx = 4, True
    cl = Rec(a)
    cl.data[:] = data
    cl.vars['prb'][:] = PRB
    cl.vars['pos'][:] = POS
    cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)
    cl.vars['obj'][:] = 0
    # start from the TRUE probe.  With prb = 1 the object and the probe share a
    # global complex factor the solver is free to split either way, and the run
    # spends its iterations on probe retrieval instead of on the object -- which
    # is the thing the blur is supposed to change.
    cl.vars['prb'][:] = PRB
    cl.BH()
    return cl


REC = []
_orig = rec_mpi.mshow_approx
rec_mpi.mshow_approx = lambda t, er, ea, show=False: REC.append((np.array(t), np.array(er), np.array(ea)))
try:
    cl = run(PSF_SIGMA, DATA_BLUR, niter=13, capture=True)
finally:
    rec_mpi.mshow_approx = _orig

check("check_approximation ran", len(REC) >= 3, f"{len(REC)} curve(s)")


def model_gap(t, er, ea):
    """max|real - model| over [0, alpha], as a fraction of the model's depth.

    Two things this must NOT normalise by.  Not f0: both curves sit on a large
    constant and any relative error against it passes regardless.  Not the
    endpoint span ea[-1] - ea[0]: check_approximation samples [0, 2*alpha] and
    the model is a parabola with its minimum at exactly t = alpha, so the two
    endpoints are equal by construction and that span is identically zero.
    The depth f(0) - min f is the scale the line search actually cares about.

    Only [0, alpha] is scored -- the half the line search steps into.  Beyond
    alpha the quadratic is an extrapolation nothing relies on.
    """
    half = len(t) // 2 + 1                       # t = 0 .. alpha
    depth = ea[0] - ea.min()
    return float(np.abs(er[:half] - ea[:half]).max() / max(depth, 1e-30))


# Iteration 0 is skipped: the first step from obj = 0 is huge, and a quadratic
# model of a genuinely global move is a poor fit there for reasons that have
# nothing to do with whether the Hessian is right.
worst = max(model_gap(*c) for c in REC[1:])
check("real functional tracks the quadratic model along the descent direction",
      worst < 0.15, f"max|real-model| / model depth = {worst:.4f}")

# Discrimination: the check above is only worth having if a wrong curvature
# fails it.  ea is exactly f0 - top*t + 0.5*bottom*t^2, so refit it, double
# bottom (what a Hessian missing a term would look like) and re-score.
bad = []
for t, er, ea in REC[1:]:
    c2, c1, c0 = np.polyfit(t, ea, 2)
    bad.append(model_gap(t, er, c0 + c1 * t + (2.0 * c2) * t * t))
check("... and would NOT track a model with twice the curvature",
      max(bad) > 0.15, f"gap with 2x bottom = {max(bad):.4f}")
free(cl)

# ---------------------------------------------------------------------------
print("4. the generator and the model agree exactly, and only, at the matching sigma")
# ---------------------------------------------------------------------------
# The iteration-free version of the whole claim, and the sharpest wiring test
# in this file: evaluate each model's misfit at the TRUE obj/prb/pos.  The model
# that generated the data must score zero and the other must not.  A sigma that
# silently failed to reach one of the two paths shows up here as a zero in the
# wrong cell.
#
# F0 itself is bit-zero at the truth: gen_data writes the intensity K|y|^2 and
# F0 reads the same quantity back, F0 = 1/N sum W(K|x|^2 - d)^2, so the two are
# the same float32 numbers.  (This briefly was NOT exact, when data was stored
# as an amplitude and the intensity misfit squared it back -- sqrt-then-square
# costs an ulp.  Dropping the sqrt from reader.py and gen_data restored it.)
#
# cl.min() is F0 + F1 though, and F1 is only ~0, not bit-0: PrbfitTerm keeps
# `ref` as an AMPLITUDE and squares it at use, so gen_sqrt_ref's sqrt(K|D.prb|^2)
# and the term's own K|D.prb|^2 differ by the ulp of that round trip.  Hence the
# ~1e-16 bound rather than ==0.  Discriminating power is unaffected: the wrong
# cell reads ~6e-5, eleven orders up.
def set_proj(cl):
    """proj = R(obj / norm_const), the normalisation gen_data uses."""
    cl._ensure_tp(cl.vars)
    cl.eff_demag[:] = cl._eff_demag_from_tp(cl.vars['tp'])
    cl.vars['obj'] /= cl.norm_const
    cl.fwd_tomo(cl.vars['obj'], out=cl.proj_tmp)
    cl.redist(cl.proj_tmp, cl.vars['proj'])
    cl.vars['obj'] *= cl.norm_const


def misfit(psf_sigma, data, sol):
    """That blur model's data misfit at a given (obj, prb, pos)."""
    cl = Rec(make_args(psf_sigma))
    cl.data[:] = data
    cl.vars['obj'][:], cl.vars['prb'][:], cl.vars['pos'][:] = sol
    cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)
    set_proj(cl)
    v = float(cl.min(cl.vars['prb'], cl.vars['obj'], cl.vars['pos'],
                     cl.vars['proj'], cl.vars['tp']))
    free(cl)
    return v


TRUTH = (OBJ, PRB, POS)
m_bb = misfit(PSF_SIGMA, DATA_BLUR, TRUTH)
m_0b = misfit(0.0, DATA_BLUR, TRUTH)
m_bs = misfit(PSF_SIGMA, DATA_SHARP, TRUTH)
m_0s = misfit(0.0, DATA_SHARP, TRUTH)
check("matched model: misfit at the truth is 0 to roundoff on the data it generated",
      m_bb < 1e-15, f"{m_bb:.6e}")
check("sigma=0 model: misfit at the truth is 0 to roundoff on sharp data",
      m_0s < 1e-15, f"{m_0s:.6e}")
check("and each model is wrong on the other dataset",
      m_0b > 1e-7 and m_bs > 1e-7,
      f"sigma=0 on blurred {m_0b:.6e}, matched on sharp {m_bs:.6e}")

# ---------------------------------------------------------------------------
print("5. recovery: modelling the blur beats ignoring it")
# ---------------------------------------------------------------------------
# How NOT to score this, both learned by measuring:
#
#  * NOT each run's own final err.  The two solvers minimise two DIFFERENT
#    functionals, so their values are not comparable, and in fact the sigma=0
#    solver reaches a lower value of its own functional on blurred data (2.3e-07
#    vs 4.2e-07 at 64 iterations) -- the blur damps exactly the high-frequency
#    directions the solver needs, so the matched model simply converges slower.
#  * NOT correlation of the recovered object with the truth.  Measured, it
#    favours sigma=0 on BOTH datasets, i.e. it tracks how far a run got rather
#    than whether its model was right.
#
# The well-posed comparison is one common referee: score both candidates with
# the functional that actually generated the data.  Then the 2x2 is symmetric
# and each diagonal cell must win by a wide margin.
NIT = 24
sols, errs = {}, {}
for dtag, data, rs in (("blurred", DATA_BLUR, PSF_SIGMA),
                       ("sharp  ", DATA_SHARP, 0.0)):
    for lbl, s_ in (("matched", PSF_SIGMA), ("sigma=0", 0.0)):
        cl = run(s_, data, niter=NIT)
        e = np.asarray(cl.table['err'], dtype='float64')
        sol = (np.array(cl.vars['obj']), np.array(cl.vars['prb']),
               np.array(cl.vars['pos']))
        free(cl)
        errs[dtag, lbl] = e
        # the referee is rs -- the generating model, the same one for both
        sols[dtag, lbl] = misfit(rs, data, sol)
    check(f"{dtag} data: both solvers converge", 
          errs[dtag, 'matched'][-1] < errs[dtag, 'matched'][0]
          and errs[dtag, 'sigma=0'][-1] < errs[dtag, 'sigma=0'][0],
          f"matched {errs[dtag, 'matched'][0]:.3e} -> {errs[dtag, 'matched'][-1]:.3e},  "
          f"sigma=0 {errs[dtag, 'sigma=0'][0]:.3e} -> {errs[dtag, 'sigma=0'][-1]:.3e}")

r_bm, r_b0 = sols["blurred", "matched"], sols["blurred", "sigma=0"]
r_sm, r_s0 = sols["sharp  ", "matched"], sols["sharp  ", "sigma=0"]
check("on blurred data, modelling the blur wins under the true functional",
      r_bm < r_b0 / 5.0,
      f"{r_bm:.4e} (matched) vs {r_b0:.4e} (sigma=0)  -- {r_b0 / r_bm:.1f}x")
# The control. Without it the line above could be measuring some unrelated
# advantage of blurring rather than of blurring by the RIGHT amount.
check("control: on sharp data the sigma=0 solver wins instead, by as much",
      r_s0 < r_sm / 5.0,
      f"{r_s0:.4e} (sigma=0) vs {r_sm:.4e} (matched)  -- {r_sm / r_s0:.1f}x")

print()
if FAILED:
    print(f"{len(FAILED)} check(s) FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("all checks passed")
