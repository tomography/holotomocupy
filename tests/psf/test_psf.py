"""Checks on the Gaussian blur operator itself (_psf_* in holotomocupy.rec_mpi).

Run:
  PYTHONPATH=<repo>/src python tests/psf/test_psf.py

Check 2 is the important one: F0's gradient and Hessian apply K in both
directions, and they only get away with reusing the same call because the
periodic, symmetric-tap operator is exactly self-adjoint.  If that ever stopped
holding, the descent direction would be plausible-looking but wrong and nothing
else here would catch it.
"""
import sys
import numpy as np
import cupy as cp

from holotomocupy.rec_mpi import _psf_taps, _psf_blur

# _psf_taps takes the Gaussian sigma in binned detector pixels and does no
# conversion of its own, so these checks -- which are all stated in pixels --
# hand it the number directly.  (It used to take a FWHM in um and divide by a
# pixel size; that arithmetic now lives in the config comments, where it is
# visible, and section 4 below checks the ladder property that replaced it.)
FWHM_TO_SIGMA = 1.0 / 2.354820045


def taps(sigma_px):
    return _psf_taps(sigma_px)[1]

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        fails.append(name)


rng = np.random.default_rng(7)
SIGMAS = [0.3, 0.9, 1.7, 3.0]

# --- 1. kernel sanity ------------------------------------------------------
# The periodic boundary is what makes a constant field a fixed point; a
# zero-padded operator would darken an r-wide rim, which is exactly the artefact
# we do not want in the model intensity (the data there is ~1).
for s in SIGMAS:
    w = taps(s)
    c = cp.full((2, 40, 33), 1.7, dtype='float32')
    err = float(cp.abs(_psf_blur(c, w) - 1.7).max())
    check(f"constant is a fixed point (sigma={s})", err < 1e-5, f"max|diff|={err:.3e}")

for s in SIGMAS:
    w = taps(s)
    r = w.size // 2
    N = 8 * r + 21
    delta = cp.zeros((1, N, N), dtype='float32')
    delta[0, N // 2, N // 2] = 1.0
    out = cp.asnumpy(_psf_blur(delta, w))[0]
    # normalisation, and the second moment must reproduce the requested sigma
    tot = out.sum()
    ax = np.arange(N) - N // 2
    prof = out.sum(axis=0)
    m2 = (prof * ax**2).sum() / prof.sum()
    smeas = np.sqrt(m2)
    check(f"delta -> normalised Gaussian (sigma={s})",
          abs(tot - 1.0) < 1e-5 and abs(smeas - s) / s < 0.01,
          f"sum={tot:.6f} sigma_meas={smeas:.4f} vs {s}")

# separability: blurring a rank-1 field must stay rank-1 (row/col blur independent)
w = taps(1.7)
a = cp.asarray(rng.standard_normal((1, 32, 1)).astype('float32'))
b = cp.asarray(rng.standard_normal((1, 1, 28)).astype('float32'))
lhs = cp.asnumpy(_psf_blur(a * b, w))
rhs = cp.asnumpy(_psf_blur(a * cp.ones((1, 1, 28), 'float32'), w)
                 * _psf_blur(cp.ones((1, 32, 1), 'float32') * b, w))
rel = np.abs(lhs - rhs).max() / np.abs(lhs).max()
check("separable (rank-1 stays rank-1)", rel < 1e-5, f"rel={rel:.3e}")

# --- 2. self-adjointness ---------------------------------------------------
# Non-square nz != n on purpose: an axis mix-up survives a square test.
# Normalised by sum|terms|, not by the result: <Kx,y> for random x,y is a
# near-cancelling sum, so dividing by it measures the cancellation rather than
# the operator.  Against the term scale the bar can be much tighter (1e-7).
for s in SIGMAS:
    w = taps(s)
    x = cp.asarray(rng.standard_normal((3, 37, 29)).astype('float32'))
    y = cp.asarray(rng.standard_normal((3, 37, 29)).astype('float32'))
    Kx, Ky = _psf_blur(x, w), _psf_blur(y, w)
    lhs = float(cp.sum(Kx.astype(cp.float64) * y.astype(cp.float64)))
    rhs = float(cp.sum(x.astype(cp.float64) * Ky.astype(cp.float64)))
    scale = float(cp.sum(cp.abs(Kx.astype(cp.float64) * y.astype(cp.float64))))
    rel = abs(lhs - rhs) / scale
    check(f"self-adjoint <Kx,y> == <x,Ky> (sigma={s})", rel < 1e-7,
          f"{lhs:.8e} vs {rhs:.8e}  |d|/sum|terms|={rel:.2e}")

# mass conservation, i.e. K^T 1 == 1: the adjoint neither creates nor destroys
# weight, so the gradient it forms is not biased toward the frame edge
w = taps(2.0)
o = cp.ones((1, 24, 19), dtype='float32')
check("K conserves total mass",
      abs(float(_psf_blur(o, w).sum()) - o.size) / o.size < 1e-6,
      f"sum={float(_psf_blur(o, w).sum()):.4f} vs {o.size}")

# --- 3. sigma == 0 is the identity, not a width-0 convolution --------------
# No taps at all: the F0 fast path tests `psf_w[j] is not None` and skips the
# convolution entirely, so the no-PSF case stays bit-for-bit what it was.
check("sigma=0 -> no kernel (blur is skipped, not run at width 0)",
      _psf_taps(0.0)[1] is None and _psf_taps(-1.0)[1] is None
      and _psf_taps(1.5)[1] is not None)

# --- 4. sigma is passed through, and the config ladder is arithmetic -------
# _psf_taps no longer converts anything: the config states the sigma in binned
# detector pixels, so what the user writes is the width that is convolved.  The
# geometry that used to be in here is now done in the config comment, and what
# is checked below is that the values shipped in those configs are right --
# sigma = spot*(M-1) / 2.3548 / (pixel * 2**bin).
check("sigma is returned unchanged", _psf_taps(1.5053)[0] == 1.5053)
check("negative sigma is treated as off", _psf_taps(-1.0)[0] == 0.0)

PIX0 = 1.476015e-06                   # bin-0 detector pixel, both scans


def cfg_sigma(spot_nm, M, b):
    """What the config comment's arithmetic gives, in binned pixels."""
    return spot_nm * 1e-9 * (M - 1.0) * FWHM_TO_SIGMA / (PIX0 * 2 ** b)


# AtomiumS1 and AtomiumS1_HT, plane 1: M = 328.00 -> 5.232 um at the detector
a = [cfg_sigma(16.0, 328.00, b) for b in (0, 1, 2)]
check("AtomiumS1(_HT) 16 nm spot: the shipped per-bin sigmas",
      all(abs(x - y) < 5e-5 for x, y in zip(a, (1.5053, 0.7526, 0.3763))),
      " ".join(f"{x:.4f}" for x in a))
# ctxl_HT, plane 1: M = 196.80 -> 3.133 um at the detector
c = [cfg_sigma(16.0, 196.80, b) for b in (0, 1, 2)]
check("ctxl_HT 16 nm spot: the same arithmetic at M=196.80",
      all(abs(x - y) < 5e-5 for x, y in zip(c, (0.9013, 0.4507, 0.2253))),
      " ".join(f"{x:.4f}" for x in c))
check("the ladder halves: that is why each bin config carries its own number",
      abs(a[1] * 2 - a[0]) < 1e-12 and abs(a[2] * 4 - a[0]) < 1e-12)

# ... and the configs really do say so.  A number typed into a comment is not a
# test; reading the shipped files is.
import os
import re
CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "experimental")
# The sweep that used to live here is gone -- every shipped config runs
# psf_sigma=0 -- so what is left to check is exactly that: no config carries a
# blur by accident, and none carries the retired psf_fwhm key.
bad, seen = [], 0
for scan in sorted(os.listdir(CONF)):
    d = os.path.join(CONF, scan)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if not re.match(r"config_step[06].*\.conf$", fn):
            continue
        txt = open(os.path.join(d, fn)).read()
        v = re.search(r"^psf_sigma=([-\d.eE+]+)", txt, re.M)
        if v is None:
            continue                       # the key is optional, 0 by default
        seen += 1
        if float(v.group(1)) != 0.0:
            bad.append(f"{scan}/{fn}: {v.group(1)}")
check("every shipped config runs psf_sigma=0", not bad,
      f"{seen} config(s) carry the key" + ("" if not bad else "; " + "; ".join(bad)))


print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
