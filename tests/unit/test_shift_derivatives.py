#!/usr/bin/env python
"""Derivatives, Hessians and adjoints of BOTH shift operators.

    python tests/unit/test_shift_derivatives.py

`test_cascade.py` reaches these through the whole F0..F4 chain, which proves
they compose correctly but not which one is wrong when something breaks.
This tests each variant on its own, for `Shift` and `ShiftFFT`, at m = 1 and
on the chirp-z path:

  dcurlySc / dcurlySmc     first derivative in (c, r) and (c, r, m)
  d2curlySc / d2curlySmc   second derivative, same two
  dcurlySadjc / ...mc      the adjoints of the first derivatives

Taylor order is the test for the derivatives -- an exact first derivative
leaves O(h^2), an exact second leaves O(h^3) -- and the dot-product identity
is the test for the adjoints.  The input is band-limited: `ShiftFFT` is the
sinc interpolant and a sharp edge would measure Gibbs instead of calculus.
"""
import os
import sys

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, order, run                             # noqa: E402

from holotomocupy.shift import Shift                              # noqa: E402
from holotomocupy.shift_fft import ShiftFFT                       # noqa: E402
from holotomocupy.utils import redot                              # noqa: E402

N, NPSI, NT = 48, 72, 4
HS = np.array([1.6e-1, 8e-2, 4e-2, 2e-2])
# Measured float32 floor of these fields: the h^3 term reaches it by h ~ 0.02
# (e.g. Shift at m=1.21 flattens at 3e-5 against a field of max 26.5, i.e.
# ~10x eps*scale).  order() drops points at or under it rather than fitting
# through the flat tail.
FLOOR = 50 * float(np.finfo('float32').eps)
rng = np.random.default_rng(11)


def band_limited(shape, cut=0.01):
    f = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')
    k = (cp.fft.fftfreq(shape[-1])[None, None, :]**2
         + cp.fft.fftfreq(shape[-2])[None, :, None]**2)
    return cp.ascontiguousarray(cp.fft.ifft2(cp.fft.fft2(cp.asarray(f)) * (k < cut)
                                             ).astype('complex64'))


def ops():
    return (('Shift', Shift(N, NPSI, N, NPSI)),
            ('ShiftFFT', ShiftFFT(N, NPSI, N, NPSI)))


def setup():
    c = band_limited((NT, NPSI, NPSI))
    c1 = band_limited((NT, NPSI, NPSI))
    c2 = band_limited((NT, NPSI, NPSI))
    r = cp.asarray((2 * (rng.random((NT, 2)) - 0.5)).astype('float32'))
    dr1 = cp.asarray((rng.random((NT, 2)) - 0.5).astype('float32'))
    dr2 = cp.asarray((rng.random((NT, 2)) - 0.5).astype('float32'))
    dm1 = cp.asarray((0.05 * rng.standard_normal((NT, 2))).astype('float32'))
    dm2 = cp.asarray((0.05 * rng.standard_normal((NT, 2))).astype('float32'))
    z = cp.zeros_like(c)
    return c, c1, c2, r, dr1, dr2, dm1, dm2, z


def taylor(name, f0, g, H, step, scale):
    """Residuals of the linear and quadratic models of step(h), as orders."""
    e1, e2 = [], []
    for h in HS:
        fh = step(h)
        e1.append(float(cp.abs(fh - f0 - np.float32(h) * g).max()))
        if H is not None:
            e2.append(float(cp.abs(fh - f0 - np.float32(h) * g
                                   - np.float32(0.5 * h * h) * H).max()))
    p1 = order(HS, e1, FLOOR * scale)
    check(f'{name}: first derivative, order 2', p1 > 1.8,
          f'slope {p1:.2f}, {e1[0]:.2e} -> {e1[-1]:.2e}'
          + ('  (at float32 rounding)' if p1 == float('inf') else ''))
    if H is not None:
        p2 = order(HS, e2, FLOOR * scale)
        check(f'{name}: second derivative, order 3', p2 > 2.5,
              f'slope {p2:.2f}, {e2[0]:.2e} -> {e2[-1]:.2e}'
              + ('  (at float32 rounding)' if p2 == float('inf') else ''))


def test_dcurlySc_and_d2curlySc():
    """Derivatives in (c, r) at fixed m, both operators, both paths."""
    c, c1, c2, r, dr1, dr2, _, _, z = setup()
    for lab, op in ops():
        for mv in (1.0, 1.21):
            m = cp.full((NT, 2), mv, dtype='float32')
            # curlySc, not curlyS: these variants take COEFFICIENTS, and
            # curlyS would apply Shift's B-spline prefilter on top.
            f0 = op.curlySc(c, r, m)
            scale = float(cp.abs(f0).max())
            g = op.dcurlySc(c, r, m, c1, dr1)
            # d2 with the SAME direction in both slots is the pure curvature
            H = op.d2curlySc(c, r, m, c1, dr1, c1, dr1)
            taylor(f'{lab} dcurlySc m={mv}', f0, g, H,
                   lambda h: op.curlySc(c + np.float32(h) * c1,
                                        r + np.float32(h) * dr1, m), scale)


def test_dcurlySmc_and_d2curlySmc():
    """Derivatives in (c, r, m) -- the magnification direction as well."""
    c, c1, c2, r, dr1, dr2, dm1, dm2, z = setup()
    for lab, op in ops():
        for mv in (1.0, 1.21):
            m = cp.full((NT, 2), mv, dtype='float32')
            f0 = op.curlySc(c, r, m)
            scale = float(cp.abs(f0).max())
            g = op.dcurlySmc(c, r, m, c1, dr1, dm1)
            H = op.d2curlySmc(c, r, m, c1, dr1, dm1, c1, dr1, dm1)
            taylor(f'{lab} dcurlySmc m={mv}', f0, g, H,
                   lambda h: op.curlySc(c + np.float32(h) * c1,
                                        r + np.float32(h) * dr1,
                                        m + np.float32(h) * dm1), scale)


def test_adjoints():
    """<d(...), g> == <c1, out1> + <dr, out2_r> (+ <dm, out2_m>)."""
    c, c1, c2, r, dr1, _, dm1, _, z = setup()
    for lab, op in ops():
        for mv in (1.0, 1.21):
            m = cp.full((NT, 2), mv, dtype='float32')
            g = cp.asarray((rng.standard_normal((NT, N, N))
                            + 1j * rng.standard_normal((NT, N, N))).astype('complex64'))

            lhs = float(redot(op.dcurlySc(c, r, m, c1, dr1), g))
            o1, o2 = op.dcurlySadjc(c, r, m, g)
            rhs = float(redot(c1, o1)) + float(cp.sum(dr1 * o2))
            sc = float(cp.linalg.norm(c1.ravel()) * cp.linalg.norm(g.ravel()))
            check(f'{lab} dcurlySadjc adjoint, m={mv}', abs(lhs - rhs) <= 1e-4 * sc,
                  f'{lhs:.6g} vs {rhs:.6g}  |diff| {abs(lhs - rhs):.2e} of {sc:.2e}')

            lhs = float(redot(op.dcurlySmc(c, r, m, c1, dr1, dm1), g))
            o1, o2r, o2m = op.dcurlySadjmc(c, r, m, g)
            rhs = (float(redot(c1, o1)) + float(cp.sum(dr1 * o2r))
                   + float(cp.sum(dm1 * o2m)))
            check(f'{lab} dcurlySadjmc adjoint, m={mv}', abs(lhs - rhs) <= 1e-4 * sc,
                  f'{lhs:.6g} vs {rhs:.6g}  |diff| {abs(lhs - rhs):.2e} of {sc:.2e}')


def test_fft_matches_cubic_on_derivatives():
    """The two operators' derivatives agree on a band-limited input."""
    c, c1, c2, r, dr1, dr2, dm1, dm2, z = setup()
    a, b = Shift(N, NPSI, N, NPSI), ShiftFFT(N, NPSI, N, NPSI)
    mg = 6
    # 'c' is B-spline COEFFICIENTS for Shift and the FIELD for ShiftFFT
    # (whose coeff is the identity), so Shift gets the prefiltered input.
    ca, c1a, c2a = a.coeff(c), a.coeff(c1), a.coeff(c2)
    for mv in (1.0, 1.21):
        m = cp.full((NT, 2), mv, dtype='float32')
        for nm, fa, fb in (
                ('dcurlySc',
                 lambda o: o.dcurlySc(ca, r, m, c1a, dr1),
                 lambda o: o.dcurlySc(c, r, m, c1, dr1)),
                ('d2curlySc',
                 lambda o: o.d2curlySc(ca, r, m, c1a, dr1, c2a, dr2),
                 lambda o: o.d2curlySc(c, r, m, c1, dr1, c2, dr2)),
                ('dcurlySmc',
                 lambda o: o.dcurlySmc(ca, r, m, c1a, dr1, dm1),
                 lambda o: o.dcurlySmc(c, r, m, c1, dr1, dm1))):
            u, v = fa(a), fb(b)
            d = cp.abs(u - v)[:, mg:-mg, mg:-mg].max()
            s = cp.abs(u)[:, mg:-mg, mg:-mg].max()
            err = float(d / s)
            check(f'{nm}: ShiftFFT == Shift at m={mv}', err < 0.05,
                  f'max rel diff {err:.2e} on the interior')


if __name__ == '__main__':
    sys.exit(run(globals()))
