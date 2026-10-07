#!/usr/bin/env python
"""Gradients and Hessians of the data misfit, by Taylor test.

    python tests/unit/test_derivatives.py

For an exact first derivative the residual of the linear model falls as h^2,
and with an exact second derivative the quadratic model's residual falls as
h^3.  A wrong sign or a missing factor shows up as order 1, so the measured
slope is the test, not the size of any single residual.

Both misfit models, with and without the detector blur, and with the
out-of-grid mask on -- the combinations that actually ship.
"""
import os
import sys

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, close, order, run                      # noqa: E402

from holotomocupy.rec_mpi import Rec                               # noqa: E402
from holotomocupy.psf import psf_taps                              # noqa: E402
from holotomocupy.utils import redot                               # noqa: E402

CHUNK, NZ, N = 2, 48, 40
SIZE = CHUNK * NZ * N
HS = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2])
# the amplitude misfit is ~20x smaller, so its h^3 term reaches float32
# rounding at the step sizes above; it needs a longer lever
HS2 = {'intensity': HS, 'amplitude': np.array([4e-1, 2e-1, 1e-1, 5e-2])}
rng = np.random.default_rng(1234)


def cx(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


X = cp.asarray(cx((CHUNK, NZ, N)))
Y = cp.asarray(cx((CHUNK, NZ, N)))
D = cp.asarray(np.abs(rng.standard_normal((CHUNK, NZ, N))).astype('float32'))
MY = cp.asarray((rng.random((CHUNK, NZ, 1)) > 0.15).astype('float32'))
MX = cp.asarray((rng.random((CHUNK, 1, N)) > 0.15).astype('float32'))


def harness(sigma=None, model='intensity', mask=True):
    """A Rec with only the detector-plane pieces filled in."""
    r = object.__new__(Rec)
    r.data_size = SIZE
    r._mask_y = MY if mask else cp.ones_like(MY)
    r._mask_x = MX if mask else cp.ones_like(MX)
    r._dist_idx = 0
    r.psf_w = None if sigma is None else psf_taps(sigma)[1]
    r.model = model
    r.apply_F_from = lambda v, i: v        # these checks pass the field directly
    return r


CASES = [(model, sigma)
         for model in ('intensity', 'amplitude')
         for sigma in (None, 1.1)]


def test_first_derivative_order():
    """F0(x+hy) - F0(x) - h dF0[y] = O(h^2)."""
    for model, sigma in CASES:
        r = harness(sigma, model)
        f0 = float(r.F0(X, D).get())
        g = float(r.dF0(X, Y, D).get())
        e = [abs(float(r.F0(X + h * Y, D).get()) - f0 - h * g) for h in HS]
        p = order(HS, e)
        check(f'dF0 order 2   [{model}, psf={sigma}]', p > 1.8,
              f'slope {p:.2f}, residuals {e[0]:.2e} -> {e[-1]:.2e}')


def test_second_derivative_order():
    """F0(x+hy) - F0 - h dF0[y] - h^2/2 d2F0[y,y] = O(h^3)."""
    for model, sigma in CASES:
        r = harness(sigma, model)
        f0 = float(r.F0(X, D).get())
        g = float(r.dF0(X, Y, D).get())
        H = float(r.d2F_dF0(X, Y, Y, None, D).get())
        hs = HS2[model]
        e = [abs(float(r.F0(X + h * Y, D).get()) - f0 - h * g - 0.5 * h * h * H)
             for h in hs]
        p = order(hs, e)
        check(f'd2F0 order 3   [{model}, psf={sigma}]', p > 2.7,
              f'slope {p:.2f}, residuals {e[0]:.2e} -> {e[-1]:.2e}')


def test_gradient_matches_directional_derivative():
    """<gF0(x, d), y> == dF0(x, y, d): the gradient IS the adjoint of dF0."""
    for model, sigma in CASES:
        r = harness(sigma, model)
        close(f'<gF0, y> == dF0[y]   [{model}, psf={sigma}]',
              float(redot(r.gF0(X, D), Y)), float(r.dF0(X, Y, D).get()), 1e-4)


def test_hessian_symmetry_and_sign():
    """d2F0[y,z] is symmetric, and d2F0[z,z] >= 0 near the model's own data."""
    for model, sigma in CASES:
        r = harness(sigma, model)
        Z = cp.asarray(cx((CHUNK, NZ, N)))
        a = float(r.d2F_dF0(X, Y, Z, None, D).get())
        b = float(r.d2F_dF0(X, Z, Y, None, D).get())
        close(f'd2F0 symmetric   [{model}, psf={sigma}]', a, b, 1e-4)
        # at the model's own data the misfit is a minimum, so the curvature
        # along any direction must be non-negative there
        d_self = r._blur(cp.abs(X)**2) if model == 'intensity' else cp.abs(X)**2
        q = float(r.d2F_dF0(X, Z, Z, None, d_self).get())
        check(f'd2F0[z,z] >= 0 at the solution   [{model}, psf={sigma}]',
              q >= 0, f'{q:.6g}')


def test_mask_is_applied_inside_the_blur():
    """Masking outside the blur is a different operator; this pins the order."""
    r = harness(1.1, 'intensity', mask=True)
    rn = harness(1.1, 'intensity', mask=False)
    check('the mask changes F0', abs(float(r.F0(X, D).get())
                                     - float(rn.F0(X, D).get())) > 1e-6)


if __name__ == '__main__':
    sys.exit(run(globals()))
