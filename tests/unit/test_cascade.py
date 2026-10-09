#!/usr/bin/env python
"""The whole forward cascade F0(F1(F2(F3(F4(x))))) and its two differentials.

    python tests/unit/test_cascade.py

`test_derivatives.py` covers F0 alone.  This covers the levels above it, in
the composition they are actually used in:

    F4  tp -> the per-angle demagnification
    F3  the B-spline/FFT shift of the projection onto the object plane
    F2  exp(i psi), the fused transmission
    F1  multiply by the probe, then Fresnel propagate
    F0  the data misfit

The three chains below are copied from `gradients_cascade` and
`hessian_cascade`, so what is tested is the loop that runs in production, not
a re-derivation of it:

    value     F0(apply_F_from(x, 1), d)
    gradient  y = d;  y = gF[id](x, y)  for id = 0..4
    forward   fx, y = dF[id](x, y)      for id = 4..1,  then dF0
    hessian   w = d2F_dF[id](x, y, z, w) alongside it, then d2F_dF0

and they are checked against each other and against finite differences:

  * <grad, y> == dF[chain][y]        -- the adjoint of every gF level at once
  * value(x+hy) - value - h dF       = O(h^2)
  * ... - h^2/2 d2F                  = O(h^3)

A missing conjugate or a dropped term at any level breaks one of the three.
"""
import os
import sys

import numpy as np
import cupy as cp
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, close, order, run                      # noqa: E402

FLOOR = 2e-7          # float32 rounding of an objective of order 1

from mpi4py import MPI                                            # noqa: E402
from holotomocupy.rec_mpi import Rec                               # noqa: E402
from holotomocupy.utils import redot                               # noqa: E402

N, NZ, NTHETA, NDIST = 32, 32, 6, 2
NOBJ = 3 * N // 2
HS = np.array([8e-2, 4e-2, 2e-2, 1e-2])
rng = np.random.default_rng(5)


def _rec(model='intensity', psf_sigma=0.0):
    theta = np.linspace(0, np.pi, NTHETA, endpoint=False).astype('float32')
    return Rec(SimpleNamespace(
        energy=17.1, detector_pixelsize=1.4760147601476e-6 * 16,
        focustodetectordistance=1.217,
        z1=np.array([5.110, 5.464])[:NDIST] * 1e-3, theta=theta,
        ndist=NDIST, ntheta=NTHETA, nz=NZ, n=N, nzobj=NOBJ, nobj=NOBJ,
        mask=0.9, lam_prbfit=0, lam_laplacian=0, rho=[1, 1, 1, 1],
        niter=1, nchunk=NTHETA, checkpoint_step=-1, error_step=-1,
        start_iter=0, model=model, psf_sigma=psf_sigma, comm=MPI.COMM_WORLD))


def _cx(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


class Cascade:
    """One chunk, one distance: the production loops, callable directly."""

    def __init__(self, cl, j=0):
        self.cl, self.j = cl, j
        # a state the model can actually be at: a random object and probe, a
        # small displacement, and a non-zero shrink so F4 is not the identity
        self.proj = cp.asarray(_cx((NTHETA, NOBJ, NOBJ)) * 0.1)
        self.prb = cp.asarray(_cx((NZ, N)) + 1.0)
        self.pos = cp.asarray((2 * (rng.random((NTHETA, 2)) - 0.5)).astype('float32'))
        self.tp = cp.asarray((1e-3 * rng.standard_normal((2, 2))).astype('float32'))
        self.d = cp.asarray(np.abs(rng.standard_normal((NTHETA, NZ, N))).astype('float32') + 0.5)
        self.t = cl.t_local        # [chunk, 1], as the gpu_batch loops pass it

    def x(self):
        return [self.prb, self.proj, self.pos, self.tp]

    def direction(self, scale=1.0):
        return [cp.asarray(_cx((NZ, N))) * scale,
                cp.asarray(_cx((NTHETA, NOBJ, NOBJ))) * 0.1 * scale,
                cp.asarray((rng.random((NTHETA, 2)) - 0.5).astype('float32')) * scale,
                cp.asarray(rng.standard_normal((2, 2)).astype('float32')) * 1e-2 * scale]

    def _arm(self):
        cl = self.cl
        cl.cl_shift.coeff_cache_reset()
        cl.apply_F_cache_reset()
        cl._t_chunk = self.t
        cl._dist_idx = self.j
        # mask_1d is pinned HOST memory; @gpu_batch uploads it in production
        cl._set_mask_chunk(cp.asarray(cl.mask_1d[self.j]))

    def value(self, x):
        self._arm()
        return float(self.cl.F0(self.cl.apply_F_from(x, 1), self.d).get())

    def dvalue(self, x, y):
        """The chained first directional derivative, as gradients_cascade runs it."""
        cl = self._arm() or self.cl
        for i in range(1, len(cl.F))[::-1]:
            fx, y = cl.dF[i](x, y)
            x = fx
        return float(cl.dF0(x, y, self.d).get())

    def d2value(self, x, y):
        """The chained second directional derivative, as hessian_cascade runs it."""
        cl = self._arm() or self.cl
        z, w = y, [None, None, None, None]
        for i in range(1, len(cl.F))[::-1]:
            w = cl.d2F_dF[i](x, y, z, w)
            fx, y = cl.dF[i](x, y)
            z = y
            x = fx
        return float(cl.d2F_dF[0](x, y, z, w, self.d).get())

    def gradient(self, x):
        """gF0..gF4, as gradients_cascade runs it.  Returns [prb, proj, pos, tp]."""
        cl = self._arm() or self.cl
        y = self.d
        for i in range(len(cl.gF)):
            y = cl.gF[i](x, y)
        # gF3 returns the un-coeff'd Deltapsi; gradients_cascade applies the
        # B-spline prefilter once, after summing the distances
        return [y[0], cl.cl_shift.coeff(y[1]), y[2], y[3]]

    @staticmethod
    def step(x, h, y):
        """x + h*y, keeping every dtype: a python float promotes complex64."""
        return [(a + np.float32(h) * b).astype(a.dtype) for a, b in zip(x, y)]

    @staticmethod
    def dot(g, y):
        """<g, y> with the right inner product per slot: complex, complex, real, real."""
        return (float(redot(g[0], y[0])) + float(redot(g[1], y[1]))
                + float(cp.sum(g[2] * y[2])) + float(cp.sum(g[3] * y[3])))


def _axis(c, k):
    """A direction that moves only variable k, so a broken level is named."""
    y = [cp.zeros_like(a) for a in c.x()]
    y[k] = c.direction()[k]
    return y


SLOTS = ('prb', 'proj (obj)', 'pos', 'tp (shrink)')


def test_gradient_is_the_adjoint_of_the_cascade():
    """<gF chain, y> == dF chain [y], per variable and for all of them at once."""
    for model in ('intensity', 'amplitude'):
        cl = _rec(model)
        c = Cascade(cl)
        g = c.gradient(c.x())
        for k, name in enumerate(SLOTS):
            y = _axis(c, k)
            close(f'<grad, y> == dF[y] for {name}   [{model}]',
                  c.dot(g, y), c.dvalue(c.x(), y), 2e-3)
        y = c.direction()
        close(f'<grad, y> == dF[y], all variables   [{model}]',
              c.dot(g, y), c.dvalue(c.x(), y), 2e-3)


def test_first_derivative_order():
    """value(x+hy) - value - h dF = O(h^2), one variable at a time."""
    for model in ('intensity', 'amplitude'):
        cl = _rec(model)
        c = Cascade(cl)
        x = c.x()
        f0 = c.value(x)
        for k, name in enumerate(SLOTS):
            y = _axis(c, k)
            g = c.dvalue(x, y)
            e = [abs(c.value(c.step(x, h, y)) - f0 - h * g)
                 for h in HS]
            p = order(HS, e, FLOOR * max(abs(f0), 1.0))
            check(f'cascade dF order 2, {name}   [{model}]', p > 1.8,
                  f'slope {p:.2f}, {e[0]:.2e} -> {e[-1]:.2e}'
                  + ('  (at float32 rounding)' if p == float('inf') else ''))


def test_second_derivative_order():
    """... - h^2/2 d2F = O(h^3): the Hessian chain, one variable at a time."""
    for model in ('intensity', 'amplitude'):
        cl = _rec(model)
        c = Cascade(cl)
        x = c.x()
        f0 = c.value(x)
        for k, name in enumerate(SLOTS):
            y = _axis(c, k)
            g, H = c.dvalue(x, y), c.d2value(x, y)
            e = [abs(c.value(c.step(x, h, y)) - f0 - h * g - 0.5 * h * h * H)
                 for h in HS]
            p = order(HS, e, FLOOR * max(abs(f0), 1.0))
            check(f'cascade d2F order 3, {name}   [{model}]', p > 2.5,
                  f'slope {p:.2f}, {e[0]:.2e} -> {e[-1]:.2e}'
                  + ('  (at float32 rounding)' if p == float('inf') else ''))


def test_levels_are_all_exercised():
    """Every level must actually move the value, or the tests above are empty."""
    cl = _rec()
    c = Cascade(cl)
    x, f0 = c.x(), None
    f0 = c.value(x)
    for k, name in enumerate(SLOTS):
        y = _axis(c, k)
        f1 = c.value(c.step(x, 0.1, y))
        check(f'{name} changes the objective', abs(f1 - f0) > 1e-9 * max(abs(f0), 1),
              f'{f0:.6g} -> {f1:.6g}')


def test_nfp_hessian3_matches_three_sweeps():
    """RecNFP's fused hessian3 returns exactly what three `hessian` calls do."""
    from holotomocupy.rec_nfp_mpi import RecNFP
    n, npos = 64, 8
    nobj = n + 64
    cl = RecNFP(SimpleNamespace(
        energy=17.1, detector_pixelsize=1.4760147601476e-6 * 4,
        focustodetectordistance=1.217, z1=5.110e-3,
        ntheta=npos, nz=n, n=n, nzobj=nobj, nobj=nobj,
        rho=[1, 2, 0.1], niter=1, nchunk=4, checkpoint_step=-1, error_step=-1,
        start_iter=0, psf_sigma=0.0, shift_type='cubic',
        path_out=os.environ.get('HTC_SCRATCH', '/tmp') + '/htc_unit_nfp',
        comm=MPI.COMM_WORLD))
    cl.vars['proj'][:] = cp.asarray(_cx((nobj, nobj)) * 0.1)
    cl.vars['prb'][:] = cp.asarray(_cx((n, n)) + 1.0)
    cl.vars['pos'][:] = cp.asarray((2 * (rng.random((cl.local_ntheta, 2)) - 0.5)
                                    ).astype('float32'))
    cl.gen_data(cl.vars, cl.data)
    # move off the solution: at the truth the residual, and with it every
    # bilinear form, is identically zero and the comparison is vacuous
    cl.vars['proj'] *= 0.7
    cl.vars['prb'] += cp.asarray(_cx((n, n)) * 0.05)
    cl.precalc(cl.vars)
    cl.compute_gradient(cl.vars, cl.grads)
    # an independent direction: with e = -g the three forms coincide and the
    # comparison proves nothing
    for v in cl._var_names:
        a = cl.etas[v]
        z = rng.standard_normal(a.shape)
        if a.dtype.kind == 'c':
            z = z + 1j * rng.standard_normal(a.shape)
        a[:] = cp.asarray(z.astype(a.dtype)) * float(cp.abs(cl.grads[v]).mean())

    got = cl.hessian3(cl.vars, cl.grads, cl.etas)
    want = (cl.hessian(cl.vars, cl.grads, cl.grads),
            cl.hessian(cl.vars, cl.grads, cl.etas),
            cl.hessian(cl.vars, cl.etas, cl.etas))
    for nm, a, b in zip(('B(g,g)', 'B(g,e)', 'B(e,e)'), got, want):
        close(f'NFP hessian3 {nm} == three-sweep', float(a), float(b), 1e-5)
    check('the three forms are not degenerate',
          len({round(float(v), 12) for v in want}) == 3, f'{want}')


if __name__ == '__main__':
    sys.exit(run(globals()))
