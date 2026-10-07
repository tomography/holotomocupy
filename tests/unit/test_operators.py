#!/usr/bin/env python
"""Adjointness of every linear operator in the forward model.

    python tests/unit/test_operators.py

The dot-product test: for A and its claimed adjoint A*,

    <A u, v> == <u, A* v>

to float32 rounding, for random u and v.  It is cheap and it catches the
mistakes that matter -- a missing conjugate, a transposed index, a wrong
normalisation -- none of which show up in a forward-only check.

Inner products: `redot(a, b) = Re sum conj(a) b` for complex fields, plain
`sum a*b` for real ones.  The tolerance is 1e-5 relative, which is what
float32 FFTs and a NUFFT gather give on these sizes.
"""
import os
import sys

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, close, run                             # noqa: E402

from holotomocupy.utils import redot                              # noqa: E402

RTOL = 1e-5
rng = np.random.default_rng(3)


def dot_close(name, lhs, rhs, a, b, rtol=RTOL):
    """Adjointness check scaled by ||a||*||b||, not by the result.

    <A u, v> is a near-cancelling sum over random fields, so it can land
    close to zero by accident; a relative test against it then fails for a
    correct operator.  The meaningful scale is the size of the terms.
    """
    scale = float(cp.linalg.norm(a.ravel()) * cp.linalg.norm(b.ravel()))
    err = abs(lhs - rhs)
    check(name, err <= rtol * scale,
          f'{lhs:.8g} vs {rhs:.8g}   |diff| {err:.2e} of scale {scale:.2e}')


def cx(*shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


def re(*shape):
    return rng.standard_normal(shape).astype('float32')


def test_tomo_R_adjoint():
    """<R u, d> == <u, RT d>, at nd = n and at the oversampled nd = 2n."""
    from holotomocupy.tomo import Tomo
    n, nz, ntheta = 48, 4, 31
    theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
    for nd in (n, 2 * n):
        cl = Tomo(n, nz, theta, -1.0, nd=nd)
        u = cp.asarray(cx(nz, n, n))
        d = cp.asarray(cx(ntheta, nz, nd))
        Ru, RTd = cl.R(u), cl.RT(d)
        dot_close(f'Tomo R/RT adjoint (nd={nd})',
                  redot(Ru, d), redot(u, RTd), Ru, d)


def test_shift_adjoint():
    """<S c, y> == <c, Sadj y> for the B-spline shift, at unit magnification."""
    from holotomocupy.shift import Shift
    n, npsi, nz, nzpsi, nt = 32, 48, 32, 48, 5
    cl = Shift(n, npsi, nz, nzpsi)
    psi = cp.asarray(cx(nt, nzpsi, npsi))
    y = cp.asarray(cx(nt, nz, n))
    r = cp.asarray((4 * (rng.random((nt, 2)) - 0.5)).astype('float32'))
    m = cp.ones((nt, 2), dtype='float32')
    c = cl.coeff(psi)
    dot_close('Shift S/Sadj adjoint', redot(cl.S(c, r, m), y),
              redot(c, cl.Sadj(y, r, m)), c, y)


def test_shift_fft_matches_shift():
    """ShiftFFT is a drop-in for Shift: same curlyS, and self-adjoint pairing."""
    from holotomocupy.shift import Shift
    from holotomocupy.shift_fft import ShiftFFT
    n, npsi, nz, nzpsi, nt = 32, 48, 32, 48, 5
    a, b = Shift(n, npsi, nz, nzpsi), ShiftFFT(n, npsi, nz, nzpsi)
    # a band-limited field, so the two interpolants must agree
    psi = cp.zeros((nt, nzpsi, npsi), dtype='complex64')
    f = cp.asarray(cx(nt, nzpsi, npsi))
    k = cp.fft.fftfreq(npsi)[None, None, :]**2 + cp.fft.fftfreq(nzpsi)[None, :, None]**2
    psi[:] = cp.fft.ifft2(cp.fft.fft2(f) * (k < 0.02))
    r = cp.asarray((2 * (rng.random((nt, 2)) - 0.5)).astype('float32'))
    m = cp.ones((nt, 2), dtype='float32')
    ga, gb = a.curlyS(psi, r, m), b.curlyS(psi, r, m)
    err = float(cp.abs(ga - gb).max() / cp.abs(ga).max())
    check('ShiftFFT.curlyS == Shift.curlyS on a band-limited field',
          err < 2e-2, f'max rel diff {err:.2e}')

    y = cp.asarray(cx(nt, nz, n))
    dot_close('ShiftFFT S/Sadj adjoint', redot(b.S(psi, r, m), y),
              redot(psi, b.Sadj(y, r, m)), psi, y)


def test_shift_fft_magnified():
    """The chirp-z path (m != 1): adjointness, and Sback against Shift."""
    from holotomocupy.shift import Shift
    from holotomocupy.shift_fft import ShiftFFT
    n, npsi, nz, nzpsi, nt = 48, 72, 48, 72, 4
    a, b = Shift(n, npsi, nz, nzpsi), ShiftFFT(n, npsi, nz, nzpsi)
    r = cp.asarray((2 * (rng.random((nt, 2)) - 0.5)).astype('float32'))
    for mv in (0.8, 0.6366):
        m = cp.full((nt, 2), mv, dtype='float32')
        c = cp.asarray(cx(nt, nzpsi, npsi))
        y = cp.asarray(cx(nt, nz, n))
        dot_close(f'ShiftFFT S/Sadj adjoint at m={mv}', redot(b.S(c, r, m), y),
                  redot(c, b.Sadj(y, r, m)), c, y)

        # Sback resamples the other way; compare with Shift only where the
        # back-map lands inside the small grid -- Shift zero-fills outside,
        # ShiftFFT wraps.
        f = cp.asarray(cx(nt, nz, n))
        k = (cp.fft.fftfreq(n)[None, None, :]**2
             + cp.fft.fftfreq(nz)[None, :, None]**2)
        psi = cp.fft.ifft2(cp.fft.fft2(f) * (k < 0.01)).astype('complex64')
        ga, gb = a.curlySback(psi, r, m), b.curlySback(psi, r, m)
        h = int(mv * (n / 2 - 4))
        sl = (slice(None), slice(nzpsi // 2 - h, nzpsi // 2 + h),
              slice(npsi // 2 - h, npsi // 2 + h))
        err = float(cp.abs(ga[sl] - gb[sl]).max() / cp.abs(ga[sl]).max())
        check(f'ShiftFFT.curlySback == Shift.curlySback at m={mv}', err < 0.05,
              f'max rel diff {err:.2e} inside the valid region')


def test_chirpz_against_cubic_on_a_rectangle():
    """chirp-z vs B-spline on the shift+scale test from holotomocupy_mpi.

    A band-limited rectangle resampled from a 3N/2 grid onto an N grid at
    several magnifications.  Both operators interpolate at the same sample
    positions, so away from the edges they must agree; a sharp rectangle
    would ring in the FFT path and measure Gibbs instead of the operator.
    """
    import scipy.ndimage as snd
    from holotomocupy.shift import Shift
    from holotomocupy.shift_fft import ShiftFFT

    n = 64
    npsi = 3 * n // 2
    h, c = 18, npsi // 2
    img = np.zeros((npsi, npsi), dtype='float32')
    for ys, xs, v in ((slice(c - h, c), slice(c - h, c), 0.25),
                      (slice(c - h, c), slice(c, c + h), 0.50),
                      (slice(c, c + h), slice(c - h, c), 0.75),
                      (slice(c, c + h), slice(c, c + h), 1.00)):
        img[ys, xs] = v
    img = snd.gaussian_filter(img, sigma=1.5, mode='constant', cval=0.0)
    a_gpu = cp.asarray((img + 0j).astype('complex64'))[cp.newaxis]

    cub, chz = Shift(n, npsi, n, npsi), ShiftFFT(n, npsi, n, npsi)
    for mv, (ry, rx) in ((0.85, (2.5, -1.5)), (1.00, (4.0, 3.0)),
                         (1.15, (-3.0, 2.0)), (1.30, (6.0, -4.0))):
        r = cp.asarray([[ry, rx]], dtype='float32')
        m = cp.full((1, 2), mv, dtype='float32')
        d = (cub.curlyS(a_gpu, r, m)[0].real - chz.curlyS(a_gpu, r, m)[0].real)
        inner = d[6:-6, 6:-6]
        err = float(cp.abs(inner).max())
        check(f'chirp-z == cubic at m={mv}, r=({ry:+.1f},{rx:+.1f})', err < 0.02,
              f'max|cubic-chirpz| {err:.3e} on the interior')


def test_propagation_adjoint():
    """<D psi, big> == <psi, DT big> for the Fresnel propagator."""
    from holotomocupy.propagation import Propagation
    n, nz, nt, nd = 32, 32, 3, 2
    cl = Propagation(n, nz, nt, nd, wavelength=7.25e-11, voxelsize=1.5e-8,
                     distance=np.array([1.0e-3, 2.0e-3]))
    for j in range(nd):
        psi = cp.asarray(cx(nt, nz, n))
        big = cp.asarray(cx(nt, nz, n))
        dot_close(f'Propagation D/DT adjoint (distance {j})',
                  redot(cl.D(psi, j), big), redot(psi, cl.DT(big, j)), psi, big)


def test_psf_self_adjoint():
    """The Gaussian blur is circulant and symmetric, so K == K^T exactly."""
    from holotomocupy.psf import psf_blur, psf_taps
    sig, w = psf_taps(1.3)
    check('psf_taps returns taps for sigma > 0', w is not None and sig == 1.3,
          f'{len(w)} taps')
    check('psf_taps returns None for sigma = 0', psf_taps(0.0)[1] is None)
    close('psf taps sum to 1', float(w.sum()), 1.0, 1e-6)
    t = cp.arange(len(w)) - len(w) // 2
    close('psf taps have the asked-for second moment',
          float(cp.sum(w * t.astype('float32')**2))**0.5, sig, 2e-3)
    x = cp.asarray(re(3, 24, 20))
    y = cp.asarray(re(3, 24, 20))
    check('psf_blur actually blurs', float(cp.abs(psf_blur(x, w) - x).max()) > 0.1)
    close('psf_blur self-adjoint', float(cp.sum(psf_blur(x, w) * y)),
          float(cp.sum(x * psf_blur(y, w))), RTOL)
    close('psf_blur preserves the mean', float(psf_blur(x, w).mean()),
          float(x.mean()), 1e-4)


def test_fbp_against_rec_tomo():
    """FBP of R u is close to u, which is the only end-to-end check of the pair."""
    from holotomocupy.tomo import Tomo
    n, nz, ntheta = 64, 2, 128
    theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
    cl = Tomo(n, nz, theta, -1.0)
    # a SMOOTH disc: a binary one rings, and this test is about the transform
    # pair, not about Gibbs
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
    r = np.sqrt(yy**2 + xx**2)
    disc = (0.5 * (1 - np.tanh((r - n / 5) / 1.5))).astype('complex64')
    u = cp.asarray(np.broadcast_to(disc, (nz, n, n)).copy())
    rec = cl.fbp(cl.R(u), 'ramp').real
    m = cp.asarray(np.broadcast_to(r < n / 2.2, (nz, n, n)).copy())
    # the ramp filter kills the zero frequency, so FBP recovers the object up
    # to a constant: compare the zero-mean fields, not the raw ones
    a, b = rec[m] - rec[m].mean(), u.real[m] - u.real[m].mean()
    err = float(cp.linalg.norm(a - b) / cp.linalg.norm(b))
    corr = float((a * b).sum() / cp.sqrt((a * a).sum() * (b * b).sum()))
    check('FBP(R u) ~ u in shape', corr > 0.999, f'correlation {corr:.5f}')
    check('FBP(R u) ~ u in amplitude', err < 0.12, f'zero-mean rel err {err:.3f}')


if __name__ == '__main__':
    sys.exit(run(globals()))
