"""ShiftFFT against Shift, and against its own definition.

ShiftFFT is meant to be a drop-in for Shift inside RecNFP, so what has to hold
is not "the two agree to machine precision" -- they cannot, one interpolates
with a 4x4 cubic stencil and the other is exact for a band-limited input --
but:

  1. on a band-limited object both operators return the same thing (the cubic
     stencil's error is the only difference, and it vanishes as the object gets
     smoother);
  2. ShiftFFT's own adjoints really are adjoints, and its derivatives really
     are the derivatives of its own forward operator;
  3. d2curlySc follows the SAME slot-pairing convention as Shift.d2curlySc,
     i.e. the caller crosses the coefficients.  This is the one thing a Taylor
     test cannot see (on the diagonal the two conventions agree), and getting
     it backwards would silently wreck the Hessian in RecNFP.d2F_dF3;
  4. the magnification-aware family (dcurlySmc / dcurlySadjmc / d2curlySmc),
     which rec_mpi.Rec calls and RecNFP does not, uses the same tau convention
     as Shift's CUDA kernels: d/dm_a = -tau_a * d/dr_a with tau measured from
     the OUTPUT tile centre.  A Taylor test in r alone cannot see a tau sign
     error, so those checks drive Delta_m and compare against Shift directly.

Run:  ./run.sh          (or: PYTHONPATH=<repo>/src python test_shift_fft.py)
"""

import numpy as np
import cupy as cp

from holotomocupy.shift import Shift
from holotomocupy.shift_fft import ShiftFFT

N, NPSI, NZ, NZPSI, NTHETA = 32, 48, 32, 48, 3


def bandlimited(shape, kmax, seed, dtype='complex64'):
    """Random object whose spectrum is zero above |k| = kmax on each axis.

    Both operators are exact on such an input -- the FFT shift by construction,
    the cubic stencil to the extent that the object is oversampled -- so the
    two can be compared without the interpolation error swamping everything.
    Also keeps the object away from the grid edges implicitly: a smooth field
    is what the periodic BC of ShiftFFT needs.
    """
    rng = np.random.default_rng(seed)
    nt, ny, nx = shape
    spec = np.zeros((nt, ny, nx), dtype='complex128')
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    mask = (np.abs(ky)[:, None] <= kmax) & (np.abs(kx)[None, :] <= kmax)
    n_on = int(mask.sum())
    for t in range(nt):
        spec[t][mask] = (rng.standard_normal(n_on) + 1j * rng.standard_normal(n_on))
    out = np.fft.ifft2(spec, axes=(-2, -1))
    out /= np.abs(out).max()
    return cp.asarray(out.astype(dtype))


def taper(a, edge=6):
    """Cosine taper to zero at the border.

    ShiftFFT wraps at the boundary, Shift mirrors.  Neither is wrong; they are
    simply different boundary conditions, and the only inputs on which they can
    be compared at all are ones that vanish there.  RecNFP gets the same
    guarantee from sizing nobj = n + 2*max|pos| + margin.
    """
    ny, nx = a.shape[-2:]
    wy = np.ones(ny); wx = np.ones(nx)
    r = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
    wy[:edge] = r; wy[-edge:] = r[::-1]
    wx[:edge] = r; wx[-edge:] = r[::-1]
    w = cp.asarray((wy[:, None] * wx[None, :]).astype('float32'))
    return a * w


def rel(a, b):
    return float(cp.linalg.norm(a - b) / cp.linalg.norm(b))


def make_case(seed=0):
    c  = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, seed))
    r  = cp.asarray(np.random.default_rng(seed + 1).uniform(-3, 3, (NTHETA, 2)), dtype='float32')
    m  = cp.ones((NTHETA, 2), dtype='float32')
    return c, r, m


def check(name, value, tol):
    ok = value < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<44s} {value:.3e}  (tol {tol:.0e})")
    return ok


def test_forward_matches_cubic():
    """S agrees with the cubic stencil on a band-limited, tapered object."""
    print("forward S: ShiftFFT vs Shift")
    c, r, m = make_case()
    fft   = ShiftFFT(N, NPSI, NZ, NZPSI)
    cubic = Shift(N, NPSI, NZ, NZPSI)
    a = fft.curlySc(c, r, m)
    b = cubic.curlySc(cubic.coeff(c), r, m)
    return check("rel|S_fft - S_cubic|", rel(a, b), 2e-2)


def test_exact_integer_shift():
    """At an integer shift the FFT path must reproduce the samples exactly.

    Sharper than the band-limited comparison above: no interpolation is
    involved at all, so any error here is a wrong grid offset, not a kernel.
    """
    print("forward S: integer shift is a pure re-index")
    rng = np.random.default_rng(7)
    c = taper(bandlimited((NTHETA, NZPSI, NPSI), 8, 11))
    r = cp.asarray(rng.integers(-3, 4, (NTHETA, 2)).astype('float32'))
    m = cp.ones((NTHETA, 2), dtype='float32')
    fft = ShiftFFT(N, NPSI, NZ, NZPSI)
    got = fft.curlySc(c, r, m)

    # s_kernel convention: out[t,ty,tx] = c[t, ty + (nzpsi-nz)/2 - ry,
    #                                        tx + (npsi-n)/2  - rx]
    dy, dx = (NZPSI - NZ) // 2, (NPSI - N) // 2
    want = cp.empty_like(got)
    rn = cp.asnumpy(r).astype(int)
    for t in range(NTHETA):
        iy = (np.arange(NZ) + dy - rn[t, 0]) % NZPSI
        ix = (np.arange(N) + dx - rn[t, 1]) % NPSI
        want[t] = c[t][cp.asarray(iy)][:, cp.asarray(ix)]
    return check("rel|S_fft - reindex|", rel(got, want), 1e-5)


def test_adjoint():
    """<S c, phi> == <c, S* phi> in the real inner product."""
    print("adjoint: Sadj is the adjoint of S")
    c, r, m = make_case(2)
    phi = bandlimited((NTHETA, NZ, N), 6, 21)
    fft = ShiftFFT(N, NPSI, NZ, NZPSI)
    lhs = float(cp.sum(cp.real(cp.conj(fft.S(c, r, m)) * phi)))
    rhs = float(cp.sum(cp.real(cp.conj(c) * fft.Sadj(phi, r, m))))
    return check("|<Sc,phi> - <c,S*phi>| / |lhs|", abs(lhs - rhs) / abs(lhs), 1e-5)


def test_first_derivative():
    """dcurlySc is the derivative of curlySc: the Taylor remainder is O(t^2)."""
    print("derivative: dcurlySc vs finite differences")
    c, r, m = make_case(3)
    c1 = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 31))
    dr = cp.asarray(np.random.default_rng(32).uniform(-1, 1, (NTHETA, 2)), dtype='float32')
    fft = ShiftFFT(N, NPSI, NZ, NZPSI)

    f0 = fft.curlySc(c, r, m)
    df = fft.dcurlySc(c, r, m, c1, dr)
    ok = True
    prev = None
    for t in (1e-2, 5e-3):
        ft = fft.curlySc(c + t * c1, r + t * dr, m)
        res = float(cp.linalg.norm(ft - f0 - t * df))
        if prev is not None:
            # halving t must shrink the second-order remainder ~4x
            ok &= check(f"remainder ratio t=1e-2 -> 5e-3 (want ~4)",
                        abs(prev / max(res, 1e-30) - 4.0) / 4.0, 0.25)
        prev = res
    return ok


def test_derivative_adjoint():
    """dcurlySadjc is the adjoint of (c1, dr) -> dcurlySc(c, r, m, c1, dr)."""
    print("adjoint: dcurlySadjc vs dcurlySc")
    c, r, m = make_case(4)
    c1 = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 41))
    dr = cp.asarray(np.random.default_rng(42).uniform(-1, 1, (NTHETA, 2)), dtype='float32')
    phi = bandlimited((NTHETA, NZ, N), 6, 43)
    fft = ShiftFFT(N, NPSI, NZ, NZPSI)

    lhs = float(cp.sum(cp.real(cp.conj(fft.dcurlySc(c, r, m, c1, dr)) * phi)))
    out1, out2 = fft.dcurlySadjc(c, r, m, phi)
    rhs = float(cp.sum(cp.real(cp.conj(c1) * out1)) + cp.sum(dr * out2))
    return check("|<dS,phi> - <(c1,dr),dS*phi>| / |lhs|", abs(lhs - rhs) / abs(lhs), 1e-4)


def _mixed_fd(cl, c, r, m, cy, dry, cz, drz, h=3e-2):
    """d2/ds dt  curlySc(c + s*cy + t*cz, r + s*dry + t*drz)  at s = t = 0."""
    def f(s, t):
        return cl.curlySc(c + s * cy + t * cz, r + s * dry + t * drz, m)
    return (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)


def test_second_derivative_slot_pairing():
    """d2curlySc, CROSSED by the caller, is the true mixed second derivative.

    Also reports what the un-crossed call gives, so the test is visibly able
    to tell the two conventions apart -- on the diagonal they coincide and a
    Taylor test would pass either way.  Both classes are checked: RecNFP swaps
    one for the other and calls them identically.
    """
    print("second derivative: slot pairing (caller crosses)")
    c, r, m = make_case(5)
    cy = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 51))
    cz = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 52))
    rng = np.random.default_rng(53)
    dry = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
    drz = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')

    ok = True
    for name, cl, prep in (("fft",   ShiftFFT(N, NPSI, NZ, NZPSI), lambda x: x),
                           ("cubic", Shift(N, NPSI, NZ, NZPSI),    None)):
        cl_prep = prep if prep is not None else cl.coeff
        C, Cy, Cz = cl_prep(c), cl_prep(cy), cl_prep(cz)
        want = _mixed_fd(cl, C, r, m, Cy, dry, Cz, drz)
        crossed   = cl.d2curlySc(C, r, m, Cz, dry, Cy, drz)
        uncrossed = cl.d2curlySc(C, r, m, Cy, dry, Cz, drz)
        ok &= check(f"[{name}] crossed vs mixed FD", rel(crossed, want), 2e-2)
        gap = rel(uncrossed, want)
        print(f"       (un-crossed call would be off by {gap:.3e} -- test discriminates)")
        ok &= gap > 0.1
    return ok


# ---------------------------------------------------------------------------
# Magnification-aware family (dcurlySmc / dcurlySadjmc / d2curlySmc).
#
# These are what rec_mpi.Rec calls -- RecNFP holds m fixed and never touches
# them.  The thing they can get wrong that nothing above would catch is the
# tau convention: d/dm_a = -tau_a(pixel) * d/dr_a with tau measured from the
# OUTPUT tile centre.  A sign error, or measuring tau from the input grid,
# still passes a Taylor test in r alone, so every check below drives Delta_m
# non-zero and two of them compare against Shift's CUDA kernels directly.
# ---------------------------------------------------------------------------

def m_case(seed, mag=1.0):
    """Object, positions, and a per-projection magnification near `mag`."""
    c = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, seed))
    rng = np.random.default_rng(seed + 1)
    r = cp.asarray(rng.uniform(-2, 2, (NTHETA, 2)), dtype='float32')
    if mag == 1.0:
        m = cp.ones((NTHETA, 2), dtype='float32')
    else:
        m = cp.asarray(rng.uniform(mag - 0.01, mag + 0.01, (NTHETA, 2)), dtype='float32')
    return c, r, m


def test_m_zero_shortcut():
    """Delta_m == 0 must reproduce the m-free methods bit for bit.

    The *mc methods short-circuit to the fused *c ones in that case, which is
    the path every rho[tp] = 0 config takes, so it is worth pinning that the
    shortcut is not merely close but identical.
    """
    print("magnification: Delta_m = 0 falls back to the *c family")
    c, r, m = m_case(60)
    c1 = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 61))
    c2 = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 62))
    rng = np.random.default_rng(63)
    dr1 = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
    dr2 = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
    dm0 = cp.zeros((NTHETA, 2), dtype='float32')
    fft = ShiftFFT(N, NPSI, NZ, NZPSI)

    a = fft.dcurlySmc(c, r, m, c1, dr1, dm0)
    b = fft.dcurlySc(c, r, m, c1, dr1)
    ok = check("|dcurlySmc - dcurlySc|", rel(a, b), 1e-12)

    a2 = fft.d2curlySmc(c, r, m, c1, dr1, dm0, c2, dr2, dm0)
    b2 = fft.d2curlySc(c, r, m, c1, dr1, c2, dr2)
    ok &= check("|d2curlySmc - d2curlySc|", rel(a2, b2), 1e-12)
    return ok


def test_m_first_derivative():
    """dcurlySmc is the derivative of curlySc in (c, r, m) jointly.

    Taylor test with Delta_m non-zero, on both dispatch paths: m == 1 (the
    FFT fast path, reached because Delta_m alone does not change m) and
    m != 1 (chirp-z).
    """
    print("magnification: dcurlySmc vs finite differences")
    ok = True
    for label, mag in (("m=1", 1.0), ("m!=1", 1.02)):
        c, r, m = m_case(64, mag)
        c1 = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 65))
        rng = np.random.default_rng(66)
        dr = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
        dm = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
        fft = ShiftFFT(N, NPSI, NZ, NZPSI)

        f0 = fft.curlySc(c, r, m)
        df = fft.dcurlySmc(c, r, m, c1, dr, dm)
        prev = None
        # Same step pair as the r-only Taylor test.  Smaller steps do not help:
        # the O(t^2) remainder drops below the float32 noise of the two
        # curlySc evaluations and the ratio stops being meaningful.
        for t in (1e-2, 5e-3):
            ft = fft.curlySc(c + t * c1, r + t * dr, m + t * dm)
            res = float(cp.linalg.norm(ft - f0 - t * df))
            if prev is not None:
                ok &= check(f"[{label}] remainder ratio (want ~4)",
                            abs(prev / max(res, 1e-30) - 4.0) / 4.0, 0.25)
            prev = res
    return ok


def test_m_matches_cubic():
    """dcurlySmc agrees with Shift.dcurlySmc -- pins the tau convention.

    A Taylor test only says ShiftFFT is consistent with ITSELF.  This is the
    check that it is consistent with the CUDA kernels rec_mpi was written
    against: tau measured from the output tile centre, and
    d/dm = -tau * d/dr rather than +tau.
    """
    print("magnification: dcurlySmc vs Shift.dcurlySmc")
    ok = True
    for label, mag in (("m=1", 1.0), ("m!=1", 1.02)):
        c, r, m = m_case(67, mag)
        c1 = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 68))
        rng = np.random.default_rng(69)
        dr = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
        dm = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
        fft   = ShiftFFT(N, NPSI, NZ, NZPSI)
        cubic = Shift(N, NPSI, NZ, NZPSI)

        a = fft.dcurlySmc(c, r, m, c1, dr, dm)
        b = cubic.dcurlySmc(cubic.coeff(c), r, m, cubic.coeff(c1), dr, dm)
        ok &= check(f"[{label}] rel|dSm_fft - dSm_cubic|", rel(a, b), 3e-2)
        # Flipping tau's sign is the failure this test exists for; show that
        # it would be caught rather than absorbed by the tolerance.
        bad = fft.dcurlySmc(c, r, m, c1, dr, -dm)
        ok &= rel(bad, b) > 0.1
    return ok


def test_m_derivative_adjoint():
    """dcurlySadjmc is the adjoint of (c1, dr, dm) -> dcurlySmc."""
    print("magnification: dcurlySadjmc vs dcurlySmc")
    ok = True
    for label, mag in (("m=1", 1.0), ("m!=1", 1.02)):
        c, r, m = m_case(70, mag)
        c1 = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 71))
        rng = np.random.default_rng(72)
        dr = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
        dm = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
        phi = bandlimited((NTHETA, NZ, N), 6, 73)
        fft = ShiftFFT(N, NPSI, NZ, NZPSI)

        lhs = float(cp.sum(cp.real(cp.conj(fft.dcurlySmc(c, r, m, c1, dr, dm)) * phi)))
        out1, out2_r, out2_m = fft.dcurlySadjmc(c, r, m, phi)
        rhs = float(cp.sum(cp.real(cp.conj(c1) * out1))
                    + cp.sum(dr * out2_r) + cp.sum(dm * out2_m))
        ok &= check(f"[{label}] |<dSm,phi> - <(c1,dr,dm),dSm*phi>| / |lhs|",
                    abs(lhs - rhs) / abs(lhs), 1e-4)
    return ok


def _mixed_fd_m(cl, c, r, m, cy, dry, dmy, cz, drz, dmz, h=3e-2):
    """d2/ds dt curlySc(c + s*cy + t*cz, r + s*dry + t*drz, m + s*dmy + t*dmz)."""
    def f(s, t):
        return cl.curlySc(c + s * cy + t * cz, r + s * dry + t * drz,
                          m + s * dmy + t * dmz)
    return (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)


def test_m_second_derivative_slot_pairing():
    """d2curlySmc, CROSSED by the caller, is the true mixed second derivative.

    Same argument as test_second_derivative_slot_pairing, now with the
    magnification direction live, and again against both classes -- Rec swaps
    one for the other and calls d2curlySmc identically.
    """
    print("magnification: d2curlySmc slot pairing (caller crosses)")
    c, r, m = m_case(74)
    cy = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 75))
    cz = taper(bandlimited((NTHETA, NZPSI, NPSI), 4, 76))
    rng = np.random.default_rng(77)
    dry = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
    drz = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
    dmy = cp.asarray(rng.uniform(-0.2, 0.2, (NTHETA, 2)), dtype='float32')
    dmz = cp.asarray(rng.uniform(-0.2, 0.2, (NTHETA, 2)), dtype='float32')

    ok = True
    for name, cl, prep in (("fft",   ShiftFFT(N, NPSI, NZ, NZPSI), lambda x: x),
                           ("cubic", Shift(N, NPSI, NZ, NZPSI),    None)):
        cl_prep = prep if prep is not None else cl.coeff
        C, Cy, Cz = cl_prep(c), cl_prep(cy), cl_prep(cz)
        want = _mixed_fd_m(cl, C, r, m, Cy, dry, dmy, Cz, drz, dmz)
        crossed   = cl.d2curlySmc(C, r, m, Cz, dry, dmy, Cy, drz, dmz)
        uncrossed = cl.d2curlySmc(C, r, m, Cy, dry, dmy, Cz, drz, dmz)
        ok &= check(f"[{name}] crossed vs mixed FD", rel(crossed, want), 3e-2)
        gap = rel(uncrossed, want)
        print(f"       (un-crossed call would be off by {gap:.3e} -- test discriminates)")
        ok &= gap > 0.1
    return ok


if __name__ == '__main__':
    results = [
        test_forward_matches_cubic(),
        test_exact_integer_shift(),
        test_adjoint(),
        test_first_derivative(),
        test_derivative_adjoint(),
        test_second_derivative_slot_pairing(),
        test_m_zero_shortcut(),
        test_m_first_derivative(),
        test_m_matches_cubic(),
        test_m_derivative_adjoint(),
        test_m_second_derivative_slot_pairing(),
    ]
    print()
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    raise SystemExit(0 if all(results) else 1)
