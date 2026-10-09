"""Compare holotomocupy.Shift.curlyS with scipy.ndimage.shift.

When npsi == n and magnification m == 1, curlyS becomes a pure subpixel shift,
which can be cross-checked against scipy.ndimage.shift (cubic B-spline, order=3).

The kernel maps: output[ty, tx] = input[ty - ry, tx - rx], which is the same
sign convention as scipy.ndimage.shift(arr, [ry, rx]).
"""

import os
import numpy as np
import cupy as cp
import scipy.ndimage as snd
import matplotlib.pyplot as plt

from holotomocupy.shift import Shift


def make_input(n, seed=0):
    rng = np.random.default_rng(seed)
    # Smooth-ish complex test object (sum of a few sinusoids + noise)
    yy, xx = np.mgrid[0:n, 0:n].astype('float32')
    re = (np.sin(2 * np.pi * xx / n * 3) * np.cos(2 * np.pi * yy / n * 2)
          + 0.3 * rng.standard_normal((n, n), dtype='float32'))
    im = (np.cos(2 * np.pi * xx / n * 4) * np.sin(2 * np.pi * yy / n * 5)
          + 0.3 * rng.standard_normal((n, n), dtype='float32'))
    return (re + 1j * im).astype('complex64')


def shift_scipy(arr, ry, rx, order=3):
    # mode='mirror' matches the kernel's sym_idx (cuda_kernels.py:140):
    #   i<0 → -i  and  i>=N → 2N-2-i  (reflect without duplicating the edge).
    # scipy's 'reflect' duplicates the edge and would NOT match.
    re = snd.shift(arr.real, [ry, rx], order=order, mode='mirror')
    im = snd.shift(arr.imag, [ry, rx], order=order, mode='mirror')
    return (re + 1j * im).astype('complex64')


def shift_holotomo(cl, arr_gpu, ry, rx):
    pos = cp.asarray([[[ry, rx]]], dtype='float32')   # [ntheta=1, ndist=1, 2]
    # magnification is per axis: [ntheta, 2] = (y, x)
    m   = cp.asarray([[1.0, 1.0]], dtype='float32')   # unit magnification
    return cl.curlyS(arr_gpu, pos[:, 0], m)


def interior_error(a, b, margin):
    """Max-abs error inside the array, ignoring the boundary `margin` pixels
    where the two methods handle out-of-bounds samples differently."""
    s = (slice(margin, -margin), slice(margin, -margin))
    return float(np.max(np.abs(a[s] - b[s])))


def check_mag_derivatives(n=64, npsi=96, ntheta=3, seed=3):
    """dcurlySmc / dcurlySadjmc / d2curlySmc against finite differences.

    curlySc is smooth in (c, r, m) jointly, so with f(t) = curlySc(c + t*c1,
    r + t*dr, m + t*dm) the analytic pieces must satisfy

        f(t) - f(0)                            = O(t^2)
        f(t) - f(0) - t*df                     = O(t^3)
        f(t) - f(0) - t*df - t^2/2 * d2f       = O(t^4)

    with df = dcurlySmc(...) and d2f = d2curlySmc(...) taken along the same
    direction twice. Halving t must therefore shrink the three residuals by
    4x, 8x and 16x -- that is what pins down the tau-folding identity
    d/dm_axis = -tau_axis * d/dr_axis inside the kernels.

    npsi > n so that a magnification of ~1.1 still samples inside the object
    grid and the mirror boundary condition stays out of the comparison.
    """
    cl = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi)
    rng = cp.random.RandomState(seed)

    def cplx(shape):
        return (rng.rand(*shape) + 1j * rng.rand(*shape)).astype('complex64')

    psi  = cplx((ntheta, npsi, npsi))
    dpsi = cplx((ntheta, npsi, npsi))
    r    = ((rng.rand(ntheta, 2) - 0.5) * 6).astype('float32')
    dr   = ((rng.rand(ntheta, 2) - 0.5) * 2).astype('float32')
    m    = (1.05 + 0.1 * rng.rand(ntheta, 2)).astype('float32')
    dm   = ((rng.rand(ntheta, 2) - 0.5) * 0.2).astype('float32')

    # coeff is a linear filter, so coeff(psi + t*dpsi) == c + t*c1 exactly.
    c, c1 = cl.coeff(psi), cl.coeff(dpsi)

    f0  = cl.curlySc(c, r, m)
    df  = cl.dcurlySmc(c, r, m, c1, dr, dm)
    d2f = cl.d2curlySmc(c, r, m, c1, dr, dm, c1, dr, dm)

    print()
    print(f"{'t':>10} {'|f(t)-f0|':>12} {'-t*df':>12} {'-t^2/2 d2f':>12}")
    print("-" * 50)
    res = []
    for t in (1e-1, 5e-2, 2.5e-2):
        ft = cl.curlySc(c + np.float32(t) * c1,
                        r + np.float32(t) * dr,
                        m + np.float32(t) * dm)
        d = ft - f0
        row = (float(cp.abs(d).max()),
               float(cp.abs(d - t * df).max()),
               float(cp.abs(d - t * df - 0.5 * t * t * d2f).max()))
        print(f"{t:10.4g} {row[0]:12.4e} {row[1]:12.4e} {row[2]:12.4e}")
        res.append(row)
    # the last two rows halve t: the residuals must drop by ~2x / ~4x / ~8x
    ratio = [a / max(b, 1e-30) for a, b in zip(res[-2], res[-1])]
    print(f"ratios on the last halving: O(t)={ratio[0]:5.2f} (want ~2)  "
          f"O(t^2)={ratio[1]:5.2f} (want ~4)  O(t^3)={ratio[2]:5.2f} (want ~8)")
    assert res[-1][1] < 0.3 * res[-1][0], "dcurlySmc is not the first derivative"
    assert res[-1][2] < 0.3 * res[-1][1], "d2curlySmc is not the second derivative"

    # adjoint: <dcurlySmc(c1, dr, dm), g> == <c1, o1> + <dr, o_r> + <dm, o_m>
    g = cplx((ntheta, n, n))
    o1, o_r, o_m = cl.dcurlySadjmc(c, r, m, g)
    lhs = float(cp.sum(cp.real(cp.conj(df) * g)))
    rhs = (float(cp.sum(cp.real(cp.conj(c1) * o1)))
           + float(cp.sum(dr * o_r)) + float(cp.sum(dm * o_m)))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"Adjoint test (dcurlySmc/dcurlySadjmc): lhs={lhs:.6e}  rhs={rhs:.6e}  "
          f"rel={rel:.3e}")
    assert rel < 1e-4, "dcurlySadjmc is not the adjoint of dcurlySmc"

    check_polarization(cl, c, r, m)


def check_polarization(cl, c, r, m):
    """d2curlySmc off the diagonal.

    The Taylor check above only ever probes B(y, y): it takes the same
    direction twice. But the kernel pairs each coefficient with the geometry in
    its OWN slot, so the mixed term is only right if the caller crosses them --
    and on the diagonal crossed and uncrossed are identical. Polarization,

        B(y+z, y+z) = B(y,y) + 2 B(y,z) + B(z,z),

    pins the off-diagonal against the two diagonals and separates the two.
    """
    ntheta = c.shape[0]
    rng = cp.random.RandomState(11)

    def cplx(shape):
        return (rng.rand(*shape) + 1j * rng.rand(*shape)).astype('complex64')

    cy, cz = cplx(c.shape), cplx(c.shape)
    yr = ((rng.rand(ntheta, 2) - 0.5) * 2).astype('float32')
    zr = ((rng.rand(ntheta, 2) - 0.5) * 2).astype('float32')
    ym = ((rng.rand(ntheta, 2) - 0.5) * 0.2).astype('float32')
    zm = ((rng.rand(ntheta, 2) - 0.5) * 0.2).astype('float32')

    d2 = cl.d2curlySmc
    Byy = d2(c, r, m, cy, yr, ym, cy, yr, ym)
    Bzz = d2(c, r, m, cz, zr, zm, cz, zr, zm)
    Bss = d2(c, r, m, cy + cz, yr + zr, ym + zm,
                      cy + cz, yr + zr, ym + zm)
    ref = Bss - Byy - Bzz          # == 2 B(y, z)
    nrm = float(cp.abs(ref).max())

    crossed   = d2(c, r, m, cz, yr, ym, cy, zr, zm)   # what Rec.d2F_dF3 does
    uncrossed = d2(c, r, m, cy, yr, ym, cz, zr, zm)
    e_ok  = float(cp.abs(2 * crossed   - ref).max()) / nrm
    e_bad = float(cp.abs(2 * uncrossed - ref).max()) / nrm
    print(f"Polarization (d2curlySmc off-diagonal): crossed rel={e_ok:.3e}  "
          f"uncrossed rel={e_bad:.3e}")
    assert e_ok < 1e-4, "d2curlySmc violates B(y+z,y+z)=B(y,y)+2B(y,z)+B(z,z)"
    assert e_bad > 1e-2, "the uncrossed pairing should fail -- test is blind"


def main():
    n = 64
    margin = 4          # cubic B-spline support is 4 pixels — ignore boundary
    arr_np  = make_input(n)
    arr_gpu = cp.asarray(arr_np)[cp.newaxis]   # [1, n, n]

    cl = Shift(n=n, npsi=n, nz=n, nzpsi=n)

    shifts = [(0.0, 0.0),
              (0.5, 0.0),
              (0.0, 0.5),
              (1.3, -2.7),
              (-3.4, 4.1),
              (7.0, 7.0)]

    print(f"{'ry':>8} {'rx':>8}   {'max|holo-scipy|':>18}   {'rel':>10}")
    print("-" * 56)
    results = []
    for ry, rx in shifts:
        b_holo  = shift_holotomo(cl, arr_gpu, ry, rx)[0].get()
        b_scipy = shift_scipy(arr_np, ry, rx, order=3)

        err = interior_error(b_holo, b_scipy, margin)
        ref = float(np.max(np.abs(b_scipy[margin:-margin, margin:-margin]))) + 1e-30
        print(f"{ry:8.3f} {rx:8.3f}   {err:18.3e}   {err/ref:10.3e}")
        results.append((ry, rx, b_holo, b_scipy))

    # ------------------------------------------------------------------
    # Adjoint test: <S(c), b> == <c, Sadj(b)>
    # ------------------------------------------------------------------
    ntheta = 4
    pos_adj = (cp.random.random([ntheta, 2], dtype='float32') - 0.5) * 20
    m_adj   = cp.ones((ntheta, 2), dtype='float32')

    c = (cp.random.random([ntheta, n, n], dtype='float32')
         + 1j * cp.random.random([ntheta, n, n], dtype='float32')).astype('complex64')
    b = (cp.random.random([ntheta, n, n], dtype='float32')
         + 1j * cp.random.random([ntheta, n, n], dtype='float32')).astype('complex64')

    Sc    = cl.S(c, pos_adj, m_adj)
    Sadjb = cl.Sadj(b, pos_adj, m_adj)
    lhs = float(cp.real(cp.sum(cp.conj(Sc) * b)))
    rhs = float(cp.real(cp.sum(cp.conj(c) * Sadjb)))
    print()
    print(f"Adjoint test (S/Sadj): lhs={lhs:.6e}  rhs={rhs:.6e}  "
          f"rel={abs(lhs - rhs) / (abs(lhs) + 1e-30):.3e}")

    check_mag_derivatives()

    # ------------------------------------------------------------------
    # Visualization — restrict to the central window so boundary effects
    # (mirror-FFT vs mirror-IIR prefilter differences) don't dominate the
    # color scale of the 2D error map.
    # ------------------------------------------------------------------
    vis_margin = n // 4                   # show only the central half
    s   = (slice(vis_margin, -vis_margin), slice(vis_margin, -vis_margin))
    row = (n // 2) - vis_margin           # central row inside the cropped view

    fig, axs = plt.subplots(len(results), 4, figsize=(14, 3 * len(results)))
    for i, (ry, rx, b_holo, b_scipy) in enumerate(results):
        h_in = b_holo[s]
        s_in = b_scipy[s]
        diff = np.abs(h_in - s_in)

        im0 = axs[i, 0].imshow(h_in.real, cmap='gray')
        axs[i, 0].set_title(f'holo Re  (ry={ry}, rx={rx})  central')
        im1 = axs[i, 1].imshow(s_in.real, cmap='gray')
        axs[i, 1].set_title('scipy Re  central')
        im2 = axs[i, 2].imshow(diff,      cmap='hot')
        axs[i, 2].set_title(f'|holo - scipy|  central  (max {diff.max():.2e})')

        axs[i, 3].plot(h_in.real[row],       label='holo')
        axs[i, 3].plot(s_in.real[row], '--', label='scipy')
        axs[i, 3].set_title(f'row {n//2} (real, central)')
        axs[i, 3].legend(); axs[i, 3].grid()

        for ax in axs[i, :3]:
            ax.axis('off')
        fig.colorbar(im2, ax=axs[i, 2], fraction=0.046)
    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'test_shift_compare.png')
    plt.savefig(out_png, dpi=110)
    print(f"\nSaved figure: {out_png}")


if __name__ == '__main__':
    main()
