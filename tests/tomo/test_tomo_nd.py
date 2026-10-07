#!/usr/bin/env python
"""Checks for Tomo's oversampled-detector option (`nd`).

    PYTHONPATH=../../src python test_tomo_nd.py

One GPU, no MPI, no data files.  Every check prints PASS/FAIL and the script
exits non-zero if any of them failed.

The point of `nd` is to let the object sit on a coarser x/y grid than the
projection plane: R maps [nz, n, n] -> [ntheta, nz, nd] with nd = n or 2n, the
detector spanning the same field of view either way.  Only two things have to
hold for the solver to be able to use it:

  * nothing changes at nd == n (check 1), and
  * R and RT stay an exact adjoint pair at nd == 2n (check 2),

and one thing has to hold for Rec.norm_const to need no new factor: R's *values*
must not depend on nd, only their sampling density (check 3).
"""
import math
import sys

import numpy as np
import cupy as cp

from holotomocupy.tomo import Tomo

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def rand_obj(nz, n, seed):
    rng = np.random.default_rng(seed)
    return cp.asarray((rng.random((nz, n, n)) + 1j * rng.random((nz, n, n))).astype('complex64'))


def rand_sino(ntheta, nz, nd, seed):
    rng = np.random.default_rng(seed)
    return cp.asarray((rng.random((ntheta, nz, nd)) + 1j * rng.random((ntheta, nz, nd))
                       ).astype('complex64'))


def main():
    n, nz, ntheta = 64, 4, 32
    theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
    mask_r = 0.9

    # ---- 1. regression: nd == n must be bit-for-bit what it was ------------
    print("1. nd == n reproduces the un-parameterised operator")
    base = Tomo(n, nz, theta, mask_r)
    u, d = rand_obj(nz, n, 0), rand_sino(ntheta, nz, n, 1)
    r0, t0 = base.R(u).copy(), base.RT(d).copy()
    # RT scatters with atomicAdd, so its summation order -- and hence its last
    # float32 bit -- varies between two runs of the same call.  Bit-exactness is
    # only meaningful for R; RT is held to its own run-to-run noise floor.
    floor = float(cp.linalg.norm(base.RT(d) - t0) / cp.linalg.norm(t0))
    print(f"     (RT atomicAdd noise floor, same call twice: rel={floor:.2e})")
    cl = Tomo(n, nz, theta, mask_r, nd=n)
    dr = float(cp.abs(cl.R(u) - r0).max())
    dt = float(cp.linalg.norm(cl.RT(d) - t0) / cp.linalg.norm(t0))
    check("R identical", dr == 0.0, f"max|diff|={dr:g}")
    check("RT within atomicAdd noise", dt < 1e-6, f"rel={dt:.2e}")

    # ---- 2. adjoint identity at both samplings -----------------------------
    print("2. <Ru, d> == <u, RTd>")
    for nd in (n, 2 * n):
        cl = Tomo(n, nz, theta, mask_r, nd=nd)
        u, d = rand_obj(nz, n, 2), rand_sino(ntheta, nz, nd, 3)
        a = complex(cp.sum(cl.R(u) * cp.conj(d)))
        b = complex(cp.sum(u * cp.conj(cl.RT(d))))
        rel = abs(a - b) / abs(a)
        check(f"nd={nd}", rel < 1e-6, f"{a:.8e} vs {b:.8e}  rel={rel:.2e}")

    # ---- 3. R's values do not depend on nd ---------------------------------
    # The object is band-limited to |f| < 1/4 of the *coarse* grid and sampled on
    # both grids (the fine one by exact Fourier interpolation), with the value
    # doubled on the coarse grid because obj scales with the in-plane voxel size.
    # Band-limiting is what makes the comparison fair: the aliased replicas the
    # kernel picks up above the coarse Nyquist are then empty, so the fine and
    # coarse operators are looking at the same object.
    # The line integral sum_j obj_j is then identical on the two grids, so any
    # residual scaling is the operator's own -- and it must come out at exactly
    # sqrt(n_fine/n_coarse), the 1/sqrt(n) that Rec.norm_const already carries.
    print("3. R value scaling is 1/sqrt(n_obj) and independent of nd")
    nc = n // 2
    rng = np.random.default_rng(4)
    F = np.fft.fftshift(np.fft.fft2(rng.random((nz, nc, nc))), axes=(-2, -1))
    kk = np.fft.fftshift(np.fft.fftfreq(nc))
    KY, KX = np.meshgrid(kk, kk, indexing='ij')
    F[:, (np.abs(KY) > 0.25) | (np.abs(KX) > 0.25)] = 0
    blk = np.fft.ifft2(np.fft.ifftshift(F, axes=(-2, -1))).real
    Fp = np.zeros((nz, n, n), 'complex128')
    Fp[:, (n - nc) // 2:(n + nc) // 2, (n - nc) // 2:(n + nc) // 2] = F
    # ifft2 divides by the transform length, so undo it to keep sample values
    fine = np.fft.ifft2(np.fft.ifftshift(Fp, axes=(-2, -1))).real * (n / nc) ** 2
    u_fine = cp.asarray(fine.astype('complex64'))
    u_coarse = cp.asarray((2.0 * blk).astype('complex64'))

    # 3a. the object-grid half.  Both operators write onto the same n-sample
    # detector, so this is a straight elementwise fit.
    A = Tomo(n, nz, theta, -1)                                 # obj n,   det n
    ra = A.R(u_fine)
    B = Tomo(nc, nz, theta, -1, nd=n)                          # obj n/2, det n
    rb = B.R(u_coarse)
    k = float(cp.sum(rb * ra).real / cp.sum(ra * ra).real)
    check("best-fit scalar == sqrt(2)",
          abs(k - math.sqrt(2)) < 1e-3, f"k={k:.6f} vs {math.sqrt(2):.6f}")

    # 3b. the nd half: same object, same object grid, nd = n vs 2n.  The finer
    # detector must carry the *same values*, only more densely -- so the shared
    # frequency band has to match once the DFT length is divided out.
    C = Tomo(nc, nz, theta, -1)                                # obj n/2, det n/2
    fc = cp.fft.fftshift(cp.fft.fft(C.R(u_coarse), axis=-1), axes=-1)
    D = Tomo(nc, nz, theta, -1, nd=n)                          # obj n/2, det n
    fd = cp.fft.fftshift(cp.fft.fft(D.R(u_coarse), axis=-1), axes=-1)
    fd = fd[..., nc // 2: nc // 2 + nc] * (nc / n)             # undo the DFT length
    rel = float(cp.linalg.norm(fd - fc) / cp.linalg.norm(fc))
    check("shared spectrum of nd=2n matches nd=n", rel < 1e-3, f"rel={rel:.2e}")

    # ---- 4. the |f| >= 1/2 bins, and no combs at theta = 0 / 90 ------------
    # Bins with |f| >= 1/2 only exist at nd > n.  The gather keeps the ones
    # whose Cartesian coordinate is still inside the padded FFT square -- the
    # corners of the square are real, recoverable content -- and SKIPS the
    # rest rather than wrapping them.  So the outer band is populated, but by
    # signal, not by aliased replicas.
    print("4. outer bins populated, and no combs on the axes")
    u = rand_obj(nz, nc, 5)
    cl = Tomo(nc, nz, theta, -1, nd=n)
    d = cl.R(u)
    f = cp.fft.fftshift(cp.fft.fft(d, axis=-1), axes=-1)
    outer = cp.concatenate([f[..., :nc // 2], f[..., nc // 2 + nc:]], axis=-1)
    frac = float(cp.linalg.norm(outer) / cp.linalg.norm(f))
    check("outer-bin energy fraction is non-zero", frac > 1e-3, f"frac={frac:.3e}")

    # THE REGRESSION THIS GUARDS.  When the gather WRAPPED instead of skipping,
    # theta = 0 and 90 deg -- and only there -- wrapped exactly onto themselves
    # (the shift 2n*cos(theta) is 0 mod 2n only for cos/sin in {0,+-1}), so
    # those two rows read back an n-periodic spectrum and came out of the
    # length-nd inverse transform as combs, every odd detector sample exactly
    # zero.  Two projections per scan were destroyed, the two aligned with x
    # and y, which is what put high-frequency vertical and horizontal line
    # artifacts in the ctxl tomo_upsample=2 reconstructions.  All three rows
    # must now be fully populated.
    th2 = np.array([0.0, np.pi / 4, np.pi / 2], dtype='float32')
    cl2 = Tomo(nc, 1, th2, -1, nd=n)
    s2 = cl2.R(u[:1]).real
    ratio = [float(cp.abs(s2[k, 0, 1::2]).mean() / cp.abs(s2[k, 0, 0::2]).mean())
             for k in range(3)]
    check("theta=0 is not a comb", ratio[0] > 0.5, f"odd/even={ratio[0]:.4f}")
    check("theta=45deg is not a comb", ratio[1] > 0.5, f"odd/even={ratio[1]:.4f}")
    check("theta=90deg is not a comb", ratio[2] > 0.5, f"odd/even={ratio[2]:.4f}")

    # ---- 5. fbp is consistent at both samplings ----------------------------
    print("5. fbp round trip")
    u = rand_obj(nz, nc, 6).real.astype('complex64')
    rels = {}
    for nd in (nc, n):
        cl = Tomo(nc, nz, theta, mask_r, nd=nd)
        d = cl.R(u * cp.asarray(cl.mask))
        rels[nd] = float(
            cp.linalg.norm(cl.R(cl.fbp(d, 'ramp')) - d) / cp.linalg.norm(d))
    check(f"nd={nc} (nd == n) unchanged", abs(rels[nc] - 0.341) < 0.05,
          f"||R fbp(d) - d||/||d|| = {rels[nc]:.4f}")
    # Informational: the ramp filter runs out to |f| = nd/(2n), so nd = 2n
    # weights the outer band the nd = n case never sees.  With the skip guard
    # in place that band is signal rather than aliased replicas, and the two
    # round trips now agree to about 1%.  The BH solver never calls fbp;
    # step 5's initial guess does.
    print(f"     (informational) nd=2n: ||R fbp(d) - d||/||d|| = {rels[n]:.3f}, "
          f"against {rels[nc]:.3f} at nd=n")

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: " + ", ".join(FAILED))
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
