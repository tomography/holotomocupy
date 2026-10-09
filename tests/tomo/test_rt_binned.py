#!/usr/bin/env python
"""Correctness and speed of Tomo's binned adjoint Radon.

    PYTHONPATH=../../src python test_rt_binned.py            # default sizes
    PYTHONPATH=../../src python test_rt_binned.py --quick    # small, fast

One GPU, no MPI, no data files.  Every check prints PASS/FAIL and the script
exits non-zero if any of them failed.

RT scatters polar samples back into the padded Cartesian spectrum with a
tile-binned shared-memory kernel.  There is no second implementation to
compare it against, so the check of record is the adjoint identity
<R u, d> == <u, RT d>: R and RT share no code, so a wrong d_ax phase, a
dropped sample, a mis-clipped stencil or a tile written to the wrong place
all break it.  Three geometries matter:

  * nd == n, the usual sampling;
  * nd == 2n, where the out-of-square guard drops samples -- RT's index must
    drop exactly the same ones;
  * n not a multiple of the tile size, where the last tile row and column hang
    off the grid and get clipped.

Then CG, because a single-call bound does not settle whether feeding RT back
into a descent direction drifts, and a timing table.
"""
import argparse
import math
import sys
import time

import numpy as np
import cupy as cp

from holotomocupy.tomo import Tomo, TILE

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def rand_obj(nz, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.random((nz, n, n)) + 1j * rng.random((nz, n, n))
    return cp.asarray(a.astype('complex64'))


def rand_sino(ntheta, nz, nd, seed):
    rng = np.random.default_rng(seed)
    a = rng.random((ntheta, nz, nd)) + 1j * rng.random((ntheta, nz, nd))
    return cp.asarray(a.astype('complex64'))


def rel_l2(a, b):
    return float(cp.linalg.norm(a - b) / cp.linalg.norm(b))


def time_rt(cl, d, reps):
    """Median wall time of one RT, in ms."""
    cl.RT(d)                       # warm up (also builds the index)
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        cl.RT(d)
        cp.cuda.Device().synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


# ---------------------------------------------------------------- correctness

def correctness(n, nz, ntheta, nd, label):
    print(f"\n{label}: n={n} nd={nd} nz={nz} ntheta={ntheta}")
    theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
    cl = Tomo(n, nz, theta, mask_r=0.9, nd=nd)

    u = rand_obj(nz, n, 0)
    d = rand_sino(ntheta, nz, nd, 1)

    lhs = complex(cp.sum(cl.R(u) * cp.conj(d)))
    rhs = complex(cp.sum(u * cp.conj(cl.RT(d))))
    err = abs(lhs - rhs) / abs(lhs)
    check("adjoint identity", err < 1e-5,
          f"|<Ru,d>-<u,RTd>|/|<Ru,d>| = {err:.2e}")

    # RT's own run-to-run spread: the shared-memory accumulation order is not
    # fixed, so this is the floor any comparison can reach.  It also catches a
    # race that only shows up sometimes.
    spread = rel_l2(cl.RT(d), cl.RT(d))
    check("RT reproducible to float32 noise", spread < 1e-5,
          f"RT-vs-itself l2 {spread:.2e}")

    print(f"  index: {cl.bin_stats()}")
    del cl, u, d
    cp.get_default_memory_pool().free_all_blocks()


def adjoint_sweep(nz=3, ntheta=97):
    """<R u, d> == <u, RT d> across sizes, both samplings.

    What varies with n is how the padded 2n grid divides into tiles: whether
    the tile divides it, how wide the partial last tile is, and whether that
    stub is narrower than the 2m+1 stencil -- which is the case the build has
    to avoid, since a stub inside a stencil would be missed by sample_tiles.
    For n a multiple of 4 that happens exactly when n % 32 == 4 (2n % 64 == 8),
    so 36, 100 and 2500 are in the list on purpose; 2500 is a real nobj.

    ntheta is prime and not a multiple of the block size, also on purpose.
    """
    print(f"\n0. adjoint identity sweep: nz={nz} ntheta={ntheta}")
    print(f"  {'n':>6} {'tile':>5} {'stub':>5} | {'nd==n':>10} {'nd==2n':>10}")
    for n in (36, 48, 64, 100, 128, 156, 192, 252, 256, 316, 384, 500, 512,
              632, 1000, 1024, 1184, 2368, 2500):
        errs = []
        for nd in (n, 2 * n):
            theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
            cl = Tomo(n, nz, theta, mask_r=0.9, nd=nd)
            u = rand_obj(nz, n, 0)
            d = rand_sino(ntheta, nz, nd, 1)
            lhs = complex(cp.sum(cl.R(u) * cp.conj(d)))
            rhs = complex(cp.sum(u * cp.conj(cl.RT(d))))
            errs.append(abs(lhs - rhs) / abs(lhs))
            b = cl._bins['b']
            del cl, u, d
            cp.get_default_memory_pool().free_all_blocks()
        ok = max(errs) < 1e-5
        if not ok:
            FAILED.append(f"adjoint n={n}")
        stub = 2 * n % b or b
        print(f"  {n:6d} {b:5d} {stub:5d} | {errs[0]:10.2e} {errs[1]:10.2e}"
              f"   {'' if ok else '  <-- FAIL'}")
    check("adjoint identity at every size", not any(
        f.startswith('adjoint n=') for f in FAILED))


def cg_convergence(n, nz, ntheta, niter):
    """CG on a sharp phantom has to actually converge, and to the phantom.

    The adjoint identity is necessary but not sufficient: RT could be the exact
    adjoint of a *different* R' and still pass it.  Recovering a known object
    pins down that R and RT are the pair the solver thinks they are.
    """
    print(f"\n4. CG, {niter} iterations: n={n} nz={nz} ntheta={ntheta}")
    theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
    cl = Tomo(n, nz, theta, mask_r=0.9)

    # Sharp edges: smooth data gives the high frequencies no weight, and the
    # high frequencies are where a scatter bug would show.
    rng = np.random.default_rng(3)
    t = np.linspace(-1, 1, n)
    x, y = np.meshgrid(t, t)
    ph = np.zeros((nz, n, n), 'float32')
    for cx, cy, r in rng.random((6, 3)):
        ph += ((x - (2 * cx - 1) * .6)**2 + (y - (2 * cy - 1) * .6)**2
               < (.05 + .2 * r)**2)
    u0 = cp.asarray(ph.astype('complex64'))
    d  = cl.R(u0)

    rec = cl.rec_tomo(d, niter)
    res = float(cp.linalg.norm(cl.R(rec) - d) / cp.linalg.norm(d))
    sol = rel_l2(rec, u0)
    check("CG residual converges", res < 1e-3, f"||Ru-d||/||d|| = {res:.4e}")
    check("CG recovers the phantom", sol < 0.1, f"||u-u0||/||u0|| = {sol:.4e}")
    del cl, u0, d, rec
    cp.get_default_memory_pool().free_all_blocks()


# ------------------------------------------------------------------- timing

def timing(sizes, reps):
    """RT, and the scatter kernel alone -- the rest of RT is two FFTs."""
    print("\ntiming, median of %d, ms" % reps)
    print(f"  {'n':>6} {'nd':>6} {'nz':>4} {'ntheta':>7} | "
          f"{'RT':>8} {'kernel':>8} | "
          f"{'build':>6} {'idx MB':>7} {'bld MB':>7} | dupl  %comb")
    from holotomocupy.cuda_kernels import scatter_binned_kernel
    for (n, nz, ntheta, nd) in sizes:
        theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
        cl = Tomo(n, nz, theta, mask_r=0.9, nd=nd)
        d  = rand_sino(ntheta, nz, nd, 1)

        cl._build_bins()                   # warm up: the first one JITs
        cl._bins = None
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        held0 = pool.total_bytes()
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        cl._build_bins()
        cp.cuda.Device().synchronize()
        t_build = (time.perf_counter() - t0) * 1e3
        # The pool never shrinks on its own, so what it holds now is the
        # build's high-water mark.
        mb_peak = (pool.total_bytes() - held0) / 2**20
        z = cl._bins
        mb_idx = sum(a.nbytes for a in (z['samples'], z['tile'], z['beg'],
                                        z['end'], z['atomic'])) / 2**20
        t_rt = time_rt(cl, d, reps)

        m, mua = cl.pars[0], cl.pars[1]
        fn = lambda: scatter_binned_kernel(
            (z['nsub'], nz), (256,),
            (cl._buf_fde, cl._buf_sino, cl.theta, z['samples'], z['tile'],
             z['beg'], z['end'], z['atomic'], m, mua, n, nd, ntheta, nz,
             z['b'], z['ntx']), shared_mem=z['shmem'])
        fn()
        cp.cuda.Device().synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            cp.cuda.Device().synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        k = float(np.median(ts))

        print(f"  {n:6d} {nd:6d} {nz:4d} {ntheta:7d} | "
              f"{t_rt:8.2f} {k:8.2f} | "
              f"{t_build:6.0f} {mb_idx:7.0f} {mb_peak:7.0f} | "
              f"{z['counts'].sum() / (ntheta * nd):4.2f}x "
              f"{100 * float(cp.mean(z['atomic'])):4.0f}%")
        del cl, d
        cp.get_default_memory_pool().free_all_blocks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='small sizes only')
    ap.add_argument('--reps', type=int, default=5)
    args = ap.parse_args()

    dev = cp.cuda.Device()
    print(f"GPU {dev.id}: {cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()}, "
          f"{dev.mem_info[1] / 2**30:.0f} GB, tile {TILE}")

    adjoint_sweep()
    correctness(64,  4, 32,  64,   "1. small, nd == n")
    correctness(256, 4, 400, 256,  "2. nd == n")
    correctness(128, 4, 200, 256,  "3. nd == 2n (out-of-square guard)")
    # Partial last tile: 252 leaves a 56-wide stub at tile 64; 100 would leave
    # an 8-wide one, so the build has to drop to a smaller tile instead.
    correctness(252, 4, 300, 252,  "4. partial tiles")
    correctness(100, 4, 200, 200,  "5. stub narrower than the stencil")
    cg_convergence(256, 4, 400, 32)

    if args.quick:
        sizes = [(256, 8, 1500, 256), (512, 8, 3000, 512)]
    else:
        sizes = [(256,  8,  1500,  256),
                 (512,  8,  3000,  512),
                 (1024, 8,  6000, 1024),
                 (2048, 4, 12000, 2048),
                 (1024, 4,  6000, 2048),   # tomo_upsample = 2
                 (2500, 4, 12000, 2500),   # stub narrower than the stencil
                 (2560, 4, 12000, 2560),   # the same scale, tile divides 2n
                 (4736, 2, 12000, 4736)]   # an AtomiumS1-sized nobj
    timing(sizes, args.reps)

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} check(s): " + ", ".join(FAILED))
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
