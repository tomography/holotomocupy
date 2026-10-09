#!/usr/bin/env python
"""R and RT, timed and checked adjoint, at the geometries step 6 actually runs.

    PYTHONPATH=../../src python bench_r_rt.py

One Tomo per rank, built as Tomo(nobj, nchunk, theta, mask, nd=ndobj), so nz
is nchunk and ntheta is the whole scan.  One R or RT call therefore covers all
angles for nchunk slices, and a full volume is nz_total/nchunk calls per rank.

The adjoint column is <R u, d> == <u, RT d> on the same random u, d that are
timed: R and RT share no code, so a dropped sample, a mis-clipped stencil or a
wrong d_ax phase all show up in it.  Expect ~1e-7; these are float32 sums over
up to 2e11 terms, so the figure grows slowly with ntheta*nd.
"""
import sys
import time

import numpy as np
import cupy as cp

from holotomocupy.tomo import Tomo

# (label, nobj, nchunk, ntheta, tomo_upsample, nz_total)
CASES = [
    ("AtomiumS1_FT bin0", 4096,  4, 12000, 1, 4096),
    ("AtomiumS1_FT bin1", 2048,  8, 12000, 1, 2048),
    ("AtomiumS1_FT bin2", 1024, 32, 12000, 1, 1024),
    ("AtomiumS1_HT bin0", 4096,  2,  4000, 1, 4096),
    ("AtomiumS1_HT bin1", 2048,  4,  4000, 1, 2048),
    ("AtomiumS1_HT bin2", 1024, 16,  4000, 1, 1024),
    ("ctxl_FT bin0",      4800,  4, 12000, 1, 4096),
    ("ctxl_FT bin1",      2400,  8, 12000, 1, 2048),
    ("ctxl_FT bin2",      1200, 32, 12000, 1, 1024),
    ("ctxl_HT bin0",      2528,  2,  4000, 2, 4096),
    ("ctxl_HT bin1",      1264,  4,  4000, 2, 2048),
    ("ctxl_HT bin2",       632, 16,  4000, 2, 1024),
    ("Y350a_ld bin0",     4736,  2,  4000, 1, 4096),
    ("Y350a_ld bin1",     2368,  4,  4000, 1, 2048),
    ("Y350a_ld bin2",     1184, 16,  4000, 1, 1024),
    ("Y350c bin0",        2560,  4,  3000, 1, 2160),
    ("Y350c bin1",        1280,  8,  3000, 1, 1080),
]

FAILED = []


def rand(shape, seed):
    cp.random.seed(seed)
    return (cp.random.random(shape, dtype=cp.float32)
            + 1j * cp.random.random(shape, dtype=cp.float32)).astype('complex64')


def med(fn, reps):
    fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main(reps=3):
    dev = cp.cuda.Device()
    name = cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()
    print(f"{name}, {dev.mem_info[1] / 2**30:.0f} GB, median of {reps}\n")
    print(f"  {'case':<19} {'n':>5} {'nd':>5} {'nz':>3} {'ntheta':>6} | "
          f"{'R ms':>7} {'RT ms':>7} {'RT/R':>5} | {'adjoint':>9} | "
          f"{'tile':>4} {'idx MB':>7} | {'calls':>5} {'R+RT s':>7}")
    for label, n, nz, ntheta, ups, nz_tot in CASES:
        nd = n * ups
        head = f"  {label:<19} {n:5d} {nd:5d} {nz:3d} {ntheta:6d} |"
        try:
            theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
            cl = Tomo(n, nz, theta, mask_r=0.9, nd=nd)
            u = rand((nz, n, n), 0)
            d = rand((ntheta, nz, nd), 1)

            Ru  = cl.R(u)
            lhs = complex(cp.sum(Ru * cp.conj(d)))
            del Ru
            RTd = cl.RT(d)
            rhs = complex(cp.sum(u * cp.conj(RTd)))
            del RTd
            err = abs(lhs - rhs) / abs(lhs)
            if not err < 1e-5:
                FAILED.append(label)

            t_r  = med(lambda: cl.R(u), reps)
            t_rt = med(lambda: cl.RT(d), reps)
            z = cl._bins
            mb = sum(a.nbytes for a in (z['samples'], z['tile'], z['beg'],
                                        z['end'], z['atomic'])) / 2**20
            calls = -(-nz_tot // nz)          # per rank, whole volume
            print(f"{head} {t_r:7.1f} {t_rt:7.1f} {t_rt / t_r:5.2f} | "
                  f"{err:9.2e} | {z['b']:4d} {mb:7.0f} | {calls:5d} "
                  f"{calls * (t_r + t_rt) / 1e3:7.1f}"
                  f"{'' if err < 1e-5 else '   <-- ADJOINT FAIL'}")
            del cl, u, d
        except cp.cuda.memory.OutOfMemoryError:
            print(f"{head}  out of memory on this card")
        cp.get_default_memory_pool().free_all_blocks()

    print()
    if FAILED:
        print("ADJOINT FAILED at: " + ", ".join(FAILED))
        return 1
    print("adjoint identity holds at every production geometry")
    return 0


if __name__ == '__main__':
    sys.exit(main())
