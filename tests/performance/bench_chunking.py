#!/usr/bin/env python
"""Is Chunking.run keeping the three streams busy?

    PYTHONPATH=<repo>/src python tests/performance/bench_chunking.py
    ... --n 1024 --size 32 --chunk 4 --work 3

Chunking.run overlaps H2D(k), compute(k-1) and D2H(k-2) on three non-blocking
streams, two buffers deep.  The best it can do is the slowest single stage;
the worst is their sum.  This reports where it actually lands.

All three stages of chunk j run on stream[j % 3], so h2d -> compute -> d2h is
stream-ordered for free.  The loop used to end every step with a host
synchronize() on all three streams, which drained the device once per chunk;
replacing it with events was worth 1.1-1.4x here, most at large nchunk where
the per-step cost is paid most often.  Efficiency below ~50% means the barrier
is back or the host buffers are not pinned -- an unpinned D2H is synchronous
however it is issued, and the overlap collapses.
"""
import argparse
import time

import numpy as np
import cupy as cp

from holotomocupy.chunking import Chunking
from holotomocupy.utils import make_pinned


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=512, help='frame size')
    p.add_argument('--size', type=int, default=64, help='frames in total')
    p.add_argument('--chunk', type=int, default=8, help='frames per chunk')
    p.add_argument('--work', type=int, default=3, help='fft pairs per chunk')
    p.add_argument('--reps', type=int, default=3)
    return p.parse_args()


def main():
    a = parse()
    n, size, chunk = a.n, a.size, a.chunk

    def kernel(cl, out, inp):
        t = inp
        for _ in range(a.work):
            t = cp.fft.ifft2(cp.fft.fft2(t))
        out[:] = t

    rng = np.random.default_rng(0)
    inp = make_pinned((size, n, n), 'complex64')
    inp[:] = (rng.standard_normal((size, n, n))
              + 1j * rng.standard_normal((size, n, n))).astype('complex64')
    out = make_pinned((size, n, n), 'complex64')
    out[:] = 0

    cl = Chunking(4 * chunk * n * n * 8 + 4096, chunk)

    @cl.gpu_batch(axis_out=0, axis_inp=0, nout=1)
    def run_all(cl, out, inp):
        kernel(cl, out, inp)

    def timeit(fn, reps):
        fn()
        cp.cuda.Device().synchronize()
        ts = []
        for _ in range(reps):
            cp.cuda.Device().synchronize()
            t = time.perf_counter()
            fn()
            cp.cuda.Device().synchronize()
            ts.append(time.perf_counter() - t)
        return min(ts) * 1e3

    got = timeit(lambda: run_all(cl, out, inp), a.reps)

    # the same stages in isolation, summed over the chunks
    g_in = cp.empty((chunk, n, n), 'complex64')
    g_out = cp.empty((chunk, n, n), 'complex64')
    nck = int(np.ceil(size / chunk))

    def one(fn, reps=10):
        fn()
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        cp.cuda.Device().synchronize()
        return (time.perf_counter() - t) / reps * 1e3

    h2d = one(lambda: g_in.set(inp[0:chunk])) * nck
    cmp_ = one(lambda: kernel(None, g_out, g_in)) * nck
    d2h = one(lambda: g_out.get(out=out[0:chunk], blocking=False)) * nck
    ideal, serial = max(h2d, cmp_, d2h), h2d + cmp_ + d2h

    print(f'n={n} size={size} chunk={chunk} nchunk={nck} work={a.work} fft pairs')
    print(f'  h2d {h2d:.1f} + compute {cmp_:.1f} + d2h {d2h:.1f} ms')
    print(f'  serial (no overlap)  {serial:8.1f} ms')
    print(f'  ideal (slowest stage){ideal:8.1f} ms')
    print(f'  Chunking.run         {got:8.1f} ms')
    print(f'  overlap efficiency   {ideal / got:7.1%}   '
          f'(1.0 = perfectly pipelined, {ideal / serial:.0%} = no overlap)')


if __name__ == '__main__':
    main()
