"""Peak detector-shaped temporaries in the F0 group, counted in units of one
float32 detector chunk.

    python tests/psf/bench_f0_mem.py

_estimate_gpu_mem charges the cascade candidate a fixed number of these, so the
number printed here is the one that belongs in that formula -- and, times 2.1
for double-buffering and slack, the GPU memory the chunking pool reserves.

Measured, not counted by eye: free the pool, call the method once, and read how
far the pool had to grow.  CuPy reuses freed blocks within the call, so the
growth IS the peak of simultaneously-live temporaries.

Both misfit models are measured.  They can differ: d2F_dF0 keeps J = K|x|^2 live
across the whole call (the amplitude curvature weight needs it, not just p), and
the amplitude kernels take two more sqrt-ing passes over it.  The formula has to
cover the larger of the two.
"""
import os
import sys

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from holotomocupy.rec_mpi import Rec                      # noqa: E402
from holotomocupy.psf import psf_taps                     # noqa: E402

CHUNK, NZ, N = 8, 512, 512


class Shim(Rec):
    """Just the F0 group: the attributes those methods actually read."""

    def __init__(self, sigma, model='intensity'):
        _, self.psf_w = psf_taps(sigma)
        self.model = model
        self._mask_y = cp.ones((CHUNK, NZ, 1), dtype='float32')
        self._mask_x = cp.ones((CHUNK, 1, N), dtype='float32')
        self.data_size = CHUNK * NZ * N
        self.apply_F_from = lambda v, i: v   # gF0's cascade step, identity here


def measure(fn, unit):
    mp = cp.get_default_memory_pool()
    fn()                                  # warm up: JIT, kernel cache
    cp.cuda.Stream.null.synchronize()
    mp.free_all_blocks()
    base = mp.total_bytes()
    r = fn()
    cp.cuda.Stream.null.synchronize()
    peak = (mp.total_bytes() - base) / unit
    del r
    return peak


def main():
    rnd = lambda: cp.asarray((np.random.rand(CHUNK, NZ, N)
                              + 1j * np.random.rand(CHUNK, NZ, N)).astype('complex64'))
    x, y, z, w = rnd(), rnd(), rnd(), rnd()
    d = cp.asarray(np.random.rand(CHUNK, NZ, N).astype('float32'))
    unit = d.nbytes

    print(f"chunk [{CHUNK}, {NZ}, {N}] float32 = {unit/2**20:.1f} MiB\n")
    print(f"{'method':14s} {'int noPSF':>10s} {'int PSF':>10s}"
          f" {'amp noPSF':>10s} {'amp PSF':>10s}   (units of one float32 chunk)")
    worst = 0.0
    for name in ('F0', 'dF0', 'd2F_dF0', 'd2F_dF0+w', 'gF0'):
        row = []
        for model in ('intensity', 'amplitude'):
            for sigma in (0.0, 1.8):
                s = Shim(sigma, model)
                call = {'F0':        lambda: s.F0(x, d),
                        'dF0':       lambda: s.dF0(x, y, d),
                        'd2F_dF0':   lambda: s.d2F_dF0(x, y, z, None, d),
                        'd2F_dF0+w': lambda: s.d2F_dF0(x, y, z, w, d),
                        'gF0':       lambda: s.gF0(x, d),
                        }[name]
                row.append(measure(call, unit))
        worst = max(worst, max(row))
        print(f"{name:14s} " + " ".join(f"{v:10.2f}" for v in row))
    print("\n(gF0's own complex64 result is 2 units and is an output, not a temporary.)")
    print(f"peak over all methods, models and sigmas: {worst:.2f} chunks"
          f"  ->  _estimate_gpu_mem should charge at least {int(np.ceil(worst))}")


if __name__ == '__main__':
    main()
