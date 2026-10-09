"""
test_redist.py — timing test for MPIClass.redist

Run with:
    mpirun -np <nranks> python test_redist.py
"""

import time
import numpy as np
from mpi4py import MPI


def get_local_chunk(total_size, rank, size):
    chunk_size = total_size // size
    remainder = total_size % size
    if rank < remainder:
        start_idx = rank * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
        end_idx = start_idx + chunk_size
    return start_idx, end_idx


class MPIClass:
    def __init__(self, comm, nzobj, ntheta, nobj, dtype):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.nzobj = int(nzobj)
        self.ntheta = int(ntheta)
        self.nobj = int(nobj)
        self.dtype = np.dtype(dtype)

        self.st_obj, self.end_obj = get_local_chunk(self.nzobj, self.rank, self.size)
        self.st_theta, self.end_theta = get_local_chunk(self.ntheta, self.rank, self.size)
        self.local_nzobj = self.end_obj - self.st_obj
        self.local_ntheta = self.end_theta - self.st_theta

        self._chunks_theta = [get_local_chunk(self.ntheta, p, self.size) for p in range(self.size)]
        self._chunks_zobj = [get_local_chunk(self.nzobj, p, self.size) for p in range(self.size)]

        base = MPI._typedict[self.dtype.char]
        # fwd-send / bwd-recv: slice rank p's theta rows from (ntheta, local_nzobj, nobj)
        self._types_theta = [
            base.Create_subarray(
                [self.ntheta, self.local_nzobj, self.nobj],
                [te - ts, self.local_nzobj, self.nobj],
                [ts, 0, 0],
            ).Commit()
            for ts, te in self._chunks_theta
        ]
        # fwd-recv / bwd-send: slice rank p's zobj columns from (local_ntheta, nzobj, nobj)
        self._types_zobj = [
            base.Create_subarray(
                [self.local_ntheta, self.nzobj, self.nobj],
                [self.local_ntheta, ze - zs, self.nobj],
                [0, zs, 0],
            ).Commit()
            for zs, ze in self._chunks_zobj
        ]
        self._ones  = [1] * self.size
        self._zeros = [0] * self.size

    def redist(self, src, dst, direction="forward"):
        """
        Redistribute a 3D array between zobj-slab and theta-slab layouts via MPI_Alltoallw.

        forward:  src (ntheta, local_nzobj, nobj) -> dst (local_ntheta, nzobj, nobj)
        backward: src (local_ntheta, nzobj, nobj) -> dst (ntheta, local_nzobj, nobj)
        """
        if direction == "forward":
            self.comm.Alltoallw(
                (src, self._ones, self._zeros, self._types_theta),
                (dst, self._ones, self._zeros, self._types_zobj),
            )
        elif direction == "backward":
            self.comm.Alltoallw(
                (src, self._ones, self._zeros, self._types_zobj),
                (dst, self._ones, self._zeros, self._types_theta),
            )
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

    def close(self):
        for t in self._types_theta + self._types_zobj:
            t.Free()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── Parameters ───────────────────────────────────────────────────────────────
nzobj  = 1024
ntheta = 1500
nobj   = 1024
dtype  = np.complex64
NREP   = 3
WARM   = 1

# ── Setup ────────────────────────────────────────────────────────────────────
cl = MPIClass(comm, nzobj, ntheta, nobj, dtype)

st_obj, end_obj = get_local_chunk(nzobj, rank, size)
st_theta, end_theta = get_local_chunk(ntheta, rank, size)
local_nzobj = end_obj - st_obj
local_ntheta = end_theta - st_theta

src_fwd = np.zeros((ntheta, local_nzobj, nobj), dtype=dtype)
dst_fwd = np.zeros((local_ntheta, nzobj, nobj), dtype=dtype)
src_bwd = np.zeros((local_ntheta, nzobj, nobj), dtype=dtype)
dst_bwd = np.zeros((ntheta, local_nzobj, nobj), dtype=dtype)

def time_redist(src, dst, direction, nrep=NREP, warm=WARM):
    """Return (max_ms, bw_GBps) across ranks."""
    for _ in range(warm):
        cl.redist(src, dst, direction=direction)
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(nrep):
        cl.redist(src, dst, direction=direction)
    comm.Barrier()
    elapsed_ms = (time.perf_counter() - t0) / nrep * 1e3
    max_ms = float(comm.allreduce(elapsed_ms, op=MPI.MAX))
    total_bytes = float(comm.allreduce(src.nbytes + dst.nbytes, op=MPI.SUM))
    bw_GBps = total_bytes / (max_ms * 1e-3) / 1e9
    return max_ms, bw_GBps

# ── Timing ───────────────────────────────────────────────────────────────────
t_fwd, bw_fwd = time_redist(src_fwd, dst_fwd, direction="forward")
t_bwd, bw_bwd = time_redist(src_bwd, dst_bwd, direction="backward")

if rank == 0:
    print(f"nranks={size}, nzobj={nzobj}, ntheta={ntheta}, nobj={nobj}", flush=True)
    print(f"  forward  {t_fwd:.3f} ms/call  {bw_fwd:.2f} GB/s  ({NREP} reps, {WARM} warm-up)", flush=True)
    print(f"  backward {t_bwd:.3f} ms/call  {bw_bwd:.2f} GB/s  ({NREP} reps, {WARM} warm-up)", flush=True)

cl.close()


# Viktor: My results

# (holotomocupy) bash-5.1$ mpirun -np 4 python test_redist.py 
# nranks=4, nzobj=1024, ntheta=1500, nobj=1024
#   forward  838.736 ms/call  30.00 GB/s  (3 reps, 1 warm-up)
#   backward 805.518 ms/call  31.24 GB/s  (3 reps, 1 warm-up)
# (holotomocupy) bash-5.1$ mpirun -np 8 python test_redist.py 
# nranks=8, nzobj=1024, ntheta=1500, nobj=1024
#   forward  526.180 ms/call  47.83 GB/s  (3 reps, 1 warm-up)
#   backward 502.221 ms/call  50.11 GB/s  (3 reps, 1 warm-up)
# (holotomocupy) bash-5.1$ mpirun -np 16 python test_redist.py 
# nranks=16, nzobj=1024, ntheta=1500, nobj=1024
#   forward  350.391 ms/call  71.82 GB/s  (3 reps, 1 warm-up)
#   backward 339.156 ms/call  74.20 GB/s  (3 reps, 1 warm-up)