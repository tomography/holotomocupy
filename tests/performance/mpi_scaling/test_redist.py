"""
Test correctness and timing of MPIClass.redist.

Run with:
    mpirun -n <nranks> python test_redist.py

Forward:  src (n_dst, local_n_src, nz) -> dst (local_n_dst, n_src, nz)
Backward: src (local_n_dst, n_src, nz) -> dst (n_dst, local_n_src, nz)

Correctness check:
  src[i, j, k] = (st_dst_global + i) * n_src + (st_src + j)
  After forward, dst[i, j, k] should equal (st_dst + i) * n_src + j  (global indices)
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
    def __init__(self, comm, n_src, n_dst, nz, dtype):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.n_src = int(n_src)
        self.n_dst = int(n_dst)
        self.nz    = int(nz)
        self.dtype = np.dtype(dtype)

        self.st_src, self.end_src = get_local_chunk(self.n_src, self.rank, self.size)
        self.st_dst, self.end_dst = get_local_chunk(self.n_dst, self.rank, self.size)
        self.local_n_src = self.end_src - self.st_src
        self.local_n_dst = self.end_dst - self.st_dst

        try:
            self._mpi_type = MPI._typedict[self.dtype.char]
        except KeyError as e:
            raise TypeError(f"Unsupported dtype: {self.dtype}") from e

        # Per-rank chunk boundaries
        st_src_all  = np.array([get_local_chunk(n_src, p, self.size)[0] for p in range(self.size)])
        end_src_all = np.array([get_local_chunk(n_src, p, self.size)[1] for p in range(self.size)])
        st_dst_all  = np.array([get_local_chunk(n_dst, p, self.size)[0] for p in range(self.size)])
        end_dst_all = np.array([get_local_chunk(n_dst, p, self.size)[1] for p in range(self.size)])
        local_n_src_all = end_src_all - st_src_all
        local_n_dst_all = end_dst_all - st_dst_all
        self._st_src_all  = st_src_all
        self._end_src_all = end_src_all
        self._st_dst_all  = st_dst_all

        nz = self.nz

        # Forward: src (n_dst, local_n_src, nz) -> dst (local_n_dst, n_src, nz)
        #   Send to p:   src[st_dst[p]:end_dst[p], :, :]  — contiguous in src
        #   Recv from p: (local_n_dst, local_n_src[p], nz) — into staging buf, then scatter
        self._fwd_sc = (local_n_dst_all * self.local_n_src * nz).astype(np.int32)
        self._fwd_sd = (st_dst_all      * self.local_n_src * nz).astype(np.int32)
        self._fwd_rc = (self.local_n_dst * local_n_src_all * nz).astype(np.int32)
        self._fwd_rd = np.zeros(self.size, np.int32)
        self._fwd_rd[1:] = np.cumsum(self._fwd_rc[:-1])

        # Backward: src (local_n_dst, n_src, nz) -> dst (n_dst, local_n_src, nz)
        #   Send to p:   src[:, st_src[p]:end_src[p], :] — non-contiguous, pack into staging buf
        #   Recv from p: (local_n_dst[p], local_n_src, nz) — contiguous in dst
        self._bwd_sc = (self.local_n_dst * local_n_src_all * nz).astype(np.int32)
        self._bwd_sd = np.zeros(self.size, np.int32)
        self._bwd_sd[1:] = np.cumsum(self._bwd_sc[:-1])
        self._bwd_rc = (local_n_dst_all * self.local_n_src * nz).astype(np.int32)
        self._bwd_rd = (st_dst_all      * self.local_n_src * nz).astype(np.int32)

        # Staging buffer shared by fwd recv and bwd send (same size)
        self._buf = np.empty(self.local_n_dst * self.n_src * nz, dtype=self.dtype)

        # Timing accumulators (reset by caller as needed)
        self.t_comm = 0.0
        self.t_pack = 0.0

    def forward(self, src, dst):
        # Send directly from src (slices along axis 0 are contiguous)
        t0 = time.perf_counter()
        self.comm.Alltoallv(
            [src, self._fwd_sc, self._fwd_sd, self._mpi_type],
            [self._buf, self._fwd_rc, self._fwd_rd, self._mpi_type],
        )
        self.t_comm += time.perf_counter() - t0
        # Scatter received chunks into dst[:, ss:se, :]
        t0 = time.perf_counter()
        for p in range(self.size):
            ss, se = int(self._st_src_all[p]), int(self._end_src_all[p])
            off, n = int(self._fwd_rd[p]), int(self._fwd_rc[p])
            dst[:, ss:se, :] = self._buf[off:off+n].reshape(self.local_n_dst, se-ss, self.nz)
        self.t_pack += time.perf_counter() - t0

    def backward(self, src, dst):
        # Pack non-contiguous slices src[:, ss:se, :] into staging buffer
        t0 = time.perf_counter()
        for p in range(self.size):
            ss, se = int(self._st_src_all[p]), int(self._end_src_all[p])
            off, n = int(self._bwd_sd[p]), int(self._bwd_sc[p])
            self._buf[off:off+n] = src[:, ss:se, :].ravel()
        self.t_pack += time.perf_counter() - t0
        # Recv directly into dst (slices along axis 0 are contiguous)
        t0 = time.perf_counter()
        self.comm.Alltoallv(
            [self._buf, self._bwd_sc, self._bwd_sd, self._mpi_type],
            [dst, self._bwd_rc, self._bwd_rd, self._mpi_type],
        )
        self.t_comm += time.perf_counter() - t0


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------- problem sizes (edit as needed) ----------
n_src = 816   # e.g. nzobj
n_dst = 1125   # e.g. ntheta
nz    = 816   # e.g. nobj
dtype = np.dtype('float32')
nwarmup = 1
nrepeat  = 2
# ----------------------------------------------------

cl = MPIClass(comm, n_src, n_dst, nz, dtype)
st_src, end_src = cl.st_src, cl.end_src
st_dst, end_dst = cl.st_dst, cl.end_dst
local_n_src = cl.local_n_src
local_n_dst = cl.local_n_dst

# --- allocate ---
src_fwd = np.empty((n_dst,      local_n_src, nz), dtype=dtype)
dst_fwd = np.empty((local_n_dst, n_src,      nz), dtype=dtype)
src_bwd = np.empty((local_n_dst, n_src,      nz), dtype=dtype)
dst_bwd = np.empty((n_dst,      local_n_src, nz), dtype=dtype)

# --- fill src for forward with global indices ---
# src_fwd[i, j, k] = i * n_src + (st_src + j)
idx_ndst = np.arange(n_dst,      dtype=dtype)[:, None, None]
idx_nsrc = np.arange(local_n_src, dtype=dtype)[None, :, None] + st_src
src_fwd[:] = idx_ndst * n_src + idx_nsrc

# ── correctness: forward ──────────────────────────────────────────────────────
cl.forward(src_fwd, dst_fwd)

# expected: dst_fwd[i, j, k] = (st_dst + i) * n_src + j
i_vals = np.arange(local_n_dst, dtype=dtype)[:, None, None] + st_dst
j_vals = np.arange(n_src,       dtype=dtype)[None, :, None]
expected_fwd = i_vals * n_src + j_vals
err_fwd = np.max(np.abs(dst_fwd - expected_fwd))
ok_fwd = err_fwd == 0.0
all_ok_fwd = comm.allreduce(ok_fwd, op=MPI.LAND)

# ── correctness: round-trip (forward then backward) ──────────────────────────
src_bwd[:] = dst_fwd
cl.backward(src_bwd, dst_bwd)
err_bwd = np.max(np.abs(dst_bwd - src_fwd))
ok_bwd = err_bwd == 0.0
all_ok_bwd = comm.allreduce(ok_bwd, op=MPI.LAND)

if rank == 0:
    status_fwd = "PASS" if all_ok_fwd else "FAIL"
    status_bwd = "PASS" if all_ok_bwd else "FAIL"
    print(f"Forward correctness : {status_fwd}  (max err = {err_fwd})")
    print(f"Backward round-trip : {status_bwd}  (max err = {err_bwd})")
    print()

# ── timing: forward ───────────────────────────────────────────────────────────
comm.Barrier()
for _ in range(nwarmup):
    cl.forward(src_fwd, dst_fwd)
comm.Barrier()

cl.t_comm = cl.t_pack = 0.0
t0 = time.perf_counter()
for _ in range(nrepeat):
    cl.forward(src_fwd, dst_fwd)
comm.Barrier()
t_fwd = (time.perf_counter() - t0) / nrepeat
t_fwd_comm = cl.t_comm / nrepeat
t_fwd_pack = cl.t_pack / nrepeat

# ── timing: backward ──────────────────────────────────────────────────────────
for _ in range(nwarmup):
    cl.backward(src_bwd, dst_bwd)
comm.Barrier()

cl.t_comm = cl.t_pack = 0.0
t0 = time.perf_counter()
for _ in range(nrepeat):
    cl.backward(src_bwd, dst_bwd)
comm.Barrier()
t_bwd = (time.perf_counter() - t0) / nrepeat
t_bwd_comm = cl.t_comm / nrepeat
t_bwd_pack = cl.t_pack / nrepeat

if rank == 0:
    nbytes = n_dst * local_n_src * nz * dtype.itemsize
    total_bytes = nbytes * size  # all ranks combined
    bw_fwd = total_bytes / t_fwd / 1e9
    bw_bwd = total_bytes / t_bwd / 1e9
    print(f"Sizes  : n_src={n_src}  n_dst={n_dst}  nz={nz}  ranks={size}  dtype={dtype}")
    print(f"Buffer : {nbytes / 1e6:.1f} MB per rank  ({total_bytes / 1e6:.1f} MB total)")
    print(f"Forward : {t_fwd*1e3:.2f} ms   ({bw_fwd:.2f} GB/s effective)")
    print(f"  alltoallv : {t_fwd_comm*1e3:.2f} ms   scatter: {t_fwd_pack*1e3:.2f} ms")
    print(f"Backward: {t_bwd*1e3:.2f} ms   ({bw_bwd:.2f} GB/s effective)")
    print(f"  pack      : {t_bwd_pack*1e3:.2f} ms   alltoallv: {t_bwd_comm*1e3:.2f} ms")
