
from mpi4py import MPI
import numpy as np
from .utils import timer


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
    """
    Cache MPI derived datatypes for repeated redistributions between:

      forward:
        src: (ntheta, local_nzobj, nobj)  -> dst: (local_ntheta, nzobj, nobj)

      backward:
        src: (local_ntheta, nzobj, nobj) -> dst: (ntheta, local_nzobj, nobj)

    where:
      - nzobj is block-distributed (local_nzobj) across ranks in the "zobj-slab" layout
      - ntheta is block-distributed (local_ntheta) across ranks in the "theta-slab" layout
      - nobj is replicated everywhere
      - partitioning is defined by get_local_chunk()

    Only two lists of MPI datatypes are built (not four): forward-send and
    forward-recv types are reused as backward-recv and backward-send respectively,
    since the subarray layouts are identical by symmetry.
    """

    def __init__(self, comm: MPI.Comm, nzobj: int, ntheta: int, nobj: int, dtype):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.nzobj = int(nzobj)
        self.ntheta = int(ntheta)
        self.nobj = int(nobj)
        self.dtype = np.dtype(dtype)

        # local slabs
        self.st_obj, self.end_obj = get_local_chunk(self.nzobj, self.rank, self.size)
        self.st_theta, self.end_theta = get_local_chunk(self.ntheta, self.rank, self.size)
        self.local_nzobj = self.end_obj - self.st_obj
        self.local_ntheta = self.end_theta - self.st_theta

        # base MPI datatype
        try:
            self.base = MPI._typedict[self.dtype.char]
        except KeyError as e:
            raise TypeError(f"Unsupported dtype for direct MPI: {self.dtype}") from e

        # common arrays for Alltoallw (byte displacements)
        self.sendcounts = np.ones(self.size, dtype=np.int32)
        self.recvcounts = np.ones(self.size, dtype=np.int32)
        self.sdispls = np.zeros(self.size, dtype=np.int32)
        self.rdispls = np.zeros(self.size, dtype=np.int32)

        # Two type lists (forward send == backward recv, forward recv == backward send)
        self._types_theta = None  # subarray over the ntheta axis — fwd send / bwd recv
        self._types_zobj = None   # subarray over the nzobj axis — fwd recv / bwd send

        self._build_types()

    def _build_types(self):
        src_shape_fwd = (self.ntheta, self.local_nzobj, self.nobj)
        dst_shape_fwd = (self.local_ntheta, self.nzobj, self.nobj)

        types_theta = []
        types_zobj = []

        for p in range(self.size):
            ts, te = get_local_chunk(self.ntheta, p, self.size)
            zs, ze = get_local_chunk(self.nzobj, p, self.size)

            # Subarray over the ntheta axis: used as fwd-send-to-p / bwd-recv-from-p
            t_theta = self.base.Create_subarray(
                sizes=src_shape_fwd,
                subsizes=(te - ts, self.local_nzobj, self.nobj),
                starts=(ts, 0, 0),
                order=MPI.ORDER_C,
            )
            t_theta.Commit()
            types_theta.append(t_theta)

            # Subarray over the nzobj axis: used as fwd-recv-from-p / bwd-send-to-p
            t_zobj = self.base.Create_subarray(
                sizes=dst_shape_fwd,
                subsizes=(self.local_ntheta, ze - zs, self.nobj),
                starts=(0, zs, 0),
                order=MPI.ORDER_C,
            )
            t_zobj.Commit()
            types_zobj.append(t_zobj)

        self._types_theta = types_theta
        self._types_zobj = types_zobj

    def close(self):
        """Free committed MPI datatypes."""
        for lst in (self._types_theta, self._types_zobj):
            if lst is None:
                continue
            for t in lst:
                t.Free()
        self._types_theta = None
        self._types_zobj = None

    def __del__(self):
        # best-effort cleanup; explicit close() is preferred.
        # Guard against being called after MPI.Finalize().
        try:
            if not MPI.Is_finalized():
                self.close()
        except Exception:
            pass

    def forward(self, src: np.ndarray, dst: np.ndarray):
        """
        src: (ntheta, local_nzobj, nobj) -> dst: (local_ntheta, nzobj, nobj)
        """
        if src.dtype != self.dtype or dst.dtype != self.dtype:
            raise ValueError("dtype mismatch")
        if src.shape != (self.ntheta, self.local_nzobj, self.nobj):
            raise ValueError(f"src.shape={src.shape} expected {(self.ntheta, self.local_nzobj, self.nobj)}")
        if dst.shape != (self.local_ntheta, self.nzobj, self.nobj):
            raise ValueError(f"dst.shape={dst.shape} expected {(self.local_ntheta, self.nzobj, self.nobj)}")

        self.comm.Alltoallw(
            [src, self.sendcounts, self.sdispls, self._types_theta],
            [dst, self.recvcounts, self.rdispls, self._types_zobj],
        )

    def backward(self, src: np.ndarray, dst: np.ndarray):
        """
        src: (local_ntheta, nzobj, nobj) -> dst: (ntheta, local_nzobj, nobj)
        """
        if src.dtype != self.dtype or dst.dtype != self.dtype:
            raise ValueError("dtype mismatch")
        if src.shape != (self.local_ntheta, self.nzobj, self.nobj):
            raise ValueError(f"src.shape={src.shape} expected {(self.local_ntheta, self.nzobj, self.nobj)}")
        if dst.shape != (self.ntheta, self.local_nzobj, self.nobj):
            raise ValueError(f"dst.shape={dst.shape} expected {(self.ntheta, self.local_nzobj, self.nobj)}")

        self.comm.Alltoallw(
            [src, self.sendcounts, self.sdispls, self._types_zobj],
            [dst, self.recvcounts, self.rdispls, self._types_theta],
        )

    @timer
    def redist(self, src: np.ndarray, dst: np.ndarray, direction="forward"):
        if direction == "forward":
            return self.forward(src, dst)
        elif direction == "backward":
            return self.backward(src, dst)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")
    @timer
    def allreduce(self, arr):
        self.comm.Allreduce(MPI.IN_PLACE, arr, op=MPI.SUM)
        return arr

    @timer
    def allreduce2(self, a, b):
        """Sum-reduce two scalars across ranks in a single MPI call."""
        buf = np.array([a, b], dtype='float64')
        self.comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
        return float(buf[0]), float(buf[1])

    @timer
    def allreduce_scalars(self, *vals):
        """Sum-reduce any number of scalars across ranks in a single MPI call.

        float64 throughout: the caller combines these into a Schur complement
        (see Rec._compute_step_fused), which cancels significant digits."""
        buf = np.array(vals, dtype='float64')
        self.comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
        return tuple(float(v) for v in buf)
