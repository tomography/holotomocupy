"""Per-rank sanity report for check_nodes_mn.sh."""
import os, socket, numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
host = socket.gethostname().split('.')[0]
cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '-')

try:
    import cupy as cp
    d = cp.cuda.Device(0)
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    free, total = d.mem_info
    gpu = f"{name} {free/2**30:.0f}/{total/2**30:.0f} GiB free"
    cp.arange(1000).sum()          # touch the device
except Exception as e:                                   # noqa: BLE001
    gpu = f"cupy FAILED: {type(e).__name__}: {e}"

out = os.environ.get('CHECK_PATH', '/data3/vnikitin')
data3 = 'ok' if os.path.isdir(out) else 'MISSING'

# Collective over the interconnect: every rank must contribute.
a = np.full(1 << 20, 1.0)
t = MPI.Wtime()
comm.Allreduce(MPI.IN_PLACE, a, op=MPI.SUM)
t = MPI.Wtime() - t
ok = a[0] == size

lines = comm.gather(
    f"  rank {rank:>3}/{size}  {host:<6} CUDA_VISIBLE_DEVICES={cvd}  {out}={data3}  {gpu}",
    root=0)
if rank == 0:
    for line in sorted(lines):
        print(line, flush=True)
    print(f"  allreduce 8 MiB over {size} ranks: {t*1e3:.1f} ms  correct={ok}", flush=True)
    print("  OK" if ok else "  FAILED", flush=True)
