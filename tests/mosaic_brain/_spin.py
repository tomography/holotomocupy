"""Long-running MPI job that holds GPU memory -- used to test Ctrl-C behaviour."""
import os, socket, time
from mpi4py import MPI
import cupy as cp
c = MPI.COMM_WORLD
buf = cp.zeros(int(2e9 // 8), dtype=cp.float64)   # ~2 GB on the device
buf += 1
print(f"rank {c.rank} up on {socket.gethostname().split('.')[0]} pid={os.getpid()}", flush=True)
c.Barrier()
for i in range(600):
    buf += 1
    cp.cuda.Stream.null.synchronize()
    if c.rank == 0 and i % 10 == 0:
        print(f"iter {i}", flush=True)
    time.sleep(1)
