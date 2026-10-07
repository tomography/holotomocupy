#!/usr/bin/env python
"""
Multi-distance holotomography performance benchmark on fully-synthetic data.

Generates everything in-process (no Reader / h5 I/O): rotation angles,
positions, and a random sqrt(intensity) data array. Like test_mosaic.py,
nothing is forward-modelled -- there is no object to synthesize and no
Rec.gen_data pass. The reconstruction starts from scratch (zero object,
flat probe) exactly as a from-scratch production run does, then runs BH for
`--niter` iterations and prints a timing summary.

The point is the cost of one BH iteration at this size; BH runs a fixed number
of iterations regardless of the values, so random data times the same as real
data and costs a fraction of the memory and setup to produce.

Launch with:
    mpirun -n <N> python rec_iterative_mpi_syn.py --n 1024 --ntheta 64 --nchunk 4
"""

import argparse
import logging
import time
import numpy as np
import cupy as cp
from types import SimpleNamespace
from mpi4py import MPI

from holotomocupy.rec_mpi       import Rec
from holotomocupy.logger_config import logger, set_log_level, add_file_handler

cp.cuda.set_pinned_memory_allocator(None)


# ── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--n',      type=int, required=True, help='detector size (square: nz=n=n)')
ap.add_argument('--ntheta', type=int, required=True, help='number of projection angles')
ap.add_argument('--nchunk', type=int, default=4,     help='theta chunk size for batched ops (default 4)')
ap.add_argument('--ndistchunk', type=int, default=0,
                help='distances sharing one upload of a theta chunk of proj '
                     '(0 = all ndist, the default; 1 = the old outer-distance loop)')
ap.add_argument('--niter',  type=int, default=1,
                help='BH iterations; with 2+ the reported last iteration excludes '
                     'the CuPy JIT compile that lands in iteration 0')
ap.add_argument('--plan', action='store_true',
                help='print sizes and the memory they need, then exit without allocating')
ap.add_argument('--nranks', type=int, default=0,
                help='rank count to assume in --plan (default: the actual MPI size)')
ap.add_argument('--log',    type=str, default='perf.log',
                help='log file path (rank 0; pass an empty string to disable file logging)')
args_cli = ap.parse_args()

n       = args_cli.n
ntheta  = args_cli.ntheta
nchunk  = args_cli.nchunk
ndistchunk = args_cli.ndistchunk
log_path = args_cli.log


# ── Fixed perf-test config (mirrors config1.conf style) ─────────────────────
nz              = n
nobj            = 3264*n//2048                       # small padding around detector for tomo
nzobj           = nobj                          # cubic object volume
ndist           = 4
# 0 on the command line means "all of them"; resolve it here so the plan, the
# log header and Rec all quote the same number.
ndistchunk = min(ndistchunk, ndist) if ndistchunk > 0 else ndist
niter           = args_cli.niter                # short — measures steady-state per-iter cost
rho             = [1, 0.05, 0.02, 0]
mask            = 1.1
lam_prbfit      = 1e-2                           # disable probe-fit regularization
lam_laplacian   = 0
start_iter      = 0
checkpoint_step = -1                            # no disk I/O
error_step      = 1                            # no cost computation in hot loop
log_level       = 'DEBUG'
shift_type ='cubic'

# Physics (brain-Y350 style; values are illustrative — perf timing doesn't depend on them)
energy                  = 17.1
detector_pixelsize      = 1.4760147601476e-6 * n / 4096
focustodetectordistance = 1.217
z1                      = np.linspace(5.10e-3, np.pi/2*5.10e-3, ndist)


# ── MPI ─────────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
set_log_level(log_level)


# ── Memory plan (--plan): sizes only, no allocation, no GPU ─────────────────


def _local(total, r, nranks):
    """holotomocupy.mpi_functions.get_local_chunk — size of rank r's slab."""
    q, rem = divmod(total, nranks)
    return q + (1 if r < rem else 0)


def _plan(nranks):
    """Pinned-host and (approximate) GPU bytes on the heaviest rank."""
    GiB  = 1024.0**3
    item = np.dtype('complex64').itemsize
    lnz  = _local(nzobj,  0, nranks)
    lth  = _local(ntheta, 0, nranks)

    host = {
        f'obj  3 x [{lnz}, {nobj}, {nobj}]':        3 * lnz * nobj * nobj * item,
        f'proj 3 x [{lth}, {nzobj}, {nobj}]':       3 * lth * nzobj * nobj * item,
        f'proj_tmp [{ntheta}, {lnz}, {nobj}]':      ntheta * lnz * nobj * item,
        f'data [{ndist}, {lth}, {nz}, {n}]':        ndist * lth * nz * n * 4,
        'prb + pos (3 x each)':                     3 * ndist * nz * n * 8 + 4 * ndist * lth * 2 * 4,
    }
    proj_chunk = nchunk * nzobj * nobj * item
    obj_chunk  = nchunk * nobj  * nobj * item
    data_chunk = nchunk * nz    * n    * 4
    dist_chunk = data_chunk + 3 * nchunk * 2 * 4 + nchunk * 4
    _pool      = lambda nd: int(2.1 * max(3 * proj_chunk + nd * dist_chunk, 3 * obj_chunk))
    gpu = {
        'chunking pool':                             _pool(ndistchunk),
        f'prb staging 3 x [{ndistchunk}, {nz}, {n}]': 3 * ndistchunk * nz * n * 8,
        f'tomo fde [{nchunk}, {2*nobj}, {2*nobj}]':  nchunk * (2 * nobj)**2 * 8,
        f'tomo sino [{ntheta}, {nchunk}, {nobj}]':   ntheta * nchunk * nobj * 8,
        f'prop big [{nchunk}, {2*nz}, {2*n}]':       nchunk * (2 * nz) * (2 * n) * 8,
        f'shift plan [{nchunk}, {nzobj}, {nobj}]':   nchunk * nzobj * nobj * 8,
        f'ref [{ndist}, {nz}, {n}]':                 ndist * nz * n * 4,
    }
    return host, gpu, sum(host.values()) / GiB, sum(gpu.values()) / GiB


if args_cli.plan:
    if rank == 0:
        nranks = args_cli.nranks if args_cli.nranks else size
        host, gpu, host_gb, gpu_gb = _plan(nranks)
        print(f'perf plan: n={n}  nranks={nranks}  nchunk={nchunk}  '
              f'ndistchunk={ndistchunk}')
        print(f'  detector   : {nz} x {n}   angles: {ntheta}   distances: {ndist}')
        print(f'  object     : {nzobj} x {nobj} x {nobj}')
        print(f'  local slab : ntheta {_local(ntheta, 0, nranks)}  '
              f'nzobj {_local(nzobj, 0, nranks)}')
        print('  pinned host memory, heaviest rank:')
        for k, v in host.items():
            print(f'    {k:<42s} {v/1024**3:10.2f} GB')
        print(f'    {"TOTAL":<42s} {host_gb:10.2f} GB  '
              f'({host_gb*nranks:.2f} GB over {nranks} ranks)')
        print('  GPU memory per rank (approx, plan work areas not counted):')
        for k, v in gpu.items():
            print(f'    {k:<42s} {v/1024**3:10.2f} GB')
        print(f'    {"TOTAL":<42s} {gpu_gb:10.2f} GB')
    raise SystemExit


# Tee logger output to a file on rank 0 so we can parse timings/memory later.
if rank == 0 and log_path:
    add_file_handler(log_path)
    logger.warning(f"writing log to {log_path}")


# ── Machine info (rank 0 only — written to the perf log for reproducibility) ─


def _log_machine_info():
    """Log CPU model, total RAM, GPU model + memory, and GPU count."""
    # CPU model
    cpu_model = 'unknown'
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
    except Exception:
        pass

    # CPU count + total RAM (psutil — already a runtime dep of holotomocupy.utils)
    cpu_count = ram_gb = None
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        ram_gb    = psutil.virtual_memory().total / 1024**3
    except Exception:
        pass

    # GPU model, memory, and how many devices CUDA sees
    gpu_name = gpu_mem_gb = gpu_count = dev_id = None
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        dev_id    = cp.cuda.Device().id
        props     = cp.cuda.runtime.getDeviceProperties(dev_id)
        raw_name  = props.get('name', b'')
        gpu_name  = raw_name.decode() if isinstance(raw_name, (bytes, bytearray)) else str(raw_name)
        gpu_mem_gb = props.get('totalGlobalMem', 0) / 1024**3
    except Exception:
        pass

    logger.warning(f"machine: CPU={cpu_model} ({cpu_count} logical cores)")
    if ram_gb is not None:
        logger.warning(f"machine: RAM total={ram_gb:.1f} GB")
    if gpu_name is not None:
        logger.warning(f"machine: GPU={gpu_name}  memory={gpu_mem_gb:.1f} GB  "
                       f"count_visible={gpu_count}  (this rank uses dev {dev_id})")

# Gather (hostname, physical_device_id) from every rank so rank 0 can count
# actual GPUs in use. Note: when CUDA_VISIBLE_DEVICES pins one GPU per rank
# (the usual bind.sh layout) cp.cuda.Device().id always returns 0 because the
# pinned GPU is the only visible one — so we derive the *physical* device id
# from CUDA_VISIBLE_DEVICES instead.
import os as _os
def _physical_dev_id():
    cvd = _os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    local_id = cp.cuda.Device().id if cp.cuda.is_available() else -1
    if not cvd:
        return local_id
    parts = [p.strip() for p in cvd.split(',') if p.strip()]
    if 0 <= local_id < len(parts):
        try:    return int(parts[local_id])
        except ValueError: return parts[local_id]
    return local_id

rank_devices = comm.gather((MPI.Get_processor_name(), _physical_dev_id()), root=0)

if rank == 0:
    _log_machine_info()
    # Distinct (host, device) pairs == distinct physical GPUs across the job.
    unique_gpus = sorted(set(rank_devices))
    from collections import defaultdict
    per_host_ranks = defaultdict(int)
    per_host_gpus  = defaultdict(set)
    for h, d in rank_devices:
        per_host_ranks[h] += 1
        per_host_gpus[h].add(d)
    logger.warning(f"job: ranks={size} hosts={len(per_host_ranks)} gpus_used={len(unique_gpus)}")
    for h in sorted(per_host_ranks):
        devs = sorted(per_host_gpus[h])
        logger.warning(f"job: host={h}  ranks={per_host_ranks[h]}  "
                       f"gpus_used={len(devs)}  devs={devs}")
    for i, (h, d) in enumerate(rank_devices):
        logger.warning(f"job: rank={i}  host={h}  dev={d}")


# ── Assemble Rec args ───────────────────────────────────────────────────────
args = SimpleNamespace(
    # sizes
    nz=nz, n=n, nzobj=nzobj, nobj=nobj,
    ntheta=ntheta, ndist=ndist,
    nchunk=nchunk, ndistchunk=ndistchunk, niter=niter, start_iter=start_iter,
    # dtypes / regs
    rho=rho,
    lam_prbfit=lam_prbfit, lam_laplacian=lam_laplacian,
    # logging / I/O
    checkpoint_step=checkpoint_step, error_step=error_step,
    # physics (Rec also expects theta, mask, z1, ...)
    energy=energy,
    focustodetectordistance=focustodetectordistance,
    z1=z1,
    detector_pixelsize=detector_pixelsize,
    theta=np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32'),
    mask=mask,
    shift_type=shift_type,
    # MPI
    comm=comm,
)


# ── Build the reconstruction class ──────────────────────────────────────────
if rank == 0:
    logger.warning(f"perf-test: n={n} nz={nz} nobj={nobj} nzobj={nzobj} "
                   f"ntheta={ntheta} ndist={ndist} nchunk={nchunk} "
                   f"ndistchunk={ndistchunk} niter={niter} "
                   f"nranks={size}")
cl = Rec(args)


# ── Synthetic inputs ────────────────────────────────────────────────────────
# Reproducibility model (same as test_mosaic.py): one MASTER_SEED,
# SeedSequence.spawn() per GLOBAL theta index, so the inputs depend only on the
# global index and not on how MPI splits the work.  Random numbers are drawn on
# the CPU via numpy.default_rng — bit-stable across NumPy versions and machines
# (cp.random would not be).
#
# Nothing is forward-modelled.  Drawing a full random data array would be
# several hundred GB of RNG at the large sizes, so one random frame per distance
# is drawn once and modulated by a per-angle scalar: cheap, rank-count
# independent, and non-degenerate, which is all the solver needs.
MASTER_SEED = 20260525
_ss_root         = np.random.SeedSequence(MASTER_SEED)
_ss_pos, _ss_dat = _ss_root.spawn(2)


def synth_pos(st, end, ntheta_global, ss):
    """[ndist, local_ntheta, 2] positions for global theta range [st:end)."""
    nl = end - st
    if nl == 0:
        return np.empty((ndist, 0, 2), dtype='float32')
    out = np.empty((ndist, nl, 2), dtype='float32')
    for j, sseed in enumerate(ss.spawn(ntheta_global)[st:end]):
        out[:, j] = 10.0 * (np.random.default_rng(sseed).random((ndist, 2),
                                                                dtype='float32') - 0.5)
    return out


def synth_data(out, st, end, ntheta_global, ss):
    """Fill the pinned [ndist, local_ntheta, nz, n] intensity buffer.

    One random frame per distance, scaled by a per-(angle, distance) factor —
    a memory-bandwidth-bound fill rather than several hundred GB of RNG.
    """
    nl = end - st
    if nl == 0:
        return
    frames = np.empty((ndist, nz, n), dtype='float32')
    fr_rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED + 2))
    for k in range(ndist):
        frames[k] = 0.9 + 0.2 * fr_rng.random((nz, n), dtype='float32')
    for j, sseed in enumerate(ss.spawn(ntheta_global)[st:end]):
        s = np.random.default_rng(sseed).uniform(0.95, 1.05, ndist).astype('float32')
        for k in range(ndist):
            np.multiply(frames[k], s[k], out=out[k, j])


# vars['obj'] is left at the zero alloc_arrays made: the reconstruction starts
# from scratch.  Nothing is forward-modelled, so there is no object to
# synthesize and no probe to recover from.
logger.info("synthesize positions")
cl.vars['pos'][:] = synth_pos(cl.st_theta, cl.end_theta, ntheta, _ss_pos)
cl.vars['prb'][:] = 1                                   # flat probe

logger.info("synthesize data")
synth_data(cl.data, cl.st_theta, cl.end_theta, ntheta, _ss_dat)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)     # ref = |D.prb| for prb = 1


# ── Time BH ────────────────────────────────────────────────────────────────
comm.Barrier()
cp.cuda.Device().synchronize()
t0 = time.time()

cl.BH()

cp.cuda.Device().synchronize()
comm.Barrier()
elapsed = time.time() - t0


# ── Timing summary ──────────────────────────────────────────────────────────
if rank == 0:
    per_iter = elapsed / max(niter, 1)
    logger.warning(f"BH done: {niter} iters in {elapsed:.3f} s  "
                   f"[nranks={size}, n={n}, nobj={nobj}, ntheta={ntheta}, ndist={ndist}, nchunk={nchunk}]")


# ── Append the parsed summary to the log ───────────────────────────────────
# The log is now complete, so parse it and write the breakdown at its end --
# one file carries both the raw @timer lines and the summary of the iteration
# that matters (the last one, i.e. after the JIT warmup).
if rank == 0 and log_path:
    for _h in logger.handlers:
        _h.flush()
    try:
        import sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from parse_perf_log import report
        with open(log_path, 'a') as _fh:
            _fh.write('\n')
            report(log_path, max(niter - 1, 0), out=_fh,
                   show_info=False, show_header=True)
    except Exception as e:                      # never let this kill a finished run
        logger.warning(f"could not append the parsed summary: {e}")
