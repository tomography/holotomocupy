#!/usr/bin/env python
"""
Mosaic holotomography performance benchmark on fully-synthetic data.

Same idea as test.py, but the scan is a MOSAIC: a grid of ntile_v x ntile_h
tiles, each measured at ndist_tile propagation distances, flattened onto one
distance axis exactly the way tests/mosaic_brain/step6.py runs the real thing:

    ndist = ntiles * ndist_tile,   flat index = tile*ndist_tile + k
    z1    = tile(z1_tile, ntiles)

so rec_mpi.Rec sees ndist = ntiles*ndist_tile distances over one wide object
grid (the default 1 x 5 tiles x 4 distances gives ndist = 20), and
each tile's place on the mosaic lives in vars['pos'] (that is how
mosaic_reader.MosaicReader presents the tile files to the solver).

Nothing is read from disk and nothing is forward-modelled:

    prb   = 1                      (flat probe, as a from-scratch step6 starts)
    ref   = |D.prb|                (seeds the probe-fit regularizer)
    data  = random, positive       (intensity; values do not affect timing)
    pos   = tile offset + random per-angle encoder jitter
    obj   = 0                      (from-scratch start; no forward model at all)

Sizes follow the detector: at ndet = 2048 the scan has 6000 angles, and both
scale together with the binning level, so `--bin b` gives

    n = nz = 2048 >> b     ntheta = 6000 >> b     nobj, nzobj = mosaic >> b

The object grid is sized from the tile layout itself (tile pitch + the field of
view of the most-magnified distance + the jitter), so it always holds the whole
mosaic; --nobj / --nzobj override it.

Launch with:
    python test_mosaic.py --bin 3 --plan               # sizes + memory, no run
    mpirun -n 8 ./set_affinity_gpu.sh \
        python test_mosaic.py --bin 3 --nchunk 8 --log logmosaic_bin3
"""

import argparse
import os as _os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import cupy as cp
from mpi4py import MPI

from holotomocupy.rec_mpi       import Rec
from holotomocupy.logger_config import logger, set_log_level, add_file_handler

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

cp.cuda.set_pinned_memory_allocator(None)


# ── Reference acquisition, taken from tests/mosaic_brain/config_gen.conf ────
# (YY037A-like ID16A geometry; the perf numbers do not depend on the physics,
#  but keeping them identical makes this test comparable to that scan.)
NDET_REF        = 2048          # detector size the rest of the scan is quoted at
NTHETA_REF      = 6000          # angles at NDET_REF; scales with the detector
TILE_STEP_REF   = 2000          # nominal tile pitch, finest-grid object px
SHIFT_RAND_PX   = 30            # encoder jitter, uniform +-, DETECTOR px at bin 0
OBJ_GRAIN       = 1024          # object grid rounded up to this at bin 0

energy                  = 33.35                                     # keV
focustodetectordistance = 1.282                                     # m
detector_pixelsize_0    = 2.95203e-6                                # m, UNBINNED
z1_tile_all             = np.array([0.0434278, 0.0452908,
                                    0.0527430, 0.0682163])          # m, per tile


# ── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--bin',        type=int, required=True,
                help='binning level: n = nz = 2048>>bin, ntheta = 6000>>bin')
ap.add_argument('--nchunk',     type=int, default=4,  help='theta chunk size for batched ops')
ap.add_argument('--ndistchunk', type=int, default=0,
                help='distances sharing one upload of a theta chunk of proj '
                     '(0 = all ndist, the default; 1 = the old outer-distance loop)')
ap.add_argument('--ntile-h',    type=int, default=5,  help='tiles per row (default 5)')
ap.add_argument('--ntile-v',    type=int, default=1,  help='tile rows (default 1)')
ap.add_argument('--ndist-tile', type=int, default=4,  help='distances per tile (default 4)')
ap.add_argument('--niter',      type=int, default=1,  help='BH iterations (iter 0 is warmup)')
ap.add_argument('--ntheta',     type=int, default=0,  help='override the 6000>>bin angle count')
ap.add_argument('--nobj',       type=int, default=0,  help='override the mosaic width  (binned px)')
ap.add_argument('--nzobj',      type=int, default=0,  help='override the mosaic height (binned px)')
ap.add_argument('--plan', action='store_true',
                help='print sizes and the memory they need, then exit without allocating')
ap.add_argument('--nranks',     type=int, default=0,
                help='rank count to assume in --plan (default: the actual MPI size)')
ap.add_argument('--log',        type=str, default='perf_mosaic.log',
                help='log file path (rank 0; pass an empty string to disable file logging)')
args_cli = ap.parse_args()

bins       = args_cli.bin
nchunk     = args_cli.nchunk
ndistchunk = args_cli.ndistchunk
ntile_h    = args_cli.ntile_h
ntile_v    = args_cli.ntile_v
ndist_tile = args_cli.ndist_tile
niter      = args_cli.niter
log_path   = args_cli.log


# ── Derived sizes ───────────────────────────────────────────────────────────
n      = nz = NDET_REF >> bins
ntheta = args_cli.ntheta if args_cli.ntheta else NTHETA_REF >> bins
ntiles = ntile_v * ntile_h
ndist  = ntiles * ndist_tile
# 0 on the command line means "all of them"; resolve it here so the plan, the
# log header and Rec all quote the same number.
ndistchunk = min(ndistchunk, ndist) if ndistchunk > 0 else ndist

if ndist_tile > len(z1_tile_all):
    raise SystemExit(f'--ndist-tile {ndist_tile} > {len(z1_tile_all)} known distances')
z1_tile  = z1_tile_all[:ndist_tile]
z1       = np.tile(z1_tile, ntiles)                 # tile-major, as MosaicReader builds it
detector_pixelsize = detector_pixelsize_0 * 2**bins

mag_tile      = focustodetectordistance / z1_tile
norm_mag_tile = (mag_tile / mag_tile[0]).astype('float32')   # <= 1; smallest = furthest z1
voxelsize     = detector_pixelsize / mag_tile[0]

# Tile placement on the mosaic, finest-grid object px, row-major with row 0 on
# top and column 0 on the left — tests/mosaic_brain/make_geometry.py:build_tiles.
# The offsets are sample SHIFTS, so they run opposite to the object-grid axes.
_v_rows   = ((ntile_v - 1) / 2 - np.arange(ntile_v)) * TILE_STEP_REF
_h_cols   = ((ntile_h - 1) / 2 - np.arange(ntile_h)) * TILE_STEP_REF
tile_off0 = np.array([[_v_rows[r], _h_cols[c]]
                      for r in range(ntile_v) for c in range(ntile_h)], dtype='float32')
tile_names = [f'{r}_{c}' for r in range(ntile_v) for c in range(ntile_h)]

# Object grid: the shift kernel samples the object over half-extent
# 0.5*(ndet-1)/norm_mag for the least-magnified distance, so the grid must hold
# the outermost tile offset plus the jitter plus that half-FOV, both directions.
_half0   = 0.5 * (NDET_REF - 1) / norm_mag_tile.min()
_jit0    = SHIFT_RAND_PX / norm_mag_tile.min()
_reach_v = float(np.abs(tile_off0[:, 0]).max()) + _jit0 + _half0
_reach_h = float(np.abs(tile_off0[:, 1]).max()) + _jit0 + _half0
_ceil    = lambda x: int(np.ceil(2 * x / OBJ_GRAIN)) * OBJ_GRAIN
nzobj    = args_cli.nzobj if args_cli.nzobj else _ceil(_reach_v) >> bins
nobj     = args_cli.nobj  if args_cli.nobj  else _ceil(_reach_h) >> bins


# ── Fixed solver config (mirrors mosaic_brain/config_step6.conf) ────────────
rho             = [1, 0.05, 0.02, 0]
mask            = 1.2
lam_prbfit      = 3.1e-3
lam_laplacian   = 0
start_iter      = 0
checkpoint_step = -1                            # no disk I/O
error_step      = 1                             # iter markers for parse_perf_log.py
log_level       = 'DEBUG'
rotation_center_shift = 0


# ── MPI ─────────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
set_log_level(log_level)

if rank == 0 and log_path and not args_cli.plan:
    add_file_handler(log_path)
    logger.warning(f"writing log to {log_path}")


# ── Memory plan ─────────────────────────────────────────────────────────────
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
    # GPU: the chunking pool uses Rec._chunking_pool_bytes; the rest are the
    # persistent buffers of Tomo / Propagation / Shift / ref. cuFFT plan work
    # areas add roughly as much again, hence "approx".
    proj_chunk = nchunk * nzobj * nobj * item
    obj_chunk  = nchunk * nobj  * nobj * item
    data_chunk = nchunk * nz    * n    * 4
    # distances are hoisted inside the theta chunk (Rec._resolve_ndistchunk):
    # ndistchunk data/pos chunks are resident at once, plus the prb bundles.
    dist_chunk  = data_chunk + 3 * nchunk * 2 * 4 + nchunk * 4
    _pool       = lambda nd: int(2.1 * max(3 * proj_chunk + nd * dist_chunk, 3 * obj_chunk))
    gpu = {
        'chunking pool':                            _pool(ndistchunk),
        f'prb staging 3 x [{ndistchunk}, {nz}, {n}]': 3 * ndistchunk * nz * n * 8,
        f'tomo fde [{nchunk}, {2*nobj}, {2*nobj}]': nchunk * (2 * nobj)**2 * 8,
        f'tomo sino [{ntheta}, {nchunk}, {nobj}]':  ntheta * nchunk * nobj * 8,
        f'prop big [{nchunk}, {2*nz}, {2*n}]':      nchunk * (2 * nz) * (2 * n) * 8,
        f'shift plan [{nchunk}, {nzobj}, {nobj}]':  nchunk * nzobj * nobj * 8,
        f'ref [{ndist}, {nz}, {n}]':                ndist * nz * n * 4,
    }
    return host, gpu, sum(host.values()) / GiB, sum(gpu.values()) / GiB


if args_cli.plan:
    if rank == 0:
        nranks = args_cli.nranks if args_cli.nranks else size
        host, gpu, host_gb, gpu_gb = _plan(nranks)
        print(f'mosaic perf plan: bin={bins}  nranks={nranks}  nchunk={nchunk}  '
              f'ndistchunk={ndistchunk}')
        print(f'  tiles      : {ntile_v} x {ntile_h} = {ntiles}, {ndist_tile} distances each '
              f'-> ndist = {ndist}')
        print(f'  detector   : {nz} x {n}   angles: {ntheta}')
        print(f'  object     : {nzobj} x {nobj} x {nobj}   voxel {voxelsize*1e9:.3f} nm')
        print(f'  tile reach : v {_reach_v:.1f} / {(nzobj<<bins)/2:.1f}   '
              f'h {_reach_h:.1f} / {(nobj<<bins)/2:.1f}   (finest px, half-extent)')
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


# ── Machine info (rank 0 only — written to the perf log for reproducibility) ─
def _log_machine_info():
    """Log CPU model, total RAM, GPU model + memory, and GPU count."""
    cpu_model = 'unknown'
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
    except Exception:
        pass

    cpu_count = ram_gb = None
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        ram_gb    = psutil.virtual_memory().total / 1024**3
    except Exception:
        pass

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


def _physical_dev_id():
    """Physical GPU id of this rank, correct under one-GPU-per-rank pinning."""
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
    unique_gpus = sorted(set(rank_devices))
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
    # physics
    energy=energy,
    focustodetectordistance=focustodetectordistance,
    z1=z1,
    detector_pixelsize=detector_pixelsize,
    theta=np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32'),
    mask=mask,
    # MPI
    comm=comm,    
)

if rank == 0:
    _, _, host_gb, gpu_gb = _plan(size)
    logger.warning(f"perf-test: mosaic {ntile_v}x{ntile_h} tiles x {ndist_tile} dist "
                   f"-> ndist={ndist}  bin={bins}")
    logger.warning(f"perf-test: n={n} nz={nz} nobj={nobj} nzobj={nzobj} "
                   f"ntheta={ntheta} ndist={ndist} nchunk={nchunk} "
                   f"ndistchunk={ndistchunk} niter={niter} "
                   f"nranks={size}")
    logger.warning(f"perf-test: voxel={voxelsize*1e9:.3f} nm  "
                   f"tile_step={TILE_STEP_REF/2**bins:.1f} px  "
                   f"jitter={SHIFT_RAND_PX/2**bins:.2f} det px")
    logger.warning(f"perf-test: predicted pinned host {host_gb:.2f} GB/rank, "
                   f"GPU ~{gpu_gb:.2f} GB/rank")


# ── Build the reconstruction class ──────────────────────────────────────────
cl = Rec(args)


# ── Synthetic inputs ────────────────────────────────────────────────────────
# Reproducibility model (same as test.py): one MASTER_SEED, SeedSequence.spawn()
# per GLOBAL theta index, so the inputs depend only on the global index and not
# on how MPI splits the work.  Drawing a full random data array would be several
# hundred GB of RNG, so one random frame per distance is drawn once and
# modulated by a per-angle scalar: cheap, rank-count independent, and
# non-degenerate, which is all the solver needs since BH runs a fixed number of
# iterations regardless of the values.
MASTER_SEED = 20260525
_ss_root                  = np.random.SeedSequence(MASTER_SEED)
_ss_pos, _ss_dat = _ss_root.spawn(2)


def synth_pos(st, end, ntheta_global, ss):
    """[ndist, local_ntheta, 2] positions in BINNED object px, tile-major.

    tile offset (finest px) + per-angle encoder jitter (uniform +-SHIFT_RAND_PX
    detector px, divided by the distance's normalised magnification), then the
    same binning conversion mosaic_reader.MosaicReader.read_pos applies.
    """
    nl = end - st
    out = np.zeros([ndist, nl, 2], dtype='float32')
    if nl == 0:
        return out
    for j, sseed in enumerate(ss.spawn(ntheta_global)[st:end]):
        jit = np.random.default_rng(sseed).uniform(-SHIFT_RAND_PX, SHIFT_RAND_PX,
                                                   size=(ntiles, ndist_tile, 2))
        jit /= norm_mag_tile[None, :, None]                  # detector px -> finest object px
        out[:, j] = (tile_off0[:, None, :] + jit).reshape(ndist, 2)

    scale = np.float32(1.0 / 2**bins)
    out *= scale
    out[..., 1] += np.float32(rotation_center_shift * scale)
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
# from scratch, as a production step6 run does.  Nothing is
# forward-modelled, so there is no object to synthesize.
logger.info("synthesize positions")
cl.vars['pos'][:] = synth_pos(cl.st_theta, cl.end_theta, ntheta, _ss_pos)
cl.vars['prb'][:] = 1                                   # flat probe

logger.info("synthesize data")
synth_data(cl.data, cl.st_theta, cl.end_theta, ntheta, _ss_dat)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)     # ref = |D.prb| for prb = 1


# Where each tile's window lands on the object grid — the same check step6.py
# prints, so a placement mistake shows up before the first iteration.
if rank == 0 and cl.end_theta > cl.st_theta:
    pos0 = np.asarray(cl.vars['pos'][:, 0])
    logger.warning(f"perf-test: tile windows at angle 0 "
                   f"(x = (nobj-1)/2 - pos[1], nobj={nobj}):")
    for t, tl in enumerate(tile_names):
        o = tile_off0[t]
        p = pos0[t * ndist_tile]
        logger.warning(f"perf-test:   {tl:<6s} offset v={o[0]:+9.2f} h={o[1]:+10.2f} finest px"
                       f"  -> pos=({p[0]:+8.2f},{p[1]:+9.2f})"
                       f"  y={(nzobj-1)/2 - p[0]:8.2f}"
                       f"  x={(nobj-1)/2 - p[1]:9.2f}")


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
    logger.warning(f"BH done: {niter} iters in {elapsed:.3f} s ({per_iter*1e3:.1f} ms/iter)  "
                   f"[nranks={size}, bin={bins}, n={n}, nobj={nobj}, nzobj={nzobj}, "
                   f"ntheta={ntheta}, ndist={ndist}, nchunk={nchunk}]")


# ── Append the parsed summary to the log ───────────────────────────────────
# The log is now complete, so parse it and write the breakdown at its end --
# one file carries both the raw @timer lines and the summary of the iteration
# that matters (the last one, i.e. after the JIT warmup).
if rank == 0 and log_path:
    for _h in logger.handlers:
        _h.flush()
    try:
        from parse_perf_log import report
        with open(log_path, 'a') as _fh:
            _fh.write('\n')
            report(log_path, max(niter - 1, 0), out=_fh,
                   show_info=False, show_header=True)
    except Exception as e:                      # never let this kill a finished run
        logger.warning(f"could not append the parsed summary: {e}")
