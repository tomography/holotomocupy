#!/usr/bin/env python
"""
Fused-vs-classic Hessian benchmark on fully-synthetic MPI data.

B(y,z) = <y, H(vars)*z> is symmetric, so the three cascade sweeps the classic
CG step runs per BH iteration

    compute_beta  : B(g,e), B(e,e)                       -> beta
    compute_alpha : etas <- beta*etas - g,  B(e_new,e_new) -> alpha

all follow from the three bilinear forms {B(g,g), B(g,e), B(e,e)} of ONE sweep
(Rec.hessian3 / hessian_cascade3):

    beta   = B(g,e) / B(e,e)
    bottom = beta^2*B(e,e) - 2*beta*B(g,e) + B(g,g)

This script measures what that is worth, and checks that it is exact.

It builds a real Rec on synthetic data (same recipe as test.py -- nothing is
forward-modelled) and drives it to a state where a CG step is meaningful (one
BH iteration, so etas is nonzero). Then, per --mode:

  accuracy  Run BOTH routes at the same states and compare every scalar they
            produce -- the three bilinear forms, beta, bottom, alpha. The old
            side really does call Rec.hessian three times; the new side calls
            Rec.hessian3 once. Sweep counters report the 3-vs-1.

  timing    Time the classic route (Rec.compute_beta + Rec.compute_alpha,
            fused_hessian off) against the fused route
            (Rec._compute_step_fused).

  both      (default) Accuracy first; the timing table is only ever printed
            for a fusion that agreed.

The two routes cannot be *timed* from the same state -- each one overwrites
etas -- so they run as two separate blocks of complete BH iterations
(compute_gradient / step / apply_step) and only the step is timed. That is
sound because the cost here is shape-driven, not value-driven: both routes push
the same pinned proj/obj slabs over PCIe the same number of times regardless of
what is in them. (The accuracy pass has no such problem; see its own note.)

nobj and ntheta scale with n by default, from the production n = 2048 geometry
(nobj = 3072, ntheta = 1800), so each size is a realistic reconstruction rather
than a detector-only sweep. Pass --nobj / --ntheta to pin them instead.

Launch with:
    mpirun -n <N> ./set_affinity_gpu.sh \
        python test_hessian_fused.py --n 2048 --nchunk 8

or sweep n = 512 / 1024 / 2048 with run_hessian.sh.
"""

import argparse
import os
import time
import numpy as np
import cupy as cp
from types import SimpleNamespace
from mpi4py import MPI

from holotomocupy.rec_mpi       import Rec
from holotomocupy.logger_config import logger, set_log_level, add_file_handler

cp.cuda.set_pinned_memory_allocator(None)


# -- CLI ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--n',      type=int, required=True, help='detector size (square: nz = n)')
ap.add_argument('--ntheta', type=int, default=0,
                help='number of projection angles (0 = 1800*n/2048, the production '
                     'scaling: angle count grows with detector size)')
ap.add_argument('--nobj',   type=int, default=0,
                help='object width/height (0 = 3072*n/2048, i.e. a 1.5x pad around '
                     'the detector at every n)')
ap.add_argument('--nzobj',  type=int, default=0, help='object depth (0 = nobj, cubic)')
ap.add_argument('--ndist',  type=int, default=4, help='number of propagation distances')
ap.add_argument('--nchunk', type=int, default=8, help='theta chunk size for batched ops')
ap.add_argument('--ndistchunk', type=int, default=0,
                help='distances sharing one upload of a theta chunk of proj '
                     '(0 = all ndist, the default; 1 = the old outer-distance loop)')
ap.add_argument('--nrep',   type=int, default=5,
                help='timed BH iterations per route (one warmup iteration, which '
                     'carries the CuPy JIT compile, is discarded on top). The report '
                     'quotes min and mean; prefer min -- a single rep sharing the GPU '
                     'with another job drags the mean but not the min.')
ap.add_argument('--lam-laplacian', type=float, default=0.0,
                help='biharmonic regularization weight; >0 also exercises '
                     'LaplacianTerm.hessian3 (costs 4 padded obj slabs of host RAM)')
ap.add_argument('--mode', choices=('accuracy', 'timing', 'both'), default='both',
                help="'accuracy' checks that the fused route reproduces the classic "
                     "one, 'timing' measures what it saves, 'both' (default) does "
                     "accuracy first and then times from the state it leaves")
ap.add_argument('--check-iters', type=int, default=3,
                help='BH iterations to check in accuracy mode. Each costs 6 cascade '
                     'sweeps -- 1 fused, plus the classic 3 and the 2 extra forms '
                     'that pin down which one disagrees -- against the 3 of a '
                     'normal iteration')
ap.add_argument('--rtol-forms', type=float, default=1e-4,
                help='tolerance on the three bilinear forms and on B(g,e) == B(e,g); '
                     'these are plain float32 reductions over the same chunk order, '
                     'so the agreement should be near machine precision')
ap.add_argument('--rtol-derived', type=float, default=1e-2,
                help='tolerance on beta, bottom and alpha. Looser on purpose: bottom '
                     'is a Schur complement of nearly-cancelling positive terms, and '
                     'the report prints the cancellation factor next to the error so '
                     'a large value here can be read rather than merely accepted')
ap.add_argument('--plan', action='store_true',
                help='print sizes and the memory they need, then exit without allocating')
ap.add_argument('--nranks', type=int, default=0,
                help='rank count to assume in --plan (default: the actual MPI size)')
ap.add_argument('--log-level', type=str, default='INFO',
                help="'INFO' (default) keeps @timer off the hot path; 'DEBUG' adds "
                     "the per-call cascade timings at a small measurement cost")
ap.add_argument('--log',    type=str, default='hessian.log',
                help='log file path (rank 0; pass an empty string to disable file logging)')
args_cli = ap.parse_args()

n          = args_cli.n
nz         = n
# Both scale with the detector: the reference point is the production n = 2048
# geometry (nobj = 3072, ntheta = 1800) from run.sh.
ntheta     = args_cli.ntheta or 1800 * n // 2048
nobj       = args_cli.nobj   or 3072 * n // 2048
nzobj      = args_cli.nzobj or nobj
ndist      = args_cli.ndist
nchunk     = args_cli.nchunk
ndistchunk = min(args_cli.ndistchunk, ndist) if args_cli.ndistchunk > 0 else ndist
nrep       = args_cli.nrep
log_path   = args_cli.log


# -- Fixed config (mirrors test.py, which mirrors config1.conf) --------------
rho             = [1, 0.05, 0.02, 0]
mask            = 1.1
lam_prbfit      = 1e-2
lam_laplacian   = args_cli.lam_laplacian
start_iter      = 0
# The benchmark drives BH by hand (compute_gradient / step / apply_step), so
# niter only has to be large enough that Rec never treats us as finished.
niter           = 2 * nrep + 8
checkpoint_step = -1
error_step      = -1                            # no cost computation in the hot loop
shift_type      = 'cubic'

energy                  = 17.1
detector_pixelsize      = 1.4760147601476e-6 * n / 4096
focustodetectordistance = 1.217
z1                      = np.linspace(5.10e-3, np.pi / 2 * 5.10e-3, ndist)


# -- MPI ---------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
set_log_level(args_cli.log_level)


# -- Memory plan (--plan): sizes only, no allocation, no GPU -----------------
def _local(total, r, nranks):
    """holotomocupy.mpi_functions.get_local_chunk -- size of rank r's slab."""
    q, rem = divmod(total, nranks)
    return q + (1 if r < rem else 0)


def _plan(nranks):
    """Pinned-host and (approximate) GPU bytes on the heaviest rank."""
    GiB  = 1024.0**3
    item = np.dtype('complex64').itemsize
    lnz  = _local(nzobj,  0, nranks)
    lth  = _local(ntheta, 0, nranks)
    # A padded obj slab (Laplacian on) carries 2 ghost rows per z-side.
    pad  = 4 if lam_laplacian > 0 else 0

    host = {
        f'obj  3 x [{lnz + pad}, {nobj}, {nobj}]':  3 * (lnz + pad) * nobj * nobj * item,
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
        print(f'hessian plan: n={n}  nranks={nranks}  nchunk={nchunk}  '
              f'ndistchunk={ndistchunk}  lam_laplacian={lam_laplacian}')
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


if rank == 0 and log_path:
    add_file_handler(log_path)
    logger.warning(f"writing log to {log_path}")


# -- Machine info (rank 0 only) ----------------------------------------------
def _log_machine_info():
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
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
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
    logger.warning(f"job: ranks={size} gpus_used={len(set(rank_devices))}")


# -- Assemble Rec args -------------------------------------------------------
args = SimpleNamespace(
    nz=nz, n=n, nzobj=nzobj, nobj=nobj,
    ntheta=ntheta, ndist=ndist,
    nchunk=nchunk, ndistchunk=ndistchunk, niter=niter, start_iter=start_iter,
    rho=rho,
    lam_prbfit=lam_prbfit, lam_laplacian=lam_laplacian,
    checkpoint_step=checkpoint_step, error_step=error_step,
    energy=energy,
    focustodetectordistance=focustodetectordistance,
    z1=z1,
    detector_pixelsize=detector_pixelsize,
    theta=np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32'),
    mask=mask,
    shift_type=shift_type,
    comm=comm,
)

if rank == 0:
    logger.warning(f"hessian-bench: n={n} nz={nz} nobj={nobj} nzobj={nzobj} "
                   f"ntheta={ntheta} ndist={ndist} nchunk={nchunk} "
                   f"ndistchunk={ndistchunk} nrep={nrep} "
                   f"lam_laplacian={lam_laplacian} nranks={size}")
cl = Rec(args)


# -- Synthetic inputs (identical recipe to test.py) --------------------------
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
    """Fill the pinned [ndist, local_ntheta, nz, n] intensity buffer."""
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


logger.info("synthesize positions")
cl.vars['pos'][:] = synth_pos(cl.st_theta, cl.end_theta, ntheta, _ss_pos)
cl.vars['prb'][:] = 1
logger.info("synthesize data")
synth_data(cl.data, cl.st_theta, cl.end_theta, ntheta, _ss_dat)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)


# -- Sweep counters ----------------------------------------------------------
# Both hessian() and hessian3() end in a .get(), so wrapping them needs no extra
# synchronize; the counts confirm 3 sweeps per classic step against 1 per fused
# step rather than taking the claim on trust.
_sweeps = {'hessian': 0, 'hessian3': 0}
_orig_hessian, _orig_hessian3 = cl.hessian, cl.hessian3


def _counted_hessian(*a, **kw):
    _sweeps['hessian'] += 1
    return _orig_hessian(*a, **kw)


def _counted_hessian3(*a, **kw):
    _sweeps['hessian3'] += 1
    return _orig_hessian3(*a, **kw)


cl.hessian, cl.hessian3 = _counted_hessian, _counted_hessian3


def _sweeps_snapshot():
    return dict(_sweeps)


def _sweeps_since(snap):
    return {k: _sweeps[k] - snap[k] for k in _sweeps}


# -- Timing helpers ----------------------------------------------------------
def _tic():
    """Barrier + device sync, so a rank cannot start its clock while another is
    still finishing the previous phase."""
    cp.cuda.Device().synchronize()
    comm.Barrier()
    return time.time()


def _toc(t0):
    """Wall time of the slowest rank -- the one the whole job waits on."""
    cp.cuda.Device().synchronize()
    return comm.allreduce(time.time() - t0, op=MPI.MAX)


# -- Drive BH to a state where a CG step is meaningful -----------------------
# etas is zero until the first step runs, and at i == start_iter both routes
# short-circuit to steepest descent (no B(g,e) / B(e,e) sweep at all), so every
# timed step below uses i = 1.
vars, grads, etas = cl.vars, cl.grads, cl.etas
cl.precalc(vars)
cl.compute_gradient(vars, grads)
alpha, _, _ = cl.compute_step(vars, grads, etas, 0)     # i == start_iter: etas <- -g
cl.apply_step(vars, etas, alpha)

I_STEP = 1                                              # any i > start_iter behaves alike


# -- Accuracy: three independent sweeps vs one fused sweep -------------------
# The old route really does run cl.hessian three times, so that is what the
# reference here does -- no reassembled sum, no fused kernel anywhere on the
# classic side:
#
#   sweep 1   Bge_c = B(g, e)            \  compute_beta
#   sweep 2   Qee_c = B(e, e)            /  beta_c = Bge_c / Qee_c
#             etas <- beta*etas - grads     _update_etas
#   sweep 3   bottom_c = B(e_new, e_new)    compute_alpha
#
# against ONE hessian3 sweep giving (Qgg, Bge, Qee) and the expansion
#   bottom_f = beta^2*Qee - 2*beta*Bge + Qgg,
# which is what bilinearity buys: B(e_new, e_new) with e_new = beta*e - g
# expands into the three forms of the pre-update pair, so the two extra sweeps
# and the post-update sweep are all redundant.
#
# The pieces are driven directly rather than through compute_beta /
# compute_alpha / _compute_step_fused because both routes mutate etas in place,
# and a second copy of etas['obj'] + etas['proj'] is ~90 GB/rank at the
# production size. Ordering the work as "all pre-update sweeps, one shared
# _update_etas, then the post-update sweep" needs no copy and keeps both routes
# on bit-identical inputs. The arithmetic is the same as in the two solver
# functions; the sweep counters below confirm 3 vs 1.
#
# Both routes are then handed the SAME beta, so the bottom/alpha rows isolate
# the expansion itself from any disagreement about beta (which gets its own row).
_DIRS = {'g': grads, 'e': etas}


class _LapSecondDirection:
    """Point LaplacianTerm's second direction at g instead of e for one call.

    `LaplacianTerm.hessian(dobj1)` computes B_lap(dobj1, e_pad[2:-2]); the second
    direction is baked in because the classic route only ever needs e there. The
    two diagnostic forms below whose second direction is g -- B(g,g) and B(e,g)
    -- are therefore unreachable through cl.hessian unless e_pad is repointed at
    the gradient's padded buffer for the duration of the call. `_DIRS['e']['obj']`
    is a view on the ORIGINAL e_pad, so the y argument is unaffected.

    A no-op when z is 'e', and when the Laplacian term is inactive.
    """

    def __init__(self, z):
        self.z     = z
        self.term  = getattr(cl, 'cl_lap_term', None)
        self.saved = None

    def __enter__(self):
        if self.term is not None and self.z == 'g':
            self.saved, self.term.e_pad = self.term.e_pad, self.term.g_pad
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            self.term.e_pad = self.saved
        return False


def classic_form(y, z):
    """One classic sweep: B(y, z) for the full functional, allreduced."""
    with _LapSecondDirection(z):
        return cl.allreduce_scalars(cl.hessian(vars, _DIRS[y], _DIRS[z]))[0]


def rel(got, ref):
    """|got - ref| / |ref|, falling back to the absolute error when ref is 0."""
    return abs(got - ref) / abs(ref) if ref else abs(got)


def accuracy_pass(nchk, rtol_forms, rtol_derived):
    """Run both routes at each of `nchk` BH iterations and compare every scalar."""
    worst, ok = {}, True
    counts = {}

    def check(name, new, old, rtol, extra=''):
        nonlocal ok
        r = rel(new, old)
        worst[name] = max(worst.get(name, 0.0), r)
        good = np.isfinite(r) and r <= rtol
        ok = ok and good
        if rank == 0:
            logger.info(f"  {name:<9s} fused={new:+.9e} classic={old:+.9e} "
                        f"rel={r:.3e} {'ok ' if good else 'FAIL'}{extra}")
        return r

    for it in range(1, nchk + 1):
        cl.compute_gradient(vars, grads)
        if rank == 0:
            logger.warning(f"accuracy iter {it}/{nchk}")

        # --- new route: one sweep -------------------------------------------
        snap = _sweeps_snapshot()
        qgg_f, bge_f, qee_f = cl.allreduce_scalars(*cl.hessian3(vars, grads, etas))
        n_new = _sweeps_since(snap)

        # --- old route, part 1: the two compute_beta sweeps ------------------
        snap = _sweeps_snapshot()
        bge_c = classic_form('g', 'e')
        qee_c = classic_form('e', 'e')

        # --- diagnostics: the third form, and the bilinear symmetry ----------
        # Neither is part of the old route (Qgg never appears there); they are
        # what tells you WHICH form is wrong if `bottom` disagrees, and whether
        # B(y,z) == B(z,y) -- the property the whole fusion rests on. Symmetry is
        # meaningful GLOBALLY only: per rank the Laplacian's two orderings differ
        # by a biharmonic flux across the slab boundary, which is exactly why
        # _compute_step_fused allreduces before expanding `bottom`.
        qgg_c = classic_form('g', 'g')
        beg_c = classic_form('e', 'g')

        check('B(g,e)', bge_f, bge_c, rtol_forms)
        check('B(e,e)', qee_f, qee_c, rtol_forms)
        check('B(g,g)', qgg_f, qgg_c, rtol_forms, '   [diagnostic, not in the old route]')
        check('symmetry', beg_c, bge_c, rtol_forms, '   [B(e,g) vs B(g,e), global]')

        # --- beta ------------------------------------------------------------
        # The fused route floors beta at 0 (Powell safeguard) and compute_beta
        # does not, so the comparison is against max(beta_classic, 0) and the
        # report says when that clamp actually fired. A clamp is a deliberate
        # behavioural difference between the routes, not an accuracy failure.
        beta_f  = max(bge_f / qee_f, 0.0) if qee_f else 0.0
        beta_c  = bge_c / qee_c if qee_c else 0.0
        check('beta', beta_f, max(beta_c, 0.0), rtol_derived,
              '   [Powell clamp fired: classic beta was negative]' if beta_c < 0 else '')

        # --- shared direction update ------------------------------------------
        # Identical in both routes (same call, same beta), so `top` is shared and
        # alpha differs only through `bottom`.
        top = cl.allreduce_scalars(cl._update_etas(grads, etas, beta_f))[0]

        # --- old route, part 3: the compute_alpha sweep ------------------------
        bottom_c = cl.allreduce_scalars(cl.hessian(vars, etas, etas))[0]
        n_old = _sweeps_since(snap)

        # --- new route: the expansion, no sweep --------------------------------
        bottom_f = beta_f * beta_f * qee_f - 2.0 * beta_f * bge_f + qgg_f
        # The three terms are large and nearly cancel; the achievable relative
        # error on the expansion is roughly the accumulation epsilon times this.
        scale  = abs(beta_f * beta_f * qee_f) + abs(2.0 * beta_f * bge_f) + abs(qgg_f)
        cancel = scale / abs(bottom_c) if bottom_c else float('inf')
        check('bottom', bottom_f, bottom_c, rtol_derived,
              f'   [cancellation x{cancel:.3g}]')

        alpha_f = top / bottom_f if bottom_f else float('nan')
        alpha_c = top / bottom_c if bottom_c else float('nan')
        check('alpha', alpha_f, alpha_c, rtol_derived)

        counts = {'old': n_old, 'new': n_new}
        if rank == 0:
            logger.info(f"  sweeps    classic hessian={n_old['hessian']} "
                        f"hessian3={n_old['hessian3']}   "
                        f"fused hessian={n_new['hessian']} "
                        f"hessian3={n_new['hessian3']}")

        # Step with the fused alpha and go round again, so later iterations are
        # checked at states the solver would actually reach.
        cl.apply_step(vars, etas, alpha_f)

    if rank == 0:
        # The diagnostic sweeps are counted too; subtract them to report the
        # 3-vs-1 the routes themselves use.
        old_sweeps = counts['old']['hessian'] - 2 if counts else 0
        new_sweeps = counts['new']['hessian3'] if counts else 0
        print()
        print(f"hessian accuracy: n={n} nobj={nobj} nzobj={nzobj} ntheta={ntheta} "
              f"ndist={ndist} lam_laplacian={lam_laplacian} nranks={size} "
              f"iters={nchk}")
        print(f"  sweeps per step: classic={old_sweeps} x hessian, "
              f"fused={new_sweeps} x hessian3")
        print(f"  {'quantity':<12s}{'worst rel':>12s}{'tol':>12s}   verdict")
        for name in ('B(g,e)', 'B(e,e)', 'B(g,g)', 'symmetry', 'beta', 'bottom', 'alpha'):
            if name not in worst:
                continue
            tol = rtol_derived if name in ('beta', 'bottom', 'alpha') else rtol_forms
            v = worst[name]
            print(f"  {name:<12s}{v:12.3e}{tol:12.1e}   "
                  f"{'PASS' if np.isfinite(v) and v <= tol else 'FAIL'}")
        print()
        print(f"ACCURACY n={n} nobj={nobj} ntheta={ntheta} ndist={ndist} "
              f"nranks={size} lam_laplacian={lam_laplacian} iters={nchk} "
              f"sweeps_old={old_sweeps} sweeps_new={new_sweeps} "
              + ' '.join(f"{k.translate(str.maketrans('', '', '(),'))}={v:.3e}"
                         for k, v in worst.items())
              + f" verdict={'PASS' if ok else 'FAIL'}")
    return ok


accuracy_ok = True
if args_cli.mode in ('accuracy', 'both'):
    if rank == 0:
        logger.warning(f"accuracy pass: {args_cli.check_iters} iteration(s)")
    accuracy_ok = accuracy_pass(args_cli.check_iters,
                                args_cli.rtol_forms, args_cli.rtol_derived)
    # Every rank sees the same allreduced scalars, so this should be unanimous;
    # reduce anyway so a rank-local surprise cannot exit 0.
    accuracy_ok = comm.allreduce(accuracy_ok, op=MPI.LAND)

if args_cli.mode == 'accuracy':
    raise SystemExit(0 if accuracy_ok else 1)


# -- Timed blocks ------------------------------------------------------------
def run_block(fused, nrep):
    """`nrep` timed BH iterations on one route, after one warmup iteration.

    Returns (grad, beta, alpha, step) lists of per-iteration wall times, in
    seconds, plus the sweep counts of a single timed step. On the fused route
    beta and alpha are not separable -- one sweep produces both -- so those two
    columns come back as NaN and only `step` is meaningful.
    """
    cl.fused_hessian = fused
    out = {'grad': [], 'beta': [], 'alpha': [], 'step': []}
    counts, snap = {'hessian': 0, 'hessian3': 0}, None
    for rep in range(nrep + 1):                        # rep 0 = warmup, discarded
        t0 = _tic()
        cl.compute_gradient(vars, grads)
        t_grad = _toc(t0)

        if rep == 1:
            snap = _sweeps_snapshot()

        if fused:
            t0 = _tic()
            alpha, _, _ = cl._compute_step_fused(vars, grads, etas, I_STEP)
            t_step = _toc(t0)
            t_beta = t_alpha = float('nan')
        else:
            t0 = _tic()
            beta = cl.compute_beta(vars, grads, etas, I_STEP)
            t_beta = _toc(t0)
            t0 = _tic()
            alpha, _, _ = cl.compute_alpha(vars, grads, etas, beta)
            t_alpha = _toc(t0)
            t_step = t_beta + t_alpha

        if rep == 1:
            counts = _sweeps_since(snap)

        cl.apply_step(vars, etas, alpha)

        if rep:                                        # skip the JIT-carrying warmup
            out['grad'].append(t_grad)
            out['beta'].append(t_beta)
            out['alpha'].append(t_alpha)
            out['step'].append(t_step)
        if rank == 0:
            tag = 'fused ' if fused else 'classic'
            kind = 'warmup' if rep == 0 else f'rep {rep}'
            logger.info(f"{tag} {kind}: grad={t_grad:.3f}s beta={t_beta:.3f}s "
                        f"alpha={t_alpha:.3f}s step={t_step:.3f}s")
    return out, counts


# Fused first, classic second: the routes cannot share a state (each overwrites
# etas), and running fused on the fresher state is the conservative order --
# any drift in the machine over the run inflates the classic numbers' *rival*,
# not the classic numbers themselves.
if rank == 0:
    logger.warning(f"timing the fused route ({nrep} reps + 1 warmup)")
res_new, cnt_new = run_block(True,  nrep)
if rank == 0:
    logger.warning(f"timing the classic route ({nrep} reps + 1 warmup)")
res_old, cnt_old = run_block(False, nrep)


# -- Report ------------------------------------------------------------------
if rank == 0:
    def stat(v):
        """(min, mean) over the reps; NaN-safe for the fused beta/alpha columns."""
        a = np.asarray(v, dtype='float64')
        if not np.isfinite(a).any():
            return float('nan'), float('nan')
        return float(np.nanmin(a)), float(np.nanmean(a))

    g_min,  g_mean  = stat(res_old['grad'] + res_new['grad'])
    ob_min, ob_mean = stat(res_old['beta'])
    oa_min, oa_mean = stat(res_old['alpha'])
    os_min, os_mean = stat(res_old['step'])
    ns_min, ns_mean = stat(res_new['step'])

    sw_old = cnt_old['hessian'] + cnt_old['hessian3']
    sw_new = cnt_new['hessian'] + cnt_new['hessian3']

    print()
    print(f"hessian fusion: n={n} nobj={nobj} nzobj={nzobj} ntheta={ntheta} "
          f"ndist={ndist} nchunk={nchunk} ndistchunk={ndistchunk} "
          f"lam_laplacian={lam_laplacian} nranks={size} nrep={nrep}")
    print(f"{'':<26s}{'min [s]':>12s}{'mean [s]':>12s}")
    print(f"  {'classic  beta':<24s}{ob_min:12.3f}{ob_mean:12.3f}"
          f"   (2 sweeps: B(g,e), B(e,e))")
    print(f"  {'classic  alpha':<24s}{oa_min:12.3f}{oa_mean:12.3f}"
          f"   (etas update + 1 sweep: B(e_new,e_new))")
    print(f"  {'classic  beta+alpha':<24s}{os_min:12.3f}{os_mean:12.3f}"
          f"   ({sw_old} sweep(s) total)")
    print(f"  {'fused    step':<24s}{ns_min:12.3f}{ns_mean:12.3f}"
          f"   ({sw_new} sweep(s) total: B(g,g), B(g,e), B(e,e))")
    print(f"  {'compute_gradient':<24s}{g_min:12.3f}{g_mean:12.3f}"
          f"   (unchanged by the fusion, for scale)")
    print()
    sp_step = os_min / ns_min if ns_min else float('nan')
    it_old, it_new = g_min + os_min, g_min + ns_min
    sp_iter = it_old / it_new if it_new else float('nan')
    print(f"  step speedup     : {sp_step:.2f}x   ({os_min:.3f} s -> {ns_min:.3f} s, "
          f"saving {os_min - ns_min:.3f} s per iteration)")
    print(f"  BH iter speedup  : {sp_iter:.2f}x   (gradient + step: "
          f"{it_old:.3f} s -> {it_new:.3f} s)")
    print(f"  step share of it : {os_min/it_old*100:.1f}% classic -> "
          f"{ns_min/it_new*100:.1f}% fused")
    print()
    # One machine-readable line per run, so run_hessian.sh can build the
    # cross-n table by grepping the logs of separate mpirun invocations.
    verdict = ('skipped' if args_cli.mode == 'timing'
               else 'PASS' if accuracy_ok else 'FAIL')
    print(f"RESULT n={n} nobj={nobj} ntheta={ntheta} ndist={ndist} nranks={size} "
          f"nchunk={nchunk} sweeps_old={sw_old} sweeps_new={sw_new} "
          f"beta_old={ob_min:.4f} alpha_old={oa_min:.4f} step_old={os_min:.4f} "
          f"step_new={ns_min:.4f} grad={g_min:.4f} "
          f"speedup_step={sp_step:.3f} speedup_iter={sp_iter:.3f} "
          f"accuracy={verdict}")

    if log_path:
        for _h in logger.handlers:
            _h.flush()

# A timing run that silently produced wrong numbers is worse than no run.
raise SystemExit(0 if accuracy_ok else 1)
