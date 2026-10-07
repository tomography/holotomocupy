#!/usr/bin/env python
"""Step 1 of the displacement study — generate synthetic holotomography data
with random sample displacements.

The phantom, the probe and the acquisition geometry are the ones from
../holotomo3d/test.py.  Three study parameters:

* `--amp` - the half-width (in detector pixels) of the uniform random
  displacement applied to the sample at every angle and every distance;
* `--prb-smooth` - the standard deviation (detector px) of the Gaussian blur
  applied to the ID16A probe, i.e. how much structure the illumination still
  has.  0 = the measured probe, the default 0.2251 px = the mild filter
  ../holotomo3d/test.py uses, several px = an almost flat beam.
* `--obj-smooth` - the same, in object voxels, for the phantom itself: how
  sharp the edges are that the reconstruction has to recover.  0 = the raw
  hard-edged shells, the default 0.2251 voxel = the same mild filter.  Has no
  effect with --obj-vol, which uses the volume as it is.

`--ndist 1` (the default) gives the single-distance case the study is about.

Run (one rank per GPU):

    mpirun -np 4 ./set_affinity_gpu.sh python gen_data.py --amp 16 --prb-smooth 2

Output: {out}/amp{amp}_ndist{ndist}[_prbs{s}][_objs{s}]/data.h5  (layout in common.py).
"""

import argparse
import os
import sys
import time

import numpy as np
import cupy as cp
import h5py
from mpi4py import MPI
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, _HERE)

# Pinned buffers here are big and long-lived; the cupy pinned pool only adds
# fragmentation on top of that (same reasoning as tests/mosaic_brain/gen_data.py).
cp.cuda.set_pinned_memory_allocator(None)

from holotomocupy.rec_mpi import Rec                              # noqa: E402
from holotomocupy.logger_config import logger, set_log_level      # noqa: E402

import common as C                                                # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--amp',    type=float, default=12.0,
                   help='half-width of the uniform random displacement [detector px]')
    p.add_argument('--prb-smooth', type=float, default=C.PRB_SMOOTH,
                   help='sigma [detector px] of the Gaussian blur applied to the probe; '
                        '0 = the measured probe, the default reproduces holotomo3d/test.py')
    p.add_argument('--obj-smooth', type=float, default=C.OBJ_SMOOTH,
                   help='sigma [object voxels] of the Gaussian blur applied to the phantom; '
                        '0 = hard edges, the default reproduces holotomo3d/test.py.  '
                        'Ignored with --obj-vol, which takes the volume as it is.')
    p.add_argument('--ndist',  type=int,   default=1,   help='number of propagation distances')
    p.add_argument('--n',      type=int,   default=512, help='detector size [px]')
    p.add_argument('--ntheta', type=int,   default=900, help='number of projection angles')
    p.add_argument('--nobj', type=int, default=0,
                   help='object grid in px, set directly; 0 = derive it from --nobj-factor')
    p.add_argument('--nobj-factor', type=float, default=0,
                   help='object grid as a multiple of n (nobj = round(factor*n/2)*2), used '
                        'only when --nobj is 0; 0 = the tight value n + 2*amp, the smallest '
                        'grid the sliding crop can read without touching mirrored edge data')
    p.add_argument('--delta',  type=float, default=1.0,  help='phantom delta scale')
    p.add_argument('--beta',   type=float, default=1e-2, help='phantom beta scale')
    p.add_argument('--obj-vol', default=None,
                   help='use a real sample volume instead of the synthetic phantom: '
                        'a path, or "path::dataset" for HDF5 (see common.open_volume). '
                        'The file holds the real part of the object directly; the '
                        'imaginary part follows from delta/beta.')
    p.add_argument('--obj-scale', type=float, default=1.0,
                   help='multiplies --obj-vol; the file carries arbitrary grey levels, '
                        'and this is what sets the projected phase excursion (the '
                        '"projected phase" line printed below is the thing to tune it on)')
    p.add_argument('--photons', type=float, default=0.0,
                   help='mean photons/pixel for Poisson noise (0 = noiseless)')
    p.add_argument('--shrink-a', default='0',
                   help='ground-truth shrinkage SLOPE A in shrink(t) = A*t + B, '
                        't = theta_idx/(ntheta-1).  One value (all distances, both '
                        'axes), two ("y,x"), or 2*ndist ("y0,x0,y1,x1,...").  '
                        'A = 1e-3 means the sample is 0.1%% smaller at the last '
                        'angle than at the first.')
    p.add_argument('--shrink-b', default='0',
                   help='ground-truth shrinkage OFFSET B, same broadcasting as '
                        '--shrink-a.  B is the shrink already present at the first '
                        'angle, which for real data grows with the distance index.')
    p.add_argument('--seed',   type=int,   default=10,  help='RNG seed for the displacements')
    p.add_argument('--nchunk', type=int,   default=4,  help='angles per GPU pass')
    p.add_argument('--prb-dir', default=C.PRB_DIR, help='directory with the ID16A probe TIFFs')
    p.add_argument('--phantom-cache', default=None,
                   help='HDF5 file the ground-truth object (phantom or rescaled --obj-vol) '
                        'is cached in, so a sweep builds it only once (default: beside the '
                        "datasets, named for n -- nobj with --obj-vol -- and a hash of "
                        "everything it depends on; "
                        "'none' rebuilds it every time)")
    p.add_argument('--out',    default=None,
                   help=f'output directory (default {C.OUT_ROOT}/amp<amp>_ndist<ndist>)')
    p.add_argument('--log-level', default='INFO')
    return p.parse_args()


a = parse()
set_log_level(a.log_level)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n     = a.n
if a.nobj:
    nobj = int(a.nobj) // 2 * 2                  # the grid, in px, as asked for
else:
    nobj = int(round((a.nobj_factor or (n + 2.0 * a.amp) / n) * n / 2)) * 2
nobj_factor = nobj / float(n)
nzobj = nobj
amp_max = (nobj - n) / 2
detector_pixelsize = C.detector_pixelsize(n)
# The ground truth is built at the DETECTOR size and then centred in the object
# grid with zeros around it, so nobj adds a blank margin and nothing else:
#   phantom   C.gen_object(span) on a span^3 grid -> C.write_centered
#   --obj-vol the whole source array rescaled to span px wide by C.fill_volume,
#             which centres and zero-pads it the same way; the sample is then
#             whatever fraction of that array is not air.
# The alternative -- generating on the nobj grid -- ties the sample's size to
# the margin, so raising the margin silently shrinks the sample relative to the
# field of view and changes what is being reconstructed.
span       = n
# ... plus the delta/beta ratio that turns --obj-vol's (already -delta-like)
# values into obj_im.
delta_beta = a.delta / a.beta
out = a.out or os.path.join(C.OUT_ROOT,
                            C.case_name(a.amp, a.ndist, a.prb_smooth, a.obj_smooth))
path = os.path.join(out, 'data.h5')

# The ground-truth object depends only on (n, delta, beta) for the phantom (it
# is built at the detector size and padded, so not on the margin), and on
# (source file, nobj, span, scale, delta/beta) for a real volume -- in
# both cases on nothing that varies across a sweep, so the whole sweep can share
# one copy: build it on the first run, read it back on every later one.  Worth
# it either way, but most of all for --obj-vol: rescaling the 3072^3 source
# reads the entire file (~150 s, almost all of it disk) and the dose comparison
# alone generates twice.
# obj_id hashes everything the result depends on -- the phantom's layer list,
# or the source's size/mtime plus the rescale parameters -- so an edited
# phantom or a replaced source file can never make an old cache silently
# reappear.
# next to the datasets (the parent of --out), not in C.OUT_ROOT: run_study.sh
# points --out at its own root, and the file is a full n^3 volume (nobj^3 with
# --obj-vol)
obj_id = (C.volume_id(a.obj_vol, span, a.obj_scale, delta_beta, nobj, nzobj)
          if a.obj_vol else C.phantom_id(smooth=a.obj_smooth))
# The phantom cache holds the span^3 CORE, not the padded grid: the core does
# not depend on the margin, so one cached phantom serves every nobj, and the
# file is (nobj/n)^3 smaller.  The --obj-vol cache still holds the whole
# rescaled nobj^3 grid, which is what fill_volume streams out.
cache_shape = (nzobj, nobj, nobj) if a.obj_vol else (span, span, span)
default_cache = os.path.join(
    os.path.dirname(os.path.normpath(out)),
    (f'objvol_nobj{nobj}_scale{a.obj_scale:g}_{obj_id}.h5' if a.obj_vol else
     f'phantom_n{span}_delta{a.delta:g}_beta{a.beta:g}_{obj_id}.h5'))
cache = None if a.phantom_cache == 'none' else (a.phantom_cache or default_cache)

if rank == 0:
    os.makedirs(out, exist_ok=True)
    logger.info('=' * 62)
    logger.info(f'  displacement amplitude : +-{a.amp:g} px  (edge-extension limit {amp_max:g} px)')
    logger.info(f'  probe smoothing        : sigma={a.prb_smooth:g} px'
                f'{" (as measured)" if a.prb_smooth <= 0 else ""}')
    if a.obj_vol:
        if abs(a.obj_smooth - C.OBJ_SMOOTH) > 1e-6:
            logger.warning(f'  --obj-smooth {a.obj_smooth:g} is ignored with --obj-vol: the '
                           f'volume is used as it is, only the phantom is filtered')
    else:
        logger.info(f'  object smoothing       : sigma={a.obj_smooth:g} voxel'
                    f'{" (hard edges)" if a.obj_smooth <= 0 else ""}')
    logger.info(f'  geometry               : {C.GEOMETRY}  '
                f'({C.ENERGY:g} keV, focus-to-detector {C.FOCUSTODETECTORDISTANCE:g} m)')
    logger.info(f'  distances              : {a.ndist}  z1={C.Z1_ALL[:a.ndist]*1e3} mm')
    logger.info(f'  detector / object grid : {n} x {n}   /   {nzobj} x {nobj} x {nobj}'
                f'  (nobj = {nobj_factor:.4g} x n)')
    logger.info(f'  ... sample / margin    : built at {span} px, centred in the grid '
                f'with a {(nobj - span) // 2} px blank border')
    logger.info(f'  detector pixel         : {detector_pixelsize*1e9:.2f} nm '
                f'({C.DETECTOR_NDET // n}x binned)')
    m0  = C.FOCUSTODETECTORDISTANCE / C.Z1_ALL[0]
    vox = detector_pixelsize / m0
    logger.info(f'  object voxel           : {vox*1e9:.2f} nm  '
                f'(x{m0:.1f} magnification, field of view {n*vox*1e6:.1f} um)')
    logger.info(f'  angles                 : {a.ntheta}')
    if a.obj_vol:
        logger.info(f'  sample volume          : {a.obj_vol}')
        logger.info(f'  ... span / scale       : {span} px = {span*vox*1e6:.1f} um  '
                    f'(the source array, rescaled to the detector width)  '
                    f'x {a.obj_scale:g}, delta/beta={delta_beta:g}')
    logger.info(f'  photons per pixel      : {a.photons:g} (0 = noiseless)')
    logger.info(f'  MPI ranks              : {comm.Get_size()}')
    logger.info(f'  output                 : {path}')
    logger.info(f'  object cache           : {cache or "disabled"}'
                f'{"" if not cache or os.path.isfile(cache) else "  (building it)"}')
    logger.info('=' * 62)
    if span > min(nobj, nzobj):
        logger.warning(f'the sample is {span} px wide but the object grid is only '
                       f'{nzobj} x {nobj}: it is cropped. Raise --nobj to at least {span}.')
    if a.amp > amp_max:
        logger.warning(f'amp={a.amp:g} exceeds (nobj-n)/2={amp_max:g}: the crop will sample '
                       f'mirrored edge data. Increase --nobj-factor.')
comm.Barrier()

# --- Rec instance (used only as the forward operator here) ------------------
args = SimpleNamespace()
args.energy                  = C.ENERGY
args.detector_pixelsize      = C.detector_pixelsize(n)
args.focustodetectordistance = C.FOCUSTODETECTORDISTANCE
args.z1                      = C.Z1_ALL[:a.ndist]
args.theta                   = np.linspace(0, np.pi, a.ntheta, dtype='float32')
args.ndist   = a.ndist
args.ntheta  = a.ntheta
args.nz      = n
args.n       = n
args.nzobj   = nzobj
args.nobj    = nobj
args.mask            = 0.9
args.lam_prbfit      = 0.0
args.lam_laplacian   = 0.0
args.rho             = [1, 0.05, 0.02, 0]
args.niter           = 0
args.nchunk          = a.nchunk
args.checkpoint_step = -1
args.error_step      = -1
args.start_iter      = 0
args.comm            = comm
# generation only touches vars / data / proj_tmp — skip the gradient buffers
args.alloc_mode      = 'gen'

cl = Rec(args)
local_nzobj  = cl.end_obj - cl.st_obj
local_ntheta = cl.end_theta - cl.st_theta


# --- ground-truth shrinkage -------------------------------------------------
def _shrink_par(text, name):
    """Parse --shrink-a / --shrink-b into [ndist, 2] (y, x)."""
    v = np.array([float(x) for x in str(text).split(',') if x.strip()], dtype='float32')
    if v.size == 1:
        return np.repeat(v, 2 * a.ndist).reshape(a.ndist, 2)
    if v.size == 2:
        return np.tile(v, (a.ndist, 1))
    if v.size == 2 * a.ndist:
        return v.reshape(a.ndist, 2)
    raise SystemExit(f'{name}: give 1, 2 or {2*a.ndist} values, got {v.size}')

tp_true = np.zeros((a.ndist, 2, 2), dtype='float32')
tp_true[:, 0, :] = _shrink_par(a.shrink_a, '--shrink-a')
tp_true[:, 1, :] = _shrink_par(a.shrink_b, '--shrink-b')
# shrink over all angles, the shape read_shrink / the writer's plot expect
t_all       = np.arange(a.ntheta, dtype='float32') / max(a.ntheta - 1, 1)
shrink_true = (tp_true[None, :, 0, :] * t_all[:, None, None]
               + tp_true[None, :, 1, :]).astype('float32')   # [ntheta, ndist, 2]
has_shrink  = bool(np.any(tp_true))
if rank == 0 and has_shrink:
    for k in range(a.ndist):
        logger.info(f'  shrink dist {k}: A=({tp_true[k,0,0]:+.4e}, {tp_true[k,0,1]:+.4e})  '
                    f'B=({tp_true[k,1,0]:+.4e}, {tp_true[k,1,1]:+.4e})  (y, x)')
cl.vars['tp'][:] = cp.asarray(tp_true)
# shrink_nd is the diagnostic copy Rec keeps of the same profile; filling it
# keeps the data mask and gen_data's demagnification consistent.
cl.shrink_nd[:] = cp.asarray(
    np.ascontiguousarray(shrink_true[cl.st_theta:cl.end_theta].transpose(1, 0, 2)))

# --- probe and true positions (identical on every rank) ---------------------
prb = C.load_probe(n, a.ndist, a.prb_dir, smooth=a.prb_smooth)
prb_contrast = C.probe_contrast(prb)
if rank == 0:
    logger.info(f'  probe contrast std|prb|/mean|prb| : '
                + ', '.join(f'{c:.4f}' for c in prb_contrast))
pos = C.gen_positions(a.ntheta, a.ndist, a.amp, a.seed)

# --- create the file and write the ground-truth object (rank 0) -------------
# The phantom needs several nobj^3 work arrays, so only rank 0 builds it; the
# other ranks pick up their z-slice from the file.
def fill_object(ds_re, ds_im):
    """Fill /obj_re,/obj_im of the new data file with the ground-truth object.

    With --obj-vol that is a rescaled slice-by-slice copy of a real volume,
    otherwise the synthetic phantom.  Either way the sample is built at the
    detector size and centred in the grid, and it is a deterministic function of
    arguments that do not change across a sweep, so it is built on the first
    call and cached, and later calls copy it back slab by slab (which also keeps
    a cache hit down to one z-batch of memory instead of a whole nobj^3 volume).
    """
    if cache and os.path.isfile(cache):
        with h5py.File(cache, 'r') as g:
            if (tuple(g['obj_re'].shape) == cache_shape
                    and float(g.attrs.get('delta', np.nan)) == a.delta
                    and float(g.attrs.get('beta',  np.nan)) == a.beta
                    and str(g.attrs.get('phantom_id', '')) == obj_id):
                logger.info(f'reading the object from {cache}')
                # straight through the datasets: write_centered reads them one
                # z-batch at a time, so the core never has to be resident either
                C.write_centered(ds_re, ds_im, g['obj_re'], g['obj_im'], nzobj, nobj)
                return
        logger.warning(f'{cache} holds an object of a different (shape, delta, beta, source) '
                       f'- rebuilding and overwriting it')

    if a.obj_vol:
        # streamed slice by slice: an nobj^3 volume does not fit in memory at n=2048
        logger.info(f'reading the sample volume from {a.obj_vol}')
        C.fill_volume(a.obj_vol, ds_re, ds_im, nzobj, nobj, span,
                      scale=a.obj_scale, delta_beta=delta_beta, log=logger.info)
    else:
        logger.info('generating the phantom')
        obj = C.gen_object(span, a.delta, a.beta, smooth=a.obj_smooth)
        C.write_centered(ds_re, ds_im, obj.real, obj.imag, nzobj, nobj)
        del obj

    if cache:
        # copied back out of the data file rather than held in memory: the
        # --obj-vol path never has the whole volume at once, and at n=2048 a
        # cache write that did would need 67 GB.
        # via a temporary file: an interrupted run must not leave a half-written
        # cache behind for the next run to read back as if it were complete
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        tmp = f'{cache}.tmp{os.getpid()}'
        cz, cy, cx = cache_shape
        z0, y0, x0 = (nzobj - cz) // 2, (nobj - cy) // 2, (nobj - cx) // 2
        with h5py.File(tmp, 'w') as g:
            g.attrs['n']     = n
            g.attrs['span']  = span
            if a.obj_vol:                 # the padded grid; the phantom cache is
                g.attrs['nobj']  = nobj   # the bare core, and does not depend on it
                g.attrs['nzobj'] = nzobj
            g.attrs['delta'] = a.delta
            g.attrs['beta']  = a.beta
            g.attrs['phantom_id'] = obj_id
            gre = g.create_dataset('obj_re', cache_shape, dtype='float32')
            gim = g.create_dataset('obj_im', cache_shape, dtype='float32')
            batch = C.h5_batch(cy * cx * 4)
            for i0 in range(0, cz, batch):
                i1 = min(i0 + batch, cz)
                gre[i0:i1] = ds_re[z0 + i0:z0 + i1, y0:y0 + cy, x0:x0 + cx]
                gim[i0:i1] = ds_im[z0 + i0:z0 + i1, y0:y0 + cy, x0:x0 + cx]
        os.replace(tmp, cache)
        logger.info(f'object cached -> {cache}  '
                    f'({2 * cz * cy * cx * 4 / 2**30:.1f} GiB)')


t0 = time.time()
if rank == 0:
    with h5py.File(path, 'w') as f:
        f.attrs['n'] = n; f.attrs['nz'] = n
        f.attrs['nobj'] = nobj; f.attrs['nzobj'] = nzobj
        f.attrs['ntheta'] = a.ntheta; f.attrs['ndist'] = a.ndist
        f.attrs['energy'] = C.ENERGY
        f.attrs['detector_pixelsize'] = detector_pixelsize
        f.attrs['focustodetectordistance'] = C.FOCUSTODETECTORDISTANCE
        f.attrs['amp'] = a.amp; f.attrs['seed'] = a.seed
        f.attrs['photons'] = a.photons
        f.attrs['delta'] = a.delta; f.attrs['beta'] = a.beta
        f.attrs['nobj_factor'] = nobj_factor
        f.attrs['span'] = span      # the sample's width; nobj - span = 2 * margin
        f.attrs['prb_smooth'] = a.prb_smooth
        f.attrs['shrink_a'] = a.shrink_a
        f.attrs['shrink_b'] = a.shrink_b
        f.attrs['obj_smooth'] = a.obj_smooth
        if a.obj_vol:
            f.attrs['obj_vol']   = a.obj_vol
            f.attrs['obj_scale'] = a.obj_scale
            f.attrs['obj_span']  = span
        f.attrs['prb_contrast'] = prb_contrast.astype('float32')

        f.create_dataset('theta', data=args.theta)
        f.create_dataset('z1',    data=np.asarray(args.z1))
        f.create_dataset('pos',   data=pos)
        # [ntheta, ndist, 2] (y, x), the same layout read_shrink expects, plus
        # the (A, B) that generated it so rec.py can score the recovery.
        f.create_dataset('shrink',  data=shrink_true)
        f.create_dataset('tp_true', data=tp_true)
        f.create_dataset('prb_abs',   data=np.abs(prb).astype('float32'))
        f.create_dataset('prb_phase', data=np.angle(prb).astype('float32'))
        # datasets filled later / collectively
        f.create_dataset('data', shape=(a.ndist, a.ntheta, n, n), dtype='float32')
        f.create_dataset('ref',  shape=(a.ndist, n, n),           dtype='float32')
        ds_re = f.create_dataset('obj_re', shape=(nzobj, nobj, nobj), dtype='float32')
        ds_im = f.create_dataset('obj_im', shape=(nzobj, nobj, nobj), dtype='float32')
        fill_object(ds_re, ds_im)
    logger.info(f'ground-truth object ready in {time.time()-t0:.1f} s')
comm.Barrier()

# --- every rank loads its own slice of the object ---------------------------
with h5py.File(path, 'r', driver='mpio', comm=comm) as f:
    batch = C.h5_batch(nobj * nobj * 4)
    for i0 in range(0, local_nzobj, batch):
        i1 = min(i0 + batch, local_nzobj)
        sl  = slice(cl.st_obj + i0, cl.st_obj + i1)
        dst = cl.vars['obj'][i0:i1]
        dst.real[:] = f['obj_re'][sl]
        dst.imag[:] = f['obj_im'][sl]

cl.vars['prb'][:] = prb          # prb is numpy; vars['prb'] is pinned numpy
C.set_pos(cl, pos)

# --- forward model ----------------------------------------------------------
logger.info('generating data')
t0 = time.time()
cl.gen_data(cl.vars, cl.data)

# The projected phase actually seen by exp(1j*proj), measured rather than
# derived: the Radon normalisation folds n, ntheta and norm_const together, so
# this is the only reliable way to check the magnitude.  A few rad to a few tens
# of rad is a well-conditioned test; rescale --obj-scale if it is off.
pr, pi = cl.vars['proj'].real, cl.vars['proj'].imag
_lo  = comm.allreduce(float(pr.min()), op=MPI.MIN)
_hi  = comm.allreduce(float(pr.max()), op=MPI.MAX)
_alo = comm.allreduce(float(pi.min()), op=MPI.MIN)
_ahi = comm.allreduce(float(pi.max()), op=MPI.MAX)
if rank == 0:
    logger.info(f'projected phase  [{_lo:+.4g}, {_hi:+.4g}] rad')
    logger.info(f'projected absorp [{_alo:+.4g}, {_ahi:+.4g}]   transmission '
                f'exp(-imag) in [{np.exp(-_ahi):.4g}, {np.exp(-_alo):.4g}]')

ref = C.gen_ref(cl, prb)
comm.Barrier()
if rank == 0:
    logger.info(f'forward model done in {time.time()-t0:.1f} s')

# --- noise ------------------------------------------------------------------
if a.photons > 0:
    C.add_poisson_noise(cl.data, a.photons, a.seed + 1000 + rank)
    C.add_poisson_noise(ref,     a.photons, a.seed + 500)

# --- write the measurements -------------------------------------------------
# cl.data is [ndist, local_ntheta, nz, n]; /data has the same distance-major order.
with h5py.File(path, 'a', driver='mpio', comm=comm) as f:
    ds = f['data']
    batch = C.h5_batch(n * n * 4)
    for k in range(a.ndist):
        for i0 in range(0, local_ntheta, batch):
            i1 = min(i0 + batch, local_ntheta)
            ds[k, cl.st_theta + i0:cl.st_theta + i1] = cl.data[k, i0:i1]
comm.Barrier()
if rank == 0:
    with h5py.File(path, 'a') as f:
        f['ref'][:] = ref
comm.Barrier()

if rank == 0:
    d = cl.data
    logger.info(f'rank-0 data range [{float(d.min()):.4f}, {float(d.max()):.4f}] '
                f'(intensity), mean {float(d.mean()):.4f}')
    logger.info(f'true displacements: max |y| = {np.abs(pos[...,0]).max():.3f} px, '
                f'max |x| = {np.abs(pos[...,1]).max():.3f} px')
    if has_shrink:
        logger.info(f'true shrink: max |y| = {np.abs(shrink_true[...,0]).max():.4e}, '
                    f'max |x| = {np.abs(shrink_true[...,1]).max():.4e}')
    logger.info(f'saved -> {path}  ({os.path.getsize(path)/1024**3:.2f} GB)')
