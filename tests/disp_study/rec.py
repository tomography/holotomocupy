#!/usr/bin/env python
"""Step 2 of the displacement study — reconstruct object + probe (+ positions)
from a dataset produced by gen_data.py.

All geometry is read back from the dataset file, so the only thing to point at
is the directory that gen_data.py wrote:

    mpirun -np 4 ./set_affinity_gpu.sh python rec.py --in <dir_or_data.h5> --niter 257

Reconstruction starts from a heavily blurred ground-truth object (see
--obj-init / --obj-blur), prb = 1 (i.e. nothing is assumed about the
illumination), and the true displacements perturbed by --pos-err.
Checkpoints, convergence table and position-drift plots land in --out.

Notes
-----
* BH refines obj, prb and pos jointly.  The step for each one scales with
  rho^2, so --freeze-prb / --freeze-pos (and --rho 1,0.05,1e-3) just make that
  variable's step negligible; rho = 0 itself is not allowed (0/0 in alpha).
* With --gt the masked NRMSE against the ground-truth phantom is reported at
  the end — that is the number to compare across displacement amplitudes.
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

from holotomocupy.rec_mpi import Rec                              # noqa: E402
from holotomocupy.writer import Writer                            # noqa: E402
from holotomocupy.logger_config import logger, set_log_level      # noqa: E402

import common as C                                                # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--in', dest='inp', required=True,
                   help='dataset directory from gen_data.py, or the data.h5 itself')
    p.add_argument('--out', default=None,
                   help='output directory (default <in>/rec_n<n>_ntheta<ntheta>)')
    p.add_argument('--niter',  type=int, default=513, help='number of BH iterations')
    p.add_argument('--nchunk', type=int, default=8,  help='angles per GPU pass')
    p.add_argument('--rho', default='1,0.05,0.00001,0',
                   help='step-size scales for obj,prb,pos (comma separated)')
    p.add_argument('--freeze-prb', action='store_true',
                   help='keep the probe at its initial value (rho_prb <- --frozen-rho). '
                        'Needed at --amp 0, where an unknown probe is exactly ambiguous '
                        'with a rotationally symmetric object; combine with --prb-init true.')
    p.add_argument('--freeze-pos', action='store_true',
                   help='keep the positions at their initial value (rho_pos <- --frozen-rho)')
    p.add_argument('--freeze-shrink', action='store_true',
                   help='keep the shrinkage parameters at the fit of the stored '
                        'profile (rho_tp <- --frozen-rho); the default rho already '
                        'ends in 0, so this only matters with an explicit --rho')
    p.add_argument('--estimate-rho', action='store_true',
                   help='before the main loop, coordinate-search rho_prb and rho_pos '
                        'on a geometric grid around --rho, scoring each candidate with '
                        'a short --rho-estimate-niter trial run (see '
                        'Rec.estimate_rho_coord). rho_obj is left alone.')
    p.add_argument('--rho-estimate-niter', type=int, default=16,
                   help='iterations per trial of --estimate-rho')
    p.add_argument('--frozen-rho', type=float, default=1e-3,
                   help='rho used for a frozen variable; the step scales with rho^2, so '
                        '1e-3 means 1e-6 of the object step. 0 is not allowed (0/0 in alpha).')
    p.add_argument('--warmup', type=int, default=0,
                   help='run this many first iterations with the probe and the positions '
                        'frozen, then release them. Tames the very first BH step when the '
                        'obj/prb ambiguity is still nearly unbroken (small --amp).')
    p.add_argument('--mask', type=float, default=1.1, help='support mask radius (fraction of FOV)')
    p.add_argument('--lam-prbfit',    type=float, default=2e-3, help='probe-fit regularisation')
    p.add_argument('--lam-laplacian', type=float, default=0.0,  help='Laplacian regularisation')
    p.add_argument('--pos-err', type=float, default=0.0,
                   help='half-width of a uniform error added to the true positions [px]')
    p.add_argument('--pos-err-seed', type=int, default=77)
    p.add_argument('--shrink-err', type=float, default=0.0,
                   help='start the shrinkage parameters off by up to this much '
                        '(uniform, added to both A and B of every distance and '
                        'axis) instead of at the fit of the stored profile. '
                        'This is what turns a --shrink-a/--shrink-b dataset '
                        'into a recovery test for rho[3] > 0.')
    p.add_argument('--shrink-err-seed', type=int, default=91)
    p.add_argument('--prb-init', default='ones', choices=['ones', 'true'],
                   help="'ones' = flat illumination, 'true' = start from the true probe")
    p.add_argument('--obj-init', default='blur', choices=['blur', 'zeros', 'true'],
                   help="'blur' = ground truth smoothed by --obj-blur (low-frequency "
                        "starting point, the high frequencies are still unknown), "
                        "'zeros' = nothing assumed, 'true' = the phantom itself")
    p.add_argument('--obj-blur', type=float, default=8.0,
                   help='sigma [px] of the isotropic 3D Gaussian for --obj-init blur')
    p.add_argument('--checkpoint-step', type=int, default=32)
    p.add_argument('--check-approx', action='store_true',
                   help='on every checkpoint step, plot the real functional '
                        'along the descent direction against the quadratic '
                        'model that picked alpha, into '
                        '{out}/check_approximation/. Costs 6 extra evaluations '
                        'of the functional per triggered iteration.')
    p.add_argument('--error-step',      type=int, default=32)
    p.add_argument('--start-iter',      type=int, default=0)
    p.add_argument('--gt', action=argparse.BooleanOptionalAction, default=True,
                   help='report the masked NRMSE against the ground-truth phantom')
    p.add_argument('--log-level', default='INFO')
    return p.parse_args()

 
a = parse()
set_log_level(a.log_level)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

in_path = a.inp if a.inp.endswith('.h5') else os.path.join(a.inp, 'data.h5')
meta    = C.read_attrs(in_path, comm)

n      = int(meta['n'])
nobj   = int(meta['nobj'])
nzobj  = int(meta['nzobj'])
ntheta = int(meta['ntheta'])
ndist  = int(meta['ndist'])
# datasets written before --prb-smooth / --obj-smooth existed all used the
# legacy filter, which is exactly what the two defaults reproduce
prb_smooth   = float(meta.get('prb_smooth', C.PRB_SMOOTH))
obj_smooth   = float(meta.get('obj_smooth', C.OBJ_SMOOTH))
prb_contrast = np.atleast_1d(np.asarray(meta.get('prb_contrast', [np.nan]), dtype='float64'))
# the detector size and the number of angles go into the folder name, so runs at
# different resolutions / angular sampling do not overwrite each other
out    = a.out or os.path.join(os.path.dirname(in_path), f'rec_n{n}_ntheta{ntheta}')
rho    = [float(v) for v in a.rho.split(',')]
rho   += [0.0] * max(0, 4 - len(rho))      # rho[3] = tp (linear shrink), 0 = frozen
if a.freeze_prb:
    rho[1] = a.frozen_rho
if a.freeze_pos:
    rho[2] = a.frozen_rho
if a.freeze_shrink:
    rho[3] = a.frozen_rho
# shrinkage is frozen through the warmup along with prb and pos: it is the
# slowest-moving unknown and worth nothing before the object has any structure.
rho_warmup = [rho[0], min(rho[1], a.frozen_rho), min(rho[2], a.frozen_rho), 0.0]

if rank == 0:
    os.makedirs(out, exist_ok=True)
    logger.info('=' * 62)
    logger.info(f'  dataset                : {in_path}')
    logger.info(f'  displacement amplitude : +-{float(meta["amp"]):g} px')
    logger.info(f'  probe smoothing        : sigma={prb_smooth:g} px   contrast '
                + ', '.join(f'{c:.4f}' for c in prb_contrast))
    logger.info(f'  object smoothing       : sigma={obj_smooth:g} voxel'
                f'{"  (a real volume, unfiltered)" if "obj_vol" in meta else ""}')
    logger.info(f'  distances              : {ndist}   angles: {ntheta}')
    logger.info(f'  detector / object      : {n} x {n}   /   {nzobj} x {nobj} x {nobj}')
    # from the file's own geometry, not common.py's constants: a dataset made
    # before DETECTOR_PIXELSIZE_1X changed still reports the voxel it was made on
    vox = (float(meta['detector_pixelsize'])
           * float(np.atleast_1d(meta['z1'])[0]) / float(meta['focustodetectordistance']))
    logger.info(f'  detector pixel / voxel : {float(meta["detector_pixelsize"])*1e9:.2f} nm'
                f'  /  {vox*1e9:.2f} nm   (field of view {n * vox * 1e6:.1f} um)')
    logger.info(f'  photons per pixel      : {float(meta["photons"]):g} (0 = noiseless)')
    logger.info(f'  niter / rho            : {a.niter} / {rho}'
                f'   (rho[3] = shrink)')
    if a.estimate_rho:
        logger.info(f'  rho estimation         : coordinate search on prb,pos '
                    f'({a.rho_estimate_niter} iters per trial)')
    if a.warmup > 0:
        logger.info(f'  warmup                 : {a.warmup} iters at rho={rho_warmup}')
    logger.info(f'  initial position error : +-{a.pos_err:g} px')
    obj_init_txt = (f'{a.obj_init} (sigma={a.obj_blur:g} px)'
                    if a.obj_init == 'blur' else a.obj_init)
    logger.info(f'  object init            : {obj_init_txt}')
    logger.info(f'  probe init             : {a.prb_init}')
    logger.info(f'  output                 : {out}')
    logger.info('=' * 62)
comm.Barrier()

# --- solver -----------------------------------------------------------------
args = SimpleNamespace()
args.energy                  = float(meta['energy'])
args.detector_pixelsize      = float(meta['detector_pixelsize'])
args.focustodetectordistance = float(meta['focustodetectordistance'])
args.z1     = meta['z1']
args.theta  = meta['theta']
args.ndist  = ndist
args.ntheta = ntheta
args.nz     = n
args.n      = n
args.nzobj  = nzobj
args.nobj   = nobj
args.mask            = a.mask
args.lam_prbfit      = a.lam_prbfit
args.lam_laplacian   = a.lam_laplacian
args.rho             = rho
# switched on per phase in run_bh(); the warmup phase must not search a rho it
# is about to throw away
args.estimate_rho       = False
args.rho_estimate_niter = a.rho_estimate_niter
args.niter           = a.niter
args.nchunk          = a.nchunk
args.checkpoint_step = a.checkpoint_step
args.check_approx    = a.check_approx
args.error_step      = a.error_step
args.start_iter      = a.start_iter
args.path_out        = out          # enables conv.csv
args.comm            = comm

cl = Rec(args)
local_nzobj  = cl.end_obj - cl.st_obj
local_ntheta = cl.end_theta - cl.st_theta

writer = Writer(
    path_out  = out,
    comm      = comm,
    st_obj    = cl.st_obj,   end_obj   = cl.end_obj,   nzobj  = nzobj, nobj = nobj,
    st_theta  = cl.st_theta, end_theta = cl.end_theta, ntheta = ntheta,
    ndist     = ndist, nz = n, n = n,
)

# --- measurements -----------------------------------------------------------
logger.info('reading data')
t0 = time.time()
with h5py.File(in_path, 'r', driver='mpio', comm=comm) as f:
    batch = C.h5_batch(n * n * 4)
    for k in range(ndist):
        for i0 in range(0, local_ntheta, batch):
            i1 = min(i0 + batch, local_ntheta)
            cl.data[k, i0:i1] = f['data'][k, cl.st_theta + i0:cl.st_theta + i1]
    cl.ref[:] = cp.asarray(f['ref'][:])
    pos_true  = f['pos'][:].astype('float32')
    # Datasets made before shrinkage was a variable have neither; then tp stays
    # 0 and the reconstruction is the fixed no-shrink model as before.
    shrink_gt = (f['shrink'][:].astype('float32')  if 'shrink'  in f else None)
    tp_true   = (f['tp_true'][:].astype('float32') if 'tp_true' in f else None)
    prb_true  = (f['prb_abs'][:] * np.exp(1j * f['prb_phase'][:])).astype('complex64')
comm.Barrier()
if rank == 0:
    logger.info(f'data read in {time.time()-t0:.1f} s')

# --- initial guess ----------------------------------------------------------
def set_obj_init(kind, sigma):
    """Fill this rank's slab of cl.vars['obj'] with the starting object.

    'blur' reads the ground truth plus a halo of 4*sigma slices above and below
    the rank's own z-range and Gaussian-filters that; since the filter is
    truncated at 4 sigma, the retained interior is identical to filtering the
    assembled volume.  mode='constant' is the right boundary rule at the top and
    bottom of the volume too, the phantom being embedded in zeros there.
    """
    if kind == 'zeros':
        cl.vars['obj'][:] = 0
        return
    pad = 0 if kind == 'true' else int(4.0 * sigma + 0.5) + 1
    z0  = max(cl.st_obj - pad, 0)
    z1  = min(cl.end_obj + pad, nzobj)
    re  = np.empty((z1 - z0, nobj, nobj), dtype='float32')
    im  = np.empty_like(re)
    batch = C.h5_batch(nobj * nobj * 4)
    with h5py.File(in_path, 'r', driver='mpio', comm=comm) as f:
        for i0 in range(0, z1 - z0, batch):
            i1 = min(i0 + batch, z1 - z0)
            re[i0:i1] = f['obj_re'][z0 + i0:z0 + i1]
            im[i0:i1] = f['obj_im'][z0 + i0:z0 + i1]
    if kind == 'blur':
        # C.gaussian_blur3d is scipy below GPU_BLUR_BYTES and a chunked GPU pass
        # above it -- sigma = 32 on an n = 2048 slab is hours of scipy otherwise
        re = C.gaussian_blur3d(re, sigma)
        im = C.gaussian_blur3d(im, sigma)
    lo = cl.st_obj - z0
    hi = lo + (cl.end_obj - cl.st_obj)
    cl.vars['obj'].real[:] = re[lo:hi]
    cl.vars['obj'].imag[:] = im[lo:hi]

set_obj_init(a.obj_init, a.obj_blur)
cl.vars['prb'][:] = prb_true if a.prb_init == 'true' else 1

pos_start = pos_true.copy()
if a.pos_err > 0:
    rng = np.random.default_rng(a.pos_err_seed)
    pos_start += (2.0 * a.pos_err) * (rng.random(pos_true.shape) - 0.5).astype('float32')
C.set_pos(cl, pos_start)

# Shrinkage starts from a least-squares fit of the stored profile -- exactly what
# step6 does on real data -- so with rho[3] = 0 this run reproduces the generator.
if shrink_gt is not None:
    cl.shrink_nd[:] = cp.asarray(np.ascontiguousarray(
        shrink_gt[cl.st_theta:cl.end_theta].transpose(1, 0, 2)))
    cl.init_tp_from_shrink()

# ... unless the run is a recovery test, which has to start somewhere wrong.
# tp is a global variable, so every rank must apply the SAME perturbation --
# hence a seeded generator rather than anything rank-dependent.
if a.shrink_err > 0:
    rng    = np.random.default_rng(a.shrink_err_seed)
    tp_pert = cp.asnumpy(cl.vars['tp'])
    tp_pert += ((2.0 * a.shrink_err)
                * (rng.random(tp_pert.shape) - 0.5)).astype('float32')
    cl.vars['tp'][:] = cp.asarray(tp_pert)
    # keep the diagnostic profile in step with the variable it is derived from
    cl.shrink_nd[:] = cp.asarray(np.ascontiguousarray(
        cl._tp_to_shrink_local(cl.vars['tp']).transpose(1, 0, 2)))
    if rank == 0:
        logger.warning(f'shrink start perturbed by +-{a.shrink_err:g} '
                       f'(seed {a.shrink_err_seed})')

# --- run --------------------------------------------------------------------
def run_bh(start, stop, rho_eff, estimate=False):
    """BH from iteration `start` to `stop` with the given step scales.

    rho_sq is a plain dict on Rec, so a phase can be run with different step
    scales simply by replacing it; precalc/postcalc are balanced, so calling
    BH more than once is safe.  With estimate=True the phase opens with
    Rec.estimate_rho_coord, which replaces rho_eff[1:] by the values it finds."""
    cl.rho_sq    = {'obj': rho_eff[0]**2, 'prb': rho_eff[1]**2,
                    'pos': rho_eff[2]**2, 'tp':  rho_eff[3]**2}
    cl.estimate_rho = estimate
    cl.start_iter = start
    cl.niter      = stop
    cl.BH(writer=writer, shrink_gt=shrink_gt)

pos_init0 = np.array(cl.vars['pos'])   # baseline for the final drift plot
logger.info('running BH')
t0 = time.time()
if a.warmup > 0:
    n_warm = min(a.warmup, a.niter)
    logger.info(f'warmup: {n_warm} iterations with prb and pos frozen (rho={rho_warmup})')
    run_bh(a.start_iter, n_warm, rho_warmup)
    if n_warm < a.niter:
        run_bh(n_warm, a.niter, rho, estimate=a.estimate_rho)
else:
    run_bh(a.start_iter, a.niter, rho, estimate=a.estimate_rho)
rho_used = [float(np.sqrt(cl.rho_sq[k])) for k in ('obj', 'prb', 'pos', 'tp')]
if a.estimate_rho and rank == 0:
    logger.warning(f'rho actually used: {rho_used}')
comm.Barrier()
if rank == 0:
    logger.info(f'BH finished in {time.time()-t0:.1f} s')

# Final state. obj has already been un-normalised by BH's postcalc, so the
# writer must not scale it again -> norm_const = 1.
cl.pos_init = pos_init0        # BH's precalc resets it at every phase
writer.write_checkpoint(cl.vars, a.niter, 1.0, pos_init=cl.pos_init,
                        shrink=cl._tp_to_shrink_local(cl.vars['tp']),
                        shrink_init=cl._tp_to_shrink_local(cl.tp_init),
                        shrink_gt=shrink_gt)

# --- shrinkage recovery vs ground truth -------------------------------------
if tp_true is not None and rank == 0:
    tp_rec = cp.asnumpy(cl.vars['tp'])
    for k in range(ndist):
        for ax, nm in enumerate(('y', 'x')):
            logger.warning(
                f'shrink dist {k} {nm}: A {tp_true[k,0,ax]:+.4e} -> {tp_rec[k,0,ax]:+.4e}  '
                f'B {tp_true[k,1,ax]:+.4e} -> {tp_rec[k,1,ax]:+.4e}')

# --- quality vs ground truth ------------------------------------------------
if a.gt:
    mask = cl.cl_tomo.mask
    mask = np.float32(mask) if np.isscalar(mask) else np.asarray(mask, dtype='float32')
    num = np.zeros(3, dtype='float64')   # |diff|^2, real diff^2, imag diff^2
    den = np.zeros(3, dtype='float64')
    off = np.zeros(2, dtype='float64')   # sum(mask*(rec-gt)) real, sum(mask)
    with h5py.File(in_path, 'r', driver='mpio', comm=comm) as f:
        batch = C.h5_batch(nobj * nobj * 4)
        for i0 in range(0, local_nzobj, batch):
            i1 = min(i0 + batch, local_nzobj)
            sl = slice(cl.st_obj + i0, cl.st_obj + i1)
            gt = f['obj_re'][sl].astype('complex64')
            gt.imag[:] = f['obj_im'][sl]
            rec  = np.asarray(cl.vars['obj'][i0:i1]).astype('complex64')
            diff = rec - gt
            num += [np.sum(mask * np.abs(diff)**2),
                    np.sum(mask * diff.real**2),
                    np.sum(mask * diff.imag**2)]
            den += [np.sum(mask * np.abs(gt)**2),
                    np.sum(mask * gt.real**2),
                    np.sum(mask * gt.imag**2)]
            off += [np.sum(mask * diff.real),
                    np.sum(mask * np.ones_like(diff.real))]
    num = comm.allreduce(num, op=MPI.SUM)
    den = comm.allreduce(den, op=MPI.SUM)
    off = comm.allreduce(off, op=MPI.SUM)
    if rank == 0:
        nrmse = np.sqrt(num / np.maximum(den, 1e-30))
        logger.warning(f'amp={float(meta["amp"]):g} px  prb_smooth={prb_smooth:g} px  '
                       f'obj_smooth={obj_smooth:g} voxel  '
                       f'niter={a.niter}  '
                       f'NRMSE(obj)={nrmse[0]:.4f}  '
                       f'NRMSE(delta)={nrmse[1]:.4f}  NRMSE(beta)={nrmse[2]:.4f}  '
                       f'mean(delta_rec-delta_gt)={off[0]/off[1]:.3e}')
        with open(os.path.join(out, 'summary.txt'), 'w') as fh:
            fh.write(f'dataset {in_path}\n'
                     f'amp {float(meta["amp"]):g}\n'
                     f'prb_smooth {prb_smooth:g}\n'
                     f'prb_contrast {prb_contrast[0]:.6f}\n'
                     f'obj_smooth {obj_smooth:g}\n'
                     f'ndist {ndist}\n'
                     f'n {n}\n'
                     f'ntheta {ntheta}\n'
                     f'nobj {nobj}\n'
                     f'photons {float(meta["photons"]):g}\n'
                     f'pos_err {a.pos_err:g}\n'
                     f'obj_init {a.obj_init}'
                     f'{" " + repr(a.obj_blur) if a.obj_init == "blur" else ""}\n'
                     f'prb_init {a.prb_init}\n'
                     f'niter {a.niter}\n'
                     f'rho {",".join(f"{v:g}" for v in rho_used)}'
                     f'{" (estimated)" if a.estimate_rho else ""}\n'
                     f'nrmse_obj {nrmse[0]:.6f}\n'
                     f'nrmse_delta {nrmse[1]:.6f}\n'
                     f'nrmse_beta {nrmse[2]:.6f}\n'
                     f'mean_offset_delta {off[0]/off[1]:.6e}\n')
        logger.info(f'summary -> {os.path.join(out, "summary.txt")}')
