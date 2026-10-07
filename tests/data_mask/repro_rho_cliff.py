"""
repro_rho_cliff.py -- why the rho search reports errors that jump between
~1e-4 and ~1e+4 on AtomiumL1_HT bin 2.

Small synthetic holotomography problem with the SAME geometry ratios as the
real run (nobj == n, so eff_demag > 1 pushes the outer detector ring off the
object grid and the out-of-grid mask kicks in), driven through exactly the
loop estimate_rho_coord uses -- but with per-iteration instrumentation:

    iter  err  alpha  top  bottom  |eta| per variable  max|var| per variable

    python repro_rho_cliff.py                 # mask on  (default)
    python repro_rho_cliff.py --no-mask       # mask off (comparison)
    python repro_rho_cliff.py --n 256 --niter 16

Run under the holotomocupy env, from the repo root or anywhere:
    /home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python repro_rho_cliff.py
"""
import argparse, os, sys
import numpy as np
import cupy as cp
from scipy.fft import fftn, ifftn, fftshift, fft2, ifft2
import scipy.ndimage as ndimage
from mpi4py import MPI
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from holotomocupy.rec_mpi import Rec
from holotomocupy.utils import read_tiff
from holotomocupy.logger_config import set_log_level
set_log_level(__import__('os').environ.get('LOGLVL','WARNING'))

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=256)
p.add_argument('--ntheta', type=int, default=120)
p.add_argument('--niter', type=int, default=16)
p.add_argument('--no-mask', action='store_true')
p.add_argument('--rho-prb', type=float, nargs='*',
               default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
p.add_argument('--lam-lap', type=float, default=2.5e-5)
p.add_argument('--prb-dir', default='/home/beams2/VNIKITIN/data/prb_id16a')
args_cli = p.parse_args()

n      = args_cli.n
ntheta = args_cli.ntheta
ndist  = 4
nobj   = n                      # <-- as in config_step6_bin2.conf: nobj == n

# AtomiumL1_HT geometry (config_steps15.conf / show_geometry.py), scaled to n.
energy                  = 33.35
focustodetectordistance = 1.289
z1 = np.array([0.611, 0.637, 0.742, 0.960]) * 1e-2
detector_pixelsize      = 11808.118e-9 * (512 / n)

mags = focustodetectordistance / z1
print(f"eff_demag = {np.round(mags[0]/mags, 5)}   (nobj={nobj}, n={n})")

# ---------------------------------------------------------------- phantom
def _frame(cube, p1, p2):
    cube[p1:p2, p1, p1] = 1; cube[p1:p2, p1, p2] = 1
    cube[p1:p2, p2, p1] = 1; cube[p1:p2, p2, p2] = 1
    cube[p1, p1:p2, p1] = 1; cube[p1, p1:p2, p2] = 1
    cube[p2, p1:p2, p1] = 1; cube[p2, p1:p2, p2] = 1
    cube[p1, p1, p1:p2] = 1; cube[p1, p2, p1:p2] = 1
    cube[p2, p1, p1:p2] = 1; cube[p2, p2, p1:p2] = 1

def rotate3d_once(vol, a_deg=28, b_deg=45):
    a, b = np.deg2rad(a_deg), np.deg2rad(b_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    A = np.linalg.inv(Ry @ Rz)
    c = (np.array(vol.shape) - 1) / 2.0
    return ndimage.affine_transform(vol, A, offset=c - A @ c, order=1,
                                    mode="constant", cval=0.0)

def gen_object(n, delta, beta):
    obj = np.zeros((n, n, n), dtype=np.float32)
    amps = np.array([3, -3, 1, 3, -4, 1, 4], dtype=np.float32)
    dil  = (np.array([33, 28, 25, 21, 16, 10, 3], dtype=np.float32) / 256.0) * n
    ax = np.arange(-n//2, n//2, dtype=np.float32)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = x*x + y*y + z*z
    del x, y, z
    fcirc = [fftn(fftshift((r2 < d*d).astype(np.float32)), workers=-1).astype(np.complex64)
             for d in dil]
    cube = np.zeros((n, n, n), dtype=np.float32)
    fcube = []
    for _ in amps:
        cube.fill(0.0)
        r = int(n * 0.2)
        _frame(cube, n//2 - r//2, n//2 + r//2)
        fcube.append(fftn(fftshift(cube), workers=-1).astype(np.complex64))
    work = np.empty((n, n, n), dtype=np.complex64)
    for kk, a in enumerate(amps):
        np.multiply(fcube[kk], fcirc[kk], out=work)
        obj += a * (fftshift(ifftn(work, workers=-1)).real > 1.0)
    obj = rotate3d_once(obj)
    obj = np.roll(np.roll(obj, -15*n//256, axis=2), -10*n//256, axis=1)
    np.maximum(obj, 0, out=obj)
    v = np.arange(-n//2, n//2, dtype=np.float32) / n
    vx, vy, vz = np.meshgrid(v, v, v, indexing="ij")
    filt = fftshift(np.exp(-3.0 * (vx*vx + vy*vy + vz*vz)).astype(np.float32))
    obj = ifftn(fftn(obj) * filt).real
    obj[obj < 0] = 0
    return (obj * (-delta + 1j*beta)).astype(np.complex64)

obj = gen_object(nobj, 1, 1e-2)

prb_abs   = read_tiff(f'{args_cli.prb_dir}/prb_abs_2048.tiff')[:ndist]
prb_phase = read_tiff(f'{args_cli.prb_dir}/prb_phase_2048.tiff')[:ndist]
prb = (prb_abs * np.exp(1j * prb_phase)).astype('complex64')
c = prb.shape[1] // 2
prb = prb[:, c-n//2:c+n//2, c-n//2:c+n//2]
v = np.arange(-n//2, n//2, dtype=np.float32) / n
vx, vy = np.meshgrid(v, v, indexing="ij")
prb = ifft2(fft2(prb) * fftshift(np.exp(-4.0*(vx*vx+vy*vy)).astype(np.float32)))
prb /= np.mean(np.abs(prb), axis=(1, 2))[:, None, None]

np.random.seed(10)
pos     = 30 * (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)
pos_err =      (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)
theta   = np.linspace(0, np.pi, ntheta, dtype='float32')

# ---------------------------------------------------------------- Rec
a = SimpleNamespace()
a.energy = energy; a.detector_pixelsize = detector_pixelsize
a.focustodetectordistance = focustodetectordistance; a.z1 = z1
a.theta = theta; a.ndist = ndist; a.ntheta = ntheta
a.nz = n; a.n = n; a.nzobj = nobj; a.nobj = nobj
a.mask = 1.1
a.lam_prbfit = 3.1e-3
a.lam_laplacian = args_cli.lam_lap
a.rho = [1, 0.05, 0.02, 0]
a.niter = args_cli.niter; a.start_iter = 0
a.nchunk = 16
a.checkpoint_step = -1; a.error_step = -1
a.mask_oob = not args_cli.no_mask
a.mask_oob_margin = 2
a.check_fused_hessian = bool(int(__import__('os').environ.get('CHECKFH','0')))
a.comm = MPI.COMM_WORLD

cl = Rec(a)
cl.vars['obj'][:] = obj[cl.st_obj:cl.end_obj]
cl.vars['prb'][:] = prb
cl.vars['pos'][:] = pos[cl.st_theta:cl.end_theta].transpose(1, 0, 2)
cl.gen_data(cl.vars, cl.data)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)

# Start from a *good* guess, like the real run does (Paganin volume + flat-field
# probe), so we are in the same regime: obj smoothed, prb flat, pos perturbed.
obj0 = ndimage.gaussian_filter(obj.real, 2) + 1j*ndimage.gaussian_filter(obj.imag, 2)
cl.vars['obj'][:] = obj0[cl.st_obj:cl.end_obj].astype('complex64')
cl.vars['prb'][:] = 1
cl.vars['pos'][:] = (pos + pos_err)[cl.st_theta:cl.end_theta].transpose(1, 0, 2)

vars = cl.vars
cl.precalc(vars)
snap = {k: v.copy() for k, v in vars.items()}
e0 = cl.min(vars['prb'], vars['obj'], vars['pos'], vars['proj'])
print(f"\nmask_oob={a.mask_oob}   initial err={e0:.5e}\n")

def energy_parts(vars):
    """(total, data-fit, prbfit, laplacian) -- single rank only."""
    tot = float(cl.min(vars['prb'], vars['obj'], vars['pos'], vars['proj']))
    ep  = float(cl.cl_prb_term.energy_local(vars['prb']))
    el  = float(cl.cl_lap_term.energy_local()) if hasattr(cl, 'cl_lap_term') else 0.0
    return tot, tot - ep - el, ep, el


def nrm(x):
    x = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
    return float(cp.linalg.norm(x.ravel()))

hdr = (f"{'it':>3} {'err':>11} {'alpha':>11} {'top':>11} {'bottom':>11} "
       f"{'F0':>11} {'prbfit':>11} {'lap':>11} "
       f"{'|e_obj|':>10} {'|e_prb|':>10} {'|e_pos|':>10} "
       f"{'max|obj|':>10} {'max|prb|':>10} {'max|pos|':>10}")

for rp in args_cli.rho_prb:
    for k, v in vars.items():
        v[:] = snap[k]
    for b in cl.grads.values(): b[:] = 0
    for b in cl.etas.values():  b[:] = 0
    cl.rho_sq = {'obj': 1.0, 'prb': rp**2, 'pos': 0.02**2, 'tp': 0.0}
    cl.start_iter = 0
    cl.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"])
    print(f"--- rho_prb = {rp:g} " + "-"*84)
    print(hdr)
    for i in range(args_cli.niter):
        cl.compute_gradient(vars, cl.grads)
        alpha, top, bottom = cl.compute_step(vars, cl.grads, cl.etas, i)
        cl.apply_step(vars, cl.etas, alpha)
        err = cl.min(vars['prb'], vars['obj'], vars['pos'], vars['proj'])
        _t, e_f0, e_pf, e_lap = energy_parts(vars)
        print(f"{i:3d} {err:11.4e} {alpha:11.4e} {top:11.4e} {bottom:11.4e} "
              f"{e_f0:11.4e} {e_pf:11.4e} {e_lap:11.4e} "
              f"{nrm(cl.etas['obj']):10.3e} {nrm(cl.etas['prb']):10.3e} "
              f"{nrm(cl.etas['pos']):10.3e} "
              f"{float(np.abs(vars['obj']).max()):10.3e} "
              f"{float(np.abs(vars['prb']).max()):10.3e} "
              f"{float(np.abs(vars['pos']).max()):10.3e}", flush=True)
        if not np.isfinite(err):
            break
    print()
