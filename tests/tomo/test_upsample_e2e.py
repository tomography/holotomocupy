#!/usr/bin/env python
"""End-to-end smoke test for tomo_upsample: a small synthetic step-6 run.

    PYTHONPATH=../../src python test_upsample_e2e.py [--scratch DIR]

One GPU, one rank, no data files: a smooth phantom is forward-modelled through
the full cascade (Radon -> shift -> propagate) and then reconstructed twice,

    A.  nobj = 160, tomo_upsample = 1      (today's behaviour)
    B.  nobj =  80, tomo_upsample = 2      (object grid halved in x/y)

from the SAME data.  Both write to a projection plane 160 wide, so B is the
configuration the ctxl bin-0 ladder will run: half the object, the same fit.

What this actually checks -- the plumbing, not the science:
  1. both runs build, precalc and iterate without a shape mismatch anywhere in
     fwd_tomo / redist / the data mask,
  2. err decreases in both,
  3. a checkpoint written by run B reads back through Reader.read_checkpoint,
     and
  4. the bin1 -> bin0 handoff read: an object stored at [nzobj/2, nobj, nobj]
     with a half-width probe lands as z x2, x/y x1, prb x2, pos x2 -- the three
     scales that used to be one.

Eight iterations is not a convergence test; it is enough for err to move.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import cupy as cp
import h5py
from mpi4py import MPI

from holotomocupy.rec_mpi import Rec
from holotomocupy.reader import Reader
from holotomocupy.writer import Writer
from holotomocupy.logger_config import set_log_level

cp.cuda.set_pinned_memory_allocator(None)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# ---- geometry, small enough to run in well under a minute -------------------
n = nz = 128
nzobj = 160                 # z is the same on both grids -- upsample is x/y only
NDOBJ = 160                 # the projection plane, identical in A and B
ndist, ntheta, nchunk, niter = 2, 16, 4, 8

energy = 17.1
detector_pixelsize = 1.4760147601476e-6 * 2 * 8
focustodetectordistance = 1.217
z1 = np.array([5.110, 6.879]) * 1e-3
theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')

comm = MPI.COMM_WORLD


def make_args(nobj, upsample, **kw):
    a = SimpleNamespace(
        nz=nz, n=n, nzobj=nzobj, nobj=nobj,
        ntheta=ntheta, ndist=ndist, nchunk=nchunk, niter=niter, start_iter=0,
        tomo_upsample=upsample,
        rho=[1, 0.05, 0.02, 0], lam_prbfit=2e-3, lam_laplacian=0,
        checkpoint_step=-1, error_step=1,
        energy=energy, focustodetectordistance=focustodetectordistance,
        z1=z1, detector_pixelsize=detector_pixelsize, theta=theta,
        mask=0.9, shift_type='cubic', comm=comm,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def phantom(nzo, nxy):
    """A few smooth Gaussian blobs: band-limited enough that the 80 grid and the
    160 grid describe the same object, so run B is not fighting the sampling."""
    rng = np.random.default_rng(7)
    zz = (np.arange(nzo) - nzo / 2) / nzo
    xx = (np.arange(nxy) - nxy / 2) / nxy
    Z, Y, X = np.meshgrid(zz, xx, xx, indexing='ij')
    v = np.zeros((nzo, nxy, nxy), 'float32')
    for _ in range(6):
        c = rng.uniform(-0.22, 0.22, 3)
        s = rng.uniform(0.08, 0.16)
        v += np.exp(-((Z - c[0])**2 + (Y - c[1])**2 + (X - c[2])**2) / (2 * s * s))
    v /= v.max()
    return (-1e-6 * v + 1j * 1e-8 * v).astype('complex64')     # delta, beta scale


def probe():
    x = np.fft.fftfreq(n) * n
    XX, YY = np.meshgrid(x, x, indexing='ij')
    p = np.exp(-(XX**2 + YY**2) / (2 * (n / 3.0)**2)).astype('complex64')
    p = np.tile(p, (ndist, 1, 1))
    return p / np.mean(np.abs(p), axis=(1, 2))[:, None, None]


def run(nobj, upsample, data, path_out=None, ckpt_step=-1):
    """One BH run from obj=0, prb=1 on the data handed in.  Returns cl."""
    a = make_args(nobj, upsample, checkpoint_step=ckpt_step)
    if path_out:
        a.path_out = path_out
    cl = Rec(a)
    cl.data[:] = data
    cl.vars['prb'][:] = PRB
    cl.vars['pos'][:] = POS + POS_ERR
    cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)
    cl.vars['obj'][:] = 0
    cl.vars['prb'][:] = 1
    w = None
    if path_out:
        w = Writer(path_out, comm, cl.st_obj, cl.end_obj, nzobj, nobj,
                   cl.st_theta, cl.end_theta, ntheta, ndist, nz, n)
    cl.BH(writer=w)
    return cl


def free(cl):
    del cl
    cp.get_default_memory_pool().free_all_blocks()


ap = argparse.ArgumentParser()
ap.add_argument('--scratch', default='/local/ssd/vnikitin/upsample_e2e')
opt = ap.parse_args()
set_log_level('WARNING')
os.makedirs(opt.scratch, exist_ok=True)

rng = np.random.default_rng(10)
POS = (12 * (rng.random((ndist, ntheta, 2), dtype='float32') - 0.5))
POS_ERR = (rng.random((ndist, ntheta, 2), dtype='float32') - 0.5)
PRB = probe()

# ---- forward-model the data once, on the FINE object grid -------------------
print(f"generating data: obj [{nzobj}, {NDOBJ}, {NDOBJ}] -> "
      f"proj [{ntheta}, {nzobj}, {NDOBJ}] -> data [{ndist}, {ntheta}, {nz}, {n}]")
gen = Rec(make_args(NDOBJ, 1))
gen.vars['obj'][:] = phantom(nzobj, NDOBJ)
gen.vars['prb'][:] = PRB
gen.vars['pos'][:] = POS
gen.gen_data(gen.vars, gen.data)
gen.cl_prb_term.gen_sqrt_ref(gen.vars['prb'], gen.ref)
DATA = np.array(gen.data)
free(gen)

# ---- 1/2. both configurations run, and err goes down ------------------------
print("1. both configurations run and err decreases")
errs = {}
for tag, nobj, up, po in (("upsample=1, nobj=160", NDOBJ, 1, None),
                          ("upsample=2, nobj=80 ", NDOBJ // 2, 2,
                           os.path.join(opt.scratch, 'up2'))):
    cl = run(nobj, up, DATA, path_out=po, ckpt_step=(4 if po else -1))
    e = np.asarray(cl.table['err'], dtype='float64')
    errs[tag] = e
    check(f"{tag}: ndobj == {NDOBJ}", cl.ndobj == NDOBJ, f"ndobj={cl.ndobj}")
    check(f"{tag}: err decreases", e[-1] < e[0],
          f"{e[0]:.4e} -> {e[-1]:.4e}  ({100*(1-e[-1]/e[0]):.1f}% down)")
    free(cl)

# The two runs fit the same data with the same projection plane, so their errors
# should be in the same ballpark -- B is not solving an easier or a broken
# problem, it just has 4x fewer unknowns in x/y.
ra, rb = (errs[k][-1] / errs[k][0] for k in errs)
print(f"     (informational) final/initial err: upsample 1 {ra:.4f}, "
      f"upsample 2 {rb:.4f}")

# ---- 3. the checkpoint run B wrote reads back -------------------------------
print("3. read_checkpoint on the upsample=2 checkpoint")
ck_dir = os.path.join(opt.scratch, 'up2', 'checkpoints')
cks = sorted(f for f in os.listdir(ck_dir) if f.startswith('checkpoint_'))
ck = os.path.join(ck_dir, cks[-1])
with h5py.File(ck, 'r') as f:
    shp = f['obj_re'].shape
check("checkpoint obj shape is the OBJECT grid", shp == (nzobj, NDOBJ // 2, NDOBJ // 2),
      f"{shp}")

# Reader wants an input file for the acquisition parameters; write a stub.
stub = os.path.join(opt.scratch, 'stub.h5')
with h5py.File(stub, 'w') as f:
    g = f.create_group('exchange')
    g.create_dataset('detector_pixelsize', data=[detector_pixelsize])
    g.create_dataset('focusdetectordistance', data=[focustodetectordistance])
    g.create_dataset('z1', data=z1)
    g.create_dataset('energy', data=[energy])
    g.create_dataset('theta', data=np.rad2deg(theta)[:, None])

rd = Reader(stub, comm, 0, nzobj, nzobj, NDOBJ // 2, 0, ntheta, ntheta,
            ndist, nz, n, 0, 0.0, 0, 0)
obj = np.zeros((nzobj, NDOBJ // 2, NDOBJ // 2), 'complex64')
prb = np.zeros((ndist, nz, n), 'complex64')
pos = np.zeros((ndist, ntheta, 2), 'float32')
rd.read_checkpoint(ck, out_obj=obj, out_prb=prb, out_pos=pos)
with h5py.File(ck, 'r') as f:
    ref = f['obj_re'][:] + 1j * f['obj_im'][:]
    pref = f['pos'][:].transpose(1, 0, 2)
check("obj round-trips unchanged at scale 1",
      float(np.abs(obj - ref).max()) == 0.0, f"max|diff|={np.abs(obj - ref).max():g}")
check("pos round-trips unchanged at scale 1", np.array_equal(pos, pref))

# ---- 4. the bin1 -> bin0 handoff: three different scales --------------------
print("4. bin1 -> bin0 read: obj z x2, obj x/y x1, prb x2, pos x2")
half = os.path.join(opt.scratch, 'half.h5')
nzh, nxy, nh = nzobj // 2, NDOBJ // 2, n // 2
r0 = np.random.default_rng(11)
o_re, o_im = (r0.random((nzh, nxy, nxy), dtype='float32') for _ in range(2))
p_ab, p_ph = (r0.random((ndist, nh, nh), dtype='float32') for _ in range(2))
p_pos = r0.random((ntheta, ndist, 2), dtype='float32').astype('float32')
with h5py.File(half, 'w') as f:
    f.attrs['iter'] = 4
    for k, v in (('obj_re', o_re), ('obj_im', o_im), ('prb_abs', p_ab),
                 ('prb_phase', p_ph), ('pos', p_pos)):
        f.create_dataset(k, data=v)

rd2 = Reader(stub, comm, 0, nzobj, nzobj, nxy, 0, ntheta, ntheta,
             ndist, nz, n, 0, 0.0, 0, 0)
obj2 = np.zeros((nzobj, nxy, nxy), 'complex64')
prb2 = np.zeros((ndist, nz, n), 'complex64')
pos2 = np.zeros((ndist, ntheta, 2), 'float32')
rd2.read_checkpoint(half, out_obj=obj2, out_prb=prb2, out_pos=pos2)

exp_obj = np.repeat(o_re + 1j * o_im, 2, axis=0)          # z doubled, x/y as-is
check("obj z doubled, x/y untouched",
      obj2.shape == exp_obj.shape and float(np.abs(obj2 - exp_obj).max()) == 0.0,
      f"{obj2.shape} vs {exp_obj.shape}, max|diff|="
      f"{np.abs(obj2 - exp_obj).max() if obj2.shape == exp_obj.shape else float('nan'):g}")
exp_prb = np.repeat(np.repeat(p_ab * np.exp(1j * p_ph), 2, axis=1), 2, axis=2)
check("prb doubled in y and x",
      float(np.abs(prb2 - exp_prb.astype('complex64')).max()) == 0.0)
check("pos scaled by 2",
      np.array_equal(pos2, p_pos.transpose(1, 0, 2) * 2))

# ---- 5. read_obj bins step 5's projection-grid init onto the object grid ----
# Step 5 is untouched by tomo_upsample: it writes obj_init at the PROJECTION
# width, and read_obj averages it down 2x2 in x/y (never in z).  Averaging, not
# summing, is what keeps the obj values grid-independent.
print("5. read_obj bins the projection-grid obj_init onto the object grid")
objf = stub.replace('.h5', '_obj.h5')
r5 = np.random.default_rng(12)
init_re = r5.random((nzobj, NDOBJ, NDOBJ), dtype='float32')
init_im = r5.random((nzobj, NDOBJ, NDOBJ), dtype='float32')
with h5py.File(objf, 'w') as f:
    f.create_dataset('/exchange/obj_init_re0_0', data=init_re)
    f.create_dataset('/exchange/obj_init_im0_0', data=init_im)

exp = (init_re + 1j * init_im).reshape(
    nzobj, NDOBJ // 2, 2, NDOBJ // 2, 2).mean(axis=(2, 4)).astype('complex64')
rd5 = Reader(stub, comm, 0, nzobj, nzobj, NDOBJ // 2, 0, ntheta, ntheta,
             ndist, nz, n, 0, 0.0, 0, 0, tomo_upsample=2)
got = rd5.read_obj()
check("upsample=2: shape is the object grid", got.shape == (nzobj, NDOBJ // 2, NDOBJ // 2),
      f"{got.shape}")
check("upsample=2: x/y averaged 2x2, z untouched",
      float(np.abs(got - exp).max()) < 1e-6, f"max|diff|={np.abs(got - exp).max():g}")

rd6 = Reader(stub, comm, 0, nzobj, nzobj, NDOBJ, 0, ntheta, ntheta,
             ndist, nz, n, 0, 0.0, 0, 0)
got1 = rd6.read_obj()
check("upsample=1: read unchanged (regression)",
      float(np.abs(got1 - (init_re + 1j * init_im)).max()) == 0.0)

print()
if FAILED:
    print(f"{len(FAILED)} check(s) FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("all checks passed")
