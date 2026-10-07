"""probe_lap.py -- where does lam_laplacian>0 inject the 1e+2 data-fit error?

Same setup as repro_rho_cliff.py, but after each BH step it checks
  * is cl.min() deterministic (3 repeats)?
  * does vars['proj'] still equal fwd_tomo(vars['obj'])?
  * what does cl.min() report if proj is recomputed from obj?
  * are grads/etas finite, and how far is the lam>0 gradient from lam=0?
"""
import argparse, os, sys
import numpy as np, cupy as cp
from scipy.fft import fftn, ifftn, fftshift, fft2, ifft2
import scipy.ndimage as ndimage
from mpi4py import MPI
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from holotomocupy.rec_mpi import Rec
from holotomocupy.utils import read_tiff, make_pinned
from holotomocupy.logger_config import set_log_level
set_log_level(os.environ.get('LOGLVL', 'WARNING'))

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=512)
p.add_argument('--ntheta', type=int, default=450)
p.add_argument('--niter', type=int, default=3)
p.add_argument('--lam-lap', type=float, default=2.5e-5)
p.add_argument('--rho-prb', type=float, default=0.05)
p.add_argument('--prb-dir', default='/home/beams2/VNIKITIN/data/prb_id16a')
c = p.parse_args()

n, ntheta, ndist, nobj = c.n, c.ntheta, 4, c.n
energy = 33.35; focustodetectordistance = 1.289
z1 = np.array([0.611, 0.637, 0.742, 0.960]) * 1e-2
detector_pixelsize = 11808.118e-9 * (512 / n)

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
    cc = (np.array(vol.shape) - 1) / 2.0
    return ndimage.affine_transform(vol, A, offset=cc - A @ cc, order=1, mode="constant", cval=0.0)

def gen_object(n, delta, beta):
    obj = np.zeros((n, n, n), dtype=np.float32)
    amps = np.array([3, -3, 1, 3, -4, 1, 4], dtype=np.float32)
    dil = (np.array([33, 28, 25, 21, 16, 10, 3], dtype=np.float32) / 256.0) * n
    ax = np.arange(-n//2, n//2, dtype=np.float32)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij"); r2 = x*x + y*y + z*z; del x, y, z
    fcirc = [fftn(fftshift((r2 < d*d).astype(np.float32)), workers=-1).astype(np.complex64) for d in dil]
    cube = np.zeros((n, n, n), dtype=np.float32); fcube = []
    for _ in amps:
        cube.fill(0.0); r = int(n * 0.2); _frame(cube, n//2 - r//2, n//2 + r//2)
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
    obj = ifftn(fftn(obj) * filt).real; obj[obj < 0] = 0
    return (obj * (-delta + 1j*beta)).astype(np.complex64)

obj = gen_object(nobj, 1, 1e-2)
prb_abs = read_tiff(f'{c.prb_dir}/prb_abs_2048.tiff')[:ndist]
prb_phase = read_tiff(f'{c.prb_dir}/prb_phase_2048.tiff')[:ndist]
prb = (prb_abs * np.exp(1j * prb_phase)).astype('complex64')
cc = prb.shape[1] // 2; prb = prb[:, cc-n//2:cc+n//2, cc-n//2:cc+n//2]
v = np.arange(-n//2, n//2, dtype=np.float32) / n
vx, vy = np.meshgrid(v, v, indexing="ij")
prb = ifft2(fft2(prb) * fftshift(np.exp(-4.0*(vx*vx+vy*vy)).astype(np.float32)))
prb /= np.mean(np.abs(prb), axis=(1, 2))[:, None, None]

np.random.seed(10)
pos = 30 * (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)
pos_err = (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)
theta = np.linspace(0, np.pi, ntheta, dtype='float32')

a = SimpleNamespace()
a.energy = energy; a.detector_pixelsize = detector_pixelsize
a.focustodetectordistance = focustodetectordistance; a.z1 = z1
a.theta = theta; a.ndist = ndist; a.ntheta = ntheta
a.nz = n; a.n = n; a.nzobj = nobj; a.nobj = nobj
a.mask = 1.1; a.lam_prbfit = 3.1e-3; a.lam_laplacian = c.lam_lap
a.rho = [1, 0.05, 0.02, 0]; a.niter = c.niter; a.start_iter = 0; a.nchunk = 16
a.checkpoint_step = -1; a.error_step = -1
a.mask_oob = True; a.mask_oob_margin = 2
a.check_fused_hessian = False
a.comm = MPI.COMM_WORLD

cl = Rec(a)
if os.environ.get('NOCOEFF', '0') == '1':
    cl.cl_shift.coeff_cached = cl.cl_shift.coeff
    print("   [coeff cache DISABLED]")
if os.environ.get('NOAPPLYF', '0') == '1':
    def _nc(x, from_level):
        if from_level >= len(cl.F):
            return x
        return cl.F[from_level](_nc(x, from_level + 1))
    cl.apply_F_from = _nc
    print("   [apply_F cache DISABLED]")
cl.vars['obj'][:] = obj[cl.st_obj:cl.end_obj]
cl.vars['prb'][:] = prb
cl.vars['pos'][:] = pos[cl.st_theta:cl.end_theta].transpose(1, 0, 2)
cl.gen_data(cl.vars, cl.data)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)

obj0 = ndimage.gaussian_filter(obj.real, 2) + 1j*ndimage.gaussian_filter(obj.imag, 2)
cl.vars['obj'][:] = obj0[cl.st_obj:cl.end_obj].astype('complex64')
cl.vars['prb'][:] = 1
cl.vars['pos'][:] = (pos + pos_err)[cl.st_theta:cl.end_theta].transpose(1, 0, 2)

vars = cl.vars
cl.precalc(vars)
cl.rho_sq = {'obj': 1.0, 'prb': c.rho_prb**2, 'pos': 0.02**2, 'tp': 0.0}
cl.start_iter = 0

proj2 = make_pinned(vars['proj'].shape, dtype='complex64')

def nrm(x):
    x = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
    return float(cp.linalg.norm(x.ravel()))

def E(pr=None):
    return float(cl.min(vars['prb'], vars['obj'], vars['pos'],
                        vars['proj'] if pr is None else pr))

def report(tag):
    e = [E() for _ in range(5)]
    lap = float(cl.cl_lap_term.energy_local()) if hasattr(cl, 'cl_lap_term') else 0.0
    pf  = float(cl.cl_prb_term.energy_local(vars['prb']))
    cl.fwd_tomo(vars['obj'], out=cl.proj_tmp)
    cl.redist(cl.proj_tmp, proj2)
    d = nrm(cp.asarray(proj2) - cp.asarray(vars['proj'])) / max(nrm(proj2), 1e-30)
    e_re = E(proj2)
    print(f"{tag}")
    print("   min() x5      : " + " ".join(f"{v:.6e}" for v in e))
    print(f"   prbfit / lap  : {pf:.4e} / {lap:.4e}   -> F0 = {e[0]-pf-lap:.6e}")
    print(f"   ||proj-T obj||/||T obj|| = {d:.4e}")
    print(f"   min() with proj := T obj : {e_re:.6e}   (F0 = {e_re-pf-lap:.6e})")
    print(f"   finite: obj={np.isfinite(vars['obj'].view('float32')).all()} "
          f"proj={np.isfinite(vars['proj'].view('float32')).all()} "
          f"max|obj|={np.abs(vars['obj']).max():.4e} max|proj|={np.abs(vars['proj']).max():.4e}",
          flush=True)

RAY = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]

def ray_scan(alpha, top, bottom, snap):
    print("   ray scan: F(t*alpha) vs quadratic model  (model = F(0) - t*top^2/bottom + t^2*top^2/(2 bottom))")
    q = top * top / bottom
    f0 = None
    for t in RAY:
        for k, v in vars.items():
            v[:] = snap[k]
        if t != 0.0:
            cl.apply_step(vars, cl.etas, t * alpha)
        e = E()
        if t == 0.0:
            f0 = e
        m = f0 - t * q + t * t * q / 2
        cu = (e - f0 + t*q)/(t*t) if t else float("nan")
        print(f"      t={t:9.1e}  F={e:12.6e}   model={m:12.6e}   curvature C(t)={cu:11.4e}  (model C={q/2:.4e})", flush=True)
    for k, v in vars.items():
        v[:] = snap[k]

print(f"\n=== lam_laplacian={c.lam_lap}  n={n} ntheta={ntheta} rho_prb={c.rho_prb} ===")
report("initial")
for i in range(c.niter):
    cl.compute_gradient(vars, cl.grads)
    alpha, top, bottom = cl.compute_step(vars, cl.grads, cl.etas, i)
    if hasattr(cl, 'cl_lap_term'):
        gl = np.zeros_like(cl.grads['obj'])
        cl.cl_lap_term.gradient(gl)
        gl = cp.asarray(gl); go = cp.asarray(cl.grads['obj'])
        gd = go - gl
        print(f"   grad split: |g_total|={nrm(go):.5e} |g_data|={nrm(gd):.5e} "
              f"|g_lap|={nrm(gl):.5e}  ratio={nrm(gl)/nrm(gd):.4e}  "
              f"cos={float(cp.real(cp.vdot(gl,gd)))/(nrm(gl)*nrm(gd)+1e-30):+.4f}")
        del gl, go, gd
        cp.get_default_memory_pool().free_all_blocks()
    print(f"\n-- iter {i}: alpha={alpha:.5e} top={top:.5e} bottom={bottom:.5e} "
          f"|g_obj|={nrm(cl.grads['obj']):.4e} |e_obj|={nrm(cl.etas['obj']):.4e} "
          f"|e_proj|={nrm(cl.etas['proj']):.4e}")
    # is etas['proj'] consistent with fwd_tomo(etas['obj'])?
    cl.fwd_tomo(cl.etas['obj'], out=cl.proj_tmp)
    cl.redist(cl.proj_tmp, proj2)
    print(f"   ||e_proj - T e_obj||/||T e_obj|| = "
          f"{nrm(cp.asarray(proj2)-cp.asarray(cl.etas['proj']))/max(nrm(proj2),1e-30):.4e}")
    if os.environ.get('PROOF', '0') == '1' and i == 0:
        from holotomocupy.utils import lap as _lap, redot as _redot
        cl.apply_step(vars, cl.etas, alpha)
        self = cl; prb = vars['prb']; pos = vars['pos']; proj = vars['proj']; tp = vars['tp']
        def accum():
            out = cp.zeros(1, dtype="float32")
            for k0, k1 in self._dist_groups():
                nd = k1 - k0
                @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                def _md(self, out, c_proj, *a):
                    c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                    c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                    self.cl_shift.coeff_cache_reset()
                    for j in range(nd):
                        self._dist_idx = k0 + j
                        self._t_chunk = c_t
                        self.apply_F_cache_reset()
                        out[:] += self.F0(self.apply_F_from(
                            [c_prb[j], c_proj, c_pos[j], c_tp[j]], 1), c_data[j])
                ks = range(k0, k1)
                _md(self, out, proj, *[pos[k] for k in ks],
                    *[self.data[k] for k in ks], self.t_local,
                    prb[k0:k1], tp[k0:k1])
            return out
        lt = self.cl_lap_term
        def go(keep_ep):
            o = accum()[0]
            ep = self.cl_prb_term.energy_local(prb)
            eptr = ep.data.ptr
            o += ep
            if not keep_ep:
                del ep                      # exactly what the real `o += f()` does
            lt.exchange_ghosts(lt.u_pad)
            acc = cp.zeros(1, dtype='float32')
            recycled = (acc.data.ptr == eptr)
            @lt.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
            def _be(self, acc, u):
                l = _lap(u[1:-3], u[2:-2], u[3:-1])
                acc[:] += _redot(l, l)
            _be(lt, acc, lt.u_pad)
            return recycled, float(o), float(acc[0])
        for keep in (False, True):
            rs = [go(keep) for _ in range(4)]
            print(f"   keep_ep={keep!s:5s}: " + "  ".join(
                f"[acc==ep_block:{r[0]!s:5s} o={r[1]:.6e}]" for r in rs), flush=True)
        raise SystemExit(0)
    if os.environ.get('KERN', '0') == '1' and i == 0:
        from holotomocupy.utils import lap as _lap, redot as _redot, reprod as _reprod
        cl.apply_step(vars, cl.etas, alpha)
        self = cl; prb = vars['prb']; pos = vars['pos']; proj = vars['proj']; tp = vars['tp']
        def accum():
            out = cp.zeros(1, dtype="float32")
            for k0, k1 in self._dist_groups():
                nd = k1 - k0
                @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                def _md(self, out, c_proj, *a):
                    c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                    c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                    self.cl_shift.coeff_cache_reset()
                    for j in range(nd):
                        self._dist_idx = k0 + j
                        self._t_chunk = c_t
                        self.apply_F_cache_reset()
                        out[:] += self.F0(self.apply_F_from(
                            [c_prb[j], c_proj, c_pos[j], c_tp[j]], 1), c_data[j])
                ks = range(k0, k1)
                _md(self, out, proj, *[pos[k] for k in ks],
                    *[self.data[k] for k in ks], self.t_local,
                    prb[k0:k1], tp[k0:k1])
            return out
        lt = self.cl_lap_term
        KS = {
          'vdot (redot)'  : lambda a, u: a.__setitem__(slice(None), a[:] + _redot(_lap(u[1:-3],u[2:-2],u[3:-1]), _lap(u[1:-3],u[2:-2],u[3:-1]))),
          'cp.sum(reprod)': lambda a, u: a.__setitem__(slice(None), a[:] + cp.sum(_reprod(_lap(u[1:-3],u[2:-2],u[3:-1]), _lap(u[1:-3],u[2:-2],u[3:-1])))),
          'no acc write'  : lambda a, u: _lap(u[1:-3],u[2:-2],u[3:-1]),
          'acc += 1.0'    : lambda a, u: a.__setitem__(slice(None), a[:] + np.float32(1.0)),
        }
        for nm, kf in KS.items():
            vals = []
            for r in range(3):
                o = accum()[0]
                o += self.cl_prb_term.energy_local(prb)
                lt.exchange_ghosts(lt.u_pad)
                acc = cp.zeros(1, dtype='float32')
                @lt.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
                def _be(self, acc, u):
                    kf(acc, u)
                _be(lt, acc, lt.u_pad)
                vals.append(float(o))
            print(f"   {nm:16s}: o after = " + " ".join(f"{v:.6e}" for v in vals), flush=True)
        raise SystemExit(0)
    if os.environ.get('CANARY', '0') == '1' and i == 0:
        from holotomocupy.utils import lap as _lap, redot as _redot
        cl.apply_step(vars, cl.etas, alpha)
        self = cl; prb = vars['prb']; pos = vars['pos']; proj = vars['proj']; tp = vars['tp']
        mp = cp.get_default_memory_pool()
        nfab = [0]
        def accum():
            out = cp.zeros(1, dtype="float32")
            for k0, k1 in self._dist_groups():
                nd = k1 - k0
                @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                def _md(self, out, c_proj, *a):
                    c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                    c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                    self.cl_shift.coeff_cache_reset()
                    for j in range(nd):
                        self._dist_idx = k0 + j
                        self._t_chunk = c_t
                        self.apply_F_cache_reset()
                        out[:] += self.F0(self.apply_F_from(
                            [c_prb[j], c_proj, c_pos[j], c_tp[j]], 1), c_data[j])
                ks = range(k0, k1)
                _md(self, out, proj, *[pos[k] for k in ks],
                    *[self.data[k] for k in ks], self.t_local,
                    prb[k0:k1], tp[k0:k1])
            return out
        lt = self.cl_lap_term
        for r in range(2):
            parent = accum()
            o = parent[0]
            can = [cp.zeros(1, dtype='float32') for _ in range(12)]
            for c in can: c.fill(-1.0)
            cptr = [c.data.ptr for c in can]
            print(f"   r{r}: o={hex(o.data.ptr)} canaries={[hex(p) for p in cptr]}", flush=True)
            o += self.cl_prb_term.energy_local(prb)
            nfab[0] = mp.used_bytes() // 2**20
            lt.exchange_ghosts(lt.u_pad)
            acc = cp.zeros(1, dtype='float32')
            @lt.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
            def _be(self, acc, u):
                l = _lap(u[1:-3], u[2:-2], u[3:-1])
                acc[:] += _redot(l, l)
            _be(lt, acc, lt.u_pad)
            bad = [(q, float(c[0])) for q, c in enumerate(can) if float(c[0]) != -1.0]
            print(f"        acc={float(acc[0]):.6e} o={float(o):.6e} "
                  f"pool_used_MB={nfab[0]} clobbered_canaries={bad}", flush=True)
        raise SystemExit(0)
    if os.environ.get('PTR', '0') == '1' and i == 0:
        from holotomocupy.utils import lap as _lap, redot as _redot
        cl.apply_step(vars, cl.etas, alpha)
        self = cl; prb = vars['prb']; pos = vars['pos']; proj = vars['proj']; tp = vars['tp']
        def accum():
            out = cp.zeros(1, dtype="float32")
            for k0, k1 in self._dist_groups():
                nd = k1 - k0
                @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                def _md(self, out, c_proj, *a):
                    c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                    c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                    self.cl_shift.coeff_cache_reset()
                    for j in range(nd):
                        self._dist_idx = k0 + j
                        self._t_chunk = c_t
                        self.apply_F_cache_reset()
                        out[:] += self.F0(self.apply_F_from(
                            [c_prb[j], c_proj, c_pos[j], c_tp[j]], 1), c_data[j])
                ks = range(k0, k1)
                _md(self, out, proj, *[pos[k] for k in ks],
                    *[self.data[k] for k in ks], self.t_local,
                    prb[k0:k1], tp[k0:k1])
            return out
        lt = self.cl_lap_term
        for r in range(3):
            parent = accum()
            o = parent[0]
            print(f"   r{r}: out ptr={hex(parent.data.ptr)} o ptr={hex(o.data.ptr)} val={float(o):.6e}")
            o += self.cl_prb_term.energy_local(prb)
            scale = np.float32(lt.lam / lt.obj_size)
            lt.exchange_ghosts(lt.u_pad)
            acc = cp.zeros(1, dtype='float32')
            print(f"        acc ptr={hex(acc.data.ptr)}   ALIAS={acc.data.ptr == o.data.ptr}")
            @lt.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
            def _be(self, acc, u):
                acc[:] += _redot(_lap(u[1:-3], u[2:-2], u[3:-1]), _lap(u[1:-3], u[2:-2], u[3:-1]))
            _be(lt, acc, lt.u_pad)
            print(f"        raw acc={float(acc[0]):.6e}  o now={float(o):.6e}")
            o += scale * float(acc[0])
            print(f"        final={float(np.array(o.get(), dtype='float32')):.6e}", flush=True)
        raise SystemExit(0)
    if os.environ.get('TAIL3', '0') == '1' and i == 0:
        cl.apply_step(vars, cl.etas, alpha)
        self = cl; prb = vars['prb']; pos = vars['pos']; proj = vars['proj']; tp = vars['tp']
        def accum():
            out = cp.zeros(1, dtype="float32")
            for k0, k1 in self._dist_groups():
                nd = k1 - k0
                @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                def _md(self, out, c_proj, *a):
                    c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                    c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                    self.cl_shift.coeff_cache_reset()
                    for j in range(nd):
                        self._dist_idx = k0 + j
                        self._t_chunk = c_t
                        self.apply_F_cache_reset()
                        out[:] += self.F0(self.apply_F_from(
                            [c_prb[j], c_proj, c_pos[j], c_tp[j]], 1), c_data[j])
                ks = range(k0, k1)
                _md(self, out, proj, *[pos[k] for k in ks],
                    *[self.data[k] for k in ks], self.t_local,
                    prb[k0:k1], tp[k0:k1])
            return out
        def V(tag, mode):
            r = []
            for _ in range(6):
                o = accum()[0]
                if mode == 'none':      pass
                elif mode == 'prb':     o += self.cl_prb_term.energy_local(prb)
                elif mode == 'lap':     o += self.cl_lap_term.energy_local()
                elif mode == 'both':
                    o += self.cl_prb_term.energy_local(prb)
                    o += self.cl_lap_term.energy_local()
                elif mode == 'both_mid':
                    o += self.cl_prb_term.energy_local(prb); float(o)
                    o += self.cl_lap_term.energy_local()
                elif mode == 'lap_first':
                    o += self.cl_lap_term.energy_local()
                    o += self.cl_prb_term.energy_local(prb)
                elif mode == 'lap_float':
                    o += float(self.cl_lap_term.energy_local())
                r.append(float(np.array(o.get(), dtype='float32')))
            print(f"   {tag:24s}: " + " ".join(f"{x:.6e}" for x in r), flush=True)
        for t in ['none', 'prb', 'lap', 'both', 'both_mid', 'lap_first']:
            V(t, t)
        raise SystemExit(0)
    if os.environ.get('TAIL2', '0') == '1' and i == 0:
        cl.apply_step(vars, cl.etas, alpha)
        self = cl; prb = vars['prb']; pos = vars['pos']; proj = vars['proj']; tp = vars['tp']
        def run(assert_g, do_ar, read_mid):
            out = cp.zeros(1, dtype="float32")
            for k0, k1 in self._dist_groups():
                nd = k1 - k0
                if assert_g: self._assert_dist_group(nd)
                @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                def _md(self, out, c_proj, *a):
                    c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                    c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                    self.cl_shift.coeff_cache_reset()
                    for j in range(nd):
                        self._dist_idx = k0 + j
                        self._t_chunk = c_t
                        self.apply_F_cache_reset()
                        out[:] += self.F0(self.apply_F_from(
                            [c_prb[j], c_proj, c_pos[j], c_tp[j]], 1), c_data[j])
                ks = range(k0, k1)
                _md(self, out, proj, *[pos[k] for k in ks],
                    *[self.data[k] for k in ks], self.t_local,
                    prb[k0:k1], tp[k0:k1])
            mid = float(out[0]) if read_mid else None
            o = out[0]
            o += self.cl_prb_term.energy_local(prb)
            if hasattr(self, 'cl_lap_term'): o += self.cl_lap_term.energy_local()
            g = np.array(o.get(), dtype='float32')
            v = float(self.allreduce(g)) if do_ar else float(g)
            return mid, v
        for nm, ag, ar, rm in [('read_mid, no assert, no allreduce', 0,0,1),
                               ('read_mid, no assert, allreduce   ', 0,1,1),
                               ('read_mid, assert,    allreduce   ', 1,1,1),
                               ('NO read_mid, assert, allreduce   ', 1,1,0)]:
            res = [run(ag, ar, rm) for _ in range(6)]
            print(f"   {nm}: " + " ".join(
                ("-" if m is None else f"{m:.4e}") + f"/{v:.6e}" for m, v in res), flush=True)
        raise SystemExit(0)
    if os.environ.get('TAIL', '0') == '1' and i == 0:
        cl.apply_step(vars, cl.etas, alpha)
        self = cl; prb = vars['prb']; pos = vars['pos']; proj = vars['proj']; tp = vars['tp']
        for r in range(12):
            out = cp.zeros(1, dtype="float32")
            for k0, k1 in self._dist_groups():
                nd = k1 - k0
                @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                def _md(self, out, c_proj, *a):
                    c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                    c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                    self.cl_shift.coeff_cache_reset()
                    for j in range(nd):
                        self._dist_idx = k0 + j
                        self._t_chunk = c_t
                        self.apply_F_cache_reset()
                        out[:] += self.F0(self.apply_F_from(
                            [c_prb[j], c_proj, c_pos[j], c_tp[j]], 1), c_data[j])
                ks = range(k0, k1)
                _md(self, out, proj, *[pos[k] for k in ks],
                    *[self.data[k] for k in ks], self.t_local,
                    prb[k0:k1], tp[k0:k1])
            a1 = float(out[0])
            o = out[0]
            ep = self.cl_prb_term.energy_local(prb)
            o += ep
            a2 = float(o)
            el = self.cl_lap_term.energy_local() if hasattr(self, 'cl_lap_term') else 0.0
            o += el
            a3 = float(o)
            a4 = float(o.get())
            print(f"   r{r:2d} accum={a1:.6e} +prbfit({float(ep):.4e})={a2:.6e} "
                  f"+lap({float(el):.3e})={a3:.6e} get={a4:.6e}", flush=True)
        raise SystemExit(0)
    if os.environ.get('BISECT', '0') == '1' and i == 0:
        cl.apply_step(vars, cl.etas, alpha)
        def mk(read_v, read_out, final_sync):
            def m(self, prb, obj, pos, proj):
                out = cp.zeros(1, dtype="float32")
                tp = self.vars['tp']
                for k0, k1 in self._dist_groups():
                    nd = k1 - k0
                    self._assert_dist_group(nd)
                    @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                    def _md(self, out, c_proj, *a):
                        c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                        c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                        self.cl_shift.coeff_cache_reset()
                        for j in range(nd):
                            self._dist_idx = k0 + j
                            self._t_chunk = c_t
                            self.apply_F_cache_reset()
                            x = [c_prb[j], c_proj, c_pos[j], c_tp[j]]
                            y = self.apply_F_from(x, 1)
                            v = self.F0(y, c_data[j])
                            if read_v: float(v)
                            out[:] += v
                            if read_out: float(out[0])
                    ks = range(k0, k1)
                    _md(self, out, proj, *[pos[k] for k in ks],
                        *[self.data[k] for k in ks], self.t_local,
                        prb[k0:k1], tp[k0:k1])
                if final_sync: cp.cuda.Device().synchronize()
                out = out[0]
                if self.rank == 0 and hasattr(self, 'cl_prb_term'):
                    out += self.cl_prb_term.energy_local(prb)
                if hasattr(self, 'cl_lap_term'):
                    out += self.cl_lap_term.energy_local()
                return self.allreduce(np.array(out.get(), dtype='float32'))
            return m
        V = [('real cl.min          ', None),
             ('copy verbatim        ', mk(0,0,0)),
             ('copy +final devsync  ', mk(0,0,1)),
             ('copy +float(v)       ', mk(1,0,0)),
             ('copy +float(out[0])  ', mk(0,1,0))]
        for nm, f in V:
            r = [(E() if f is None else f(cl, vars['prb'], vars['obj'], vars['pos'], vars['proj']))
                 for _ in range(3)]
            print(f"   {nm}: " + " ".join(f"{float(x):.6e}" for x in r), flush=True)
        raise SystemExit(0)
    if os.environ.get('ACC', '0') == '1' and i == 0:
        cl.apply_step(vars, cl.etas, alpha)
        def min_dbg(self, prb, obj, pos, proj):
            out = cp.zeros(1, dtype="float32")
            tp = self.vars['tp']
            parts = []
            for k0, k1 in self._dist_groups():
                nd = k1 - k0
                self._assert_dist_group(nd)
                @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
                def _md(self, out, c_proj, *a):
                    c_pos = a[0*nd:1*nd]; c_data = a[1*nd:2*nd]
                    c_t = a[2*nd]; c_prb = a[2*nd+1]; c_tp = a[2*nd+2]
                    self.cl_shift.coeff_cache_reset()
                    for j in range(nd):
                        self._dist_idx = k0 + j
                        self._t_chunk = c_t
                        self.apply_F_cache_reset()
                        x = [c_prb[j], c_proj, c_pos[j], c_tp[j]]
                        y = self.apply_F_from(x, 1)
                        v = self.F0(y, c_data[j])
                        parts.append(float(v))
                        before = float(out[0])
                        out[:] += v
                        after = float(out[0])
                        if abs(after - before - float(v)) > 1e-6*max(1.0, abs(after)):
                            parts.append(('BAD', before, float(v), after))
                ks = range(k0, k1)
                _md(self, out, proj, *[pos[k] for k in ks],
                    *[self.data[k] for k in ks], self.t_local,
                    prb[k0:k1], tp[k0:k1])
            acc = float(out[0])
            pysum = sum(p for p in parts if not isinstance(p, tuple))
            nbad = sum(1 for p in parts if isinstance(p, tuple))
            ep = float(self.cl_prb_term.energy_local(prb)) if self.rank == 0 and hasattr(self,'cl_prb_term') else 0.0
            el = float(self.cl_lap_term.energy_local()) if hasattr(self,'cl_lap_term') else 0.0
            print(f"       out_gpu={acc:.6e} pysum={pysum:.6e} nbad={nbad} "
                  f"prbfit={ep:.6e} lap={el:.6e} total={acc+ep+el:.6e}", flush=True)
            return acc + ep + el
        for r in range(3):
            print(f"   call{r}: min()={E():.6e}", flush=True)
            min_dbg(cl, vars['prb'], vars['obj'], vars['pos'], vars['proj'])
        raise SystemExit(0)
    if os.environ.get('PERCHUNK', '0') == '1' and i == 0:
        cl.apply_step(vars, cl.etas, alpha)
        from holotomocupy.rec_mpi import Rec as _R
        oF0 = _R.F0
        log = []
        def F0w(self, x, d):
            v = oF0(self, x, d)
            log.append((self._dist_idx, x.shape[0], float(cp.abs(x).max()),
                        float(d.max()), float(v)))
            return v
        cl.__class__.F0 = F0w
        for r in range(3):
            log.clear()
            v = E()
            tot = sum(t[4] for t in log)
            print(f"   call{r}: F={v:.6e}  sum(F0 parts)={tot:.6e}  nparts={len(log)}", flush=True)
            worst = sorted(range(len(log)), key=lambda q: -log[q][4])[:4]
            for q in sorted(worst):
                di, nn, mx, dmx, vv = log[q]
                print(f"       part{q:4d} dist={di} nth={nn} max|x0|={mx:.4e} "
                      f"max d={dmx:.4e} F0={vv:.4e}", flush=True)
        cl.__class__.F0 = oF0
        raise SystemExit(0)
    if os.environ.get('CORRUPT', '0') == '1' and i == 0:
        cl.apply_step(vars, cl.etas, alpha)
        def cks():
            d = {}
            for nm in ('obj', 'prb', 'pos', 'proj'):
                a = vars[nm]
                d[nm] = float(np.asarray(a, dtype=np.complex128).sum().real
                              if np.iscomplexobj(a) else np.float64(a).sum())
            d['data'] = float(np.float64(cl.data).sum())
            d['dmask'] = float(np.float64(cl.mask_1d).sum())
            if hasattr(cl, 'cl_lap_term'):
                d['u_pad'] = float(cl.cl_lap_term.u_pad.astype(np.complex128).sum().real)
                d['e_pad'] = float(cl.cl_lap_term.e_pad.astype(np.complex128).sum().real)
                d['g_pad'] = float(cl.cl_lap_term.g_pad.astype(np.complex128).sum().real)
            d['ptmp'] = float(cl.proj_tmp.astype(np.complex128).sum().real)
            return d
        base = cks()
        print("   base " + " ".join(f"{k}={v:+.10e}" for k, v in base.items()), flush=True)
        for r in range(4):
            v = E()
            now = cks()
            bad = [k for k in base if now[k] != base[k]]
            print(f"   call{r}: F={v:.6e}  changed={bad if bad else 'none'}", flush=True)
            for k in bad:
                print(f"          {k}: {base[k]:+.10e} -> {now[k]:+.10e}", flush=True)
            base = now
        raise SystemExit(0)
    if os.environ.get('CHUNKSWEEP', '0') == '1' and i == 0:
        cl.apply_step(vars, cl.etas, alpha)
        for ch in [1, 2, 3, 4, 8, 15, 16, 32]:
            cl.cl_chunking.chunk = ch
            try:
                vals = [E() for _ in range(4)]
                print(f"   nchunk={ch:4d}: " + " ".join(f"{v:.6e}" for v in vals), flush=True)
            except Exception as ex:
                print(f"   nchunk={ch:4d}: {type(ex).__name__}: {ex}", flush=True)
        raise SystemExit(0)
    if os.environ.get('VIEW', '0') == '1' and i == 0:
        from holotomocupy.utils import make_pinned as _mp
        o = vars['obj']
        print(f"   obj buffer: base={'view of pad' if o.base is not None else 'standalone'} "
              f"contig={o.flags['C_CONTIGUOUS']} offset_bytes="
              f"{o.ctypes.data - (o.base.ctypes.data if o.base is not None and hasattr(o.base,'ctypes') else o.ctypes.data)}")
        # 1) fwd_tomo on the view vs on an identical standalone pinned copy
        ocopy = _mp(o.shape, dtype='complex64'); ocopy[:] = o
        pA = _mp(vars['proj'].shape, dtype='complex64')
        pB = _mp(vars['proj'].shape, dtype='complex64')
        cl.fwd_tomo(o,     out=cl.proj_tmp); cl.redist(cl.proj_tmp, pA)
        cl.fwd_tomo(ocopy, out=cl.proj_tmp); cl.redist(cl.proj_tmp, pB)
        print(f"   fwd_tomo(view) vs fwd_tomo(copy): rel diff = "
              f"{nrm(cp.asarray(pA)-cp.asarray(pB))/max(nrm(pB),1e-30):.4e}")
        print(f"      F0 via view-proj = {E(pA):.6e}   via copy-proj = {E(pB):.6e}")
        # 2) apply_step on the view
        o0 = o.copy(); ee = cl.etas['obj'].copy()
        p0 = vars['proj'].copy(); ep = cl.etas['proj'].copy()
        cl.apply_step(vars, cl.etas, alpha)
        ref = cp.asarray(o0) + np.float32(alpha) * cp.asarray(ee)
        print(f"   apply_step obj : ||new-(old+a*eta)||/||a*eta|| = "
              f"{nrm(cp.asarray(o)-ref)/max(float(alpha)*nrm(ee),1e-30):.4e}")
        refp = cp.asarray(p0) + np.float32(alpha) * cp.asarray(ep)
        print(f"   apply_step proj: ||new-(old+a*eta)||/||a*eta|| = "
              f"{nrm(cp.asarray(vars['proj'])-refp)/max(float(alpha)*nrm(ep),1e-30):.4e}")
        # 3) after the step: fwd_tomo view vs copy again, and F0 both ways
        ocopy[:] = o
        cl.fwd_tomo(o,     out=cl.proj_tmp); cl.redist(cl.proj_tmp, pA)
        cl.fwd_tomo(ocopy, out=cl.proj_tmp); cl.redist(cl.proj_tmp, pB)
        print(f"   AFTER STEP fwd_tomo(view) vs (copy): rel diff = "
              f"{nrm(cp.asarray(pA)-cp.asarray(pB))/max(nrm(pB),1e-30):.4e}")
        def ck(x):
            a = cp.asarray(x).view('float32').astype('float64')
            return float(a.sum()), float((a*a).sum())
        print(f"      F0 stored proj = {E():.6e}  view-proj = {E(pA):.6e}  copy-proj = {E(pB):.6e}")
        print(f"   CHECKSUMS  obj  sum,ss = {ck(o)}")
        print(f"              proj sum,ss = {ck(vars['proj'])}")
        print(f"              T obj sum,ss= {ck(pA)}")
        print(f"              prb  sum,ss = {ck(vars['prb'])}")
        print(f"              pos  sum,ss = {ck(vars['pos'])}")
        print(f"              data sum,ss = {ck(cl.data)}")
        print(f"              mask sum    = {float(np.float64(cl.mask_1d).sum())}")
        print(f"              eff_demag   = {cp.asnumpy(cl.eff_demag).ravel()[:8]}")
        raise SystemExit(0)
    if os.environ.get('WALK', '0') == '1' and i == 0 and hasattr(cl, 'cl_lap_term'):
        gl = np.zeros_like(cl.grads['obj'])
        cl.cl_lap_term.gradient(gl)                       # gl = g_lap = -delta_eta
        eprb = cl.etas['prb'].copy(); epos = cl.etas['pos'].copy()
        # eta0 = eta_lam + g_lap  == the direction the lam=0 run would use
        cl.etas['obj'][:] = cl.etas['obj'] + gl
        cl.fwd_tomo(cl.etas['obj'], out=cl.proj_tmp); cl.redist(cl.proj_tmp, cl.etas['proj'])
        cl.apply_step(vars, cl.etas, alpha)
        print(f"   landing point of the lam=0 direction : F={E():.6e}   (lam=0 run gave 5.099363e-03)")
        # now walk from here to the lam>0 landing point along -alpha*g_lap
        cl.etas['obj'][:] = -gl
        cl.fwd_tomo(cl.etas['obj'], out=cl.proj_tmp); cl.redist(cl.proj_tmp, cl.etas['proj'])
        cl.etas['prb'][:] = 0; cl.etas['pos'][:] = 0
        snap = {k: v.copy() for k, v in vars.items()}
        f0 = E()
        print(f"   walk along -alpha*g_lap, |alpha*g_lap|={alpha*nrm(gl):.4e} vs ||obj||={nrm(vars['obj']):.4e}")
        for s_ in [0.0, 0.03, 0.1, 0.3, 0.6, 1.0, 1.5]:
            for k, v in vars.items(): v[:] = snap[k]
            if s_: cl.apply_step(vars, cl.etas, s_ * alpha)
            e = E()
            print(f"      s={s_:5.2f}  |step|={s_*alpha*nrm(gl):10.3e}  F={e:12.6e}", flush=True)
        raise SystemExit(0)
    if os.environ.get('DELTA', '0') == '1' and i == 0 and hasattr(cl, 'cl_lap_term'):
        # replace eta by the pure Laplacian-gradient direction, same total
        # magnitude the BH step gives it (alpha*|g_lap|), and scan F along it.
        gl = np.zeros_like(cl.grads['obj'])
        cl.cl_lap_term.gradient(gl)
        nl = nrm(gl)
        cl.etas['obj'][:] = gl
        cl.fwd_tomo(cl.etas['obj'], out=cl.proj_tmp)
        cl.redist(cl.proj_tmp, cl.etas['proj'])
        cl.etas['prb'][:] = 0; cl.etas['pos'][:] = 0
        snap = {k: v.copy() for k, v in vars.items()}
        f0 = E()
        print(f"   pure-delta scan: |delta|={nl:.5e}  |T delta|={nrm(cl.etas['proj']):.5e}  "
              f"step norm at s=1 is alpha*|delta| = {alpha*nl:.4e}  (||obj||={nrm(vars['obj']):.4e})")
        for s_ in [0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0]:
            for k, v in vars.items(): v[:] = snap[k]
            if s_: cl.apply_step(vars, cl.etas, s_ * alpha)
            e = E()
            print(f"      s={s_:7.0e}  |step|={s_*alpha*nl:10.3e}  F={e:12.6e}  "
                  f"C={(e-f0)/(s_*s_) if s_ else float('nan'):11.4e}", flush=True)
        for k, v in vars.items(): v[:] = snap[k]
        raise SystemExit(0)
    if os.environ.get('RAY', '0') == '1' and i == 0:
        snap = {k: v.copy() for k, v in vars.items()}
        ray_scan(alpha, top, bottom, snap)
    cl.apply_step(vars, cl.etas, alpha)
    report(f"   after step {i}")
