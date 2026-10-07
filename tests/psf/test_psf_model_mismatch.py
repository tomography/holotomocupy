#!/usr/bin/env python
"""Does modelling the detector PSF matter?  Generate WITH it, reconstruct both ways.

    PYTHONPATH=<repo>/src python tests/psf/test_psf_model_mismatch.py
    ... --sigma 1.2 --n 128 --ntheta 180 --niter 100

The data are made with d = K|psi|^2 for a known sigma, so the truth is known
exactly and so is the blur.  Three arms:

    gen sigma / rec sigma
      s / s     matched   -- the model can reproduce the data
      s / 0     MISMATCH  -- the solver must explain blurred data with a sharp
                             model, so it sharpens the object to compensate
      0 / 0     control   -- no blur anywhere

What this answers: the Fresnel propagator produces fringes finer than the
detector resolves.  If the PSF is left out of the model, that unmodelled
smoothing has to be absorbed somewhere, and the object is the only free thing
left.  The arms below say how much that costs and what it looks like.
"""
import argparse
import os
import sys

import numpy as np
import cupy as cp
from mpi4py import MPI
from types import SimpleNamespace

_DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'demo')
sys.path.insert(0, _DEMO)
from phantoms import phantom3d, id16a_probe                      # noqa: E402


def _paganin_volume():
    """demo/02_holotomography.py's step-5 stand-in; its name is not importable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'demo_holo', os.path.join(_DEMO, '02_holotomography.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.paganin_volume
from holotomocupy.rec_mpi import Rec                              # noqa: E402
from holotomocupy.psf import psf_taps                             # noqa: E402
from holotomocupy.logger_config import set_log_level              # noqa: E402

ENERGY, PIXEL, FOCUS_DET = 33.35, 3.7e-6, 1.28
Z1 = np.array([5.11e-3, 5.464e-3, 6.879e-3, 9.817e-3], dtype='float32')


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sigma', type=float, default=1.2, help='PSF sigma used to MAKE the data')
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--ntheta', type=int, default=180)
    p.add_argument('--ndist', type=int, default=4)
    p.add_argument('--niter', type=int, default=100)
    p.add_argument('--nchunk', type=int, default=16)
    p.add_argument('--paganin', type=float, default=100.0)
    p.add_argument('--tag', default='', help='suffix for the figure filename')
    p.add_argument('--photons', type=float, default=0.0,
                   help='Poisson photons per pixel at d=1; 0 = noiseless')
    p.add_argument('--error-step', type=int, default=2,
                   help='how often F0 is recorded for the convergence plot')
    return p.parse_args()


def metrics(u, truth):
    """Correlation, error, and a noise proxy that does not need a flat region."""
    u, t = u.real.astype('float64'), truth.real.astype('float64')
    uu, tt = u - u.mean(), t - t.mean()
    corr = float((uu * tt).sum() / np.sqrt((uu**2).sum() * (tt**2).sum()))
    rms = float(np.sqrt(((u - t)**2).mean()) / t.std())
    # noise proxy: FRACTION of the volume's power above f=0.25 cyc/px.  The
    # phantom is low-passed when it is built, so it has little real content
    # there -- what shows up is what the solve put in.  Orthonormal transform
    # and a ratio, so it is dimensionless and size-independent.
    F = np.fft.fftn(uu, norm='ortho')
    fr = np.sqrt(sum(np.fft.fftfreq(n)[(slice(None),) + (None,) * (u.ndim - 1 - i)]**2
                     for i, n in enumerate(u.shape)))
    hi = fr > 0.25
    p_tot = float((np.abs(F)**2).sum())
    hf = float((np.abs(F[hi])**2).sum() / p_tot)
    return corr, rms, hf


def run(a, sig_gen, sig_rec, truth, prb, pos, pos_err, theta, nobj, comm,
        paganin_volume):
    cl = Rec(SimpleNamespace(
        energy=ENERGY, detector_pixelsize=PIXEL, focustodetectordistance=FOCUS_DET,
        z1=Z1[:a.ndist], theta=theta, ndist=a.ndist, ntheta=a.ntheta,
        nz=a.n, n=a.n, nzobj=nobj, nobj=nobj,
        mask=0.9, lam_prbfit=2e-3, lam_laplacian=0, rho=[1, 0.05, 0.02, 0],
        shift_type='cubic', psf_sigma=sig_gen,          # <- data are made with THIS
        niter=a.niter, nchunk=a.nchunk, checkpoint_step=-1,
        error_step=a.error_step, start_iter=0, comm=comm))

    cl.vars['obj'][:] = truth[cl.st_obj:cl.end_obj]
    cl.vars['prb'][:] = prb
    cl.vars['pos'][:] = pos[cl.st_theta:cl.end_theta].transpose(1, 0, 2)
    cl.gen_data(cl.vars, cl.data)
    cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)   # ref also carries K

    if a.photons > 0:
        # Poisson on the INTENSITY, after the blur -- a detector counts photons
        # it has already integrated over its PSF.  The reference is left clean:
        # a measured flat is averaged over many more frames than one projection.
        d = np.asarray(cl.data)
        d[:] = (np.random.default_rng(7).poisson(np.clip(d, 0, None) * a.photons)
                / a.photons).astype('float32')

    # ---- swap the model's PSF to the RECONSTRUCTION value ----------------
    cl.psf_sigma, cl.psf_w = psf_taps(sig_rec)
    cl.cl_prb_term.psf_w = cl.psf_w

    cl.vars['obj'][:] = cp.asnumpy(paganin_volume(cl, cl.data, nobj, a.n, a.paganin))
    cl.vars['prb'][:] = 1
    cl.vars['pos'][:] = (pos + pos_err)[cl.st_theta:cl.end_theta].transpose(1, 0, 2)
    cl.BH()

    tbl = cl.table.copy()
    rec = np.asarray(cl.vars['obj'])
    e1 = float(np.abs(np.asarray(cl.vars['pos'])
                      - pos[cl.st_theta:cl.end_theta].transpose(1, 0, 2)).mean())
    del cl
    cp.get_default_memory_pool().free_all_blocks()
    return rec, e1, tbl


def _convergence(tabs, a):
    """F0 against iteration.  NOT one functional: each arm minimises its own.

    matched and MISMATCH share the SAME data, so their curves are directly
    comparable -- the gap is how much of the data each model can explain.
    control has DIFFERENT data (unblurred), so only its shape is meaningful.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sty = {'control': ('0.55', ':'), 'matched': ('0.15', '-'), 'MISMATCH': ('0.0', '--')}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for k in ('control', 'matched', 'MISMATCH'):
        t = tabs[k]
        c, ls = sty[k]
        for b in ax:
            b.plot(t['iter'], t['err'], ls, color=c, lw=1.8, label=k)
    for b, (tt, yl) in zip(ax, [('same data for matched/MISMATCH', 'log'),
                                ('first 50 iterations', 'log')]):
        b.set_yscale(yl); b.set_xlabel('iteration'); b.grid(alpha=0.3)
        b.set_ylabel('F0'); b.legend(fontsize=8); b.set_title(tt, fontsize=10)
    ax[1].set_xlim(-1, min(50, a.niter))
    fig.suptitle(f'Convergence, PSF sigma={a.sigma:g} in the DATA, {a.niter} iterations.  '
                 f'matched vs MISMATCH share the same data; control does not.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    o = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'psf_convergence{a.tag}.png')
    fig.savefig(o, dpi=125)
    print(f'  wrote {o}')
    for k in ('matched', 'MISMATCH', 'control'):
        e = tabs[k]['err'].to_numpy()
        print(f'    {k:9s} F0 start {e[0]:.5e} -> end {e[-1]:.5e}  '
              f'(x{e[0]/max(e[-1],1e-30):.1f} reduction)')


def _figure(vols, truth, out, a):
    """Central slice per arm, plus the error against the known truth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = ['control', 'matched', 'MISMATCH']
    t = truth.real[: vols[order[0]].shape[0]]
    z = t.shape[0] // 2
    lo, hi = np.percentile(t[z], [0.5, 99.5])
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    ax[0][0].imshow(t[z], cmap='gray', vmin=lo, vmax=hi)
    ax[0][0].set_title('truth', fontsize=10)
    ax[1][0].axis('off')
    for j, k in enumerate(order, start=1):
        u = vols[k]
        c, r, h = out[k]
        ax[0][j].imshow(u[z], cmap='gray', vmin=lo, vmax=hi)
        ax[0][j].set_title(f'{k}\ncorr {c:.4f}', fontsize=10)
        d = u[z] - t[z]
        lim = np.percentile(np.abs(d), 99.5)
        im = ax[1][j].imshow(d, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax[1][j].set_title(f'error  rms/std {r:.3f},  hi-f {h:.4f}', fontsize=9)
        fig.colorbar(im, ax=ax[1][j], fraction=0.046)
    for r_ in ax:
        for b in r_:
            b.set_xticks([]); b.set_yticks([])
    fig.suptitle(f'Data made with PSF sigma={a.sigma:g} (NOISELESS).  '
                 f'Reconstructed with it (matched), without it (MISMATCH), '
                 f'and with no blur anywhere (control).', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    o = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'psf_model_mismatch{a.tag}.png')
    fig.savefig(o, dpi=120)
    print(f'\n  wrote {o}')


def main():
    a = parse()
    comm = MPI.COMM_WORLD
    set_log_level('ERROR')
    nobj = a.n * 3 // 2
    truth = phantom3d(nobj)
    prb = id16a_probe(a.n, a.ndist)
    rng = np.random.default_rng(10)
    pos = (30 * (rng.random((a.ntheta, a.ndist, 2)) - 0.5)).astype('float32')
    pos_err = (rng.random((a.ntheta, a.ndist, 2)) - 0.5).astype('float32')
    theta = np.linspace(0, np.pi, a.ntheta, dtype='float32')

    pag = _paganin_volume()
    arms = [(a.sigma, a.sigma, 'matched   '),
            (a.sigma, 0.0,      'MISMATCH  '),
            (0.0,     0.0,      'control   ')]
    print(f'phantom {nobj}^3, n={a.n}, ntheta={a.ntheta}, ndist={a.ndist}, '
          f'niter={a.niter}, PSF sigma for the data = {a.sigma:g}, '
          f'photons/px = {"none (noiseless)" if a.photons == 0 else f"{a.photons:g}"}\n')
    print(f'  {"arm":10s} {"gen":>5s} {"rec":>5s} | {"corr":>8s} {"rms/std":>8s} '
          f'{"hi-f frac":>9s} {"pos err":>8s}')
    out, vols, tabs = {}, {}, {}
    for sg, sr, name in arms:
        rec, e1, tbl = run(a, sg, sr, truth, prb, pos, pos_err, theta, nobj, comm, pag)
        tabs[name.strip()] = tbl
        c, r, h = metrics(rec, truth[: rec.shape[0]])
        out[name.strip()] = (c, r, h)
        vols[name.strip()] = np.asarray(rec).real.copy()
        print(f'  {name} {sg:5.2f} {sr:5.2f} | {c:8.4f} {r:8.4f} {h:9.4f} {e1:8.4f}',
              flush=True)
    # radially-averaged power spectrum: does the MISMATCH arm come out
    # SHARPER (sharpening to fit) or SMOOTHER (absorbing the blur)?
    t = truth.real[: vols['control'].shape[0]]
    def radial(u):
        U = np.fft.fftn(u - u.mean(), norm='ortho')
        fr = np.sqrt(sum(np.fft.fftfreq(n)[(slice(None),) + (None,) * (u.ndim - 1 - i)]**2
                         for i, n in enumerate(u.shape)))
        b = np.clip((fr / 0.5 * 24).astype(int), 0, 23)
        pw = np.abs(U)**2
        return np.array([pw[b == i].mean() for i in range(24)])
    rt = radial(t)
    print(f'\n  radially-averaged power, relative to TRUTH (1.0 = same content)')
    print(f'  {"f/Nyq":>6s} {"control":>9s} {"matched":>9s} {"MISMATCH":>9s}')
    rs = {k: radial(vols[k]) for k in ('control', 'matched', 'MISMATCH')}
    for i in range(2, 24, 3):
        f = (i + .5) / 24
        print(f'  {f:6.2f} ' + ' '.join(f'{rs[k][i]/rt[i]:9.3f}'
              for k in ('control', 'matched', 'MISMATCH')))
    _convergence(tabs, a)
    _figure(vols, truth, out, a)
    m, mm = out['matched'], out['MISMATCH']
    print(f'\n  leaving the PSF out of the model: corr {m[0]:.4f} -> {mm[0]:.4f}, '
          f'hi-f power fraction {m[2]:.4f} -> {mm[2]:.4f} '
          f'({mm[2]/max(m[2],1e-12):.2f}x)')


if __name__ == '__main__':
    main()
