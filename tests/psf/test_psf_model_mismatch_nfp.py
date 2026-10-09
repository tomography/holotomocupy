#!/usr/bin/env python
"""Does modelling the detector PSF matter in NFP?  Siemens star, 2-D object.

    PYTHONPATH=<repo>/src python tests/psf/test_psf_model_mismatch_nfp.py
    ... --sigma 1.2 --n 512 --ntheta 16 --niter 256

The NFP twin of test_psf_model_mismatch.py.  Same three arms, same question,
but `RecNFP`: one rotation angle, a 2-D complex object, and the PROBE solved
for alongside it.

    gen sigma / rec sigma
      s / s     matched   -- the model can reproduce the data
      s / 0     MISMATCH  -- blurred data, sharp model; the solver must put the
                             missing smoothing somewhere
      0 / 0     control   -- no blur anywhere

Why a separate test rather than a flag on the 3-D one.  In holotomography the
probe is pinned by a measured flat (`PrbfitTerm`) and the object is a 3-D
volume, so an unmodelled blur has essentially one place to go: the object.  In
NFP there is no reference term at all (`rec_nfp_mpi` has no regularisation),
the probe is a free n x n complex array, and the object has a second channel,
beta, that the data barely constrains.  So the interesting question here is not
"how much does the object degrade" but WHICH of {delta, beta, probe} absorbs
the mismatch.  The per-channel columns below are the answer.

beta is reported separately for that reason: in the WT_cell1 scan the probe
features leak almost entirely into beta, and `corr(beta, |prb|)` is the metric
that shows it.  The truth here has beta = -delta/delta_beta exactly, so any
beta structure that does not track delta is manufactured.
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
from phantoms import star_projection, id16a_probe                 # noqa: E402

from holotomocupy.rec_nfp_mpi import RecNFP                        # noqa: E402
from holotomocupy.psf import psf_taps                              # noqa: E402
from holotomocupy.logger_config import set_log_level               # noqa: E402

# ID16A near-field ptychography, as in tests/nfp/test.py -- a different setup
# from the holotomography test above it (33.35 keV, 4 distances).
ENERGY, PIXEL, FOCUS_DET, Z1 = 17.1, 1.4760147601476e-6 * 4, 1.217, 5.110e-3

ARMS = ('control', 'matched', 'MISMATCH')
# The three places an unmodelled blur can go.  prb is |probe|: the phase shares
# a constant with the object's, the amplitude does not.
CHANNELS = ('delta', 'beta', 'prb')


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sigma', type=float, default=1.2, help='PSF sigma used to MAKE the data')
    p.add_argument('--n', type=int, default=512, help='detector / probe size')
    p.add_argument('--ntheta', type=int, default=16, help='scan positions')
    p.add_argument('--niter', type=int, default=256)
    p.add_argument('--nchunk', type=int, default=8)
    p.add_argument('--delta-beta', type=float, default=29.0, help='true delta/beta ratio')
    # The star is built with a pi/8 phase range; WT_cell1's real object is
    # ~1.5%.  Scaling delta and beta together keeps delta_beta exact.
    p.add_argument('--contrast', type=float, default=1.0,
                   help='scale on the phantom; <1 moves toward the weak-object regime')
    p.add_argument('--prb-smooth', type=float, default=4.0,
                   help='low-pass on the measured ID16A probe; 0 = raw tiffs')
    p.add_argument('--model', default='amplitude', choices=('amplitude', 'intensity'),
                   help='misfit model; amplitude is what the WT_cell1 study uses')
    # Production WT_cell1 value.  rho[prb]=32 all but freezes the positions
    # (alpha is shared), so use --rho 1,2,1e-2 to let the position channel also
    # try to absorb the mismatch.
    p.add_argument('--rho', default='1,32,1e-2', help='rho for proj,prb,pos')
    p.add_argument('--photons', type=float, default=0.0,
                   help='Poisson photons per pixel at d=1; 0 = noiseless')
    p.add_argument('--error-step', type=int, default=4,
                   help='how often F0 is recorded for the convergence plot')
    # Fixing the iteration count confounds "the model is blind here" with
    # "this arm just got further".  Scoring along the run lets the arms be
    # compared where their F0 is EQUAL, which separates the two.
    p.add_argument('--probe-step', type=int, default=0,
                   help='record channel metrics every N iters; 0 = off')
    p.add_argument('--path-out', default='/local/ssd/vnikitin/psf_nfp',
                   help='RecNFP scratch (conv_nfp.csv); never /tmp')
    p.add_argument('--tag', default='', help='suffix for the figure filenames')
    return p.parse_args()


def metrics(u, t):
    """Correlation, error, and a noise proxy that does not need a flat region.

    Means are removed from both: the probe and the object trade a constant
    freely in ptychography, so an offset is not an error.
    """
    u, t = u.astype('float64'), t.astype('float64')
    uu, tt = u - u.mean(), t - t.mean()
    corr = float((uu * tt).sum() / np.sqrt((uu**2).sum() * (tt**2).sum()))
    rms = float(np.sqrt(((uu - tt)**2).mean()) / t.std())
    # noise proxy: FRACTION of the power above f=0.25 cyc/px.  The star is
    # low-passed when it is built, so it has little real content there -- what
    # shows up is what the solve put in.
    F = np.fft.fftn(uu, norm='ortho')
    fr = np.sqrt(sum(np.fft.fftfreq(m)[(slice(None),) + (None,) * (u.ndim - 1 - i)]**2
                     for i, m in enumerate(u.shape)))
    pw = np.abs(F)**2
    return corr, rms, float(pw[fr > 0.25].sum() / pw.sum())


def _probe(cl, truth, crop, traj, step):
    """Score every channel against the truth as the run goes, not just at the end.

    Wraps the per-instance log_iter, so the BH loop is never interrupted and the
    CG direction is never reset -- which running in segments would do.
    """
    orig = cl.log_iter

    def wrapped(vars, i, writer):
        orig(vars, i, writer)
        if step <= 0 or i % step:
            return
        f0 = cl.min(vars['prb'], vars['proj'], vars['pos'])   # collective
        if cl.rank:
            return
        p = vars['proj'].get()
        got = {'delta': -p.real[crop], 'beta': p.imag[crop],
               'prb': np.abs(vars['prb'].get())}
        row = dict(iter=i, f0=f0)
        for c in CHANNELS:
            for k, v in zip(('corr', 'rms', 'hif'), metrics(got[c], truth[c])):
                row[f'{c}_{k}'] = v
        traj.append(row)

    cl.log_iter = wrapped


def run(a, sig_gen, sig_rec, proj_gt, prb_gt, pos, pos_err, nobj, comm, rho,
        truth, crop, traj):
    path_out = os.path.join(a.path_out, f'gen{sig_gen:g}_rec{sig_rec:g}')
    cl = RecNFP(SimpleNamespace(
        energy=ENERGY, detector_pixelsize=PIXEL, focustodetectordistance=FOCUS_DET,
        z1=Z1, ntheta=a.ntheta, nz=a.n, n=a.n, nzobj=nobj, nobj=nobj,
        rho=rho, model=a.model,
        shift_type='fft', psf_sigma=sig_gen,            # <- data are made with THIS
        niter=a.niter, nchunk=a.nchunk, checkpoint_step=-1,
        error_step=a.error_step, start_iter=0, path_out=path_out, comm=comm))

    cl.vars['proj'][:] = cp.asarray(proj_gt)
    cl.vars['prb'][:] = cp.asarray(prb_gt)
    cl.vars['pos'][:] = cp.asarray(pos[cl.st_theta:cl.end_theta])
    cl.gen_data(cl.vars, cl.data)

    if a.photons > 0:
        # Poisson on the INTENSITY, after the blur -- a detector counts photons
        # it has already integrated over its PSF.  Seeded by GLOBAL theta index
        # so the realisation does not depend on the rank count.
        seeds = np.random.SeedSequence(20251119).spawn(a.ntheta)
        for j in range(cl.end_theta - cl.st_theta):
            rng = np.random.default_rng(seeds[cl.st_theta + j])
            cl.data[j] = (rng.poisson(np.clip(cl.data[j], 0, None) * a.photons)
                          / a.photons).astype('float32')

    # ---- swap the model's PSF to the RECONSTRUCTION value --------------------
    # One line, unlike the 3-D test: RecNFP has no cl_prb_term to keep in step.
    cl.psf_sigma, cl.psf_w = psf_taps(sig_rec)

    cl.vars['proj'][:] = 0          # the step0 cold start, not a paganin guess
    cl.vars['prb'][:] = 1
    cl.vars['pos'][:] = cp.asarray((pos + pos_err)[cl.st_theta:cl.end_theta])
    _probe(cl, truth, crop, traj, a.probe_step)
    cl.BH()

    tbl = cl.table.copy()
    rec_proj = cl.vars['proj'].get()
    rec_prb = cl.vars['prb'].get()
    e1 = float(np.abs(cl.vars['pos'].get() - pos[cl.st_theta:cl.end_theta]).mean())
    del cl
    cp.get_default_memory_pool().free_all_blocks()
    return rec_proj, rec_prb, e1, tbl


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
    for k in ARMS:
        t = tabs[k]
        c, ls = sty[k]
        for b in ax:
            b.plot(t['iter'], t['err'], ls, color=c, lw=1.8, label=k)
    for b, tt in zip(ax, ('same data for matched/MISMATCH',
                          f'first {min(50, a.niter)} iterations')):
        b.set_yscale('log'); b.set_xlabel('iteration'); b.grid(alpha=0.3)
        b.set_ylabel('F0'); b.legend(fontsize=8); b.set_title(tt, fontsize=10)
    ax[1].set_xlim(-1, min(50, a.niter))
    fig.suptitle(f'NFP convergence, PSF sigma={a.sigma:g} in the DATA, '
                 f'{a.niter} iterations, model={a.model}.  '
                 f'matched vs MISMATCH share the same data; control does not.',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    o = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'psf_nfp_convergence{a.tag}.png')
    fig.savefig(o, dpi=125)
    print(f'  wrote {o}')
    for k in ('matched', 'MISMATCH', 'control'):
        e = tabs[k]['err'].to_numpy()
        print(f'    {k:9s} F0 start {e[0]:.5e} -> end {e[-1]:.5e}  '
              f'(x{e[0] / max(e[-1], 1e-30):.1f} reduction)')


def _at_equal_f0(trajs, a):
    """Compare matched vs MISMATCH where they fit the data EQUALLY well.

    They share the same data, so their F0 is on one scale.  matched ends far
    lower, so the fair common point is MISMATCH's floor: rewind matched to the
    first iterate that reaches it.  If matched is still worse there, the blur
    has made a subspace invisible; if it is better, it had merely got further.
    control is excluded -- its data are unblurred, so its F0 is a different
    number.
    """
    tm, tx = trajs.get('matched'), trajs.get('MISMATCH')
    if not tm or not tx:
        return
    f_common = max(tm[-1]['f0'], tx[-1]['f0'])

    def at(t, key):
        """Metric where this arm's best-so-far F0 equals f_common, interpolated.

        Picking the nearest probed iterate instead would land far off target on
        a coarse --probe-step; the trajectory is smooth in log F0, so interpolate.
        """
        f = np.minimum.accumulate([r['f0'] for r in t])   # best-so-far: monotone
        x, y = -np.log10(f), np.array([r[key] for r in t])
        xc = -np.log10(f_common)
        if not (x[0] <= xc <= x[-1]):
            return float('nan')
        return float(np.interp(xc, x, y))

    it = {k: int(np.interp(-np.log10(f_common),
                           -np.log10(np.minimum.accumulate([r['f0'] for r in t])),
                           [r['iter'] for r in t]))
          for k, t in (('matched', tm), ('MISMATCH', tx))}

    print(f'\n  --- matched vs MISMATCH at EQUAL F0 ({f_common:.5e}) ---')
    print('  the iteration-matched table above confounds model blindness with '
          'depth of descent; this does not.')
    print(f'  {"arm":9s} {"iter~":>5s} ' +
          ' '.join(f'{c + " corr":>12s}' for c in CHANNELS) +
          ' ' + ' '.join(f'{c + " hi-f":>12s}' for c in CHANNELS))
    vals = {}
    for k, t in (('matched', tm), ('MISMATCH', tx)):
        vals[k] = {c: (at(t, f'{c}_corr'), at(t, f'{c}_hif')) for c in CHANNELS}
        print(f'  {k:9s} {it[k]:5d} ' +
              ' '.join(f'{vals[k][c][0]:12.4f}' for c in CHANNELS) + ' ' +
              ' '.join(f'{vals[k][c][1]:12.4f}' for c in CHANNELS))
    d = {c: vals['matched'][c][0] - vals['MISMATCH'][c][0] for c in CHANNELS}
    if any(np.isnan(v) for v in d.values()):
        print('  => inconclusive: the arms\' F0 ranges do not overlap.')
        return
    worse = [c for c, v in d.items() if v < 0]
    print('  matched - MISMATCH corr at equal F0: ' +
          ', '.join(f'{c} {v:+.4f}' for c, v in d.items()))
    print('  => ' + ('matched is WORSE in ' + ', '.join(worse) +
                     ' at equal fit: a blind subspace, not slower descent.'
                     if worse else
                     'matched is no worse at equal fit: the end-of-run gap was '
                     'depth of descent, not a blind subspace.'))


def _trajectory(trajs, a):
    """Channel quality against F0.  x descends: left = fits worse, right = better."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sty = {'control': ('0.55', ':'), 'matched': ('0.15', '-'), 'MISMATCH': ('0.0', '--')}
    fig, ax = plt.subplots(2, len(CHANNELS), figsize=(5.0 * len(CHANNELS), 8.0))
    for j, cn in enumerate(CHANNELS):
        for k in ARMS:
            t = trajs.get(k)
            if not t:
                continue
            c, ls = sty[k]
            f = [r['f0'] for r in t]
            ax[0][j].plot(f, [r[f'{cn}_corr'] for r in t], ls, color=c, lw=1.8, label=k)
            ax[1][j].plot(f, [r[f'{cn}_hif'] for r in t], ls, color=c, lw=1.8, label=k)
        for i, yl in enumerate((f'{cn}: corr with truth', f'{cn}: hi-f power fraction')):
            ax[i][j].set_xscale('log'); ax[i][j].invert_xaxis()
            ax[i][j].set_xlabel('F0 (better fit ->)'); ax[i][j].set_ylabel(yl)
            ax[i][j].grid(alpha=0.3); ax[i][j].legend(fontsize=8)
        ax[1][j].set_yscale('log')
    fig.suptitle('NFP: reconstruction quality against how well the model fits. '
                 'A curve that turns DOWN as F0 falls is fitting invisible '
                 'structure.  matched/MISMATCH share data; control does not.',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    o = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'psf_nfp_quality_vs_f0{a.tag}.png')
    fig.savefig(o, dpi=125)
    print(f'  wrote {o}')


def _figure(chan, out, a):
    """Each channel per arm, with its error against the known truth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2 * len(CHANNELS), 4, figsize=(13.5, 6.5 * len(CHANNELS)))
    for r, name in enumerate(CHANNELS):
        t = chan[name]['truth']
        lo, hi = np.percentile(t, [0.5, 99.5])
        ax[2 * r][0].imshow(t, cmap='gray', vmin=lo, vmax=hi)
        ax[2 * r][0].set_title(f'{name}: truth', fontsize=10)
        ax[2 * r + 1][0].axis('off')
        for j, k in enumerate(ARMS, start=1):
            # Means are removed for the same reason metrics() removes them.
            u = chan[name][k] - chan[name][k].mean() + t.mean()
            c, rr, h = out[k][name]
            ax[2 * r][j].imshow(u, cmap='gray', vmin=lo, vmax=hi)
            ax[2 * r][j].set_title(f'{name}: {k}\ncorr {c:.4f}', fontsize=10)
            d = u - t
            lim = max(np.percentile(np.abs(d), 99.5), 1e-20)
            im = ax[2 * r + 1][j].imshow(d, cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax[2 * r + 1][j].set_title(f'error  rms/std {rr:.3f},  hi-f {h:.4f}',
                                       fontsize=9)
            fig.colorbar(im, ax=ax[2 * r + 1][j], fraction=0.046)
    for r_ in ax:
        for b in r_:
            b.set_xticks([]); b.set_yticks([])
    fig.suptitle(f'NFP, data made with PSF sigma={a.sigma:g} '
                 f'({"noiseless" if a.photons == 0 else f"{a.photons:g} ph/px"}).  '
                 f'Reconstructed with it (matched), without it (MISMATCH), and '
                 f'with no blur anywhere (control).', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    o = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'psf_nfp_model_mismatch{a.tag}.png')
    fig.savefig(o, dpi=110)
    print(f'\n  wrote {o}')


def main():
    a = parse()
    comm = MPI.COMM_WORLD
    set_log_level('ERROR')
    rho = [float(v) for v in a.rho.split(',')]
    # run() is collective, so every rank calls it; only rank 0 holds the tables
    # and does the scoring and plotting.
    rank = comm.Get_rank()
    pr = print if rank == 0 else (lambda *x, **k: None)

    rng = np.random.default_rng(10)
    pos = (30 / 512 * a.n * (rng.random((a.ntheta, 2)) - 0.5)).astype('float32')
    pos_err = 4 * (rng.random((a.ntheta, 2)) - 0.5).astype('float32')
    # 'fft' shifts are PERIODIC, so the object grid must clear n + 2*max|pos|.
    pos_range = int(np.ceil(np.abs(pos + pos_err).max())) + 8
    nobj = int(np.ceil((a.n + 2 * pos_range) / 32)) * 32

    proj_gt = star_projection(nobj, a.delta_beta) * a.contrast
    # The MEASURED ID16A probe, mildly low-passed (smooth=4.0).  That filter is
    # ~100x weaker than the PSF under test at high f, so it does not blunt the
    # experiment; --prb-smooth 0 uses the raw tiffs.
    prb_gt = id16a_probe(a.n, smooth=a.prb_smooth)[0]

    # Compare on the centred n x n patch: the corners of the nobj grid are seen
    # by no scan position, so they are free parameters and not a result.
    o = (nobj - a.n) // 2
    crop = (slice(o, o + a.n),) * 2
    truth = {'delta': -proj_gt.real[crop].astype('float64'),
             'beta': proj_gt.imag[crop].astype('float64'),
             'prb': np.abs(prb_gt).astype('float64')}

    pr(f'star {nobj}^2 (n={a.n}), ntheta={a.ntheta}, niter={a.niter}, '
          f'model={a.model}, rho={rho}, delta/beta={a.delta_beta:g}, '
          f'contrast x{a.contrast:g}, PSF sigma for the data = {a.sigma:g}, '
          f'photons/px = {"none (noiseless)" if a.photons == 0 else f"{a.photons:g}"}')
    pr(f'  truth: delta p-p {np.ptp(truth["delta"]):.4f} rad, '
          f'beta/delta std ratio {truth["beta"].std() / truth["delta"].std():.5f} '
          f'(= 1/{a.delta_beta:g} exactly), '
          f'|prb| std/mean {truth["prb"].std() / truth["prb"].mean():.4f}\n')

    arms = [(a.sigma, a.sigma, 'matched'), (a.sigma, 0.0, 'MISMATCH'),
            (0.0, 0.0, 'control')]
    pr(f'  {"arm":9s} {"gen":>5s} {"rec":>5s} {"chan":>6s} | {"corr":>8s} '
          f'{"rms/std":>8s} {"hi-f":>8s} {"b/d std":>8s} {"cor(.,|prb|)":>12s} '
          f'{"pos err":>8s}')
    out, tabs, trajs = {}, {}, {}
    chan = {c: dict(truth=truth[c]) for c in CHANNELS}
    for sg, sr, name in arms:
        trajs[name] = []
        rec_proj, rec_prb, e1, tbl = run(a, sg, sr, proj_gt, prb_gt, pos, pos_err,
                                         nobj, comm, rho, truth, crop, trajs[name])
        tabs[name] = tbl
        got = {'delta': -rec_proj.real[crop].astype('float64'),
               'beta': rec_proj.imag[crop].astype('float64'),
               'prb': np.abs(rec_prb).astype('float64')}
        bd = got['beta'].std() / max(got['delta'].std(), 1e-30)
        out[name] = {}
        for ci, cn in enumerate(CHANNELS):
            chan[cn][name] = got[cn]
            out[name][cn] = metrics(got[cn], truth[cn])
            c, rr, h = out[name][cn]
            # corr with the TRUE probe amplitude: the WT_cell1 leak metric.  On
            # the prb row it is just that row's own corr, so it is left blank.
            lk = '-' if cn == 'prb' else f'{metrics(got[cn], truth["prb"])[0]:+.4f}'
            head = f'  {name:9s} {sg:5.2f} {sr:5.2f}' if ci == 0 else ' ' * 22
            pr(f'{head} {cn:>6s} | {c:8.4f} {rr:8.4f} {h:8.4f} '
                  f'{f"{bd:.4f}" if cn == "beta" else "-":>8s} {lk:>12s} '
                  f'{f"{e1:.4f}" if ci == 0 else "-":>8s}', flush=True)

    # radially-averaged power: does the MISMATCH arm come out SHARPER
    # (sharpening to fit) or SMOOTHER (absorbing the blur)?
    def radial(u):
        U = np.fft.fftn(u - u.mean(), norm='ortho')
        fr = np.sqrt(sum(np.fft.fftfreq(m)[(slice(None),) + (None,) * (u.ndim - 1 - i)]**2
                         for i, m in enumerate(u.shape)))
        b = np.clip((fr / 0.5 * 24).astype(int), 0, 23)
        pw = np.abs(U)**2
        return np.array([pw[b == i].mean() for i in range(24)])

    for cn in CHANNELS:
        rt = radial(truth[cn])
        rs = {k: radial(chan[cn][k]) for k in ARMS}
        pr(f'\n  {cn}: radially-averaged power, relative to TRUTH '
              f'(1.0 = same content)')
        pr(f'  {"f/Nyq":>6s} ' + ' '.join(f'{k:>9s}' for k in ARMS))
        for i in range(2, 24, 3):
            pr(f'  {(i + .5) / 24:6.2f} '
                  + ' '.join(f'{rs[k][i] / rt[i]:9.3f}' for k in ARMS))

    # Only rank 0 has the convergence table and the trajectory; the others
    # would crash on the empty frame.
    if rank == 0:
        _convergence(tabs, a)
        if a.probe_step > 0:
            _at_equal_f0(trajs, a)
            _trajectory(trajs, a)
            import pandas as pd
            for k in ARMS:
                p = os.path.join(a.path_out, f'traj_{k}{a.tag}.csv')
                pd.DataFrame(trajs[k]).to_csv(p, index=False)
            pr(f'  wrote {a.path_out}/traj_*{a.tag}.csv')
        _figure(chan, out, a)

    pr('\n  leaving the PSF out of the model:')
    for cn in CHANNELS:
        m, mm = out['matched'][cn], out['MISMATCH'][cn]
        pr(f'    {cn:>6s}  corr {m[0]:.4f} -> {mm[0]:.4f},  '
              f'rms/std {m[1]:.4f} -> {mm[1]:.4f},  '
              f'hi-f {m[2]:.4f} -> {mm[2]:.4f} ({mm[2] / max(m[2], 1e-12):.2f}x)')


if __name__ == '__main__':
    main()
