#!/usr/bin/env python
"""Score how much PROBE structure leaked into the reconstructed object.

    python compare_arms.py <A/nfp_results.h5> <B/nfp_results.h5> ...

Pure numpy + h5py -- no cupy, no MPI, so it runs on a login node or over the
eagle mount.  Prints one block per file; the arms are comparable to each other
only, not to any absolute scale.

The question it answers: with the probe removed correctly, the object should be
featureless (transmission 1, i.e. delta = beta = 0) wherever the sample is not.
Three things give the leak away:

  corr(obj, |prb|) at low frequency   -- the probe's own envelope printed on
                                         the object; the direct signature.
  beta/delta std ratio                -- physics says ~1e-3 at 17.1 keV, so
                                         anything near 1 is not absorption.
  corner level and spread             -- empty regions should sit at 0.

Low-pass is a Gaussian on the FFT grid, cut at 5 um: probe envelope and
illumination structure live there, the cell's edges do not.
"""
import sys
import numpy as np
import h5py

VOXEL_NM = 25.0        # from the .nx; see README
CUT_UM   = 5.0         # low-pass period: above this is probe envelope
CORNER   = 256         # px, the four corner windows used as "empty"


def lowpass(a, sigma_px):
    """Gaussian low-pass, zero-mean output."""
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.fftfreq(a.shape[1])[None, :]
    g  = np.exp(-2 * (np.pi * sigma_px) ** 2 * (ky ** 2 + kx ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(a) * g))
    return out - out.mean()


def corners(a, w=CORNER):
    return np.stack([a[:w, :w], a[:w, -w:], a[-w:, :w], a[-w:, -w:]])


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = a.std() * b.std()
    return float((a * b).mean() / d) if d > 0 else np.nan


def report(path):
    with h5py.File(path, 'r') as f:
        delta = f['proj_delta'][0].astype('float64')
        beta  = f['proj_beta'][0].astype('float64')
        prbA  = np.abs(f['prb_amp'][0].astype('float64'))
        fm    = f['frame_mean'][()] if 'frame_mean' in f else None

    # The object grid is nobj (2176); the probe is n (2048).  Compare on the
    # probe's footprint, centred -- outside it the object is unconstrained.
    o = (delta.shape[0] - prbA.shape[0]) // 2
    s = slice(o, o + prbA.shape[0])
    d, b = delta[s, s], beta[s, s]

    sigma_px = (CUT_UM * 1000.0 / VOXEL_NM) / (2 * np.pi)
    dl, bl, pl = lowpass(d, sigma_px), lowpass(b, sigma_px), lowpass(prbA, sigma_px)

    print(f'\n=== {path}')
    print(f'  delta: std={d.std():.5f}  mean={d.mean():+.5f}  '
          f'ptp={np.ptp(d):.4f}')
    print(f'  beta : std={b.std():.5f}  mean={b.mean():+.5f}  '
          f'ptp={np.ptp(b):.4f}')
    print(f'  beta/delta std ratio        = {b.std()/d.std():7.3f}'
          '   (physics ~1e-3; >>that means probe leak)')
    print(f'  corr(delta, |prb|) >{CUT_UM:.0f}um   = {corr(dl, pl):+7.3f}')
    print(f'  corr(beta , |prb|) >{CUT_UM:.0f}um   = {corr(bl, pl):+7.3f}')

    cd, cb = corners(d), corners(b)
    print(f'  corners delta: level={cd.mean():+.5f}  spread={cd.std():.5f}  '
          f'window-to-window={np.ptp(cd.mean(axis=(1,2))):.5f}')
    print(f'  corners beta : level={cb.mean():+.5f}  spread={cb.std():.5f}  '
          f'window-to-window={np.ptp(cb.mean(axis=(1,2))):.5f}')
    if fm is not None:
        print(f'  frame_mean (ADU): {fm.min():.1f}..{fm.max():.1f}  '
              f'p-p={100*np.ptp(fm)/fm.mean():.2f}%   <- drift that was removed')
    else:
        print('  frame_mean: absent -- this arm predates the per-frame fix')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        report(p)
    print()
