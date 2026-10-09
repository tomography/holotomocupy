#!/usr/bin/env python
"""One compact row per rho[prb] arm of the WT_cell1 sweeps.

    python sweep_table.py [<dir holding nfp_prb*/>]

Two ladders, identical except for step0's intensity normalisation:

    flux         nfp_prb<p>_flux   every frame scaled to mean 1 (step0_flux.py)
    global mean  nfp_prb<p>        whole stack / one global mean (step0.py)

Skips arms that have not finished.  Columns are the probe-leak metrics from
compare_arms.py plus the position motion, read from each file's own pos_err --
the arms do not share a step size, so positions freeze again as rho[prb]
climbs and alpha falls.  See README.md.
"""
import os
import sys
import numpy as np
import h5py

from compare_arms import VOXEL_NM, CUT_UM, lowpass, corners, corr

ARMS = [1, 2, 4, 8, 16, 32, 64, 128]
FAMILIES = [('_flux', 'flux'), ('', 'global mean')]
ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/beams/VNIKITIN/eagle/vnikitin/20260929/PROCESSED_DATA'

hdr = (f'{"rho[prb]":>8} {"beta std":>9} {"b/d":>6} {"cor(b,prb)":>11} '
       f'{"cor(d,prb)":>11} {"corner b w2w":>13} {"corner d w2w":>13} '
       f'{"max|dpos|":>10}')

for suffix, fam in FAMILIES:
    print(f'\n=== step0 normalisation: {fam}')
    print(hdr)
    print('-' * len(hdr))
    for p in ARMS:
        f_h5 = os.path.join(ROOT, f'nfp_prb{p}{suffix}', 'scan0001',
                            'nfp_results.h5')
        if not os.path.exists(f_h5):
            print(f'{p:>8} {"-- not finished --":>60}')
            continue
        try:
            with h5py.File(f_h5, 'r') as f:
                delta = f['proj_delta'][0].astype('float64')
                beta  = f['proj_beta'][0].astype('float64')
                prbA  = f['prb_amp'][0].astype('float64')
                pe    = f['pos_err'][()]
        except (OSError, KeyError) as e:          # still being written
            print(f'{p:>8}   unreadable: {e}')
            continue

        o = (delta.shape[0] - prbA.shape[0]) // 2
        s = slice(o, o + prbA.shape[0])
        d, b = delta[s, s], beta[s, s]

        sig = (CUT_UM * 1000.0 / VOXEL_NM) / (2 * np.pi)
        dl, bl, pl = lowpass(d, sig), lowpass(b, sig), lowpass(prbA, sig)
        cd, cb = corners(d), corners(b)

        print(f'{p:>8} {b.std():>9.5f} {b.std()/d.std():>6.2f} '
              f'{corr(bl, pl):>+11.3f} {corr(dl, pl):>+11.3f} '
              f'{np.ptp(cb.mean(axis=(1,2))):>13.5f} '
              f'{np.ptp(cd.mean(axis=(1,2))):>13.5f} '
              f'{np.abs(pe).max():>10.4f}')
print()
