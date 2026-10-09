#!/usr/bin/env python
"""
Pre-flight check -- run before steps15, fails the job on a half-copied scan.

    python preflight.py config_steps15.conf

The failure this exists for is silent: Layout infers ndist by globbing
{pfile}_[0-9]_/, so a scan caught mid-copy reconstructs at whatever number of
planes happen to be on disk, with no error anywhere.  Everything checked here
is cheap -- directory listings and the .info sidecars, no frames read.

Exit 0 = go, 1 = stop.  Warnings do not fail, but they must not scroll past
unseen either.
"""

import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holotomocupy.config import parse_args_steps15
from esrf_layout import Layout

# .info keys that describe the instrument, not the plane: they must be the
# same in all four sidecars or the planes are not one scan.  PixelSize,
# Distance and SourceDistance are deliberately NOT here -- the sample moves
# between planes, so they differ by design.  Their one invariant, constant
# focus-to-detector, is checked by Layout.geometry().
INFO_MUST_MATCH = ('TOMO_N', 'Energy', 'Dim_1', 'Dim_2', 'ScanRange',
                   'REF_N', 'DARK_N', 'REF_ON')


def main():
    cfg = parse_args_steps15(sys.argv[1])
    errors, warnings = [], []

    lay = Layout(cfg.path, cfg.pfile)
    print(lay)
    if lay.nx_unresolved:
        print(f'note: {lay.nx_unresolved[0]}/{lay.nx_unresolved[1]} NXtomo '
              f'virtual sources do not resolve; reading frames from the EDFs')
    if getattr(lay, 'nx_relinked', None):
        for k, f in lay.nx_relinked.items():
            print(f'note: plane {k + 1} metadata is relinked to '
                  f'{os.path.basename(f)} (re-taken scan)')

    ndist = getattr(cfg, 'ndist', None) or lay.ndist
    if lay.ndist != ndist:
        errors.append(f'{lay.ndist} plane directories on disk but the config '
                      f'wants ndist={ndist} -- the copy is incomplete')

    # Per-plane: a readable .info, and the full frame count.
    nproj0 = None
    for k in range(lay.ndist):
        info_path = f'{lay.dname(k)}/{lay.pfile}_{k + 1}_.info'
        if not os.path.exists(info_path):
            errors.append(f'plane {k + 1}: no .info sidecar at {info_path}')
            continue
        info = lay.info(k)
        nproj = len(glob.glob(f'{lay.dname(k)}/{lay.pfile}_{k + 1}_[0-9]*.edf'))
        nref = len(glob.glob(f'{lay.dname(k)}/ref[0-9]*.edf'))
        ndark = len(glob.glob(f'{lay.dname(k)}/dark*.edf'))
        print(f'plane {k + 1}: {nproj} proj + {nref} ref + {ndark} dark EDFs')
        if nproj0 is None:
            nproj0 = nproj
        elif nproj != nproj0:
            errors.append(f'plane {k + 1}: {nproj} projection EDFs but plane 1 '
                          f'has {nproj0} -- the copy is incomplete')
        # The scan writes ntheta+3 frames: the last scan frame plus two
        # post-scan retakes at omega=90 and omega=0.
        want = int(float(info['TOMO_N'])) + 3
        if nproj < want:
            errors.append(f'plane {k + 1}: {nproj} projection EDFs, expected '
                          f'{want} (TOMO_N {info["TOMO_N"]} + 3 retake frames)')

        for key in INFO_MUST_MATCH:
            a, b = lay.info(0).get(key), info.get(key)
            if a != b:
                errors.append(f'plane {k + 1}: .info {key}={b} but plane 1 '
                              f'has {a}')

    # Geometry, and the independent .info cross-check of the distance signs.
    try:
        geo = lay.geometry()
        print(f'energy {geo["energy"]:.3f} keV  det_px '
              f'{geo["detector_pixelsize"] * 1e6:.6f} um  voxel '
              f'{geo["voxelsize"] * 1e9:.4f} nm')
        print(f'norm_magnifications {geo["norm_magnifications"]}')
        for m in lay.info_check(geo):
            errors.append(f'geometry: {m}')
        if getattr(lay, 'geom_from_info', None):
            print(f'note: planes {[k + 1 for k in lay.geom_from_info]} take '
                  f'geometry from the .info sidecar, no NXtomo')
    except SystemExit as e:
        errors.append(f'geometry: {e}')

    # The commanded random displacement, one table per plane.  The row count
    # is worth reporting because it is not consistent across scans: FT writes
    # ntheta+3 rows with the last three explicitly zero (the piezo is parked
    # for the anchor and both retakes), while HT stops at ntheta.  The motion
    # estimator pads a short table with zeros, which FT's explicit zeros say is
    # the right fill -- this just makes it visible before the run.
    for k in range(lay.ndist):
        src = lay.shift_source(k)
        if not os.path.exists(src):
            # Zero commanded displacement means ESRF writes no file at all, so
            # on an RD000 scan this is correct and steps15 substitutes zeros.
            # Anything else is a half-copied projections/ dir.
            (warnings if 'RD000' in cfg.pfile else errors).append(
                f'plane {k + 1}: no shift table at {src}'
                + ('  -- RD000, so zero displacement was commanded and step 3 '
                   'will use zeros' if 'RD000' in cfg.pfile else ''))
            continue
        try:
            nrow = len(np.loadtxt(src, dtype='float64'))
        except Exception as e:
            errors.append(f'plane {k + 1}: cannot read shift table {src}: {e}')
            continue
        want = lay.ntheta + 3
        if nrow >= want:
            print(f'plane {k + 1}: shift table {nrow} rows (ntheta+3 = {want})')
        elif nrow >= lay.ntheta:
            print(f'plane {k + 1}: shift table {nrow} rows, short of ntheta+3 '
                  f'= {want}; the motion estimator will pad {want - nrow} '
                  f'zero row(s)')
        else:
            errors.append(f'plane {k + 1}: shift table has {nrow} rows, fewer '
                          f'than ntheta = {lay.ntheta}')

    for w in warnings:
        print(f'WARNING: {w}')
    for e in errors:
        print(f'ERROR: {e}')
    if errors:
        print(f'preflight FAILED with {len(errors)} error(s)')
        return 1
    print(f'preflight OK ({len(warnings)} warning(s))')
    return 0


if __name__ == '__main__':
    sys.exit(main())
