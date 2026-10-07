#!/usr/bin/env python
"""Write the two preview slices of a step-6 checkpoint out as TIFFs.

    python preview_tiff.py --src .../checkpoint_1280_binnedz.h5 \
                           --out .../checkpoints_tiff_binned

These are the same two cuts Writer makes during a run (writer.py), so a
checkpoint produced outside the solver -- by bin_z.py, say -- can be given the
same preview pair without re-running anything:

    <stem>_obj_re.tiff       (ny, nx)   horizontal, z = nz // 2
    <stem>_obj_re_vert.tiff  (nz, nx)   vertical,   y = ny // 2, through the axis

<stem> is the input file's basename, so a binned volume lands as
checkpoint_1280_binnedz_obj_re.tiff and never collides with the run's own
checkpoint_1280_obj_re.tiff.

Serial and cheap: the horizontal cut is one contiguous slice, and the vertical
one is read in z blocks, so peak memory is a block, not the volume.
"""
import argparse
import os
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='checkpoint .h5 to read')
    p.add_argument('--out', required=True, help='output directory for the TIFFs')
    p.add_argument('--part', default='obj_re',
                   help='dataset to slice (default obj_re, the real part)')
    p.add_argument('--zblock', type=int, default=256,
                   help='z slices per read for the vertical cut (default 256)')
    p.add_argument('--force', action='store_true',
                   help='rewrite TIFFs that are already there')
    a = p.parse_args()

    import h5py
    import tifffile

    if not os.path.exists(a.src):
        sys.exit(f'missing checkpoint: {a.src}')
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.src))[0]
    h_path = os.path.join(a.out, f'{stem}_{a.part}.tiff')
    v_path = os.path.join(a.out, f'{stem}_{a.part}_vert.tiff')

    with h5py.File(a.src, 'r') as f:
        if a.part not in f:
            sys.exit(f'{a.src} has no {a.part} (has: {", ".join(f.keys())})')
        d = f[a.part]
        nz, ny, nx = d.shape
        it = int(f.attrs['iter']) if 'iter' in f.attrs else None
        zbin = int(f.attrs['zbin']) if 'zbin' in f.attrs else None
        print(f'{a.src}', flush=True)
        print(f'  {a.part} {nz} x {ny} x {nx} {d.dtype}'
              + (f'   iter={it}' if it is not None else '')
              + (f'  zbin={zbin}' if zbin is not None else ''), flush=True)

        t = time.time()
        if a.force or not os.path.exists(h_path):
            tifffile.imwrite(h_path, np.asarray(d[nz // 2], dtype='float32'))
            print(f'  horizontal z={nz//2}  ({ny} x {nx})  -> {h_path}'
                  f'   {time.time()-t:.1f}s', flush=True)
        else:
            print(f'  horizontal exists, skipped: {h_path}', flush=True)

        t = time.time()
        if a.force or not os.path.exists(v_path):
            # Read in z blocks: d[:, ymid, :] as one fancy read would pull the
            # whole volume through HDF5's selection machinery.
            vert = np.empty((nz, nx), dtype='float32')
            ymid = ny // 2
            for z0 in range(0, nz, a.zblock):
                z1 = min(z0 + a.zblock, nz)
                vert[z0:z1] = d[z0:z1, ymid, :]
            tifffile.imwrite(v_path, vert)
            print(f'  vertical   y={ymid}  ({nz} x {nx})  -> {v_path}'
                  f'   {time.time()-t:.1f}s', flush=True)
        else:
            print(f'  vertical exists, skipped: {v_path}', flush=True)


if __name__ == '__main__':
    main()
