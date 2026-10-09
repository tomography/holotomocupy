"""The shift_dir file format, in one place.

make_geometry.py writes these files, gen_data.py, steps15.py and
plot_geometry.py read them back:

    {shift_dir}/tile_offsets.txt   where each tile sits on the mosaic
    {shift_dir}/<tile>.txt         the per-angle sample shift of that tile

Both are object pixels on the FINEST grid (bin 0), the same units and sign
convention as /exchange/cshifts_final in the real YY037A files, and the two add
up to exactly that:

    cshifts_final[itheta, tile*ndist + k] = tile_offset[tile] + shift[itheta, k]

They are plain text on purpose -- edit either by hand and every script downstream
picks the change up, since nothing regenerates them.
"""

import os
import numpy as np


def build_tiles(ntile_v, ntile_h, step_v, step_h):
    """[(name, v, h)] for the whole ntile_v x ntile_h grid, row-major.

    Nominal placement: a regular grid of step_v x step_h object px centred on
    the object grid, generated from the tile index alone.

    The name is "{row}_{col}", both zero-based, ordered as the tile lands in the
    composed mosaic: row 0 on top, column 0 on the left.  The offsets are sample
    SHIFTS, so they run opposite to the object-grid axes -- a tile shifted by +h
    appears at smaller x -- hence both steps count down from the centre.
    """
    v_rows = ((ntile_v - 1) / 2 - np.arange(ntile_v)) * step_v
    h_cols = ((ntile_h - 1) / 2 - np.arange(ntile_h)) * step_h
    return [(f'{r}_{c}', float(v_rows[r]), float(h_cols[c]))
            for r in range(ntile_v) for c in range(ntile_h)]


# --------------------------------------------------------------------- write

def write_tile_offsets(path, tiles, ntile_v, ntile_h, step_v, step_h, ndist,
                       voxelsize_nm):
    """Write tile_offsets.txt for `tiles` as build_tiles returns them."""
    with open(path, 'w') as f:
        f.write('# Tile placement for the synthetic YY037A-like mosaic.\n')
        f.write(f'# Object px on the FINEST grid (bin 0, {voxelsize_nm:.3f} nm voxels).\n')
        f.write(f'# {ntile_v} rows x {ntile_h} columns, row-major; the flat\n')
        f.write(f'# distance index used in the HDF5 file is tile*{ndist} + k.\n')
        f.write(f'# Nominal grid, {step_v:g} x {step_h:g} px steps.\n')
        f.write('# name is "{row}_{col}", zero-based, as the tile lands in the\n')
        f.write('# composed mosaic: row 0 on top, column 0 on the left.\n')
        f.write('#\n# index  name       v            h\n')
        for i, (name, v, h) in enumerate(tiles):
            f.write(f'{i:5d}  {name:<8s} {v:12.4f} {h:12.4f}\n')


def write_tile_shifts(shift_dir, name, shift, ndist, voxelsize_nm, source):
    """Write <tile>.txt: [ntheta, ndist, 2] flattened to [ntheta, ndist*2]."""
    ntheta = shift.shape[0]
    cols = ' '.join(f'v{k} h{k}' for k in range(ndist))
    np.savetxt(os.path.join(shift_dir, f'{name}.txt'),
               shift.reshape(ntheta, ndist * 2), fmt='%12.5f',
               header=(f'Per-angle sample shift of tile {name}.\n'
                       f'Object px on the FINEST grid (bin 0, {voxelsize_nm:.3f} nm voxels),\n'
                       f'same units and sign as /exchange/cshifts_final.\n'
                       f'{ntheta} rows x {ndist*2} cols: {cols}\n'
                       f'source: {source}'))


# ---------------------------------------------------------------------- read

def read_tile_offsets(path):
    """tile_offsets.txt -> (names, offsets[ntiles, 2] float32), in file order."""
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found — run make_geometry.py first')
    names, off = [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            _, name, v, h = line.split()
            names.append(name)
            off.append([float(v), float(h)])
    if not names:
        raise SystemExit(f'{path}: no tile rows')
    return names, np.array(off, dtype='float32')


def read_tile_shifts(shift_dir, name, ntheta, ndist):
    """<tile>.txt -> [ntheta, ndist, 2] float32, truncated to the first ntheta."""
    path = os.path.join(shift_dir, f'{name}.txt')
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found — run make_geometry.py first')
    flat = np.loadtxt(path, dtype='float32', ndmin=2)
    if flat.shape[1] != ndist * 2:
        raise SystemExit(f'{path}: {flat.shape[1]} columns, expected {ndist*2} '
                         f'({ndist} distances x 2)')
    if flat.shape[0] < ntheta:
        raise SystemExit(f'{path}: {flat.shape[0]} angles, config asks for {ntheta}')
    return flat[:ntheta].reshape(ntheta, ndist, 2)
