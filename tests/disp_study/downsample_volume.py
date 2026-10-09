#!/usr/bin/env python
"""Block-average a sample volume once, so gen_data does not re-read it every run.

common.fill_volume downsamples the source volume with an integer block average,
which means it touches every voxel: at n = 256 the 3072^3 brain volume is
rescaled to 158 object px, so fi = 19 and every one of the 161 destination
slices pulls a fresh 19 x 3059^2 = 711 MB slab off disk.  That is the whole
116 GB file per gen_data run -- ~150 s, of which ~90 % is the read and 2 ms is
the arithmetic.  run_dose_brain.sh generates twice, so it pays it twice.

Binning the source down to a grid that is still comfortably wider than the
object span removes almost all of it: a bin-8 copy of the brain volume is
384^3 = 226 MB, and fill_volume's own antialiasing then does the remaining
2.4x.  The result is not bit-identical to reading the full volume (16 box taps
plus a 0.11 px Gaussian instead of 19 box taps), but it is band-limited to the
same 158 px grid, so anything the object grid can represent is unchanged.

    # what would be used for a 158 px object, and where it would live
    python downsample_volume.py --in /data3/.../init.h5::exchange/data \
        --span 158 --plan

    # build it (streams the source once; minutes, then never again)
    python downsample_volume.py --in /data3/.../init.h5::exchange/data \
        --factor 8 --out /data3/.../init_bin8.h5

Serial, no MPI, no GPU -- the block average is memory-bandwidth trivial next to
the read, and keeping it off the GPU lets this run beside a reconstruction.
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

MARGIN = 2.0   # keep the binned grid this many times wider than the object span


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in', dest='inp', required=True,
                   help='source volume, "path" or "path::dataset"')
    p.add_argument('--span', type=float, default=None,
                   help='object width in px the volume will be rescaled to; '
                        'picks the largest factor that keeps the binned grid '
                        f'{MARGIN:g}x wider than that')
    p.add_argument('--factor', type=int, default=None,
                   help='block-average factor, instead of deriving one from --span')
    p.add_argument('--out', default=None,
                   help='output .h5 (default: <source>_bin<factor>.h5 beside the source)')
    p.add_argument('--dir', default=None,
                   help='directory for the default output name (default: the source dir)')
    p.add_argument('--dset', default='exchange/data', help='dataset name to write')
    p.add_argument('--plan', action='store_true',
                   help='print "<factor> <spec>" and exit without reading anything')
    return p.parse_args()


def choose_factor(shape, span):
    """Largest common divisor-ish factor leaving the grid MARGIN x wider than span."""
    limit = int(min(shape) / (MARGIN * span))
    for f in range(max(1, limit), 0, -1):
        if all(s % f == 0 for s in shape):
            return f
    return 1


def default_out(path, factor, dirname=None):
    base = os.path.basename(path)
    stem = base[:-3] if base.endswith('.h5') else \
           base[:-5] if base.endswith('.hdf5') else base
    return os.path.join(dirname or os.path.dirname(os.path.abspath(path)),
                        f'{stem}_bin{factor}.h5')


a = parse()
path = a.inp.partition('::')[0]

_vol, _fh = C.open_volume(a.inp)
shape, dtype = _vol.shape, _vol.dtype
if _fh is not None:
    _fh.close()

factor = a.factor
if factor is None:
    if a.span is None:
        raise SystemExit('give --factor or --span')
    factor = choose_factor(shape, a.span)

out = a.out or default_out(path, factor, a.dir)
spec = a.inp if factor <= 1 else f'{out}::{a.dset}'

if a.plan:
    print(factor, spec)
    raise SystemExit(0)

if factor <= 1:
    raise SystemExit(f'factor {factor}: nothing to do, use {a.inp} directly')

sz, sy, sx = shape
oz, oy, ox = sz // factor, sy // factor, sx // factor
src_gb = np.prod(shape) * np.dtype(dtype).itemsize / 1e9
dst_gb = oz * oy * ox * 4 / 1e9
print(f'{a.inp}  {shape} {dtype}  {src_gb:.1f} GB')
print(f'  block-average {factor}x -> ({oz}, {oy}, {ox})  {dst_gb:.3f} GB')
print(f'  -> {out}::{a.dset}')

# via a temporary file: an interrupted run must not leave a half-written volume
# behind for the next run to read back as if it were complete
tmp = f'{out}.tmp{os.getpid()}'
vol, fh = C.open_volume(a.inp)
t0 = time.time()
try:
    with h5py.File(tmp, 'w') as g:
        d = g.create_dataset(a.dset, shape=(oz, oy, ox), dtype='float32',
                             chunks=(1, oy, ox))
        d.attrs['source'] = a.inp
        d.attrs['bin'] = factor
        step = max(1, oz // 20)
        for i in range(oz):
            if i % step == 0:
                done = i / max(1, oz)
                eta = (time.time() - t0) * (1 - done) / done if done else 0.0
                print(f'  ... slice {i}/{oz}' + (f'  eta {eta / 60:.1f} min' if done else ''),
                      flush=True)
            raw = np.asarray(vol[i * factor:(i + 1) * factor, :oy * factor, :ox * factor],
                             dtype='float32')
            d[i] = raw.mean(axis=0).reshape(oy, factor, ox, factor).mean(axis=(1, 3))
    os.replace(tmp, out)
finally:
    if fh is not None:
        fh.close()
    if os.path.exists(tmp):
        os.remove(tmp)

dt = time.time() - t0
print(f'  done in {dt / 60:.1f} min  ({src_gb / dt:.2f} GB/s)')
print(f'  use  --obj-vol {out}::{a.dset}')
