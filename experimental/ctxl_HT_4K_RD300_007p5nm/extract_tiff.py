#!/usr/bin/env python
"""Write Re(obj) from a step-6 checkpoint out as a cropped TIFF stack.

    mpiexec -n 24 python extract_tiff.py                    # all three sets
    mpiexec -n 8  python extract_tiff.py --sets even        # just one
                  python extract_tiff.py --zmax 4 --out /tmp/t   # serial smoke test

Three reconstructions of the same sample are turned into three stacks:

    full  <-  ..._rec6/checkpoints/checkpoint_<iter>.h5     all 4000 projections
    even  <-  ..._rec6_p0/checkpoints/checkpoint_<iter>.h5  projections 0,2,...,3998
    odd   <-  ..._rec6_p1/checkpoints/checkpoint_<iter>.h5  projections 1,3,...,3999

--dir <set>=<suffix> points one set somewhere else, and --sets picks a subset,
so any other arm can be extracted with the same code:

    --sets full,even --iter 1280 --crop 256 --zcrop 128
    --dir full=rec6_psf10_amp_binned
    --dir even=rec6_psf10_amp_binned_p0
    --out ..._rec6_results_binned_psf10

The arm directories are NOT named to one rule -- the baseline truncated arm is
rec6_binned / rec6_p0_binned / rec6_p1_binned, the psf arms are
rec6_psf10_amp_binned / ..._binned_p0 / ..._binned_p1 -- which is why the
suffix is spelled out per set instead of being built from an arm name.

obj_re is stored [z, y, x] at 5056^3.  --crop trims that many pixels off each
of the four y/x borders and --zcrop trims that many slices off the top and the
bottom, so the defaults 768 and 256 leave 4544 slices of 3520 x 3520.  One TIFF
per slice, so an ImageJ image-sequence import picks the stack up in order.

FILE NUMBERING RESTARTS AT ZERO: with --zcrop 256 the first file is 0000.tiff
and holds global z = 256, and the last is 4543.tiff holding global z = 4799.
Add --zcrop to a file number to get back to the index inside the checkpoint.
All three sets are cropped identically, so full/even/odd stay slice-aligned
with each other -- an FSC can be taken file-by-file.

That is 47 MiB a slice, 210 GiB a set, ~629 GiB for all three.  Uncompressed:
these go straight into segmentation and FSC, and the read cost of zlib on a
210 GiB stack is worse than the disk.

WORK SPLIT.  Ranks are dealt out over the sets (rank % nsets), and inside each
set that group of ranks takes contiguous z blocks.  Each rank opens the file
independently read-only -- no MPI-IO -- because every slice read is already a
contiguous 102 MB block of an uncompressed dataset.  Use a multiple of 3 ranks
for all three sets or the groups come out uneven; fewer ranks than sets works
too, with a rank doing several sets in turn.

RESTART.  A slice is skipped only if its TIFF is already there AND its stored
shape is the one the current --crop asks for, so a job that runs out of
walltime can simply be resubmitted.  Comparing file *size* is not enough: a
stack left over from a wider crop has larger files, which a size test reads as
"already done" and silently skips the whole job.  --force rewrites regardless.

STALE SLICES.  A previous run with a smaller --zcrop leaves files past the end
of the current stack (0 --zcrop wrote 5056, 256 writes 4544).  Mixing the two
geometries in one directory would break an image-sequence import, so the run
stops and names them unless --clean is passed to delete them first.
"""
import argparse
import os
import sys
import time

import numpy as np

SETS = {                       # output name -> reconstruction directory suffix
    'full': 'rec6',
    'even': 'rec6_p0',
    'odd':  'rec6_p1',
}
BASE = ('/eagle/APS_IRI/vnikitin/20260829/ctxl/'
        'ctxl_HT_4K_RD300_007p5nm_0001_')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--iter', type=int, default=1472,
                   help='checkpoint iteration number (default 1472)')
    p.add_argument('--crop', type=int, default=768,
                   help='pixels trimmed from each y/x border (default 768)')
    p.add_argument('--zcrop', type=int, default=256,
                   help='slices trimmed from the top and the bottom in z '
                        '(default 256); file numbering still starts at 0')
    p.add_argument('--base', default=BASE,
                   help='common prefix of the reconstruction directories')
    p.add_argument('--out', default=BASE + 'rec6_results',
                   help='output root; one subdirectory per set')
    p.add_argument('--sets', default=','.join(SETS),
                   help='comma-separated subset of: ' + ','.join(SETS))
    p.add_argument('--dir', action='append', default=[], metavar='SET=SUFFIX',
                   help='point one set at a different reconstruction directory, '
                        'as <set>=<suffix after --base>; repeatable.  The arm '
                        'directories are not named to one rule -- the baseline '
                        'truncated arm is rec6_binned / rec6_p0_binned / '
                        'rec6_p1_binned but the psf arms are '
                        'rec6_psf10_amp_binned / ..._binned_p0 / ..._binned_p1 '
                        '-- so the suffixes are given explicitly rather than '
                        'built from an arm name')
    p.add_argument('--part', default='obj_re',
                   help='dataset to extract (default obj_re, the real part)')
    p.add_argument('--zmax', type=int, default=None,
                   help='stop after this many kept slices -- for smoke tests')
    p.add_argument('--force', action='store_true',
                   help='rewrite slices whose TIFF already looks complete')
    p.add_argument('--clean', action='store_true',
                   help='delete leftover TIFFs past the end of the current '
                        'stack, and any .part temporaries')
    p.add_argument('--dry-run', action='store_true',
                   help='report sizes and the work split, write nothing')
    a = p.parse_args()

    sets = [s.strip() for s in a.sets.split(',') if s.strip()]
    bad = [s for s in sets if s not in SETS]
    if bad:
        sys.exit(f'unknown set(s): {",".join(bad)}; choose from {",".join(SETS)}')

    dirs = dict(SETS)
    for spec in a.dir:
        tag, _, suffix = spec.partition('=')
        tag, suffix = tag.strip(), suffix.strip()
        if tag not in SETS:
            sys.exit(f'--dir {spec}: unknown set {tag}; '
                     f'choose from {",".join(SETS)}')
        if not suffix:
            sys.exit(f'--dir {spec}: expected <set>=<suffix>')
        dirs[tag] = suffix

    # MPI is optional: without it this is rank 0 of 1 and does the whole job.
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        # Not just ImportError: mpi4py imports fine but raises RuntimeError
        # ("cannot load MPI library") on a workstation with no libmpi, which
        # is exactly where the documented serial smoke test gets run.
        comm, rank, size = None, 0, 1

    import h5py
    import tifffile

    def log(msg):
        print(f'[{rank:3d}] {msg}', flush=True)

    # --- shapes, and the y/x window ------------------------------------------
    plan = []
    for tag in sets:
        src = os.path.join(a.base + dirs[tag], 'checkpoints',
                           f'checkpoint_{a.iter}.h5')
        if not os.path.exists(src):
            sys.exit(f'missing checkpoint: {src}')
        with h5py.File(src, 'r') as f:
            if a.part not in f:
                sys.exit(f'{src} has no dataset {a.part}')
            shape = f[a.part].shape
        plan.append((tag, src, shape))

    nz, ny, nx = plan[0][2]
    for tag, src, shape in plan:
        if shape != (nz, ny, nx):
            sys.exit(f'{tag} is {shape}, expected {(nz, ny, nx)}')
    c = a.crop
    if 2 * c >= min(ny, nx):
        sys.exit(f'--crop {c} leaves nothing of a {ny} x {nx} slice')
    zc = a.zcrop
    if 2 * zc >= nz:
        sys.exit(f'--zcrop {zc} leaves no slices of {nz}')
    y0, y1, x0, x1 = c, ny - c, c, nx - c
    nz_out = nz - 2 * zc                        # kept slices, global z in [zc, nz-zc)
    if a.zmax is not None:
        nz_out = min(nz_out, a.zmax)
    width = max(4, len(str(nz_out - 1)))        # 4544 slices -> 0000..4543
    slice_bytes = (y1 - y0) * (x1 - x0) * 4

    if rank == 0:
        print(f'{a.part}  {nz} x {ny} x {nx} float32', flush=True)
        print(f'crop {c} px -> {y1-y0} x {x1-x0}, zcrop {zc} -> {nz_out} slices '
              f'(global z {zc}..{zc+nz_out-1}, file number + {zc}), '
              f'{slice_bytes/2**20:.1f} MiB/slice, '
              f'{nz_out*slice_bytes/2**30:.1f} GiB/set, '
              f'{len(sets)*nz_out*slice_bytes/2**30:.1f} GiB total', flush=True)
        for tag, src, _ in plan:
            print(f'  {tag:5s} <- {src}', flush=True)
            print(f'  {tag:5s} -> {os.path.join(a.out, tag)}/'
                  f'{0:0{width}d}.tiff .. {nz_out-1:0{width}d}.tiff', flush=True)
        if not a.dry_run:
            for tag in sets:
                os.makedirs(os.path.join(a.out, tag), exist_ok=True)
    # Leftovers from a run with a smaller --zcrop must go before this one
    # starts, or the directory ends up holding two different slice geometries.
    stop = np.zeros(1, dtype='i')
    if rank == 0 and not a.dry_run:
        import re
        pat = re.compile(r'^(\d+)\.tiff$')
        for tag in sets:
            outdir = os.path.join(a.out, tag)
            stale, parts = [], []
            for name in os.listdir(outdir):
                if name.endswith('.part'):
                    parts.append(name)
                    continue
                m = pat.match(name)
                if m and int(m.group(1)) >= nz_out:
                    stale.append(name)
            for name in parts:
                os.remove(os.path.join(outdir, name))
            if not stale:
                continue
            if a.clean:
                for name in sorted(stale):
                    os.remove(os.path.join(outdir, name))
                print(f'{tag}: removed {len(stale)} leftover slices past '
                      f'{nz_out-1:0{width}d}.tiff', flush=True)
            else:
                print(f'ERROR: {outdir} holds {len(stale)} slices past the end '
                      f'of this stack ({min(stale)} .. {max(stale)}), left by a '
                      f'run with a smaller --zcrop.  Rerun with --clean to '
                      f'delete them.', flush=True)
                stop[0] = 1
    if comm is not None:
        comm.Bcast(stop, root=0)
        comm.Barrier()
    if stop[0]:
        sys.exit(1)                 # must be non-zero: the PBS wrapper tests it

    # --- deal the ranks out over the sets, then z contiguously inside each ----
    my_work = []
    for gi, (tag, src, _) in enumerate(plan):
        group = [r for r in range(size) if r % len(plan) == gi]
        if not group:
            # Fewer ranks than sets: this set has no group of its own, so fold
            # it onto one rank that then does two or three sets back to back.
            group = [gi % size]
        if rank not in group:
            continue
        k, ngrp = group.index(rank), len(group)
        # near-equal contiguous blocks: the first nz_out % ngrp get one extra
        q, rem = divmod(nz_out, ngrp)
        z0 = k * q + min(k, rem)
        z1 = z0 + q + (1 if k < rem else 0)
        my_work.append((tag, src, z0, z1))

    for tag, src, z0, z1 in my_work:
        log(f'{tag}: files {z0}..{z1-1} = global z {zc+z0}..{zc+z1-1}'
            f'  ({z1-z0} slices)')
    if a.dry_run:
        return

    # --- read, crop, write ----------------------------------------------------
    def complete(dst):
        """True if dst already holds a slice of exactly the shape we want."""
        try:
            with tifffile.TiffFile(dst) as tf:
                return tuple(tf.series[0].shape) == (y1 - y0, x1 - x0)
        except Exception:
            return False                        # missing, truncated, not a TIFF

    t_start = time.time()
    written = skipped = 0
    for tag, src, z0, z1 in my_work:
        outdir = os.path.join(a.out, tag)
        with h5py.File(src, 'r') as f:
            d = f[a.part]
            for j in range(z0, z1):
                z = zc + j                      # j numbers the file, z the slice
                dst = os.path.join(outdir, f'{j:0{width}d}.tiff')
                if not a.force and os.path.exists(dst) and complete(dst):
                    skipped += 1
                    continue
                # Read the whole slice: it is one contiguous 102 MB block, which
                # beats letting HDF5 do a strided read of the cropped window.
                sl = np.ascontiguousarray(d[z, y0:y1, x0:x1])
                tmp = dst + '.part'
                tifffile.imwrite(tmp, sl)
                os.replace(tmp, dst)            # never leave a short TIFF behind
                written += 1
                if written % 50 == 0:
                    el = time.time() - t_start
                    log(f'{tag}: {written} written, {el/60:.1f} min, '
                        f'{written*slice_bytes/2**30/el:.2f} GiB/s')

    el = time.time() - t_start
    log(f'done: {written} written, {skipped} skipped, {el/60:.1f} min')
    if comm is not None:
        tot = comm.reduce(written, op=MPI.SUM, root=0)
        tsk = comm.reduce(skipped, op=MPI.SUM, root=0)
        if rank == 0:
            print(f'TOTAL {tot} written, {tsk} skipped in {el/60:.1f} min '
                  f'({tot*slice_bytes/2**30/el:.2f} GiB/s aggregate)', flush=True)


if __name__ == '__main__':
    main()
