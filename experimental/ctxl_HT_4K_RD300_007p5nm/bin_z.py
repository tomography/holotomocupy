#!/usr/bin/env python
"""Bin a step-6 checkpoint down in z, turning an anisotropic volume isotropic.

    mpiexec -n 8 python bin_z.py --iter 1536 --zbin 2
                 python bin_z.py --iter 1536 --zbin 2 --dry-run

With tomo_upsample = 2 the reconstruction runs on an ANISOTROPIC grid: the
object is 5056 x 2528 x 2528, i.e. z voxels of one detector pixel v and x/y
voxels of 2v.  That is deliberate -- z costs nothing extra in the Radon
transform, so there is no reason to throw it away during the fit -- but the
volume that goes into segmentation should be isotropic.  This script does that
last step, outside the reconstruction:

    obj_re/obj_im  (5056, 2528, 2528)  ->  (2528, 2528, 2528)

by AVERAGING each group of `--zbin` consecutive z slices.  Averaging, not
summing: obj holds a field value per voxel (the same value whatever the grid --
R's 1/sqrt(nobj) and norm_const's sqrt(nobj) cancel), so the representative
value of a merged voxel is the mean of its parts.  Summing would scale the
whole volume by zbin.

Everything else in the checkpoint -- prb_abs, prb_phase, pos, tp and the `iter`
attribute -- is copied through untouched, so the output is a checkpoint in the
same layout and extract_tiff.py can be pointed straight at it with
--part obj_re.  It is NOT a checkpoint you can restart a reconstruction from:
its z sampling no longer matches the data.

WORK SPLIT.  Contiguous blocks of OUTPUT slices, one block per rank, written
collectively through MPI-IO.

RESTART.  A `z_done` uint8 dataset in the output marks each output slice that
has been written, so a job killed by walltime can simply be resubmitted and
picks up where it stopped -- at any rank count, since the marks are per slice
and not per block.  --force ignores the marks and rewrites everything.
"""
import argparse
import os
import sys
import time

import numpy as np

# Default --path: the tomo_upsample=2 arm, which is the only one that produces
# an anisotropic volume.  The u1 arm (..._rec6_u1) is already isotropic and has
# nothing for this script to do.
BASE = ('/eagle/APS_IRI/vnikitin/20260829/ctxl/'
        'ctxl_HT_4K_RD300_007p5nm_0001_rec6_u2')
# copied through verbatim; obj_re/obj_im are the ones that get binned
PASSTHROUGH = ('prb_abs', 'prb_phase', 'pos', 'tp')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--path', default=BASE,
                   help='reconstruction directory holding checkpoints/')
    p.add_argument('--iter', type=int, required=True,
                   help='checkpoint iteration number')
    p.add_argument('--zbin', type=int, default=2,
                   help='consecutive z slices averaged into one (default 2)')
    p.add_argument('--src', default=None,
                   help='read this checkpoint instead of deriving it from '
                        '--path/--iter')
    p.add_argument('--out', default=None,
                   help='write here instead of <src>_zbin<N>.h5')
    p.add_argument('--force', action='store_true',
                   help='rewrite slices already marked done')
    p.add_argument('--dry-run', action='store_true',
                   help='report shapes and the work split, write nothing')
    a = p.parse_args()

    if a.zbin < 1:
        sys.exit(f'--zbin must be >= 1, got {a.zbin}')

    # MPI is optional: without it this is rank 0 of 1 and does the whole job.
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except ImportError:
        MPI, comm, rank, size = None, None, 0, 1

    import h5py

    def log(msg):
        print(f'[{rank:3d}] {msg}', flush=True)

    src = a.src or os.path.join(a.path, 'checkpoints', f'checkpoint_{a.iter}.h5')
    if not os.path.exists(src):
        sys.exit(f'missing checkpoint: {src}')
    dst = a.out or f'{os.path.splitext(src)[0]}_zbin{a.zbin}.h5'
    if os.path.abspath(dst) == os.path.abspath(src):
        sys.exit('--out is the input file; refusing to bin in place')

    # --- shapes ---------------------------------------------------------------
    with h5py.File(src, 'r') as f:
        if 'obj_re' not in f:
            sys.exit(f'{src} has no obj_re')
        nz, ny, nx = f['obj_re'].shape
        has_im = 'obj_im' in f
        present = [k for k in PASSTHROUGH if k in f]
        src_iter = int(f.attrs['iter']) if 'iter' in f.attrs else a.iter
        src_attrs = {k: f.attrs[k] for k in f.attrs}
    nz_out = nz // a.zbin
    if nz_out < 1:
        sys.exit(f'--zbin {a.zbin} leaves no slices of {nz}')
    dropped = nz - nz_out * a.zbin

    slice_bytes = ny * nx * 4
    if rank == 0:
        print(f'{src}', flush=True)
        print(f'  obj_re {nz} x {ny} x {nx} float32'
              + (' (+ obj_im)' if has_im else ' (real only)'), flush=True)
        print(f'  zbin {a.zbin} -> {nz_out} x {ny} x {nx}'
              + (f'   DROPPING the last {dropped} slice(s): {nz} is not a '
                 f'multiple of {a.zbin}' if dropped else ''), flush=True)
        print(f'  copied through: {", ".join(present) or "(none)"}', flush=True)
        print(f'  -> {dst}   '
              f'{(1 + has_im) * nz_out * slice_bytes / 2**30:.1f} GiB', flush=True)
    if a.dry_run:
        # still report the split, it is the thing worth checking before a long run
        q, rem = divmod(nz_out, size)
        for r in range(size):
            z0 = r * q + min(r, rem)
            print(f'  rank {r:3d}: out z {z0}..{z0 + q + (1 if r < rem else 0) - 1}',
                  flush=True)
        return

    # --- create the output (rank 0, serial: prb/tp are small and global) ------
    if rank == 0 and (a.force or not os.path.exists(dst)):
        tmp = dst + '.part'
        with h5py.File(src, 'r') as fi, h5py.File(tmp, 'w') as fo:
            for k, v in src_attrs.items():
                fo.attrs[k] = v
            fo.attrs['iter'] = src_iter
            fo.attrs['zbin'] = a.zbin
            fo.attrs['zbin_src'] = os.path.basename(src)
            for k in present:
                fo.create_dataset(k, data=fi[k][...])
            fo.create_dataset('obj_re', shape=(nz_out, ny, nx), dtype='float32')
            if has_im:
                fo.create_dataset('obj_im', shape=(nz_out, ny, nx), dtype='float32')
            # per-output-slice completion marks; see RESTART in the docstring
            fo.create_dataset('z_done', shape=(nz_out,), dtype='uint8',
                              data=np.zeros(nz_out, dtype='uint8'))
        os.replace(tmp, dst)
        print(f'created {dst}', flush=True)
    if comm is not None:
        comm.Barrier()

    # --- deal the ranks out over contiguous blocks of output slices -----------
    q, rem = divmod(nz_out, size)
    z0 = rank * q + min(rank, rem)
    z1 = z0 + q + (1 if rank < rem else 0)
    log(f'out z {z0}..{z1-1} ({z1-z0} slices) <- in z {z0*a.zbin}..{z1*a.zbin-1}')

    # ~256 MB of INPUT per batch, and the write stays far under the 2^31-byte
    # MPI-IO transfer limit
    z_batch = max(1, (1 << 28) // (a.zbin * slice_bytes))

    t_start = time.time()
    written = skipped = 0
    # The input is read per rank without MPI-IO: every read is a contiguous
    # block of an uncompressed dataset, and the ranks do not overlap.
    with h5py.File(src, 'r') as fi, \
         h5py.File(dst, 'a', driver='mpio', comm=comm) as fo:
        d_re, d_im = fi['obj_re'], (fi['obj_im'] if has_im else None)
        o_re, o_im = fo['obj_re'], (fo['obj_im'] if has_im else None)
        o_done = fo['z_done']
        done = (np.zeros(z1 - z0, dtype='uint8') if a.force or z1 <= z0
                else o_done[z0:z1])
        for j0 in range(z0, z1, z_batch):
            j1 = min(j0 + z_batch, z1)
            if not a.force and done[j0 - z0:j1 - z0].all():
                skipped += j1 - j0
                continue
            # whole batch at once: a partly-done batch is cheaper to redo than
            # to read around, and a batch is at most a few hundred MB
            for ds_in, ds_out in ((d_re, o_re), (d_im, o_im)):
                if ds_in is None:
                    continue
                blk = ds_in[j0 * a.zbin:j1 * a.zbin]
                # AVERAGE, not sum -- see the module docstring
                ds_out[j0:j1] = blk.reshape(j1 - j0, a.zbin, ny, nx).mean(axis=1)
                del blk
            o_done[j0:j1] = 1
            written += j1 - j0
            el = time.time() - t_start
            log(f'{written} written, {el/60:.1f} min, '
                f'{written * (1 + has_im) * slice_bytes / 2**30 / max(el, 1e-9):.2f} GiB/s out')

    el = time.time() - t_start
    log(f'done: {written} written, {skipped} skipped, {el/60:.1f} min')
    if comm is not None:
        tot = comm.reduce(written, op=MPI.SUM, root=0)
        tsk = comm.reduce(skipped, op=MPI.SUM, root=0)
        if rank == 0:
            print(f'TOTAL {tot} written, {tsk} skipped in {el/60:.1f} min -> {dst}',
                  flush=True)
            if tot + tsk == nz_out:
                print(f'{dst} is complete: obj {nz_out} x {ny} x {nx}', flush=True)


if __name__ == '__main__':
    main()
