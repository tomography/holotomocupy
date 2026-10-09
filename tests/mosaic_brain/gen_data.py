#!/usr/bin/env python
"""Synthetic mosaic holotomography data, YY037A-like acquisition geometry.

    mpirun -n <ngpus> python gen_data.py config_gen.conf

One self-contained HDF5 file per tile -- the same arrangement as the real
YY037A scan -- each in exactly the layout holotomocupy.reader.Reader expects:

    {path_out}/{pfile}_{tile}.h5         one per tile, e.g. ..._0_0.h5
        /exchange/energy                 [1]              keV
        /exchange/focusdetectordistance  [1]              m
        /exchange/z1                     [ndist_tile]     m
        /exchange/detector_pixelsize     [1]              m, UNBINNED
        /exchange/theta                  [ntheta,1]       degrees
        /exchange/cshifts_final          [ntheta, ndist_tile, 2]  finest obj px
        /exchange/pref_{bin}             [ndist_tile, nz, n]      intensity
        /exchange/pdata{k}_{bin}         [ntheta, nz, n]  k = 0..ndist_tile-1
        /exchange attrs: tile, tile_index, tile_offset, tiles, ndist_tile, bin

    {path_out}/{pfile}_prb.h5            the ground-truth probe (prb_file=),
                                         ndist entries, tile-major

There is no combined file and no metadata master: {pfile}.h5 is only a name
stem, used to find {pfile}_obj.h5, exactly as {pfile}_mosaic.h5 is in the real
pipeline.

steps15.py then stitches the tiles into one mosaic, runs multi-distance Paganin
and FBP, and writes the real {pfile}_obj.h5 that step6.py starts from.

step6.py reads the tiles back with mosaic_reader.MosaicReader, which flattens
them onto one distance axis, tile-major, as step6 does for the real scan:
ndist = ntile_v*ntile_h * len(z1) and the flat index is tile*len(z1) + k.
Where each tile sits is baked into its cshifts_final, so read_pos needs no
separate tile_offsets term.

No shrinkage: /exchange/shrink is deliberately absent, so shrink_nd stays 0 and
the effective demagnification is 1/norm_magnification for every angle.

All angles are generated in one pass.  vars['proj'] is [ntheta, nzobj, nobj]
complex64 -- one full mosaic plane per angle -- and proj_tmp and data scale the
same way, so check the "in memory" lines --plan prints against the node before
launching; these are pinned (page-locked) allocations.
"""

import os
import sys
import time
import contextlib
import numpy as np
import cupy as cp
import h5py
from types import SimpleNamespace
from mpi4py import MPI

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, _HERE)
from holotomocupy.rec_mpi import Rec                       # noqa: E402
from holotomocupy.config import parse_args_gen             # noqa: E402
from holotomocupy.logger_config import logger, set_log_level   # noqa: E402
from holotomocupy.utils import read_tiff                   # noqa: E402
from mosaic_geometry import read_tile_offsets, read_tile_shifts   # noqa: E402

cp.cuda.set_pinned_memory_allocator(None)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
args = parse_args_gen(sys.argv[1])
set_log_level(args.log_level)


def info(msg):
    if rank == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Sample volume
# ---------------------------------------------------------------------------
def open_sample(spec, nzobj, nobj):
    """Open a sample volume for slice-wise reading.

    spec is a path, optionally "path::dataset" for HDF5.  Returns
    (accessor, file_or_None); the accessor is indexable by z-slice and has
    .shape / .dtype, so an h5py dataset, a memmap and an mmap'd .npy all work.
    Close the second element when done.

      .h5/.hdf5   the named dataset, or the largest 3-D one in the file
      .npy        memory-mapped
      anything    raw float32 of exactly (nzobj, nobj, nobj)
    """
    path, _, dset = spec.partition('::')
    if path.endswith(('.h5', '.hdf5')):
        f = h5py.File(path, 'r')
        if not dset:
            cand = []
            f.visititems(lambda k, v: cand.append((v.size, k))
                         if isinstance(v, h5py.Dataset) and v.ndim == 3 else None)
            if not cand:
                f.close()
                raise SystemExit(f'{path}: no 3-D dataset found')
            dset = max(cand)[1]
            if len(cand) > 1:
                info(f'  {len(cand)} 3-D datasets in {path}, using the largest: '
                     f'/{dset}  {[c[1] for c in sorted(cand, reverse=True)]}')
        d = f[dset]
        info(f'  /{dset}  shape {d.shape}  dtype {d.dtype}')
        return d, f
    if path.endswith('.npy'):
        return np.load(path, mmap_mode='r'), None
    return np.memmap(path, dtype='float32', mode='r',
                     shape=(nzobj, nobj, nobj)), None


def fill_scaled(vol, cl, nzobj, nobj, span):
    """Fill this rank's cl.vars['obj'] slices from vol, rescaled to span px.

    Two steps, in this order:

      1. CROP VERTICALLY.  The volume is usually taller than the scanned band,
         so only the central `nzobj / q` source rows can land inside the mosaic
         grid.  They are selected first; the rest are never read.
      2. SCALE.  One isotropic factor q = span / sx on all three axes, so the
         x-y extent becomes `span` binned object px and the aspect ratio is
         preserved.  The result is centred on the mosaic grid; whatever still
         falls outside is cropped and whatever is not covered stays 0
         (transmission 1).

    Downsampling is an integer block-average followed by a spline zoom for the
    remainder, so a 3x-smaller grid does not alias.  Work is done one
    destination slice at a time: only a few source slices are ever resident.
    """
    from cupyx.scipy.ndimage import zoom as gpu_zoom

    sz, sy, sx = vol.shape
    q  = span / sx                     # overall scale factor, source -> dest px

    # 1. vertical crop: the central band of source rows that lands in nzobj.
    keep = min(sz, int(np.ceil(nzobj / q)))
    z0   = (sz - keep) // 2
    info(f'  crop z {z0}:{z0 + keep} of {sz}  ({keep} source rows -> {nzobj} px)')

    # 2. isotropic scale by q on z, y and x.
    fi = max(1, int(1.0 / q))          # integer pre-average, antialiasing
    q2 = q * fi                        # residual zoom applied after averaging
    szp, syp, sxp = keep // fi, sy // fi, sx // fi
    info(f'  block-average {fi}x -> {(szp, syp, sxp)}, then zoom {q2:.4f}x')

    def pre_slice(j):
        """Pre-averaged slice j of the CROPPED band, on the GPU, as float32."""
        s0 = z0 + j * fi
        raw = np.asarray(vol[s0:s0 + fi, :syp * fi, :sxp * fi],
                         dtype='float32')
        g = cp.asarray(raw).mean(axis=0)
        return g.reshape(syp, fi, sxp, fi).mean(axis=(1, 3))

    cache = {}
    for i in range(cl.end_obj - cl.st_obj):
        # Destination row cl.st_obj+i, measured from the grid centre, mapped
        # back to the pre-averaged source grid.
        zc = (cl.st_obj + i - (nzobj - 1) / 2) / q2 + (szp - 1) / 2
        j0 = int(np.floor(zc))
        if j0 < 0 or j0 > szp - 1:
            cl.vars['obj'][i] = 0
            continue
        for j in (j0, min(j0 + 1, szp - 1)):
            if j not in cache:
                cache[j] = pre_slice(j)
        w = np.float32(zc - j0)
        s = cache[j0] * (1 - w) + cache[min(j0 + 1, szp - 1)] * w
        for j in list(cache):
            if j < j0:
                del cache[j]

        s = gpu_zoom(s, q2, order=1, mode='constant', cval=0.0)
        # centre s inside the (nobj, nobj) grid, cropping the overhang
        out = cp.zeros((nobj, nobj), dtype='float32')
        sh, sw = s.shape
        y0, x0 = (nobj - sh) // 2, (nobj - sw) // 2
        sy0, sx0 = max(0, -y0), max(0, -x0)
        dy0, dx0 = max(0, y0), max(0, x0)
        h = min(sh - sy0, nobj - dy0)
        wd = min(sw - sx0, nobj - dx0)
        out[dy0:dy0 + h, dx0:dx0 + wd] = s[sy0:sy0 + h, sx0:sx0 + wd]

        cl.vars['obj'][i].real = cp.asnumpy(out)
        cl.vars['obj'][i].imag = cp.asnumpy(-out/args.delta_beta)


PRB_CROP = 256      # unbinned detector px cropped from every side of the probe


def prep_probe(a, n):
    """[ndist, ndet, ndet] probe plane -> [ndist, n, n].

    The recorded ID16A probe has a hard frame at the edge of the window.  Crop
    PRB_CROP px from every side and resample what is left onto the working grid,
    so the illumination structure survives and the border does not.  The
    resample is the same two-step one fill_scaled uses: an integer
    block-average, then a spline zoom for the residual factor.

    Amplitude and phase are passed through separately -- zooming the complex
    field would mix them wherever the phase is steep.
    """
    from cupyx.scipy.ndimage import zoom as gpu_zoom
    c = PRB_CROP
    a = a[:, c:a.shape[1] - c, c:a.shape[2] - c]
    fi = max(1, a.shape[1] // n)
    if fi > 1:
        m = (a.shape[1] // fi) * fi
        a = a[:, :m, :m].reshape(a.shape[0], m // fi, fi, m // fi, fi).mean(axis=(2, 4))
    if a.shape[1] != n:
        g = gpu_zoom(cp.asarray(a, dtype='float32'),
                     (1, n / a.shape[1], n / a.shape[2]), order=1)
        a = cp.asnumpy(g)
    if a.shape[1] < n or a.shape[2] < n:      # spline rounding, at most 1 px
        a = np.pad(a, ((0, 0), (0, n - a.shape[1]), (0, n - a.shape[2])), mode='edge')
    return a[:, :n, :n].astype('float32')


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
ndist_t = len(args.z1)                       # distances per tile
scale   = 1.0 / 2**args.bin
n       = args.ndet  >> args.bin             # detector, binned
nobj    = args.nobj  >> args.bin             # mosaic width, binned
nzobj   = args.nzobj >> args.bin             # mosaic height, binned

z1_t = np.array(args.z1, dtype='float32')
mag  = args.focustodetectordistance / z1_t
norm_mag = mag / mag[0]
voxelsize = args.detector_pixelsize / mag[0]

# --- tiles --------------------------------------------------------------
tile_names, tile_off = read_tile_offsets(args.tile_file)
ntiles   = len(tile_names)
ndist    = ntiles * ndist_t                  # flat distance axis

# --- per-angle sample shifts -> cshifts_final ---------------------------
# cshifts_final[itheta, tile*ndist_t + k] = tile_offset[tile] + shift[itheta, k]
cshifts = np.empty([args.ntheta, ndist, 2], dtype='float32')
for t, name in enumerate(tile_names):
    cshifts[:, t * ndist_t:(t + 1) * ndist_t] = (
        read_tile_shifts(args.shift_dir, name, args.ntheta, ndist_t) + tile_off[t])

# Positions on the generation grid, via exactly the transform Reader.read_pos
# applies, so reading the file back at this bin reproduces them.
pos_all = cshifts * np.float32(scale)
pos_all = np.ascontiguousarray(pos_all.transpose(1, 0, 2))   # [ndist, ntheta, 2]

theta_deg = (np.arange(args.ntheta, dtype='float32')
             * (args.theta_range / args.ntheta))
theta_all = (-theta_deg / 180 * np.pi).astype('float32')   # Reader's sign convention

info('=' * 68)
info(f'  energy               : {args.energy} keV')
info(f'  focus-det distance   : {args.focustodetectordistance*100:.3f} cm')
info(f'  z1 per tile          : {[f"{v*100:.3f} cm" for v in z1_t]}')
info(f'  norm magnifications  : {np.round(norm_mag, 6).tolist()}')
info(f'  detector pixel size  : {args.detector_pixelsize*2**args.bin*1e9:.3f} nm (bin={args.bin})')
info(f'  voxel size           : {voxelsize*2**args.bin*1e9:.3f} nm (bin={args.bin})')
info(f'  detector size        : {n} x {n}')
info(f'  object size          : {nzobj} x {nobj} x {nobj}')
info(f'  mosaic               : {args.ntile_v} rows x {args.ntile_h} cols = {ntiles} tiles')
info(f'  n distances          : {ndist} = {ntiles} tiles x {ndist_t}')
info(f'  n angles             : {args.ntheta}')
info('  shrinkage            : none')
info(f'  sample               : {args.obj_vol or "zero (transmission 1)"}')
info(f'  data on disk         : '
     f'{ndist*args.ntheta*n*n*4/2**30:.1f} GiB total, '
     f'{ndist_t*args.ntheta*n*n*4/2**30:.1f} GiB per tile file')
info(f'  output               : {ntiles} x '
     f'{os.path.basename(args.out_file).replace(".h5", "_<tile>.h5")} '
     f'in {args.path_out}')
info(f'  object in memory     : '
     f'{nzobj*nobj*nobj*8/2**30:.1f} GiB over {comm.Get_size()} rank(s)')
info(f'  proj  in memory      : '
     f'{2*args.ntheta*nzobj*nobj*8/2**30:.1f} GiB over {comm.Get_size()} rank(s)  '
     f'(vars[proj] + proj_tmp)')
info(f'  data  in memory      : '
     f'{ndist*args.ntheta*n*n*4/2**30:.1f} GiB over {comm.Get_size()} rank(s)')
_tot = (nzobj*nobj*nobj*8 + 2*args.ntheta*nzobj*nobj*8
        + ndist*args.ntheta*n*n*4) / 2**30
info(f'  total pinned         : {_tot:.1f} GiB node-wide, '
     f'{_tot/comm.Get_size():.1f} GiB per rank -- page-locked, must fit in RAM')
info('=' * 68)

if '--plan' in sys.argv:
    # Sizing only: everything above is pure geometry, so this runs anywhere.
    sys.exit(0)

# ---------------------------------------------------------------------------
# Rec
# ---------------------------------------------------------------------------
rargs = SimpleNamespace(
    energy                  = args.energy,
    detector_pixelsize      = args.detector_pixelsize * 2**args.bin,
    focustodetectordistance = args.focustodetectordistance,
    z1                      = np.tile(z1_t, ntiles),          # tile-major
    theta                   = theta_all,
    ndist                   = ndist,
    ntheta                  = args.ntheta,
    nz                      = n,
    n                       = n,
    nzobj                   = nzobj,
    nobj                    = nobj,
    mask                    = 1.0,
    lam_prbfit              = 0.0,
    lam_laplacian           = 0.0,
    rho                     = [1, 1, 1, 0],
    niter                   = 0,
    nchunk                  = args.nchunk,
    checkpoint_step         = -1,
    error_step              = -1,
    start_iter              = 0,
    comm                    = comm,
    # gen_data touches only vars / data / proj_tmp.  'gen' tells
    # Rec.alloc_arrays to skip the gradient and conjugate-direction buffers --
    # two more obj-sized and two more proj-sized pinned slabs, plus etas['obj'] --
    # so the run never peaks above what the lines above report.  Clearing
    # cl.grads / cl.etas after the fact did not help: the peak is inside
    # alloc_arrays.
    alloc_mode              = 'gen',
)
info('Create class')
cl = Rec(rargs)

st_th, end_th = cl.st_theta, cl.end_theta
logger.info(f'theta-range [{st_th}:{end_th}), obj-range [{cl.st_obj}:{cl.end_obj})')

# ---------------------------------------------------------------------------
# Object
# ---------------------------------------------------------------------------
if args.obj_vol:
    info(f'Reading sample from {args.obj_vol}')
    vol, vf = open_sample(args.obj_vol, nzobj, nobj)
    if args.obj_span_px:
        # Scale the volume so its x-y extent spans obj_span_px finest object px
        # (the physical width of the sample), centred on the mosaic grid.
        span = args.obj_span_px / 2**args.bin
        info(f'  scaling {vol.shape} -> x-y span {span:.1f} binned px '
             f'({args.obj_span_px:g} finest px, '
             f'{args.obj_span_px*voxelsize*1e3:.3f} mm)')
        fill_scaled(vol, cl, nzobj, nobj, span)
    else:
        if vol.shape != (nzobj, nobj, nobj):
            raise SystemExit(f'{args.obj_vol} has shape {vol.shape}, expected '
                             f'{(nzobj, nobj, nobj)} (the binned generation grid); '
                             f'set obj_span_px to rescale it instead')
        for i in range(cl.end_obj - cl.st_obj):
            s = np.asarray(vol[cl.st_obj + i], dtype='float32')
            cl.vars['obj'][i].real = s
            cl.vars['obj'][i].imag = -s / args.delta_beta
    if vf is not None:
        vf.close()
else:
    cl.vars['obj'][:] = 0          # transmission exp(0) = 1 everywhere

# obj_scale multiplies the whole volume.  The sample file carries arbitrary
# grey levels, so this is what sets the projected phase excursion; the
# "projected phase" line printed after gen_data is the thing to tune it
# against (a few rad to a few tens of rad is well conditioned).
if args.obj_scale != 1.0:
    info(f'scaling the object by {args.obj_scale:g}')
    cl.vars['obj'][:] *= np.float32(args.obj_scale)
# ---------------------------------------------------------------------------
# Probe — the ID16A probe, one copy per tile
# ---------------------------------------------------------------------------
info(f'Reading probe from {args.prb_abs}')
prb_abs   = read_tiff(args.prb_abs).astype('float32')
prb_phase = read_tiff(args.prb_phase).astype('float32')
if prb_abs.shape[0] < ndist_t or prb_phase.shape[0] < ndist_t:
    raise SystemExit(f'probe has {min(prb_abs.shape[0], prb_phase.shape[0])} '
                     f'distances, need {ndist_t}')
prb_abs, prb_phase = prb_abs[:ndist_t], prb_phase[:ndist_t]

nz0, n0 = prb_abs.shape[1:]
if nz0 != args.ndet or n0 != args.ndet:
    raise SystemExit(f'probe is {nz0}x{n0}, config says the detector is {args.ndet}')
info(f'  crop {PRB_CROP} px per side -> {nz0 - 2*PRB_CROP}^2, resample -> {n}^2')
prb_t = (prep_probe(prb_abs, n)
         * np.exp(1j * prep_probe(prb_phase, n))).astype('complex64')
prb_t /= np.mean(np.abs(prb_t), axis=(1, 2))[:, None, None]

cl.vars['prb'][:] = np.tile(prb_t, (ntiles, 1, 1))          # tile-major

# Flat field: ref[j] = |D prb_j|, one per flat distance index
ref = np.empty([ndist, n, n], dtype='float32')
for j in range(ndist):
    cl._dist_idx = j
    ref[j] = cp.abs(cl.cl_prop.D(cp.asarray(cl.vars['prb'][j])[cp.newaxis], j)[0]).get()

# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------
if rank == 0:
    os.makedirs(args.path_out, exist_ok=True)
comm.Barrier()

def tile_file(name):
    """{path_out}/{pfile}_{tile}.h5"""
    return args.out_file.replace('.h5', f'_{name}.h5')


def write_meta(g, z1v, cs, refv):
    """The acquisition metadata Reader.__init__ / read_pos / read_ref need."""
    g.create_dataset('energy',                data=np.array([args.energy], 'float32'))
    g.create_dataset('focusdetectordistance',
                     data=np.array([args.focustodetectordistance], 'float32'))
    g.create_dataset('z1',                    data=np.asarray(z1v, 'float32'))
    g.create_dataset('detector_pixelsize',
                     data=np.array([args.detector_pixelsize], 'float32'))
    g.create_dataset('theta',                 data=theta_deg[:, None])
    g.create_dataset('cshifts_final',         data=cs)
    g.create_dataset(f'pref_{args.bin}',      data=refv)
    g.attrs['ndist_tile']  = ndist_t
    g.attrs['bin']         = args.bin
    g.attrs['shrinkage']   = 'none'


if rank == 0:
    # One file per tile, each self-contained: its own ndist_tile distances, with
    # the tile offset already in its cshifts_final, so a plain Reader with
    # ndist=ndist_tile can open it on its own.
    for t, name in enumerate(tile_names):
        sl = slice(t * ndist_t, (t + 1) * ndist_t)
        with h5py.File(tile_file(name), 'w') as f:
            g = f.create_group('exchange')
            write_meta(g, z1_t, cshifts[:, sl], (ref**2)[sl])
            g.attrs['tile']         = str(name)
            g.attrs['tile_index']   = t
            g.attrs['tile_offset']  = tile_off[t]
            g.attrs['tiles']        = [str(x) for x in tile_names]
            g.attrs['note'] = (f'synthetic; tile {name} of {ntiles}, flat distance '
                               f'index in the mosaic = {t}*{ndist_t} + k; '
                               f'cshifts_final already includes the tile offset')
    info(f'wrote {ntiles} tile files, {tile_file(tile_names[0])} ...')

    with h5py.File(args.out_file.replace('.h5', '_prb.h5'), 'w') as f:
        f.create_dataset('prb_amp',   data=np.abs(cl.vars['prb']).astype('float32'))
        f.create_dataset('prb_phase', data=np.angle(cl.vars['prb']).astype('float32'))
comm.Barrier()

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
nrow   = max(1, (1 << 28) // (n * n * 4))          # rows per MPI-IO call (<2 GiB)
t_start = time.time()
with contextlib.ExitStack() as stack:
    files = [stack.enter_context(h5py.File(tile_file(name), 'a',
                                           driver='mpio', comm=comm))
             for name in tile_names]
    # Created here rather than on rank 0 above: a dataset written collectively
    # has to be created collectively. Chunked one frame at a time so HDF5
    # allocates as we go instead of zero-filling the whole file up front.
    # dsets[k] is the flat distance k = tile*ndist_t + kk, same order as
    # cl.data, z1 and cshifts_final.
    dsets = [files[k // ndist_t]['/exchange'].create_dataset(
                 f'pdata{k % ndist_t}_{args.bin}',
                 shape=(args.ntheta, n, n), dtype='float32', chunks=(1, n, n))
             for k in range(ndist)]
    cl.vars['pos'][:] = pos_all[:, st_th:end_th]
    cl.gen_data(cl.vars, cl.data)             # intensity K|y|^2, as read back
    info(f'generated in {time.time()-t_start:.0f}s, writing')

    # The projected phase actually seen by exp(1j*proj), measured rather than
    # derived: the Radon normalisation folds n, ntheta and norm_const together,
    # so this is the only reliable way to check the magnitude.  A few rad to a
    # few tens of rad is a well-conditioned test; rescale obj_vol if it is off.
    pr = cl.vars['proj'].real
    pi = cl.vars['proj'].imag
    lo  = comm.allreduce(float(pr.min()), op=MPI.MIN)
    hi  = comm.allreduce(float(pr.max()), op=MPI.MAX)
    alo = comm.allreduce(float(pi.min()), op=MPI.MIN)
    ahi = comm.allreduce(float(pi.max()), op=MPI.MAX)
    npx = comm.allreduce(pr.size, op=MPI.SUM)
    avg  = comm.allreduce(float(pr.sum(dtype='float64')), op=MPI.SUM) / npx
    aavg = comm.allreduce(float(pi.sum(dtype='float64')), op=MPI.SUM) / npx
    info(f'projected phase  [{lo:+.4g}, {hi:+.4g}] rad   mean {avg:+.4g}')
    info(f'projected absorp [{alo:+.4g}, {ahi:+.4g}] mean {aavg:+.4g}   '
         f'transmission exp(-imag) in '
         f'[{np.exp(-ahi):.4g}, {np.exp(-alo):.4g}]')

    # Each rank owns angles [st_th:end_th) and no other rank touches them, so
    # these are independent writes even though the file is open with mpio.
    for k in range(ndist):
        for i0 in range(0, end_th - st_th, nrow):
            i1 = min(i0 + nrow, end_th - st_th)
            dsets[k][st_th + i0:st_th + i1] = cl.data[k, i0:i1]
        if (k + 1) % ndist_t == 0:
            info(f'  wrote tile {k // ndist_t + 1}/{ntiles}  '
                 f'({time.time()-t_start:.0f}s elapsed)')

comm.Barrier()
info(f'done in {time.time()-t_start:.0f}s -> {ntiles} tile files next to '
     f'{args.out_file}')
