#!/usr/bin/env python
"""Steps 1-5 for the synthetic mosaic, i.e. in practice only STEP 5.

    mpirun -n <ngpus> python steps15.py config_steps15.conf

The real pipeline's steps 1-4 (EDF conversion, outlier removal and flat-field
normalisation, shift estimation, multi-level binning) have no counterpart here:
gen_data.py already writes clean, binned, flat-field-ready frames, and the
shifts are known exactly.  What is left is STEP 5, done the same way as on real
data:

    per angle, per tile
        rdata = pdata_k / pref_k                       flat-field division
        demagnify + shift each distance onto the tile object grid  (curlySback)
        blend the 4 distances into one intensity stack
    composite the tiles into one mosaic-wide intensity stack
    multi-distance Paganin  ->  phase
    redistribute angle-slabs -> z-slabs
    FBP with  sinogram = phase - 1j*phase/paganin

and writes the initial object step6.py starts from:

    {path_out}/{pfile}_obj.h5   /exchange/obj_init_re{paganin}_{bin}
                                /exchange/obj_init_im{paganin}_{bin}
                                (nzobj>>bin, nobj>>bin, nobj>>bin) float32

    {path_out}/{pfile}_srdata.h5  /exchange/srdata_bin{bin}  first few mosaics
    {path_out}/{pfile}_proj.h5    /exchange/proj_bin{bin}    Paganin phases

Differences from the real YY037A steps15, all because the data is synthetic:

  * shifts are read straight from shift_dir (tile_offsets.txt + <tile>.txt) and
    treated as cshifts_final -- there is no shift estimation and no check
  * no shrinkage, so the effective demagnification is 1/norm_magnification for
    every angle and every tile, and the Paganin distances are angle-independent
  * no tile-overlap estimation, no seam shift combination, no magnification
    correction, no per-tile grey-scale matching -- every tile sees the same
    probe and the same normalisation
  * no Gaussian pre-smoothing of data/reference: the reference is exact
  * one bin level only, the one gen_data.py produced (bin= in the config)
  * the imaginary channel is -phase/paganin, not +phase/paganin: gen_data.py
    builds obj = -delta + 1j*delta/delta_beta, so the sign is known here

ntheta_rec= must equal ntheta= in config_step6.conf.  Tomo.fbp carries a
sqrt(nobj/ntheta) factor and Rec.norm_const carries the same one, so the two
cancel only when both scripts use the same angle count; otherwise the initial
object comes out scaled by sqrt(ntheta_step6/ntheta_rec).
"""

import os
import sys
import time
import numpy as np
import cupy as cp
import h5py
from mpi4py import MPI

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, _HERE)
from holotomocupy.config import parse_args_steps15         # noqa: E402
from holotomocupy.shift import Shift                       # noqa: E402
from holotomocupy.tomo import Tomo                         # noqa: E402
from holotomocupy.chunking import Chunking                 # noqa: E402
from holotomocupy.mpi_functions import MPIClass            # noqa: E402
from holotomocupy.logger_config import logger, set_log_level   # noqa: E402
from mosaic_geometry import read_tile_offsets, read_tile_shifts   # noqa: E402

cp.cuda.set_pinned_memory_allocator(None)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
args = parse_args_steps15(sys.argv[1])
set_log_level(args.log_level)


def info(msg):
    if rank == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Geometry: tiles, shifts, distances
# ---------------------------------------------------------------------------
if not args.tiles:
    raise SystemExit(f'{sys.argv[1]}: tiles= is required')
if not args.shift_dir:
    raise SystemExit(f'{sys.argv[1]}: shift_dir= is required')

tiles      = args.tiles
ntiles     = len(tiles)
path_out   = args.path_out or args.path
tile_paths = [f'{args.path}/{args.pfile}_{t}.h5' for t in tiles]
for p in tile_paths:
    if not os.path.exists(p):
        raise SystemExit(f'{p} not found — run gen_data.py first')

# tile placement, finest-grid object px, indexed by the config's tile order
_names, _off = read_tile_offsets(args.tile_file)
_by_name = dict(zip(_names, _off))
missing = [t for t in tiles if t not in _by_name]
if missing:
    raise SystemExit(f'{args.tile_file}: no offset for tile(s) {missing}')
tile_off = np.array([_by_name[t] for t in tiles], dtype='float32')

with h5py.File(tile_paths[0], 'r') as fid:
    energy           = float(fid['/exchange/energy'][0])
    fdd              = float(fid['/exchange/focusdetectordistance'][0])
    z1               = fid['/exchange/z1'][:].astype('float64')
    detector_pixelsize = float(fid['/exchange/detector_pixelsize'][0])
    theta_deg        = fid['/exchange/theta'][:, 0].astype('float64')
    ndist_t          = len(z1)
    n_bin            = fid[f'/exchange/pdata0_{args.bin}'].shape[-1]
    ntheta0          = fid[f'/exchange/pdata0_{args.bin}'].shape[0]

wavelength          = 1.24e-09 / energy
z2                  = fdd - z1
magnifications      = fdd / z1
norm_magnifications = (magnifications / magnifications[0]).astype('float32')
distances           = ((z1 * z2) / fdd * norm_magnifications**2).astype('float32')
voxelsize           = abs(detector_pixelsize / magnifications[0])

# per-angle sample shift, finest-grid object px: shifts[t, j, k, (v,h)]
shifts = np.empty([ntiles, ntheta0, ndist_t, 2], dtype='float32')
for t, name in enumerate(tiles):
    shifts[t] = read_tile_shifts(args.shift_dir, name, ntheta0, ndist_t)

# --- grids -----------------------------------------------------------------
# nobj / nzobj / nobj_tile in the config are BINNED px, as in config_step6.conf.
# The txt shifts and tile offsets stay on the finest grid, the cshifts_final
# convention, so they are the only thing multiplied by scale.
b             = args.bin
scale         = 1.0 / 2**b
nobj_bin      = args.nobj
nzobj_bin     = args.nzobj
voxelsize_bin = voxelsize * 2**b

# The tile object grid must hold the widest distance's footprint plus the
# per-angle shift excursion.  nobj_tile=0 in the config -> compute it here.
if args.nobj_tile:
    nobj_tile_bin = args.nobj_tile
else:
    reach = (n_bin / norm_magnifications.min()
             + 2 * float(np.abs(shifts).max()) * scale + 8)
    nobj_tile_bin = int(np.ceil(reach / 64)) * 64
if nobj_tile_bin > min(nzobj_bin, nobj_bin):
    raise SystemExit(f'nobj_tile {nobj_tile_bin} does not fit in the '
                     f'{nzobj_bin}x{nobj_bin} mosaic grid')

# --- angle subset ----------------------------------------------------------
ntheta5 = args.ntheta_rec if 0 < args.ntheta_rec < ntheta0 else ntheta0
# same thinning rule as Reader.__init__, so step6 sees the same angles
ids     = np.arange(0, ntheta0, ntheta0 / ntheta5)[:ntheta5].astype('int')
theta5  = np.ascontiguousarray(-theta_deg[ids] / 180 * np.pi).astype('float32')
proj_step = max(1, ntheta5 // 32)   # keep the _proj.h5 diagnostic ~32 planes

# The angle slab must match MPIClass's own split, or redist misaligns.
cl_mpi5   = MPIClass(comm, nzobj_bin, ntheta5, nobj_bin, 'float32')
st5, end5 = cl_mpi5.st_theta, cl_mpi5.end_theta
nloc      = end5 - st5
local_ids = ids[st5:end5]

# --- paste geometry --------------------------------------------------------
# The constant tile offset becomes an integer paste origin; only its sub-pixel
# remainder, plus the per-angle shift, goes into the interpolation.
off_bin  = tile_off * scale
ioff     = np.round(off_bin).astype(int)
frac     = (off_bin - ioff).astype('float32')
origin_y = (nzobj_bin - nobj_tile_bin) // 2 - ioff[:, 0]
origin_x = (nobj_bin  - nobj_tile_bin) // 2 - ioff[:, 1]

r_np  = shifts[:, local_ids] * np.float32(scale) + frac[:, None, None, :]
r_np[..., 1] += np.float32(args.rotation_center_shift * scale)
r_gpu = [cp.asarray(r_np[t]) for t in range(ntiles)]

# tile row / column, from the "{row}_{col}" names, for the seam feather
rowcol = []
for t, name in enumerate(tiles):
    rr, _, cc = name.partition('_')
    rowcol.append((int(rr), int(cc)) if cc else (0, t))
nrow = max(r for r, _ in rowcol) + 1
ncol = max(c for _, c in rowcol) + 1

def _gap(origin, ncount):
    if ncount < 2:
        return nobj_tile_bin
    u = np.unique(origin)
    return int(np.abs(np.diff(u)).min()) if len(u) > 1 else nobj_tile_bin

gap_y = _gap(origin_y, nrow)
gap_x = _gap(origin_x, ncol)

info(f'energy {energy} keV, voxelsize {voxelsize*1e9:.3f} nm '
     f'({voxelsize_bin*1e9:.3f} nm at bin {b})')
info(f'norm_magnifications {np.array2string(norm_magnifications, precision=4)}')
info(f'{ntiles} tiles ({nrow}x{ncol}), {ndist_t} distances, '
     f'frames {n_bin}x{n_bin}')
info(f'tile grid {nobj_tile_bin} px, mosaic {nzobj_bin}x{nobj_bin} px, '
     f'tile gap {gap_y}x{gap_x} px')
info(f'{ntheta5}/{ntheta0} angles, {nloc} on rank 0, paganin={args.paganin}')

# ---------------------------------------------------------------------------
# GPU operators and buffers for the stitch + Paganin phase
# ---------------------------------------------------------------------------
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

cl_shift = Shift(n_bin, nobj_tile_bin, n_bin, nobj_tile_bin)

npad_bin = max(4, n_bin // 16)     # blend width between distances, in _stitch
_quintic_cache = {}


def _quintic(m):
    """Quintic smoothstep 0 -> 1 over m samples, cached per width.

    _tile_window asks for ramps up to nobj_tile_bin//8, far wider than the
    npad_bin _stitch uses, so the curve is generated at the requested width
    rather than sliced out of one fixed-length table -- a truncated quintic
    stops short of 1 and puts a step at the end of the ramp.
    """
    v = _quintic_cache.get(m)
    if v is None:
        t = cp.linspace(0, 1, m, endpoint=False, dtype='float32')
        v = t**5 * (126 - 420*t + 540*t**2 - 315*t**3 + 70*t**4)
        _quintic_cache[m] = v
    return v

cref = []
for p in tile_paths:
    with h5py.File(p, 'r') as fid:
        cref.append(cp.asarray(fid[f'/exchange/pref_{b}'][:ndist_t]))

srdata = cp.empty([ndist_t, nobj_tile_bin, nobj_tile_bin], dtype='float32')
mosaic = cp.empty([ndist_t, nzobj_bin, nobj_bin], dtype='float32')
wsum   = cp.empty([nzobj_bin, nobj_bin], dtype='float32')


def _clip(v):
    return min(nobj_tile_bin, max(0, int(v))) + 5


def _overlap_slices(o, w_src, w_dst):
    """src/dst slice pair for pasting a length-w_src array at offset o."""
    s0 = max(0, -o)
    s1 = min(w_src, w_dst - o)
    return slice(s0, s1), slice(o + s0, o + s1)


def _ramp(w, lo, hi, m, up):
    """Quintic ramp of width m inside w[lo:hi], rising at lo or falling at hi."""
    lo, hi = max(0, lo), min(w.shape[0], hi)
    m = min(m, hi - lo)          # a footprint narrower than the ramp gets a short one
    if m <= 0:
        return
    v = _quintic(m)
    if up:
        w[lo:lo + m] = v
    else:
        w[hi - m:hi] = 1 - v


def _stitch(fids, jl, t):
    """Flat-divide, demagnify+shift each distance, blend them on the tile grid.

    Fills srdata[0..ndist_t-1] and returns the footprint (pady0, pady1, padx0,
    padx1) of the widest distance -- the region the tile actually covers.
    """
    jg     = local_ids[jl]
    data_j = cp.asarray(np.stack([fids[t][f'/exchange/pdata{k}_{b}'][jg]
                                  for k in range(ndist_t)]))
    rdata  = data_j / (cref[t] + 1e-5)

    box = None
    for k in range(ndist_t - 1, -1, -1):           # widest footprint first
        eff_mag = norm_magnifications[k]           # no shrinkage
        # (1, 2): curlySback reads mag[2*tz + axis] -- y and x, equal here.
        mag = cp.array([[1.0 / eff_mag, 1.0 / eff_mag]], dtype='float32')
        tmp = cl_shift.curlySback(
            cp.log(rdata[k][None].astype('complex64')), r_gpu[t][jl:jl + 1, k], mag
        )[0].real
        tmp = cp.exp(tmp)

        half  = int((nobj_tile_bin - n_bin / eff_mag) / 2)
        rv, rh = r_np[t][jl, k]
        pady0, pady1 = _clip(half - int(rv)), _clip(half + int(rv))
        padx0, padx1 = _clip(half - int(rh)), _clip(half + int(rh))
        if box is None:
            box = (pady0, pady1, padx0, padx1)

        # extend the frame over the rest of the tile grid so Paganin sees data
        tmp = cp.pad(tmp[pady0:-pady1], ((pady0, pady1), (0, 0)), 'edge')
        tmp = cp.pad(tmp[:, padx0:-padx1], ((0, 0), (padx0, padx1)),
                     'linear_ramp', end_values=((1, 1), (1, 1)))

        if k < ndist_t - 1:
            prev  = srdata[k + 1]
            denom = tmp[pady0:-pady1, padx0:-padx1].mean() + 1e-10
            tmp  *= float(prev[pady0:-pady1, padx0:-padx1].mean() / denom)
            wy = cp.ones(nobj_tile_bin, dtype='float32')
            wx = cp.ones(nobj_tile_bin, dtype='float32')
            wy[:pady0] = 0; wy[nobj_tile_bin - pady1:] = 0
            wx[:padx0] = 0; wx[nobj_tile_bin - padx1:] = 0
            _ramp(wy, pady0, nobj_tile_bin - pady1, npad_bin, True)
            _ramp(wy, pady0, nobj_tile_bin - pady1, npad_bin, False)
            _ramp(wx, padx0, nobj_tile_bin - padx1, npad_bin, True)
            _ramp(wx, padx0, nobj_tile_bin - padx1, npad_bin, False)
            w   = cp.outer(wy, wx)
            tmp = tmp * w + prev * (1 - w)
        srdata[k] = tmp
    return box


def _tile_window(box, t):
    """1 inside the tile footprint, quintic ramp on every side facing a neighbour."""
    pady0, pady1, padx0, padx1 = box
    row, col = rowcol[t]
    y0, y1 = pady0, nobj_tile_bin - pady1
    x0, x1 = padx0, nobj_tile_bin - padx1

    ovy = (y1 - y0) - gap_y
    ovx = (x1 - x0) - gap_x
    tfy = max(8, min(ovy // 2, nobj_tile_bin // 8)) if ovy > 16 else 0
    tfx = max(8, min(ovx // 2, nobj_tile_bin // 8)) if ovx > 16 else 0

    wy = cp.zeros(nobj_tile_bin, dtype='float32'); wy[y0:y1] = 1
    wx = cp.zeros(nobj_tile_bin, dtype='float32'); wx[x0:x1] = 1
    if row > 0:        _ramp(wy, y0, y1, tfy, True)
    if row < nrow - 1: _ramp(wy, y0, y1, tfy, False)
    if col > 0:        _ramp(wx, x0, x1, tfx, True)
    if col < ncol - 1: _ramp(wx, x0, x1, tfx, False)
    return cp.outer(wy, wx)


def _mosaic(fids, jl):
    """Composite every tile of local angle jl onto the mosaic-wide grid."""
    mosaic.fill(0)
    wsum.fill(0)
    for t in range(ntiles):
        box = _stitch(fids, jl, t)
        wt  = _tile_window(box, t)
        sy, dy = _overlap_slices(int(origin_y[t]), nobj_tile_bin, nzobj_bin)
        sx, dx = _overlap_slices(int(origin_x[t]), nobj_tile_bin, nobj_bin)
        wts = wt[sy, sx]
        mosaic[:, dy, dx] += srdata[:, sy, sx] * wts
        wsum[dy, dx]      += wts
    mosaic[:] /= cp.maximum(wsum, 1e-2)   # [:] -- a bare `mosaic /=` would make it local
    mosaic[:, wsum < 1e-2] = 1.0        # uncovered = unit transmission


def multiPaganin(data, distances, wavelength, voxelsize, delta_beta, alpha):
    """Multi-distance Paganin phase retrieval on GPU. data: [ndist, ny, nx]."""
    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    fy = cp.fft.fftfreq(data.shape[-2], d=voxelsize).astype('float32')
    fx, fy = cp.meshgrid(fx, fy)
    numerator = 0
    denominator = 0
    for j in range(data.shape[0]):
        rad_freq     = cp.fft.fft2(data[j].astype('complex64'))
        taylorExp    = 1 + wavelength * distances[j] * cp.pi * delta_beta * (fx**2 + fy**2)
        numerator   += taylorExp * rad_freq
        denominator += taylorExp**2
    numerator   /= len(distances)
    denominator  = denominator / len(distances) + alpha
    phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
    phase *= delta_beta * 0.5
    return phase


# object-plane propagation distances; no shrinkage, so the same for every angle
pag_distances = distances / norm_magnifications**2
pad8 = min(nzobj_bin, nobj_bin) // 8
PAG_ALPHA = 1e-3        # Paganin regularisation, as in the experimental steps15


def _paganin():
    pj = cp.pad(mosaic, ((0, 0), (pad8, pad8), (pad8, pad8)), 'reflect')
    ph = multiPaganin(pj, pag_distances, wavelength, voxelsize_bin,
                      args.paganin, PAG_ALPHA)
    return ph[pad8:pad8 + nzobj_bin, pad8:pad8 + nobj_bin]


# ---------------------------------------------------------------------------
# Phase A: stitch + Paganin, one angle slab per rank
# ---------------------------------------------------------------------------
os.makedirs(path_out, exist_ok=True)
stem = f'{path_out}/{args.pfile}'
if rank == 0:
    for suf in ('_srdata.h5', '_proj.h5', '_obj.h5'):
        if os.path.exists(stem + suf):
            os.remove(stem + suf)
comm.Barrier()

fids = [h5py.File(p, 'r') for p in tile_paths]

# air background, calibrated on the first angle by rank 0
global_bg = np.float32(0)
if rank == 0 and nloc > 0:
    _mosaic(fids, 0)
    ph0  = _paganin()
    rows = max(8, 16 * n_bin // 512)
    cov  = cp.asnumpy(wsum[:rows] > 1e-2)
    p0   = cp.asnumpy(ph0[:rows])
    global_bg = np.float32(np.median(p0[cov]) if cov.any() else 0.0)
    logger.info(f'air background {float(global_bg):+.6g} rad')
global_bg = comm.bcast(global_bg, root=0)

nsr = min(8, ntheta5)
local_recPag = np.empty([nloc, nzobj_bin, nobj_bin], dtype='float32')
proj_ids     = list(range(0, ntheta5, proj_step))

with h5py.File(stem + '_srdata.h5', 'a', driver='mpio', comm=comm) as fsr, \
     h5py.File(stem + '_proj.h5',   'a', driver='mpio', comm=comm) as fpr:
    sr_ds = fsr.create_dataset(f'/exchange/srdata_bin{b}',
                               shape=(nsr, nzobj_bin, nobj_bin), dtype='float32')
    pr_ds = fpr.create_dataset(f'/exchange/proj_bin{b}',
                               shape=(len(proj_ids), nzobj_bin, nobj_bin),
                               dtype='float32')
    t0 = time.time()
    for jl in range(nloc):
        _mosaic(fids, jl)
        jg = st5 + jl
        if jg < nsr:
            sr_ds[jg] = cp.asnumpy(mosaic[0])
        ph = _paganin() - global_bg
        local_recPag[jl] = cp.asnumpy(ph)
        if jg % proj_step == 0:
            pr_ds[jg // proj_step] = local_recPag[jl]
        if rank == 0 and (jl % 20 == 0 or jl == nloc - 1):
            el = time.time() - t0
            logger.info(f'step5 bin={b}: {jl+1}/{nloc} angles, {el:.0f}s '
                        f'(eta {el/(jl+1)*(nloc-jl-1):.0f}s)')

for f in fids:
    f.close()
del srdata, mosaic, wsum, cref, r_gpu, cl_shift
cp.get_default_memory_pool().free_all_blocks()
comm.Barrier()
info(f'stitch + Paganin done in {time.time()-t0:.0f}s')

# ---------------------------------------------------------------------------
# Phase B: angle slabs -> z slabs, then FBP
# ---------------------------------------------------------------------------
local_nz = cl_mpi5.local_nzobj
z_start  = cl_mpi5.st_obj

psi_z = np.empty([ntheta5, local_nz, nobj_bin], dtype='float32')
cl_mpi5.redist(local_recPag, psi_z, direction='backward')
del local_recPag

psi_z_c = np.empty([ntheta5, local_nz, nobj_bin], dtype='complex64')
psi_z_c.real[:] = psi_z
# Single material: gen_data.py builds obj = -delta + 1j*delta/delta_beta, so the
# imaginary part is MINUS the real part over paganin.  (The experimental
# steps15.py scripts use +psi_z/paganin, which flips the sign of the 1%-sized
# imaginary channel; here the ground truth is known, so it is done right.)
psi_z_c.imag[:] = -psi_z / args.paganin
del psi_z

rec_loc = np.zeros([local_nz, nobj_bin, nobj_bin], dtype='complex64')
cl_tomo = Tomo(nobj_bin, args.nchunk, theta5, mask_r=args.mask)
nbytes  = 2 * (ntheta5 * args.nchunk * nobj_bin + args.nchunk * nobj_bin**2) \
    * np.dtype('complex64').itemsize
cl = Chunking(nbytes, args.nchunk)


@cl.gpu_batch(axis_out=0, axis_inp=1, nout=1)
def _fbp(_, rec_loc, psi_z_c):
    rec_loc[:] = cl_tomo.fbp(psi_z_c, 'ramp')


t1 = time.time()
_fbp(cl, rec_loc, psi_z_c)
del psi_z_c
comm.Barrier()
info(f'FBP done in {time.time()-t1:.0f}s')

# ---------------------------------------------------------------------------
# Write the initial object step6.py reads
# ---------------------------------------------------------------------------
pag_tag = int(args.paganin) if args.paganin == int(args.paganin) else args.paganin
wbatch  = max(1, (1 << 28) // (nobj_bin * nobj_bin * 4))
with h5py.File(stem + '_obj.h5', 'a', driver='mpio', comm=comm) as fid:
    re_ds = fid.create_dataset(f'/exchange/obj_init_re{pag_tag}_{b}',
                               shape=(nzobj_bin, nobj_bin, nobj_bin), dtype='float32')
    im_ds = fid.create_dataset(f'/exchange/obj_init_im{pag_tag}_{b}',
                               shape=(nzobj_bin, nobj_bin, nobj_bin), dtype='float32')
    for i0 in range(0, local_nz, wbatch):
        i1 = min(i0 + wbatch, local_nz)
        re_ds[z_start + i0:z_start + i1] = rec_loc[i0:i1].real
        im_ds[z_start + i0:z_start + i1] = rec_loc[i0:i1].imag
comm.Barrier()
info(f'wrote {stem}_obj.h5  /exchange/obj_init_re{pag_tag}_{b} and _im{pag_tag}_{b}  '
     f'({nzobj_bin}, {nobj_bin}, {nobj_bin})')
