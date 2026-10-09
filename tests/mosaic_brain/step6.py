"""
Reconstruct the synthetic mosaic dataset (MPI) — the step6 of this test.

    mpirun -np <ngpus> ./set_affinity_gpu.sh python step6.py config_step6.conf

This is the same Bilinear-Hessian driver as
experimental/Y350a_dist1234/step6.py, pointed at what gen_data.py wrote.  No
shrink solver is involved:

  * gen_data.py wrote one self-contained file per tile; the local
    mosaic_reader.MosaicReader flattens them onto one distance axis of
    ndist = ntiles * ndist_tile entries in tile-major order.  ndist= in the
    config is the count PER TILE, as in the real config_step6, and tiles=
    lists the files.
  * Each tile's place on the mosaic is baked into its /exchange/cshifts_final,
    so read_pos returns positions that already carry the tile placement and
    rec_mpi.Rec derives everything else from z1 alone.
  * /exchange/shrink is deliberately absent, so shrink_nd stays 0.

The starting object is whatever read_obj finds in {pfile}_obj.h5, i.e.
obj_init_re{paganin}_{bin} + 1j*obj_init_im{paganin}_{bin}.  steps15.py writes
that file (mosaic stitch -> multi-distance Paganin -> FBP), so paganin= and bin=
here must match config_steps15.conf.  Set init_vol= in the config to start from
a raw .vol instead.

Checkpoints go to path_out; an existing one is resumed from automatically.
"""

import os
import sys
import numpy as np
import cupy as cp
from mpi4py import MPI

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, _HERE)
from holotomocupy.rec_mpi import Rec                        # noqa: E402
from holotomocupy.config import parse_args                  # noqa: E402
from holotomocupy.mpi_functions import MPIClass             # noqa: E402
from holotomocupy.reader import find_latest_checkpoint      # noqa: E402
from holotomocupy.writer import Writer                      # noqa: E402
from holotomocupy.logger_config import logger, set_log_level    # noqa: E402
from mosaic_reader import MosaicReader                      # noqa: E402

cp.cuda.set_pinned_memory_allocator(None)

# --- Parse configuration file -------------------------------------------
args = parse_args(sys.argv[1])
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
args.comm = comm
set_log_level(args.log_level)

# ndist= in the config is the number of distances PER TILE; the mosaic axis is
# ntiles times longer, tile-major, so index t*ndist_tile + k is tile t distance
# k -- the same convention step6 uses for the real scan.
if not args.tiles:
    raise SystemExit(f'{sys.argv[1]}: tiles= is required, it names the '
                     f'{{pfile}}_{{tile}}.h5 files gen_data.py wrote')
args.ndist_tile = args.ndist
args.ndist      = args.ndist_tile * len(args.tiles)
for _p in args.tile_files:
    if not os.path.exists(_p):
        raise SystemExit(f'{_p} not found — run gen_data.py first')

# --- Distribute object and projection slices across MPI ranks -----------
cl_mpi = MPIClass(comm, args.nzobj, args.ntheta, args.nobj, 'complex64')

# --- Build I/O helpers --------------------------------------------------
reader = MosaicReader(
    args.tile_files, args.mosaic_file, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist_tile, args.nz, args.n,
    args.paganin, args.rotation_center_shift, args.start_theta, args.bin,
    tiles=args.tiles,
)
tiles, ndist_tile, tile_off = reader.tiles, reader.ndist_tile, reader.tile_offsets
writer = Writer(
    args.path_out, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist, args.nz, args.n,
)

# Physics parameters come from the generated /exchange/* archive
args.energy                  = args.energy if args.energy is not None else reader.energy
args.focustodetectordistance = reader.focustodetectordistance
args.z1                      = reader.z1
args.detector_pixelsize      = reader.detector_pixelsize
args.theta                   = reader.theta

# --- Run summary (rank 0) -----------------------------------------------
# The tile names and offsets gen_data.py stored as attributes, so a placement
# mistake shows up before the first iteration rather than as a plausible but
# meaningless result 500 iterations later.
if rank == 0:
    mag = args.focustodetectordistance / args.z1[:ndist_tile]
    logger.info("=" * 68)
    logger.info(f"  energy               : {args.energy:.4f} keV")
    logger.info(f"  detector pixel size  : {args.detector_pixelsize*1e9:.3f} nm  (bin={args.bin})")
    logger.info(f"  voxel size           : {args.detector_pixelsize/mag[0]*1e9:.3f} nm")
    logger.info(f"  focus-det distance   : {args.focustodetectordistance*100:.3f} cm")
    logger.info(f"  z1 per tile          : {[f'{v*100:.3f} cm' for v in args.z1[:ndist_tile]]}")
    logger.info(f"  detector size        : {args.nz} x {args.n}")
    logger.info(f"  object size          : {args.nzobj} x {args.nobj} x {args.nobj}")
    logger.info(f"  n angles             : {args.ntheta}  (start={args.start_theta})")
    logger.info(f"  n distances          : {args.ndist} = {len(tiles)} tiles x {ndist_tile}")
    logger.info(f"  tiles                : {tiles}")
    logger.info(f"  rotation center shift: {args.rotation_center_shift:.4f} px")
    logger.info(f"  mask                 : {args.mask}")
    logger.info(f"  rho                  : {args.rho}")
    logger.info(f"  n MPI ranks          : {comm.Get_size()}")
    logger.info(f"  tile files           : "
                f"{os.path.basename(args.tile_files[0])} ... "
                f"{os.path.basename(args.tile_files[-1])}")
    logger.info(f"  initial object       : "
                f"{args.mosaic_file.replace('.h5', '_obj.h5')}")
    logger.info(f"  path_out             : {args.path_out}")
    logger.info("=" * 68)

# --- Initialise the reconstruction class --------------------------------
logger.info("Create class")
cl = Rec(args)
cl.method = args.method
logger.info(f"obj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.nobj} x {cl.nobj}")
logger.info(f"projt-range [{cl.st_theta}:{cl.end_theta}), local size: {cl.end_theta-cl.st_theta} x {cl.nzobj} x {cl.nobj}")

# --- Load measurements and reference (flat-field) data -----------------
logger.info("Read data")
reader.read_data(out=cl.data)
reader.read_ref(out=cl.ref)

# --- Load initial variables (object, probe, positions) ------------------
logger.info("Read initial variables")
ckpt = find_latest_checkpoint(args.path_out, args.start_iter)
if ckpt:
    logger.info(f"Resuming from checkpoint: {ckpt}")
    reader.read_checkpoint(ckpt, out_obj=cl.vars['obj'], out_pos=cl.vars['pos'],
                           out_prb=cl.vars['prb'], out_bd=cl.vars.get('bd'), out_tp=cl.vars['tp'])
else:
    if getattr(args, 'init_vol', None):
        logger.info(f"Reading initial object from vol file: {args.init_vol}")
        reader.read_vol_obj(args.init_vol, out=cl.vars["obj"])
    else:
        reader.read_obj(out=cl.vars['obj'])
    reader.read_pos(out=cl.vars['pos'])
    if args.prb_file:
        logger.info(f"Loading {args.ndist} probes from: {args.prb_file}")
    reader.read_prb(prb_file=args.prb_file, out=cl.vars['prb'])
if args.pos_checkpoint:
    logger.info(f"Overriding positions from: {args.pos_checkpoint}")
    reader.read_pos_checkpoint(args.pos_checkpoint, out=cl.vars['pos'],
                               out_tp=cl.vars['tp'])

# Where each tile's window lands on the object grid, before iterating.
if rank == 0:
    pos0 = cp.asnumpy(cl.vars['pos'][:, 0])
    logger.info(f"  tile windows at angle {args.start_theta} "
                f"(x = (nobj-1)/2 - pos[1], nobj={args.nobj}):")
    for t, tl in enumerate(tiles):
        o = tile_off[t] if len(tile_off) else np.zeros(2)
        p = pos0[t * ndist_tile]
        logger.info(f"    {tl:<6s} offset v={o[0]:+9.2f} h={o[1]:+10.2f} finest px"
                    f"  -> pos=({p[0]:+8.2f},{p[1]:+9.2f})"
                    f"  y={(args.nzobj-1)/2 - p[0]:8.2f}"
                    f"  x={(args.nobj-1)/2 - p[1]:9.2f}")

# --- Run iterative reconstruction ---------------------------------------
logger.info("Run reconstruction")
vars = cl.BH(writer)
