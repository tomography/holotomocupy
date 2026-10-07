"""
Step 6 — Full iterative holographic tomographic reconstruction (MPI).

Launch with mpirun / mpiexec, one rank per GPU:

    mpirun -np <ngpus> ./bind.sh python step6_rec_iterative_mpi.py conifgs/config1.conf

The solver reads the Paganin initial guess written by Step 5 and iteratively
refines the 3-D complex refractive-index distribution (object), the illumination
probe, and the sample positions using the Bilinear-Hessian method.

Checkpoints are written periodically to `path_out`; if a checkpoint exists the
run resumes automatically from the latest saved iteration.
"""

import sys
import h5py
from mpi4py import MPI
from holotomocupy.rec_mpi import Rec
from holotomocupy.config import parse_args
from holotomocupy.mpi_functions import MPIClass
from holotomocupy.reader import Reader, find_latest_checkpoint
from holotomocupy.writer import Writer
from holotomocupy.logger_config import logger, set_log_level

import numpy as np
import cupy as cp
cp.cuda.set_pinned_memory_allocator(None)

# --- Parse configuration file -------------------------------------------
args = parse_args(sys.argv[1])
comm = MPI.COMM_WORLD
args.comm = comm
set_log_level(args.log_level)

# --- Distribute object and projection slices across MPI ranks -----------
cl_mpi = MPIClass(comm, args.nzobj, args.ntheta, args.nobj, 'complex64')

# --- Build I/O helpers --------------------------------------------------
reader = Reader(
    args.in_file, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist, args.nz, args.n,
    args.paganin, args.rotation_center_shift, args.start_theta, args.bin,
    correct3d_extra=args.correct3d_extra,
    correct3d_extra_file=args.correct3d_extra_file,
    correct3d_bin=args.correct3d_bin,
)
writer = Writer(
    args.path_out, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist, args.nz, args.n,
)

# Physics parameters are stored in the HDF5 file and forwarded to the solver
args.energy                  = args.energy if args.energy is not None else reader.energy
args.focustodetectordistance = reader.focustodetectordistance
args.z1                      = reader.z1
args.detector_pixelsize      = reader.detector_pixelsize
args.theta                   = reader.theta

# --- Print run summary (rank 0 only) ------------------------------------
if comm.Get_rank() == 0:
    mag  = args.focustodetectordistance / args.z1[0]
    voxel_nm = args.detector_pixelsize / mag * 1e9
    logger.info("=" * 60)
    logger.info(f"  energy               : {args.energy:.4f} keV")
    logger.info(f"  detector pixel size  : {args.detector_pixelsize*1e9:.3f} nm  (bin={args.bin})")
    logger.info(f"  voxel size           : {voxel_nm:.3f} nm")
    logger.info(f"  focus-det distance   : {args.focustodetectordistance*100:.3f} cm")
    logger.info(f"  z1 distances         : {[f'{v*100:.3f} cm' for v in args.z1]}")
    logger.info(f"  detector size        : {args.nz} x {args.n}")
    logger.info(f"  object size          : {args.nzobj} x {args.nobj} x {args.nobj}")
    logger.info(f"  n angles             : {args.ntheta}  (start={args.start_theta})")
    logger.info(f"  n distances          : {args.ndist}")
    logger.info(f"  rotation center shift: {args.rotation_center_shift:.4f} px")
    logger.info(f"  paganin              : {args.paganin}")
    logger.info(f"  n MPI ranks          : {comm.Get_size()}")
    logger.info(f"  pfile                : {args.pfile or args.in_file}")
    logger.info(f"  path_out             : {args.path_out}")
    logger.info("=" * 60)

# --- Initialise the reconstruction class --------------------------------
logger.info("Create class")
cl = Rec(args)
logger.info(f"obj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.nobj} x {cl.nobj}")
logger.info(f"proj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.ntheta} x {cl.nobj}")
logger.info(f"projt-range [{cl.st_theta}:{cl.end_theta}), local size: {cl.end_theta-cl.st_theta} x {cl.nzobj} x {cl.nobj}")

# --- Load measurements and reference (flat-field) data -----------------
logger.info("Read data")
reader.read_data(out=cl.data)
reader.read_ref(out=cl.ref)
reader.read_shrink(out=cl.shrink_nd)
logger.debug(f"shrink[:3] (dist, theta, [y x]) = {cl.shrink_nd[:3]}")
# Fit shrink(t) = A*t + B per (distance, axis) -- this is the starting
# point of the shrinkage variable, refined further when rho[3] > 0.
cl.init_tp_from_shrink()

# --- Load initial variables (object, probe, positions) ------------------
# Resume from the latest checkpoint if one exists; otherwise use the
# Paganin reconstruction from Step 5 as the starting object.
logger.info("Read initial variables")
ckpt = find_latest_checkpoint(args.path_out, args.start_iter)
if ckpt:
    logger.info(f"Resuming from checkpoint: {ckpt}")
    reader.read_checkpoint(ckpt, out_obj=cl.vars['obj'], out_pos=cl.vars['pos'], out_prb=cl.vars['prb'],
                           out_bd=cl.vars.get('bd'), out_tp=cl.vars['tp'])
elif getattr(args, 'init_vol', None):
    logger.info(f"Reading initial object from vol file: {args.init_vol}")
    reader.read_vol_obj(args.init_vol, out=cl.vars["obj"], scale=getattr(args, "init_vol_scale", 1.0))
    reader.read_pos(out=cl.vars['pos'])
    if args.prb_file:
        logger.info(f"Loading {args.ndist} probes from: {args.prb_file}")
    reader.read_prb(prb_file=args.prb_file, out=cl.vars['prb'])
else:
    reader.read_obj(out=cl.vars['obj'])
    reader.read_pos(out=cl.vars['pos'])
    if args.prb_file:
        logger.info(f"Loading {args.ndist} probes from: {args.prb_file}")
    reader.read_prb(prb_file=args.prb_file, out=cl.vars['prb'])
if args.pos_checkpoint:
    logger.info(f"Overriding positions from: {args.pos_checkpoint}")
    logger.info(f'before {cl.vars['pos'][:1]=}')
    reader.read_pos_checkpoint(args.pos_checkpoint, out=cl.vars['pos'],
                               out_tp=cl.vars['tp'])
    logger.info(f'after {cl.vars['pos'][:1]=}')
    
# --- Run iterative reconstruction ---------------------------------------
logger.info("Run reconstruction")
vars = cl.BH(writer)
