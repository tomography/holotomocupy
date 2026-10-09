import sys
from mpi4py import MPI
from holotomocupy.rec_mpi import Rec
from holotomocupy.config import parse_args
from holotomocupy.mpi_functions import MPIClass
from holotomocupy.reader import Reader
from holotomocupy.writer import Writer
from holotomocupy.logger_config import logger, set_log_level

import cupy as cp
cp.cuda.set_pinned_memory_allocator(None)


args = parse_args(sys.argv[1])
set_log_level(args.log_level)
comm = MPI.COMM_WORLD
args.comm = comm

cl_mpi = MPIClass(comm, args.nzobj, args.ntheta, args.nobj, 'complex64')

reader = Reader(
    args.in_file, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist, args.nz, args.n,
    args.paganin, args.rotation_center_shift, args.start_theta, args.bin,
)
writer = Writer(
    args.path_out, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist, args.nz, args.n,
)

# physics parameters read from the data file
args.energy                  = reader.energy
args.focustodetectordistance = reader.focustodetectordistance
args.z1                      = reader.z1
args.detector_pixelsize      = reader.detector_pixelsize
args.theta                   = reader.theta

logger.info("Create class")
cl = Rec(args)

logger.info(f"obj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.nobj} x {cl.nobj}")
logger.info(f"proj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.ntheta} x {cl.nobj}")
logger.info(f"projt-range [{cl.st_theta}:{cl.end_theta}), local size: {cl.end_theta-cl.st_theta} x {cl.nzobj} x {cl.nobj}")


logger.info(f'Read data with unbinning')
reader.read_prb_unbin(out=cl.vars['prb'])
reader.read_pos(out=cl.vars['pos'])
reader.read_obj_unbin(out=cl.vars['obj'])

logger.debug(f"{cl.vars['pos'].shape=}")
logger.debug(f"{cl.vars['prb'].shape=}")
logger.debug(f"{cl.vars['obj'].shape=}")


comm.Barrier()
logger.info(f'Generate data')

cl.gen_data(cl.vars, cl.data)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)


comm.Barrier()
logger.info(f'Set initial guess for obj, prb, pos')
reader.read_pos_error_unbin(out=cl.vars['pos'])
cl.vars['prb'][:] = 1
cl.vars['obj'][:] = 0


comm.Barrier()
logger.info(f'Run reconstruction')
vars = cl.BH(writer)
