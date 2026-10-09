import numpy as np
import cupy as cp
import os
import time
import warnings
import pandas as pd
import nvtx
import cupy.fft
from mpi4py import MPI

from .tomo import Tomo
from .propagation import Propagation
from .shift import Shift
from .shift_fft import ShiftFFT
from .chunking import Chunking, _axpby
from .extra_terms import LaplacianTerm, PrbfitTerm
from .utils import make_pinned, mshow_approx, redot, reprod, timer
from .psf import psf_taps as _psf_taps, psf_blur as _psf_blur
from .mpi_functions import MPIClass
from .logger_config import logger
from .conv2d_cufftdx import precompile as cufftdx_precompile

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=".*peer.*")
cupy.fft.config.get_plan_cache().set_size(0) # dont waste GPU memory

class Rec:
    # B(y,z) = <y, H(vars)·z> is symmetric, so the three sweeps the classic path
    # runs per iteration all follow from {B(g,g), B(g,e), B(e,e)} of ONE sweep:
    #     beta   = B(g,e) / B(e,e)
    #     bottom = b^2*B(e,e) - 2*b*B(g,e) + B(g,g)
    # Every term expands the same way (hessian_cascade3, PrbfitTerm.hessian3,
    # LaplacianTerm.hessian3), cutting the PCIe-bound streaming to a third.
    # False falls back to the classic path; subclasses that override
    # hessian_cascade or compute_alpha must opt out.
    fused_hessian = True

    # Distance loop INSIDE the theta-chunk kernel: one upload of a proj chunk
    # serves `ndistchunk` distances, instead of re-streaming the whole pinned
    # proj slab once per distance. Only detector-plane arrays (data, prb) are
    # per-distance, and those are far smaller than the object plane.
    # See _resolve_ndistchunk; ndistchunk = 1 restores the old outer loop.
    # Subclasses whose cascade kernels are genuinely per-distance set False.
    hoist_dist_loop = True

    # How often to log internal instrumentation (cache hit/miss counters).
    # -1 = never. Overridden by args.debug_step when the config supplies one;
    # declared here so callers that build args by hand need not set it.
    debug_step = -1

    # Quadratic-model diagnostic, off unless a config asks for it. Class-level
    # for the same reason as debug_step: a hand-built args (tests, notebooks)
    # must not be required to carry it.
    check_approx = False

    # Laplacian regularization weight. 0 = off. Same rationale as debug_step:
    # _pool_bytes_for reads it before __init__ has finished copying args, so a
    # hand-built args (tests, notebooks) must not be required to carry it.
    lam_laplacian = 0.0

    def __init__(self, args):

        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)

        if len(args.rho) != 4:
            raise ValueError(
                f"rho must have 4 entries [obj, prb, pos, tp]; got {list(args.rho)}. "
                f"Use 0 for tp to freeze shrinkage at its initial fit.")

        # list of functionals, gradients, differentials, and second-order differentials
        self.F = [self.F0, self.F1, self.F2, self.F3, self.F4]
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF3, self.gF4]
        self.dF = [self.dF0, self.dF1, self.dF2, self.dF3, self.dF4]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2, self.d2F_dF3, self.d2F_dF4]

        # Detector oversampling of the Radon transform.  `nobj` always means the
        # OBJECT width; `ndobj` is the projection/psi width, i.e. what R writes
        # onto and what the whole cascade below fwd_tomo sees.  At the default
        # tomo_upsample = 1 they are equal and nothing changes.  At 2 the object
        # sits on a 2x coarser x/y grid than the projection plane, whose FOV,
        # sampling and centre convention are untouched -- so pos,
        # rotation_center_shift, eff_demag, the probe and Propagation all keep
        # their meaning.  Object z stays at nzobj: the run is anisotropic by
        # design (z voxel v, x/y voxel 2v) and is binned in z afterwards.
        self.tomo_upsample = int(getattr(args, 'tomo_upsample', 1))
        if self.tomo_upsample not in (1, 2):
            raise ValueError(
                f"tomo_upsample must be 1 or 2, got {self.tomo_upsample}")
        self.ndobj = self.nobj * self.tomo_upsample

        self.ndistchunk = self._resolve_ndistchunk()
        nbytes = self._chunking_pool_bytes()

        ### multinode processing
        self.cl_mpi = MPIClass(args.comm, self.nzobj, self.ntheta, self.ndobj, 'complex64')
        self.local_nzobj = self.cl_mpi.local_nzobj
        self.local_ntheta = self.cl_mpi.local_ntheta
        self.rank      = self.cl_mpi.rank
        self.st_obj    = self.cl_mpi.st_obj
        self.end_obj   = self.cl_mpi.end_obj
        self.st_theta  = self.cl_mpi.st_theta
        self.end_theta = self.cl_mpi.end_theta
        if self.rank == 0:
            # Report the ACTUAL cost of the hoist, not the sizing budget: the
            # chunking pool often does not grow at all (it is capped by the
            # obj-shape linear_batch candidate, not by the cascade one), so the
            # only real increase is the per-group prb bundles.
            nd = self.ndistchunk
            pool_1  = self._pool_bytes_for(1)
            pool_nd = self._pool_bytes_for(nd)
            prb_1   = 3 * self.nz * self.n * 8
            prb_nd  = 3 * nd * self.nz * self.n * 8
            logger.info(
                f"ndistchunk={nd}/{self.ndist}: "
                f"{len(self._dist_groups())} proj upload(s) per cascade sweep "
                f"(was {self.ndist}); GPU "
                f"{(pool_nd - pool_1 + prb_nd - prb_1)/2**20:+.0f} MiB "
                f"(pool {pool_1/2**30:.2f}->{pool_nd/2**30:.2f} GiB, "
                f"prb staging {prb_1/2**20:.0f}->{prb_nd/2**20:.0f} MiB); "
                f"host RAM unchanged")
            logger.warning(
                f"tomo geometry: obj [{self.nzobj}, {self.nobj}, {self.nobj}] "
                f"-> proj [{self.ntheta}, {self.nzobj}, {self.ndobj}]  "
                f"(tomo_upsample={self.tomo_upsample}, "
                f"voxel aspect z:x = 1:{self.tomo_upsample})")
            if self.tomo_upsample > 1 and self.lam_laplacian > 0:
                logger.warning(
                    f"tomo_upsample={self.tomo_upsample} with lam_laplacian="
                    f"{self.lam_laplacian:g}: LaplacianTerm's 3-D stencil assumes "
                    f"isotropic voxels, but this run is z:x = 1:"
                    f"{self.tomo_upsample}, so it penalises z gradients "
                    f"{self.tomo_upsample**2}x too strongly relative to x/y.")

        # X-ray propagation and magnification parameters for classes
        wavelength = 1.24e-09 / self.energy
        z2 = self.focustodetectordistance - self.z1
        magnifications = self.focustodetectordistance / self.z1
        norm_magnifications = magnifications / magnifications[0]
        distance = (self.z1 * z2) / self.focustodetectordistance * norm_magnifications**2
        voxelsize = self.detector_pixelsize / magnifications[0]

        # Variable scalings. args.rho are the initial values, used as-is unless
        # args.estimate_rho asks for a coordinate search first (estimate_rho_coord).
        # rho[3] == 0 freezes tp, i.e. keeps shrinkage at its initial fit.
        self.rho    = list(args.rho)
        self.rho_sq = {'obj': self.rho[0]**2, 'prb': self.rho[1]**2,
                       'pos': self.rho[2]**2, 'tp':  self.rho[3]**2}
        if not hasattr(self, 'estimate_rho'):
            self.estimate_rho = False
        if not hasattr(self, 'rho_estimate_niter'):
            self.rho_estimate_niter = 16


        # cuFFTDx JIT compile: rank 0 builds the .so, then all ranks proceed
        if self.rank == 0:
            cufftdx_precompile(2 * self.nz, 2 * self.n)
        self.cl_mpi.comm.Barrier()

        # sizes for normalization
        self.data_size = self.ntheta * self.ndist * self.nz * self.n
        self.prb_size  = self.ndist * self.nz * self.n
        self.obj_size  = self.nzobj * self.nobj**2
        # normalization constant to address work with normal operators
        self.norm_const = np.float32(np.sqrt(self.nobj / self.ntheta))
        self.norm_magnifications = norm_magnifications

        # create classes (one GPU per MPI rank via CUDA_VISIBLE_DEVICES)
        self.cl_chunking = Chunking(nbytes, self.nchunk)
        self.cl_tomo  = Tomo(self.nobj, self.nchunk, self.theta, self.mask,
                             nd=self.ndobj)
        self.cl_prop  = Propagation(self.n, self.nz, self.nchunk, self.ndist, wavelength, voxelsize, distance)
        shift_type = getattr(args, 'shift_type', 'cubic')
        self.shift_type = shift_type
        if shift_type == 'fft':
            self.cl_shift = ShiftFFT(self.n, self.ndobj, self.nz, self.nzobj, self.nchunk)
        elif shift_type == 'cubic':
            self.cl_shift = Shift(self.n, self.ndobj, self.nz, self.nzobj, self.nchunk)
        else:
            raise ValueError(f"shift_type must be 'cubic' or 'fft', got {shift_type!r}")
        if self.lam_laplacian > 0:
            self.cl_lap_term = LaplacianTerm(self.lam_laplacian, self.obj_size,
                                             self.local_nzobj, self.nobj,
                                             self.cl_mpi, self.cl_chunking.gpu_batch,
                                             grad_pad=getattr(self, 'alloc_mode', 'full') != 'gen')
        # Detector PSF: ONE Gaussian on the detector intensity (see the F0
        # block), the same at every distance.  An absent key gives sigma = 0,
        # psf_w is then None and F0 takes its original fast path.
        # MUST come before PrbfitTerm: the probe-fit term takes the same taps.
        self.psf_sigma, self.psf_w = _psf_taps(getattr(args, 'psf_sigma', 0.0))
        if self.psf_w is not None and self.rank == 0:
            logger.info(
                f"PSF: one Gaussian on the detector intensity, "
                f"sigma={self.psf_sigma:g} binned det.px ({self.psf_w.size} "
                f"taps), the same at every distance")

        # Data-misfit model, 'intensity' (default) or 'amplitude'.  Orthogonal
        # to psf_sigma above: both models blur the model intensity with the same
        # K and differ only in the space the comparison is made in (see the F0
        # block).  An absent key gives 'intensity', so old configs are
        # bit-for-bit unaffected.  PrbfitTerm follows the same knob, so the two
        # misfits stay the same shape and lam_prbfit stays a relative weight.
        self.model = getattr(args, 'model', 'intensity')
        if self.model not in ('intensity', 'amplitude'):
            raise ValueError(f"model must be 'intensity' or 'amplitude', "
                             f"got {self.model!r}")
        if self.model != 'intensity' and self.rank == 0:
            logger.info(
                "misfit model: AMPLITUDE, 1/N sum W (sqrt(K|psi|^2) - sqrt(d))^2"
                " -- err is ~4x smaller than under the intensity model and the"
                " two are NOT comparable.  lam_laplacian should be ~4x smaller"
                " here for the same relative weight; lam_prbfit should NOT be,"
                " PrbfitTerm follows this knob and rescales with F0.")

        if self.lam_prbfit > 0:
            # Same K as F0.  The measured flat `ref` came off the same detector
            # through the same partially coherent beam as the data, so the model
            # for it is K|D.prb|^2, not |D.prb|^2 -- blurring one side of the
            # misfit and not the other would make the probe absorb the PSF.
            self.cl_prb_term = PrbfitTerm(self.lam_prbfit, self.prb_size,
                                          self.ndist, self.nz, self.n, self.cl_prop,
                                          psf_w=self.psf_w, model=self.model)

        self.alloc_arrays()
       
        # fast refs
        self.gpu_batch = self.cl_chunking.gpu_batch
        self.redot_batch = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.linear_redot_batch = self.cl_chunking.linear_redot_batch
        self.mulc_batch = self.cl_chunking.mulc_batch
        self.redist = self.cl_mpi.redist
        self.allreduce  = self.cl_mpi.allreduce
        self.allreduce2 = self.cl_mpi.allreduce2
        self.allreduce_scalars = self.cl_mpi.allreduce_scalars
        
        # save convergence results
        self.table = pd.DataFrame(columns=["iter", "err", "time", "dobj2", "dobj_rel"])

        # apply_F_from memoization (per-chunk scope; reset in every cascade kernel)
        self._apply_F_cache  = {}
        self._apply_F_hits   = 0
        self._apply_F_misses = 0
        # read-only zero slabs for the d2F_dF3 fallback branches, keyed by shape
        self._zero_buf_cache = {}

    def _dist_bytes(self):
        """Per-distance chunking-pool cost inside a hoisted cascade kernel:
        one data chunk, the three pos chunks, and one separable-mask chunk. The
        t chunk that F4 turns into a per-(theta, axis) demagnification is
        distance-independent, so it is counted once in _pool_bytes_for rather
        than per distance."""
        return (self.nchunk * self.nz * self.n * 4          # data   [nchunk,nz,n] f32
                + 3 * self.nchunk * 2 * 4                   # x/y/z pos [nchunk,2] f32
                + self.nchunk * (self.nz + self.n) * 4)     # mask [nchunk,nz+n] f32

    def _resolve_ndistchunk(self):
        """How many distances share one upload of a theta chunk of proj.

        1     = the old outer-distance loop: a cascade sweep costs ndist*3 proj
                slabs of PCIe traffic.
        ndist = one upload per sweep, and one B-spline coeff prefilter per chunk
                instead of ndist of them. This is the default.

        Taking every distance is safe because the growth is entirely
        detector-plane: the extra cost is ndist data chunks plus the prb
        bundles, against the three object-plane proj chunks that dominate and
        do not scale with it. For the mosaic geometries this is built for
        (nzobj*nobj >> nz*n) the cascade stays well under the obj-shape
        linear_batch call site that actually sizes the chunking pool.

        `ndistchunk` (config key or --ndistchunk) overrides it; 0 or absent
        means all of them. Lower it only if a tight GPU needs the detector-plane
        residency back.
        """
        if not self.hoist_dist_loop:
            return 1
        want = int(getattr(self, 'ndistchunk', 0) or 0)
        if want > 0:
            return max(1, min(want, self.ndist))
        return self.ndist

    def _dist_groups(self):
        """range(ndist) partitioned into consecutive groups of <= ndistchunk.
        Each group is one @gpu_batch pass over theta."""
        nd = self.ndistchunk
        return [(k0, min(k0 + nd, self.ndist)) for k0 in range(0, self.ndist, nd)]

    def _assert_dist_group(self, nd):
        """Chunking classifies inputs as proper/non-proper by shape[axis]; the
        per-group prb bundle is [nd, nz, n] and would be silently mistaken for
        a proper (theta-chunked) input if nd happened to equal local_ntheta."""
        assert nd != self.local_ntheta, (
            f"ndistchunk group size {nd} aliases local_ntheta={self.local_ntheta}; "
            f"pick a different ndistchunk")

    def _chunking_pool_bytes(self):
        """Chunking-pool bytes at the configured ndistchunk."""
        return self._pool_bytes_for(self.ndistchunk)

    def _pool_bytes_for(self, ndistchunk):
        """Worst-case chunking-pool footprint across all gpu_batch callers.

        Candidates (proper-input/output buffers per single call, double-buffered):
          cascade kernels (distances hoisted inside the chunk, ndistchunk of
          them per call):
            _hess_dists:   3 proj + 1 t + ndistchunk * (1 data + 3 pos)
            _grad_dists:   3 proj + 1 t + ndistchunk * (1 data + 3 pos)
          linear_batch on vars['obj']:     3 obj-shape  (out + x + y)
          gradient_laplacian (inp_pad=4):  3 obj-shape + 4 obj-slabs from padding
          laplacian hessian3 (inp_pad=4):  2 obj-shape + 8 obj-slabs (two haloed
                                           inputs, g_pad and e_pad)
                                           (both only sized in when lam_laplacian > 0)
          fwd_tomo / adj_tomo:             1 sinogram chunk + 1 obj-shape.  The
                                           sinogram side is proj_tmp
                                           [ntheta, local_nzobj, ndobj] chunked
                                           on axis 1, so one chunk carries EVERY
                                           angle -- nchunk*ntheta*ndobj, not
                                           nchunk*nobj^2.  It grows linearly in
                                           ndobj while every other candidate grows
                                           quadratically, so it dominates whenever
                                           ntheta > nobj: a coarse bin of a
                                           many-angle single-distance scan
                                           (AtomiumL1_largedisp bin 2 is ntheta
                                           1800 against nobj 512).  Omitting it
                                           overran the pool by ~100 MB and
                                           surfaced as cudaErrorInvalidValue
                                           inside Chunking.p2g.
        linear_batch on vars['proj'] is 3 proj-shape — dominated by cascade.
        gen_data (1 proj + ndistchunk data, as outputs) and min
        (1 proj + ndistchunk * small) are both under the cascade candidate.
        Any new @gpu_batch caller with a bigger footprint must be added.
        """
        obj_item   = np.dtype('complex64').itemsize
        obj_slab   = self.nobj  * self.nobj  * obj_item     # one z-slab of obj
        proj_bytes = self.nchunk * self.nzobj * self.ndobj * obj_item
        obj_bytes  = self.nchunk * obj_slab
        dist_bytes = ndistchunk * self._dist_bytes() + self.nchunk * 4  # + t [nchunk,1] f32
        tomo_bytes = self.nchunk * self.ntheta * self.ndobj * obj_item + obj_bytes
        candidates = [3 * proj_bytes + dist_bytes,   # cascade
                      3 * obj_bytes,                 # lin_obj
                      tomo_bytes]                    # fwd_tomo / adj_tomo
        # F0's temporaries (|x|^2, J, r, p and the Hessian's second nested
        # pair) are detector-shaped float32 and live inside the cascade kernel
        # body.  Each convolve1d allocates a fresh output -- a gather stencil
        # reads a neighbourhood, so it cannot run in place, and cupyx does not
        # guard against output aliasing the input; psf_blur is two of them.
        # Measured by tests/psf/bench_f0_mem.py: dF0 is the peak, 7 of them at
        # once without a PSF and 8 with, so the 7 below is one light in the
        # blurred case.  Counted unconditionally.
        # DEBUG MODE (2026-09-15): with the F0 group unfused (see the comment
        # above F0), that bench now measures 10 -- every elementwise step keeps
        # its own full-size temporary.  The 7 here is the FUSED number and is
        # deliberately left alone; restore the fusion before a production run.
        candidates[0] += 7 * self.nchunk * self.nz * self.n * 4
        if self.lam_laplacian > 0:
            candidates.append(3 * obj_bytes + 4 * obj_slab)        # gradient_laplacian
            candidates.append(2 * obj_bytes + 8 * obj_slab)        # laplacian hessian3
        return int(2.1 * max(candidates))   # ×2 double-buffering + 10% slack

    def alloc_arrays(self):
        """Allocate all pinned CPU and CuPy GPU buffers used during reconstruction.

        args.alloc_mode = 'gen' (gen_data.py) skips the gradient and
        conjugate-direction buffers entirely: generation only ever touches
        vars / data / proj_tmp.  Allocating grads/etas and clearing the dicts
        afterwards still peaks at 3 obj-slabs + 3 proj-slabs of pinned memory,
        two thirds of which is never used -- at bin 1 that peak is 4.2 TiB
        against 1.8 TiB for what generation actually needs.  'full' (default)
        allocates everything, so reconstruction is unaffected.
        """
        gen = getattr(self, 'alloc_mode', 'full') == 'gen'
        prb_shape = [self.ndist, self.nz, self.n]
        obj_shape = [self.local_nzobj, self.nobj, self.nobj]
        # vars/etas/grads['obj'] alias the padded scratch owned by cl_lap_term when
        # the Laplacian term is active; otherwise plain obj-shape buffers. All
        # three carry ghost rows so a single chunked pass can differentiate any
        # of them (LaplacianTerm.hessian3).
        if hasattr(self, 'cl_lap_term'):
            obj_buf   = self.cl_lap_term.obj_view
            etas_obj  = self.cl_lap_term.etas_view
            grads_obj = self.cl_lap_term.grads_view
        else:
            obj_buf  = make_pinned(obj_shape, dtype='complex64'); obj_buf[:]  = 0
            # etas/grads['obj'] are gradient-side buffers: not allocated in gen mode.
            etas_obj  = None if gen else make_pinned(obj_shape, dtype='complex64')
            grads_obj = None if gen else make_pinned(obj_shape, dtype='complex64')
            for b in (etas_obj, grads_obj):
                if b is not None:
                    b[:] = 0

        # prb / pos are pinned CPU (uploaded once per gpu_batch call). grads['prb']
        # stays on GPU: the chunking machinery only accepts cp.ndarray for
        # non-proper outputs.
        self.vars = {
            'obj':  obj_buf,
            'pos':  make_pinned([self.ndist, self.local_ntheta, 2],         dtype='float32'),
            'prb':  make_pinned(prb_shape,                                 dtype='complex64'),
            'proj': make_pinned([self.local_ntheta, self.nzobj, self.ndobj], dtype='complex64'),
            # Linear shrinkage model, shrink(t) = A*t + B with t = theta_idx/(ntheta-1):
            # tp[dist, 0, axis] = A, tp[dist, 1, axis] = B, axis 0 = y, 1 = x.
            # GLOBAL (identical on every rank, like prb) and tiny, so it lives on
            # the GPU and skips the chunking machinery in linear_batch entirely.
            'tp':   cp.zeros([self.ndist, 2, 2], dtype='float32'),
        }
        # measurement data; ref is owned by cl_prb_term — aliased here for back-compat
        # so external code (readers, gen_sqrt_ref out-arg) can keep using cl.ref.
        self.data = make_pinned([self.ndist, self.local_ntheta, self.nz, self.n], dtype='float32')
        if hasattr(self, 'cl_prb_term'):
            self.ref = self.cl_prb_term.ref
        else:
            self.ref = cp.empty([self.ndist, self.nz, self.n], dtype='float32')
        # gradient and conjugate-direction buffers (reconstruction only)
        self.grads, self.etas = {}, {}
        if not gen:
            for ge in self.grads, self.etas:
                ge["pos"]  = make_pinned([self.ndist, self.local_ntheta, 2], dtype='float32')
                ge["proj"] = make_pinned([self.local_ntheta, self.nzobj, self.ndobj], dtype='complex64')
            # vars/grads/etas['prb'] all pinned. gradients_cascade uses a small per-k GPU
            # staging buffer to accumulate y[0]*rho_sq across theta chunks for one dist,
            # then D2H's the slot to grads['prb'][k] after each k's @gpu_batch.
            self.grads["prb"] = make_pinned(prb_shape, dtype='complex64')
            self.etas["prb"]  = make_pinned(prb_shape, dtype='complex64')
            # tp mirrors prb: global, so its gradient is allreduced and only rank 0
            # contributes it to the line-search sums.
            self.grads["tp"]  = cp.zeros([self.ndist, 2, 2], dtype='float32')
            self.etas["tp"]   = cp.zeros([self.ndist, 2, 2], dtype='float32')
            self.etas["obj"]  = etas_obj
            self.grads["obj"] = grads_obj
            # etas must start at exactly zero, not at whatever the pinned pool
            # handed back: the first CG step is eta <- 0*eta - grad, and 0*NaN is NaN.
            for k, v in self.etas.items():
                if k != "obj":      # obj is zeroed at allocation above
                    v[:] = 0
        self.proj_tmp    = make_pinned([self.ntheta, self.local_nzobj, self.ndobj], dtype='complex64')

        # Shrinkage as read from /exchange/shrink (the fit target of
        # init_tp_from_shrink) and the demagnification it implies, per axis
        # ([...,0] = y, [...,1] = x). eff_demag is a diagnostic / data-mask input
        # only: inside the cascade F4 produces demag differentiably from vars['tp'].
        self.shrink_nd = cp.zeros((self.ndist, self.local_ntheta, 2), dtype='float32')
        self.eff_demag = cp.zeros((self.ndist, self.local_ntheta, 2), dtype='float32')

        # t = theta_idx / (ntheta - 1) for this rank's angles, shaped (local_ntheta, 1)
        # so it is a *proper* (theta-chunked) cascade input that broadcasts over the
        # two axes. F4 turns (tp, t) into the per-(theta, axis) demagnification.
        t_global = cp.arange(self.ntheta, dtype='float32') / max(self.ntheta - 1, 1)
        self.t_local = cp.ascontiguousarray(
            t_global[self.st_theta:self.end_theta].reshape(-1, 1))
        self.norm_magnifications_gpu = cp.asarray(self.norm_magnifications, dtype='float32')

        # Out-of-grid detector mask, filled by _build_data_mask in precalc.
        # Axis-separable per (dist, angle): [..., :nz] rows, [..., nz:] columns,
        # pixel weight = their outer product (a dense mask would cost ~30 GB).
        # Pinned: a proper, theta-chunked cascade input like self.data. All-ones
        # here so paths that skip precalc keep the unmasked objective.
        self.mask_1d = make_pinned([self.ndist, self.local_ntheta, self.nz + self.n],
                                   dtype='float32')
        self.mask_1d[:] = 1
        # The same information in its compact authoritative form: the half-open
        # keep box (y0, y1, x0, x1) per (dist, angle). Diagnostics only.
        self.mask_box = np.empty((self.ndist, self.local_ntheta, 4), dtype='int32')
        self.mask_box[..., 0], self.mask_box[..., 1] = 0, self.nz
        self.mask_box[..., 2], self.mask_box[..., 3] = 0, self.n
        # Neutral until a cascade kernel points them at its own chunk, so that
        # a path calling F0 outside the four batched call sites still sees an
        # unmasked objective rather than an AttributeError.
        self._mask_y = self._mask_x = cp.ones(1, dtype='float32')

    def _set_mask_chunk(self, mk):
        """Point the F0 family at one theta-chunk of self.mask_1d for one
        distance. mk is [chunk, nz + n]; the two factors broadcast against the
        [chunk, nz, n] detector arrays."""
        self._mask_y = mk[:, :self.nz, None]
        self._mask_x = mk[:, None, self.nz:]

    # ------------------------------------------------------------------
    # Shrinkage: the linear model, and its initialization from the stored
    # per-projection profile.
    # ------------------------------------------------------------------

    def init_tp_from_shrink(self, reader=None):
        """Least-squares fit vars['tp'] to the shrinkage profile in shrink_nd.

        shrink_nd is [ndist, local_ntheta, 2], theta-distributed; the fit is
        global, so rank 0 gathers it, solves [t, 1] . [A, B] = shrink per
        (distance, axis) and broadcasts the result. `reader`, when given, is
        read into shrink_nd first.

        load_shrink_from_mats already produces a profile exactly linear in
        theta, so RMS_fit is ~0 for every current dataset and rho[3]=0 then
        reproduces the fixed-shrinkage model bit for bit. A nonzero RMS_fit is
        the signal that the stored profile is not linear and that the linear
        model is throwing information away.
        """
        if reader is not None:
            reader.read_shrink(out=self.shrink_nd)

        local_np = cp.asnumpy(self.shrink_nd).astype('float32')   # [ndist, lt, 2]
        gathered = self.cl_mpi.comm.gather((self.st_theta, local_np), root=0)

        tp_init = np.zeros((self.ndist, 2, 2), dtype='float32')
        if self.rank == 0:
            all_shrink = np.zeros((self.ndist, self.ntheta, 2), dtype='float32')
            for st, arr in gathered:
                all_shrink[:, st:st + arr.shape[1]] = arr
            t = np.arange(self.ntheta, dtype='float64') / max(self.ntheta - 1, 1)
            M = np.column_stack([t, np.ones_like(t)])
            for d in range(self.ndist):
                for ax, name in enumerate(('y', 'x')):
                    y = all_shrink[d, :, ax].astype('float64')
                    (A_fit, B_fit), res, *_ = np.linalg.lstsq(M, y, rcond=None)
                    tp_init[d, 0, ax] = A_fit
                    tp_init[d, 1, ax] = B_fit
                    resid = float(res[0]) if len(res) else float(
                        np.sum((M @ np.array([A_fit, B_fit]) - y) ** 2))
                    rms = float(np.sqrt(resid / self.ntheta))
                    logger.warning(
                        f'init_tp_from_shrink: dist={d} axis={name}  '
                        f'A={A_fit:+.4e} B={B_fit:+.4e}  RMS_fit={rms:.3e}')
        self.cl_mpi.comm.Bcast(tp_init, root=0)
        self.vars['tp'][:] = cp.asarray(tp_init)
        self.tp_init = self.vars['tp'].copy()
        return self.vars['tp']

    def _shrink_from_tp(self, tp):
        """[ndist, local_ntheta, 2] shrink implied by tp on this rank's angles."""
        A = tp[:, 0, :][:, None, :]        # [ndist, 1, 2]
        B = tp[:, 1, :][:, None, :]
        return A * self.t_local[None] + B  # t_local[None] is [1, local_ntheta, 1]

    def _eff_demag_from_tp(self, tp):
        """[ndist, local_ntheta, 2] effective demagnification implied by tp."""
        return ((1.0 + self._shrink_from_tp(tp))
                / self.norm_magnifications_gpu[:, None, None])

    def _tp_to_shrink_global(self, tp):
        """[ntheta, ndist, 2] shrink over ALL angles (host, for plots/logs)."""
        tp_np = cp.asnumpy(tp) if isinstance(tp, cp.ndarray) else np.asarray(tp)
        A, B = tp_np[:, 0, :], tp_np[:, 1, :]                       # [ndist, 2]
        t = np.arange(self.ntheta, dtype='float32') / max(self.ntheta - 1, 1)
        return (A[None] * t[:, None, None] + B[None]).astype('float32')

    def _tp_to_shrink_local(self, tp):
        """This rank's slice of _tp_to_shrink_global: [local_ntheta, ndist, 2]."""
        return self._tp_to_shrink_global(tp)[self.st_theta:self.end_theta]

    def _log_shrink_stats(self, tp, i):
        """Rank-0 one-liner with the fitted A, B per distance and axis.

        shrink(t) = A*t + B with t = theta_index / (ntheta - 1) in [0, 1], so
        (A, B) IS the model: B is the shrink at the first angle, A+B at the
        last, and A alone is the drift across the scan. Reporting the two
        coefficients is both smaller and more informative than the mean/max of
        |shrink| this used to print, and it keeps the sign.
        """
        if self.rank != 0:
            return
        tp_np = cp.asnumpy(tp) if isinstance(tp, cp.ndarray) else np.asarray(tp)
        A, B  = tp_np[:, 0, :], tp_np[:, 1, :]                      # [ndist, 2]
        parts = "  ".join(
            f"d{j}: y=(A={A[j,0]:+.4e} B={B[j,0]:+.4e})"
            f"  x=(A={A[j,1]:+.4e} B={B[j,1]:+.4e})"
            for j in range(self.ndist))
        logger.warning(f"iter={i}: shrink A*t+B  {parts}")

    def _ensure_tp(self, vars):
        """Back-stop for callers that fill shrink_nd but never fit tp.

        Without it a script written against the fixed-shrinkage API would keep
        running and silently reconstruct at shrink = 0.
        """
        if not bool(cp.any(self.shrink_nd)) or bool(cp.any(vars['tp'])):
            return
        if self.rank == 0:
            logger.warning('shrink_nd is nonzero but vars[tp] is still zero; '
                           'fitting the linear model now (call '
                           'init_tp_from_shrink() explicitly to silence this).')
        self.init_tp_from_shrink()

    def BH(self, writer=None, shrink_gt=None):
        vars  = self.vars
        grads = self.grads
        etas  = self.etas

        self.shrink_gt = shrink_gt
        self.precalc(vars)
        self.error_debug(vars, -1)

        if self.estimate_rho:
            self.estimate_rho_coord(vars, grads, etas,
                                    niter_trial=self.rho_estimate_niter)

        self._iterate(vars, grads, etas, writer)
        self.postcalc(vars)
        return vars

    def _iterate(self, vars, grads, etas, writer=None):
        """Main BH iteration loop. Assumes precalc() has already run."""
        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            with nvtx.annotate(f"::BH:{i}"):
                self.compute_gradient(vars, grads)
                alpha, top, bottom = self.compute_step(vars, grads, etas, i)
                if self.check_approx:
                    self.check_approximation(vars, etas, top, bottom, alpha,
                                             i, writer)
                want = self._is_error_iter(i)
                res = self.apply_step(vars, etas, alpha, obj_norms=want)
                if want:
                    self._record_dobj(res)
                self.log_iter(vars, i, writer)

    def estimate_rho_coord(self, vars, grads, etas, niter_trial=16, max_extend=8):
        """Coordinate search on rho[prb, pos, tp] over a geometric grid
        {..., init/2, init, 2*init, ...} centred on the current self.rho_sq.

        For each variable in order (prb -> pos -> tp; tp is skipped when it is
        frozen at rho=0):
          - Run three short BH trials at rho = {init/2, init, 2*init}
              (`init` = the current sqrt(rho_sq[v])).
          - If the middle wins, keep it.
          - Else extend up (x2, x4, ...) or down (/2, /4, ...) until improvement
            stops, capped by max_extend rungs.
          - Adopt the winning value; move on with the winner baked into `base`.

        Each trial restores vars/grads/etas/table/start_iter to the snapshot
        taken here, runs `_iterate` for niter_trial iterations silently, and
        scores with self.min(). A trial that blows up (CUDA / RuntimeError, or a
        non-finite error) scores inf so the search steps past divergent rho.

        Trial errors are memoised on the full rho vector, so the centre probe of
        the second and third coordinates -- which is the vector that just won
        the previous one -- is free.

        Updates self.rho_sq in place and restores the state so the outer BH loop
        starts clean. Costs one extra copy of vars (obj + proj dominate) for the
        snapshot, and 3..(3 + 2*max_extend) trials of niter_trial iterations per
        coordinate, minus one for each coordinate after the first.
        """
        snap_vars       = {k: v.copy() for k, v in vars.items()}
        snap_table      = self.table.copy()
        snap_start_iter = self.start_iter
        snap_niter      = self.niter
        snap_error_step = self.error_step
        snap_ckpt_step  = self.checkpoint_step

        # Silence trial logging / disable checkpoint writes for the duration.
        self.niter           = niter_trial
        self.start_iter      = 0
        # Trials are silent by default; rho_trial_error_step=N logs the error
        # every N iterations inside each trial, which is the only way to see
        # whether a bad score is a slow descent or a first-step blow-up.
        self.error_step      = int(getattr(self, 'rho_trial_error_step', -1))
        self.checkpoint_step = -1

        def _reset_trial():
            for k, v in vars.items():
                v[:] = snap_vars[k]
            for buf in grads.values(): buf[:] = 0
            for buf in etas.values():  buf[:] = 0
            self.table      = pd.DataFrame(columns=["iter", "err", "time", "dobj2", "dobj_rel"])
            self.start_iter = 0

        def _run_trial(rho_vec):
            _reset_trial()
            self.rho_sq = {'obj': rho_vec[0]**2, 'prb': rho_vec[1]**2,
                           'pos': rho_vec[2]**2, 'tp': rho_vec[3]**2}
            try:
                self._iterate(vars, grads, etas, writer=None)
                err = float(self.min(vars['prb'], vars['obj'], vars['pos'],
                                     vars['proj'], vars['tp']))
                if not np.isfinite(err):
                    err = float('inf')
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f'rho trial {rho_vec} crashed '
                                   f'({type(e).__name__}: {e}) -> err=inf')
                err = float('inf')
            return err

        # Errors keyed by the full rho vector, shared across coordinates: the
        # winner of one coordinate is exactly the centre probe of the next, so
        # without this the switch prb->pos->tp re-runs a trial it already has.
        trial_cache = {}

        def _coord(base, idx, name, init):
            seen = {}
            def probe(val):
                rv  = list(base); rv[idx] = val
                key = tuple(rv)
                if key in trial_cache:
                    e = trial_cache[key]
                    seen.setdefault(val, e)
                    if self.rank == 0:
                        logger.warning(f'  {name}={val:g}  err={e:.4e} (cached)')
                    return e
                e = _run_trial(rv)
                trial_cache[key] = e
                seen[val] = e
                if self.rank == 0:
                    logger.warning(f'  {name}={val:g}  err={e:.4e}')
                return e
            e_c  = probe(init)
            e_up = probe(init * 2)
            e_dn = probe(init / 2)
            if e_c <= e_up and e_c <= e_dn:
                best = init
            elif e_up < e_dn:
                cur_v, cur_e = init * 2, e_up
                for _ in range(max_extend):
                    nxt = cur_v * 2
                    e_nxt = probe(nxt)
                    if e_nxt >= cur_e: break
                    cur_v, cur_e = nxt, e_nxt
                best = cur_v
            else:
                cur_v, cur_e = init / 2, e_dn
                for _ in range(max_extend):
                    nxt = cur_v / 2
                    e_nxt = probe(nxt)
                    if e_nxt >= cur_e: break
                    cur_v, cur_e = nxt, e_nxt
                best = cur_v
            if self.rank == 0:
                logger.warning(f'  -> best {name}={best:g}')
            return best, sorted(seen.items())

        base = [float(np.sqrt(self.rho_sq[k])) for k in ('obj', 'prb', 'pos', 'tp')]
        if self.rank == 0:
            logger.warning(f'estimate_rho_coord: start from {base}, '
                           f'niter_trial={niter_trial}')

        # obj stays at whatever it was; prb, pos and (when free) tp get searched.
        history = {}
        base[1], history['prb'] = _coord(base, 1, 'prb', base[1])
        base[2], history['pos'] = _coord(base, 2, 'pos', base[2])
        if base[3] > 0:
            base[3], history['tp'] = _coord(base, 3, 'tp', base[3])
        else:
            history['tp'] = []      # frozen: nothing to scale

        # Restore state so the outer BH loop starts clean.
        _reset_trial()
        self.table           = snap_table
        self.start_iter      = snap_start_iter
        self.niter           = snap_niter
        self.error_step      = snap_error_step
        self.checkpoint_step = snap_ckpt_step

        self.rho_sq = {'obj': base[0]**2, 'prb': base[1]**2,
                       'pos': base[2]**2, 'tp': base[3]**2}
        self.rho    = list(base)
        if self.rank == 0:
            logger.warning(f'estimate_rho_coord: final rho = {base}')
        return history

    def _build_data_mask(self, vars):
        """Weight out the detector pixels whose object-grid footprint leaves
        the grid.

        F3 samples the object plane at

            x = eff_demag*(tx - (n-1)/2) - r_x + (ndobj-1)/2

        and interpolates with a cubic B-spline, i.e. taps floor(x)-1 ..
        floor(x)+2, so a pixel is fully supported only for 1 <= x < ndobj-2 --
        and likewise in y against nzobj. (The plane being sampled is the
        PROJECTION plane, width ndobj, not the object grid -- tomo_upsample
        coarsens the object behind R without moving this boundary.)
        eff_demag = (1+shrink)/norm_mag is > 1 for every plane but the
        reference one, so as soon as
        ndobj < n*max(eff_demag) the outer ring of the detector back-maps past
        the edge of the projection grid. What the model predicts there is decided
        by the shift kernel's boundary condition (a mirrored copy of the
        sample), not by the sample itself, so fitting those pixels only pushes
        the residual into the probe and the in-grid object. This zeroes their
        weight in F0 and in every derivative of it. Partial support counts as
        no support: a pixel with two of its four taps in-grid returns a blend
        of sample and boundary condition, which is wrong in its own way.

        The support is a rectangle *per (distance, angle)*: since eff_demag > 0
        the map tx -> x is increasing, so the kept set on each axis is one
        contiguous interval, and the mask is the outer product of the two. Both
        intervals are asymmetric about the detector centre -- they follow the
        sample as -r_x shifts it -- which is the whole point. Collapsing them to
        one centred rectangle (the worst case over all angles and all ranks, as
        this used to do) costs nothing when the sample barely moves and
        everything when it does: at 300 px of displacement against ndobj = n it
        threw away half the detector at every angle to accommodate the two
        extreme ones.

        Stored twice: mask_1d as the two 1-D factors the kernels multiply, and
        mask_box as the four integers (y0, y1, x0, x1) they came from. Never as
        a dense [ndist, local_ntheta, nz, n] array -- that is 30 GB per rank at
        bin 0.

        Frozen for the whole level: rebuilding it mid-run would change F
        between iterations, so the errors in conv.csv would no longer be
        comparable, and F0 would stop being differentiable in pos (every
        derivative below ignores the mask's own dependence on it). Both inputs
        are optimized, so both are taken at their *initial* value -- pos as
        passed in, eff_demag as implied by the initial tp -- and
        mask_oob_margin is the slack that keeps the frozen box valid as they
        drift. Measured drift over a full 256-iteration level is ~1.2 px, so
        the 2 px default has room.
        """
        margin = float(getattr(self, 'mask_oob_margin', 2.0))
        enabled = bool(getattr(self, 'mask_oob', True))
        # ndobj is the PROJECTION width (nobj*tomo_upsample).  Same rationale as
        # margin above: this method is called unbound from hand-built stubs
        # (tests, notebooks) that only carry the object width.
        ndobj = int(getattr(self, 'ndobj', self.nobj))

        if not enabled:
            self.mask_1d[:] = 1
            self.mask_box[..., 0], self.mask_box[..., 1] = 0, self.nz
            self.mask_box[..., 2], self.mask_box[..., 3] = 0, self.n
            if self.rank == 0:
                logger.warning("data mask: disabled (mask_oob=False)")
            return

        # eff_demag is [ndist, local_ntheta, 2] and pos is [ndist, local_ntheta, 2],
        # both already this rank's own angles -- there is nothing to reduce over
        # ranks any more, each box depends only on its own (dist, angle).
        ed  = cp.asnumpy(self.eff_demag).astype('float64')          # (y, x)
        pos = np.asarray(vars['pos'], dtype='float64')              # (y, x)

        ax = np.arange(self.n,  dtype='float64') - (self.n  - 1) * 0.5
        ay = np.arange(self.nz, dtype='float64') - (self.nz - 1) * 0.5

        # A tap set {floor(v)-1 .. floor(v)+2} fits in [0, N-1] iff 1 <= v < N-2.
        # Both ends carry the margin, so the box stays valid as pos drifts either way.
        def _axis(idx, ed_a, pos_a, N):
            # v[t, i] = ed[t]*idx[i] - pos[t] + (N-1)/2
            v = ed_a[:, None] * idx[None, :] - pos_a[:, None] + (N - 1) * 0.5
            return (v >= 1.0 + margin) & (v < (N - 2) - margin)

        for k in range(self.ndist):
            my = _axis(ay, ed[k, :, 0], pos[k, :, 0], self.nzobj)   # [ntheta, nz]
            mx = _axis(ax, ed[k, :, 1], pos[k, :, 1], ndobj)   # [ntheta, n]
            self.mask_1d[k, :, :self.nz] = my
            self.mask_1d[k, :, self.nz:] = mx
            for m, o in ((my, 0), (mx, 2)):
                cnt = m.sum(axis=1)
                self.mask_box[k, :, o]     = np.where(cnt > 0, m.argmax(axis=1), 0)
                self.mask_box[k, :, o + 1] = self.mask_box[k, :, o] + cnt

        # Kept fraction per (dist, angle), and the fraction the old single
        # centred rectangle would have kept, for comparison.
        if self.local_ntheta > 0:
            loc = (self.mask_1d[:, :, :self.nz].mean(axis=2)
                   * self.mask_1d[:, :, self.nz:].mean(axis=2))     # [ndist, ntheta]
            stat = np.stack([loc.sum(axis=1), loc.min(axis=1), loc.max(axis=1),
                             np.full(self.ndist, float(self.local_ntheta))])
            gl = np.concatenate([ed.max(axis=1).ravel(),
                                 np.abs(pos).max(axis=1).ravel()])
        else:
            # neutral elements, so a rank holding no angles does not drag the
            # min to 0 or the max to 1
            stat = np.stack([np.zeros(self.ndist), np.full(self.ndist, np.inf),
                             np.full(self.ndist, -np.inf), np.zeros(self.ndist)])
            gl   = np.zeros(4 * self.ndist)
        stat = np.ascontiguousarray(stat, dtype='float64')
        self.cl_mpi.comm.Allreduce(MPI.IN_PLACE, stat[0], op=MPI.SUM)
        self.cl_mpi.comm.Allreduce(MPI.IN_PLACE, stat[1], op=MPI.MIN)
        self.cl_mpi.comm.Allreduce(MPI.IN_PLACE, stat[2], op=MPI.MAX)
        self.cl_mpi.comm.Allreduce(MPI.IN_PLACE, stat[3], op=MPI.SUM)
        gl = gl.astype('float64')
        self.cl_mpi.comm.Allreduce(MPI.IN_PLACE, gl, op=MPI.MAX)

        empty = int((self.mask_box[..., 1] <= self.mask_box[..., 0]).sum()
                    + (self.mask_box[..., 3] <= self.mask_box[..., 2]).sum())
        empty = self.cl_mpi.comm.allreduce(empty, op=MPI.SUM)

        if self.rank == 0:
            mean = stat[0] / np.maximum(stat[3], 1)
            ed_max = gl[:2 * self.ndist].reshape(self.ndist, 2)
            r_max  = gl[2 * self.ndist:].reshape(self.ndist, 2)
            cx, cy = (ndobj - 1) * 0.5, (self.nzobj - 1) * 0.5
            hx = min(cx - 1.0, (ndobj - 3) - cx)
            hy = min(cy - 1.0, (self.nzobj - 3) - cy)
            for k in range(self.ndist):
                # what one centred rectangle over the global worst case would keep
                gx = (ed_max[k, 1] * np.abs(ax) <= hx - r_max[k, 1] - margin).mean()
                gy = (ed_max[k, 0] * np.abs(ay) <= hy - r_max[k, 0] - margin).mean()
                logger.warning(
                    f"data mask dist {k}: eff_demag<=(y {ed_max[k,0]:.5f}, "
                    f"x {ed_max[k,1]:.5f}) |pos|<={r_max[k].max():.2f}px "
                    f"margin={margin:g}px -> per-angle keeps "
                    f"{100 * mean[k]:.1f}% of the pixels "
                    f"(min {100 * stat[1][k]:.1f}%, max {100 * stat[2][k]:.1f}%; "
                    f"one shared centred box would keep {100 * gx * gy:.1f}%)")
            logger.warning(
                f"data mask: keeps {100 * mean.mean():.1f}% of the data. F0 still "
                f"divides by the FULL ntheta*ndist*nz*n, so reported errors scale "
                f"with that fraction and are not comparable with an unmasked run "
                f"-- or with a run made before the mask went per-angle.")
            if mean.mean() < 0.999:
                logger.warning(
                    f"data mask: ndobj={ndobj} < n*max(eff_demag)="
                    f"{self.n * ed_max.max():.0f}, or the sample moves too far "
                    f"across it; the discarded pixels are real measurements that "
                    f"only a larger object grid can use.")
            if empty:
                logger.warning(
                    f"data mask: {empty} (dist, angle, axis) boxes are EMPTY -- "
                    f"those projections contribute nothing to the data fit.")

    def precalc(self, vars):
        """One-time setup at the start of BH: shrinkage, obj normalization,
        pos snapshot, initial proj from fwd_tomo + redist."""
        self._ensure_tp(vars)
        self.eff_demag[:] = self._eff_demag_from_tp(vars['tp'])
        self._build_data_mask(vars)
        if not hasattr(self, 'tp_init'):
            self.tp_init = vars['tp'].copy()
        self._log_shrink_stats(vars['tp'], -1)

        # normalize obj to work with normal operators (restored at BH exit)
        vars["obj"] /= self.norm_const
        if self.start_iter == 0:
            vars["obj"] *= self.cl_tomo.mask

        self.pos_init = vars['pos'].copy()

        self.fwd_tomo(vars["obj"], out=self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])

    def postcalc(self, vars):
        """Restore obj normalization after the BH loop (inverse of precalc)."""
        vars["obj"] *= self.norm_const

    def compute_gradient(self, vars, grads):
        """gradients_cascade + propagate the obj-side gradient through
        fwd_tomo + redist so grads['proj'] is ready for the hessian calls."""
        with nvtx.annotate("gradients"):
            self.gradients(vars, grads)
        with nvtx.annotate(":::BH:fwd_tomo"):
            self.fwd_tomo(grads["obj"], out=self.proj_tmp)
        with nvtx.annotate(":::BH:redist", color='red'):
            self.redist(self.proj_tmp, grads['proj'])

    def compute_step(self, vars, grads, etas, i):
        """One CG step: update the search direction with the new beta and
        return (alpha, top, bottom). Fused single-sweep route when eligible
        (see the `fused_hessian` note on the class), classic three-sweep route
        otherwise."""
        if self.fused_hessian:
            return self._compute_step_fused(vars, grads, etas, i)
        beta = self.compute_beta(vars, grads, etas, i)
        return self.compute_alpha(vars, grads, etas, beta)

    def _compute_step_fused(self, vars, grads, etas, i):
        """beta and alpha from ONE cascade sweep instead of three."""
        with nvtx.annotate(":::BH:calc_step"):
            check = getattr(self, 'check_fused_hessian', False)

            # Reduce BEFORE expanding `bottom` below: the expansion folds B(g,e)
            # and B(e,g) into one 2*beta*Bge term, and for the regularization those
            # are equal only globally -- per rank they differ by a biharmonic flux
            # across the slab boundary (tests/extra_terms/test_hessian3.py).
            Qgg, Bge, Qee = self.allreduce_scalars(
                *self.hessian3(vars, grads, etas))

            # Steepest descent on the first iteration: eta_new = -g, so
            # bottom = B(-g,-g) = Qgg, which the expansion below already gives at
            # beta = 0. etas is zero there, so the sweep measured Bge = Qee = 0
            # and only the ratio needs the special case (0/0).
            #
            # beta is floored at 0 (the Powell safeguard): Bge = B(g,e) < 0 means
            # the new gradient and the old direction have negative curvature
            # coupling, so beta*e points away from the descent direction rather
            # than along it, and dropping it falls back to steepest descent for
            # that iteration -- always a valid CG restart.
            beta = 0.0 if i == self.start_iter else max(Bge / Qee, 0.0)

            if check and i > self.start_iter:
                ref_beta = self._ref_beta(vars, grads, etas)
                self._log_fused_check(i, "beta", beta, ref_beta)

            # etas <- beta*etas - grads. Must run *after* the sweep: it
            # overwrites the direction the sweep just read (e_pad views it).
            top, = self.allreduce_scalars(self._update_etas(grads, etas, beta))

            bottom = beta * beta * Qee - 2.0 * beta * Bge + Qgg
            # A Schur complement (Qgg - Bge^2/Qee): positive quantities that nearly
            # cancel when beta is large. float64, but warn on a non-positive result
            # (nonconvex) or on heavy cancellation.
            scale = abs(beta * beta * Qee) + abs(2.0 * beta * Bge) + abs(Qgg)
            nonconvex = not bottom > 1e-6 * scale

            if check:
                # etas holds the updated direction, so this is exactly the
                # sweep compute_alpha would have run.
                ref_bottom = self.allreduce_scalars(self.hessian(vars, etas, etas))[0]
                self._log_fused_check(i, "bottom", bottom, ref_bottom)

            if nonconvex:
                # The quadratic model along eta is concave (bottom < 0) or flat.
                # Its stationary point is then a MAXIMUM, so top/bottom is a
                # negative alpha and the update walks uphill -- which is exactly
                # what a tomo_upsample=2 run does from iteration 0, where the
                # coarse object grid cannot match the data's fine detail and the
                # residual-weighted term of d2F0 goes negative over enough of the
                # detector to flip B(g,g).  Under the INTENSITY misfit that term
                # is p*Re(conj(z) y) with p = K[W(K|x|^2 - d)], sign-indefinite
                # wherever the model over-predicts; the other term,
                # 2 W K[Re(conj(x) z)]^2, is a square and always >= 0.  So the
                # indefinite part is proportional to the residual and vanishes at
                # the solution -- the Hessian is PSD there.  (The amplitude misfit
                # this replaced had the same structure with (1 - d/|x|) in place
                # of p, plus a d/|x|^3 factor that blew up at small |x|.)
                #
                # top = <-grad, eta> > 0 for any descent direction, so |bottom|
                # keeps the step downhill and of the right magnitude; the model
                # itself says "go further", but unboundedly so, and there is no
                # objective evaluation here to backtrack against.
                logger.warning(
                    f"iter={i}: non-convex alpha denominator bottom={bottom:.6e} "
                    f"from Qgg={Qgg:.6e} Bge={Bge:.6e} Qee={Qee:.6e} beta={beta:.6e}"
                    f" -- stepping along |bottom| instead of taking alpha<0")
                bottom = abs(bottom) if bottom != 0.0 else 1e-6 * scale

            alpha = top / bottom
        return alpha, top, bottom

    def _ref_beta(self, vars, grads, etas):
        """beta the classic way — two extra sweeps. Verification only."""
        t, b = self.allreduce_scalars(self.hessian(vars, grads, etas),
                                      self.hessian(vars, etas, etas))
        return t / b

    def _log_fused_check(self, i, name, got, ref):
        """Report |fused - measured| / |measured| for the fused-step check."""
        rel = abs(got - ref) / abs(ref) if ref != 0 else abs(got)
        logger.info(f"iter={i}: fused-check {name:>6}  fused={got:+.9e}  "
                    f"measured={ref:+.9e}  rel={rel:.3e}")

    def _update_etas(self, grads, etas, beta):
        """etas <- beta*etas - grads for every variable, returning the local
        alpha numerator -<eta_new, grad>/rho^2 summed over obj/pos (+prb and tp
        on rank 0)."""
        top = 0
        for v in ("obj", "pos"):
            if self.rho_sq[v]>0:
                top -= self.linear_redot_batch(etas[v], grads[v], beta, -1) / self.rho_sq[v]
        # probe is shared across ranks; only rank 0 contributes to the rank-0 sum
        dot_prb = self.linear_redot_batch(etas['prb'], grads['prb'], beta, -1)
        # tp is global too -- the direction update must run on every rank, but
        # only rank 0 may add it to the sum that gets allreduced.
        dot_tp  = self.linear_redot_batch(etas['tp'], grads['tp'], beta, -1)
        if self.rank == 0:
            if self.rho_sq['prb']>0:
                top -= dot_prb / self.rho_sq['prb']
            if self.rho_sq['tp']>0:
                top -= dot_tp / self.rho_sq['tp']
        self.linear_batch(etas['proj'], grads['proj'], beta, -1)
        return top

    def compute_beta(self, vars, grads, etas, i):
        """CG coefficient beta = <grad, H·eta> / <eta, H·eta>.
        Returns 0 on the first iter (pure steepest descent)."""
        if i == self.start_iter:
            return 0
        with nvtx.annotate(":::BH:calc beta"):
            top    = self.hessian(vars, grads, etas)
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            return top / bottom

    def compute_alpha(self, vars, grads, etas, beta):
        """Update the search direction (etas) with the new beta, then compute
        the step size alpha. Returns (alpha, top, bottom); top/bottom are
        forwarded to check_approximation."""
        with nvtx.annotate(":::BH:calc_alpha"):
            top = self._update_etas(grads, etas, beta)
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            if not bottom > 0.0:
                # Same non-convex case the fused route guards; see
                # _compute_step_fused.
                logger.warning(
                    f"non-convex alpha denominator bottom={bottom:.6e} "
                    f"-- stepping along |bottom| instead of taking alpha<0")
                bottom = abs(bottom) if bottom != 0.0 else 1.0
            alpha = top / bottom
        return alpha, top, bottom

    def apply_step(self, vars, etas, alpha, obj_norms=False):
        """var ← var + alpha·eta for every variable.

        With obj_norms, also returns (‖obj^{n+1} − obj^n‖², ‖obj^{n+1}‖²) as
        LOCAL sums for the caller to allreduce.  The step is exactly
        alpha·etas['obj'], which is already resident, so no copy of the previous
        object is needed; the two reductions ride along inside the object's own
        update pass, which streams both arrays anyway.  Zero extra memory and no
        extra host↔device traffic.
        """
        out = None
        for v in ("obj", "prb", "pos", "proj", "tp"):
            if v == "obj" and obj_norms:
                out = self._apply_step_obj(vars, etas, alpha)
            else:
                self.linear_batch(vars[v], etas[v], 1, alpha)
        return out

    def _apply_step_obj(self, vars, etas, alpha):
        """obj <- obj + alpha*eta_obj, accumulating ||eta_obj||^2 and
        ||obj_new||^2 in the same streaming pass.

        Returns the LOCAL (alpha^2*||eta_obj||^2, ||obj_new||^2); this rank owns
        only its z slab.  float64 accumulators, unlike the float32 ones in
        Chunking.redot_batch: this is a sum of ~nzobj/nchunk same-sign chunk
        norms over up to 3e10 voxels, and float32 would eat ~1e-4 of it.  They
        are 0-d scalars, so the extra bytes are free -- and 0-d, not (1,), so
        gpu_batch can never mistake them for chunked outputs."""
        x, y = vars['obj'], etas['obj']
        a2 = float(alpha) ** 2
        if isinstance(x, cp.ndarray):
            _axpby(x, x, y, 1, alpha)
            return a2 * float(redot(y, y).get()), float(redot(x, x).get())

        d2 = cp.zeros((), dtype="float64")
        o2 = cp.zeros((), dtype="float64")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _step_obj(self, out, d2, o2, x, y, alpha):
            _axpby(out, x, y, 1, alpha)
            d2[...] += redot(y, y)
            o2[...] += redot(out, out)

        _step_obj(self, x, d2, o2, x, y, alpha)
        return a2 * float(d2.get()[()]), float(o2.get()[()])

    def log_iter(self, vars, i, writer):
        """Error logging + visualization debug for this iter."""
        with nvtx.annotate(":::BH:calc error", color='gray'):
            self.error_debug(vars, i)
        with nvtx.annotate(":::BH:vis_debug", color='gray'):
            self.vis_debug(vars, i, writer)
        # Cache hit/miss for this iter.  Always drained so the counts stay
        # per-iteration rather than cumulative; only logged on debug_step.
        ch, cm = self.cl_shift.coeff_cache_stats(reset=True)
        ah, am = self.apply_F_cache_stats(reset=True)
        if self.rank == 0 and self.debug_step != -1 and i % self.debug_step == 0:
            logger.info(f"iter={i}: coeff_cache    hits={ch} misses={cm}")
            logger.info(f"iter={i}: apply_F_cache  hits={ah} misses={am}")

    def hessian(self, vars, grads, etas):
        """Hessian for the full functional, is a sum of 3 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        3. regularization term

        All three addends are LOCAL; the caller allreduces the sum once. The
        regularization term takes its second direction from the stored e_pad
        (which views etas['obj']), not from the `etas` argument."""
        with nvtx.annotate("hessian"):
            w = self.hessian_cascade(vars, grads, etas)
            if self.rank == 0 and hasattr(self, 'cl_prb_term'):
                w += self.cl_prb_term.hessian(vars["prb"], grads["prb"], etas["prb"])
            if hasattr(self, 'cl_lap_term'):
                w += self.cl_lap_term.hessian(grads["obj"])
        return w

    @timer
    def hessian_cascade(self, vars, grads, etas):
        """"Cascade computation of the hessian for the main term, following the
            composition rule (Carlsson, 2025). Distances run INSIDE the
            theta-chunk loop, ndistchunk at a time, so one upload of the x/y/z
            proj chunks serves ndistchunk distances. `out` is a scalar cupy
            accumulator across every chunk and every k."""

        out = cp.zeros(1, dtype="float32")
        # Identity check on un-sliced pinned arrays (slices per-k would never be `is`-equal).
        y_is_z = grads['prb'] is etas['prb']

        for k0, k1 in self._dist_groups():
            nd = k1 - k0
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
            def _hess_dists(self, out, *a):
                # proper inputs: nd*(d, x_pos, y_pos, z_pos, mask), the t chunk,
                # then the three dist-independent proj chunks; non-proper: the
                # three [nd, nz, n] prb bundles and the three [nd, 2, 2] tp bundles.
                d     = a[0 * nd:1 * nd]
                x_pos = a[1 * nd:2 * nd]
                y_pos = a[2 * nd:3 * nd]
                z_pos = a[3 * nd:4 * nd]
                mask  = a[4 * nd:5 * nd]
                t                      = a[5 * nd]
                x_proj, y_proj, z_proj = a[5 * nd + 1:5 * nd + 4]
                x_prb,  y_prb,  z_prb  = a[5 * nd + 4:5 * nd + 7]
                x_tp,   y_tp,   z_tp   = a[5 * nd + 7:5 * nd + 10]

                # coeff(proj) is distance-independent, so the B-spline prefilter
                # of each of the three proj chunks is computed once here and
                # reused by every k below -- nd x fewer full-grid FFT pairs.
                self.cl_shift.coeff_cache_reset()
                self._t_chunk = t
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._set_mask_chunk(mask[j])
                    # apply_F results ARE distance-dependent: reset per k, or nd
                    # detector-plane cascade states would pile up.
                    self.apply_F_cache_reset()
                    x = [x_prb[j], x_proj, x_pos[j], x_tp[j]]
                    y = [y_prb[j], y_proj, y_pos[j], y_tp[j]]
                    z = y if y_is_z else [z_prb[j], z_proj, z_pos[j], z_tp[j]]
                    w = [None, None, None, None]
                    for id in range(1, len(self.F))[::-1]:
                        w = self.d2F_dF[id](x, y, z, w)
                        fx, y = self.dF[id](x, y)
                        z = y if y_is_z else self.dF[id](x, z, return_x=False)
                        x = fx
                    out[:] += self.d2F_dF[0](x, y, z, w, d[j])

            ks = range(k0, k1)
            _hess_dists(self, out,
                        *[self.data[k]      for k in ks],
                        *[vars['pos'][k]    for k in ks],
                        *[grads['pos'][k]   for k in ks],
                        *[etas['pos'][k]    for k in ks],
                        *[self.mask_1d[k]   for k in ks],
                        self.t_local,
                        vars['proj'], grads['proj'], etas['proj'],
                        vars['prb'][k0:k1], grads['prb'][k0:k1], etas['prb'][k0:k1],
                        vars['tp'][k0:k1],  grads['tp'][k0:k1],  etas['tp'][k0:k1])

        return out[0].get()

    def hessian3(self, vars, grads, etas):
        """{B(g,g), B(g,e), B(e,e)} — the same three terms hessian() sums, but
        as the three bilinear forms of (g,e) instead of one, each term
        contributing all three from a single pass over its own data. Local (not
        allreduced), like hessian().

        The regularization term ignores its arguments: g and e are the interiors
        of the padded slabs it owns (see LaplacianTerm.hessian3)."""
        with nvtx.annotate("hessian3"):
            hgg, hge, hee = self.hessian_cascade3(vars, grads, etas)
            if self.rank == 0 and hasattr(self, 'cl_prb_term'):
                pgg, pge, pee = self.cl_prb_term.hessian3(
                    vars["prb"], grads["prb"], etas["prb"])
                hgg = hgg + pgg
                hge = hge + pge
                hee = hee + pee
            if hasattr(self, 'cl_lap_term'):
                lgg, lge, lee = self.cl_lap_term.hessian3()
                hgg = hgg + lgg
                hge = hge + lge
                hee = hee + lee
        return hgg, hge, hee

    @timer
    def hessian_cascade3(self, vars, grads, etas):
        """The three bilinear forms {B(g,g), B(g,e), B(e,e)} of the main term
        from ONE cascade sweep.

        Same composition rule (Carlsson, 2025) as hessian_cascade, but the
        x-chain — the part that streams vars/grads/etas['proj'] and `data` over
        PCIe, i.e. essentially all of the cost — is advanced once and shared by
        all three pairs instead of being recomputed per call.

        `out` is a 3-element cupy accumulator. Chunking classifies it as a
        *non-proper* output (its axis-0 length is not the chunk's theta count)
        and hands it whole to every chunk, exactly as it does the scalar in
        hessian_cascade."""

        # ...which relies on 3 != the chunked axis length. Only violated by a
        # degenerate decomposition, but a silent mis-slice would be very hard
        # to spot in the numbers.
        assert self.local_ntheta != 3, "local_ntheta==3 aliases the accumulator shape"

        out = cp.zeros(3, dtype="float32")

        for k0, k1 in self._dist_groups():
            nd = k1 - k0
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
            def _hess3_dists(self, out, *a):
                # proper inputs: nd*(d, x_pos, g_pos, e_pos, mask), the t chunk,
                # then the three dist-independent proj chunks; non-proper: the
                # three [nd, nz, n] prb bundles and the three [nd, 2, 2] tp bundles.
                d     = a[0 * nd:1 * nd]
                x_pos = a[1 * nd:2 * nd]
                g_pos = a[2 * nd:3 * nd]
                e_pos = a[3 * nd:4 * nd]
                mask  = a[4 * nd:5 * nd]
                t                      = a[5 * nd]
                x_proj, g_proj, e_proj = a[5 * nd + 1:5 * nd + 4]
                x_prb,  g_prb,  e_prb  = a[5 * nd + 4:5 * nd + 7]
                x_tp,   g_tp,   e_tp   = a[5 * nd + 7:5 * nd + 10]

                # coeff(x_proj/g_proj/e_proj) is distance-independent: three FFT
                # prefilters per chunk instead of three per (chunk, distance).
                self.cl_shift.coeff_cache_reset()
                self._t_chunk = t
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._set_mask_chunk(mask[j])
                    self.apply_F_cache_reset()   # per dist: cascade states are per-dist
                    x = [x_prb[j], x_proj, x_pos[j], x_tp[j]]
                    g = [g_prb[j], g_proj, g_pos[j], g_tp[j]]
                    e = [e_prb[j], e_proj, e_pos[j], e_tp[j]]
                    # Passing the *same* list object for y and z on the diagonal forms
                    # keeps the `y12 is z12` / cached-coeff fast paths in d2F_dF1 and
                    # d2F_dF3 alive — the y_is_z flag becomes structural.
                    wgg = [None, None, None, None]
                    wge = [None, None, None, None]
                    wee = [None, None, None, None]
                    for id in range(1, len(self.F))[::-1]:
                        # d2F_dF[id] must see the pre-update x, g, e — hence all three
                        # contractions before dF[id] advances the chains.
                        wgg = self.d2F_dF[id](x, g, g, wgg)
                        wge = self.d2F_dF[id](x, g, e, wge)
                        wee = self.d2F_dF[id](x, e, e, wee)
                        fx, gn = self.dF[id](x, g)
                        en = self.dF[id](x, e, return_x=False)
                        x, g, e = fx, gn, en
                    out[0:1] += self.d2F_dF[0](x, g, g, wgg, d[j])
                    out[1:2] += self.d2F_dF[0](x, g, e, wge, d[j])
                    out[2:3] += self.d2F_dF[0](x, e, e, wee, d[j])

            ks = range(k0, k1)
            _hess3_dists(self, out,
                         *[self.data[k]      for k in ks],
                         *[vars['pos'][k]    for k in ks],
                         *[grads['pos'][k]   for k in ks],
                         *[etas['pos'][k]    for k in ks],
                         *[self.mask_1d[k]   for k in ks],
                         self.t_local,
                         vars['proj'], grads['proj'], etas['proj'],
                         vars['prb'][k0:k1], grads['prb'][k0:k1], etas['prb'][k0:k1],
                         vars['tp'][k0:k1],  grads['tp'][k0:k1],  etas['tp'][k0:k1])

        h = out.get()
        return float(h[0]), float(h[1]), float(h[2])

    def gradients(self, vars, grads):
        """Full gradient, consists of 2 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        """
        self.gradients_cascade(vars, grads)

        with nvtx.annotate(":::BH:redist back", color='red'):
            self.redist(grads['proj'], self.proj_tmp, direction='backward')

        # part2, parallelization over object slices: adjoint Radon
        self.adj_tomo(grads['obj'], self.proj_tmp)
        if hasattr(self, 'cl_lap_term'):
            # NOTE: unlike the probe term below, this adds its gradient WITHOUT
            # rho_sq['obj'], while the misfit part of grads['obj'] carries it
            # (gradproj_out += y[1]*rho_sq['obj'] in gradients_cascade).  The
            # two halves of grads['obj'] are therefore in the same metric only
            # because rho[obj] = 1.0 in every config shipped here.  Anyone who
            # sets rho[obj] != 1 -- or turns estimate_rho back on -- must scale
            # this call the way cl_prb_term.gradient is scaled, and check
            # cl_lap_term.hessian/hessian3 at the same time.
            self.cl_lap_term.gradient(grads['obj'])

        if self.rank == 0 and hasattr(self, 'cl_prb_term'):
            self.cl_prb_term.gradient(grads["prb"], vars["prb"], self.rho_sq['prb'])

        grads['prb'][:] = self.allreduce(grads['prb'])
        # tp is global: every rank computed the contribution of its own angles.
        grads['tp'][:] = cp.asarray(self.allreduce(cp.asnumpy(grads['tp'])))

    @timer
    def gradients_cascade(self, vars, grads):
        """Cascade gradient for the main term (Carlsson, 2025).

        Distances run INSIDE the theta-chunk loop, ndistchunk per @gpu_batch
        pass: one upload of the vars['proj'] chunk feeds them all, and their
        Deltapsi contributions are summed on the GPU instead of through a
        per-distance read-modify-write of grads['proj'] across PCIe.

        grads['proj'] is still passed as both proper input and proper output so
        that successive *groups* accumulate; with ndistchunk >= ndist there is
        one group and that costs a single extra read of a zeroed slab. The
        trailing cl_shift.coeff (parent's gF3 returns un-coeff'd Deltapsi) is
        applied by the last group, once the sum over all distances is complete.
        """

        grads['proj'][:] = 0   # zero accumulator before the group loop
        grads['prb'][:]  = 0   # each k writes its own slot
        grads['tp'][:]   = 0   # accumulated over theta chunks AND ranks

        groups = self._dist_groups()
        for gi, (k0, k1) in enumerate(groups):
            nd   = k1 - k0
            last = (gi == len(groups) - 1)
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=nd + 3)
            def _grad_dists(self, gradproj_out, *a):
                # outputs: nd*gradpos (proper), gradprb [nd,nz,n] and
                # gradtp [nd,2,2] (both non-proper, so both accumulate across
                # theta chunks in place).
                gradpos     = a[0:nd]
                gradprb     = a[nd]
                gradtp      = a[nd + 1]
                # proper inputs: gradproj_in, nd*(d, pos, mask), t, proj;
                # non-proper: the [nd, nz, n] prb and [nd, 2, 2] tp bundles.
                b           = a[nd + 2:]
                gradproj_in = b[0]
                d    = b[1 + 0 * nd:1 + 1 * nd]
                pos  = b[1 + 1 * nd:1 + 2 * nd]
                mask = b[1 + 2 * nd:1 + 3 * nd]
                t    = b[1 + 3 * nd]
                proj = b[2 + 3 * nd]
                prb  = b[3 + 3 * nd]
                tp   = b[4 + 3 * nd]

                # coeff(proj) is distance-independent -- one prefilter per chunk.
                self.cl_shift.coeff_cache_reset()
                # Accumulate straight into the output chunk buffer. gF3's y[1] is
                # a freshly zeroed [chunk, nzpsi, npsi] from dcurlySadjc, so it can
                # be scaled in place -- no separate accumulator and no `acc * rho`
                # temporary, i.e. two object-plane slabs saved over the naive form.
                gradproj_out[:] = gradproj_in
                self._t_chunk = t
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._set_mask_chunk(mask[j])
                    self.apply_F_cache_reset()   # per dist: cascade states are per-dist
                    x = [prb[j], proj, pos[j], tp[j]]
                    y = d[j]
                    for id in range(len(self.gF)):
                        y = self.gF[id](x, y)

                    gradprb[j] += y[0] * self.rho_sq['prb']
                    gradpos[j][:] = y[2] * self.rho_sq['pos']
                    gradproj_out += y[1]*self.rho_sq['obj']
                    gradtp[j] += y[3] * self.rho_sq['tp']
                if last:
                    gradproj_out[:] = self.cl_shift.coeff(gradproj_out)

            ks = range(k0, k1)
            _grad_dists(self,
                        grads['proj'],                                  # out
                        *[grads['pos'][k] for k in ks],                 # out
                        grads['prb'][k0:k1],                            # out
                        grads['tp'][k0:k1],                             # out
                        grads['proj'],                                  # inp
                        *[self.data[k]    for k in ks],
                        *[vars['pos'][k]  for k in ks],
                        *[self.mask_1d[k] for k in ks],
                        self.t_local,
                        vars['proj'], vars['prb'][k0:k1], vars['tp'][k0:k1])

    @timer
    def fwd_tomo(self, obj, out):
        """Forward tomography operator"""
        
        @self.gpu_batch(axis_out=1, axis_inp=0,nout=1)
        def _fwd_tomo(self, out, obj):
            out[:] = self.cl_tomo.R(obj)
            
        _fwd_tomo(self, out, obj)
        return out    
    
    @timer
    def adj_tomo(self, gradu, gradproj):
        @self.gpu_batch(axis_out=0, axis_inp=1, nout=1)
        def _adj_tomo(self, gradu, gradproj):
            gradu[:] = self.cl_tomo.RT(gradproj)
        
        _adj_tomo(self, gradu, gradproj)

    ####################### Functions for the cascade (following math notes for variables)
    # F* - functional
    # dF* - differential
    # d2F* - second order term for hessian
    # gF* - gradient
    #######################################################################################

    def apply_F_from(self, x, from_level):
        """Apply F[from_level], F[from_level+1], ..., F[len(F)-1] inside-out, returning the
        partial cascade value at level (from_level-1). Used by gF* methods that need to
        "fast-forward" the input x through cascade levels above their own.

        Memoized recursively by (id(x), from_level): within one cascade pass gF0..gFN
        share the same outer x, so without caching F3 would be re-evaluated N times and
        F2 (N-1) times etc. MUST be paired with apply_F_cache_reset() per chunk —
        id() can be reused across distinct objects once an earlier one is GC'd.
        """
        if from_level >= len(self.F):
            return x
        key = (id(x), from_level)
        cached = self._apply_F_cache.get(key)
        if cached is not None:
            self._apply_F_hits += 1
            return cached
        self._apply_F_misses += 1
        upstream = self.apply_F_from(x, from_level + 1)
        result = self.F[from_level](upstream)
        self._apply_F_cache[key] = result
        return result

    def apply_F_cache_reset(self):
        self._apply_F_cache = {}

    def _zero_like_cached(self, a):
        """Read-only all-zeros buffer shaped like `a`. Cached: d2F_dF3's
        fallback branches would otherwise allocate an object-plane slab on every
        call, three times per (chunk, distance) in hessian_cascade3."""
        key = (a.shape, a.dtype)
        z = self._zero_buf_cache.get(key)
        if z is None:
            z = cp.zeros(a.shape, dtype=a.dtype)
            self._zero_buf_cache[key] = z
        return z

    def apply_F_cache_stats(self, reset=False):
        stats = (self._apply_F_hits, self._apply_F_misses)
        if reset:
            self._apply_F_hits = 0
            self._apply_F_misses = 0
        return stats


    ####### F0(x0) = 1/n \|m . g(K|x0|^2, d)\|_2^2
    #
    # TWO MISFIT MODELS, selected by self.model ('intensity' or 'amplitude').
    # They share everything except the last pointwise step.  With I = |x|^2,
    # J = K I the blurred MODEL intensity, a = sqrt(J) the model amplitude,
    # s = sqrt(d) the measured amplitude and W = my*mx the detector mask:
    #
    #   intensity   F0 = 1/N sum W (a^2 - d)^2 = 1/N sum W (J - d)^2
    #   amplitude   F0 = 1/N sum W (a   - s)^2
    #
    # PSF WORKS IN BOTH, and sits in the same place in both.  The focal spot and
    # the detector PSF blur INTENSITY, incoherently, so neither can be folded
    # into cl_prop.D (which is coherent) and the blur's only legal home is here,
    # on I.  Both are Gaussian, so they are carried as ONE Gaussian K (see
    # _psf_blur; it is self-adjoint, so K^T is literally K), the same at every
    # distance, and K is the identity when no blur is configured.  The amplitude
    # model then compares sqrt(K I) against sqrt(d): the blur stays on intensity
    # where the physics puts it, and only the COMPARISON moves to amplitude.  At
    # psf_sigma = 0 that is exactly the historical amplitude misfit, ||x| - s|^2.
    #
    # d IS THE MEASURED INTENSITY IN BOTH.  reader.py used to take the sqrt on
    # load, back when the only misfit was the amplitude one; it no longer does,
    # and gen_data generates intensity to match.  Nothing squares d, and the
    # amplitude path takes its own sqrt here.
    #
    # WHICH ONE IS RIGHT is a noise question, not a physics question -- the
    # forward model is identical.  (a^2 - d) = (a - s)(a + s), so the intensity
    # misfit is the amplitude misfit carrying an extra per-pixel weight
    # (a + s)^2 ~ 4d.  That is the Poisson variance, so amplitude is the
    # correctly weighted least squares under counting noise (sqrt is its
    # variance-stabilizing transform) and intensity is correct when read noise
    # dominates.  See config._parse_model for what this does to err, to
    # lam_prbfit/lam_laplacian (divide by ~4 for amplitude) and to rho (nothing).
    #
    # m = my*mx is the out-of-grid detector mask from _build_data_mask, kept as
    # the two separable factors self._mask_y [chunk,nz,1] and self._mask_x
    # [chunk,1,n] and broadcast by the product W = my*mx.  They are read off
    # self, like
    # self._dist_idx.  1/data_size still counts every pixel, masked or not:
    # renormalizing would silently rescale lam_prbfit, lam_laplacian and rho.
    #
    # ONE SET OF DERIVATIVES COVERS BOTH.  Write the misfit as 1/N sum W g(J),
    # and let u = Re(conj(x) z), v = Re(conj(x) y).  Since dJ[z] = 2 K(u),
    #
    #   F0        = 1/N sum W g(J)
    #   dF0[y]    = 4/N sum p v                                  p = K[W g'/2]
    #   d2F0[y,z] = 4/N sum { 2 t v + p Re(conj(z) y) }          t = K[W (g''/2) K u]
    #                                        (+ 4/N sum p Re(conj(x) w))
    #
    # so the two models differ ONLY in the two pointwise weights g'/2 and g''/2.
    # Below, each model gets its own four functions with those weights written
    # out, and F0/dF0/d2F_dF0/gF0 only pick between the two families:
    #
    #                g               g'/2                g''/2
    #   intensity    (J - d)^2       J - d               1
    #   amplitude    (sqrt J - s)^2  (a - s)/(2a)        s/(4 a^3)
    #
    # Substituting the intensity row reproduces the previous code exactly.
    #
    # THE AMPLITUDE PATH DIVIDES and the intensity path does not.  a can be zero
    # (a masked pixel, a true zero of the propagated field) and 1/a^3 makes the
    # curvature blow up long before that, which is the reason this misfit was
    # dropped in the first place.  Both denominators are therefore floored at
    # _AMP_FLOOR; F0 itself is left exact, so only the derivatives are modified,
    # and only on pixels where the model predicts essentially no signal.  With
    # flat-field-normalised data a ~ 1, so the floor is ~3 orders below anything
    # physical and is in practice never active.
    #
    # Note g''/2 >= 0 for BOTH models, so the t term is a genuine square in the
    # quadratic form (sum W (g''/2) (K u)^2 >= 0) and all the indefiniteness
    # lives in p, which is proportional to the residual and vanishes at the
    # solution -- see the non-convex branch in _compute_step_fused.
    #
    # Two things that are easy to get wrong:
    #  * W sits INSIDE the outer K, in both p and t.  Masking the blurred result
    #    afterwards instead would not be the derivative of anything.  The
    #    p-weighted terms must therefore NOT be masked again: p already carries
    #    W.  Likewise W sits between the Hessian's two convolutions.
    #  * the Hessian needs TWO nested convolutions; it is not a substitution
    #    into a pointwise kernel.

    # Floor on the model amplitude a = sqrt(K|x|^2) in the amplitude model's two
    # denominators.  Data are flat-field normalised (a ~ 1), so this is ~1e-3 of
    # anything physical; it caps the curvature weight s/(4a^3) at ~2.5e8*s
    # instead of letting it diverge.
    _AMP_FLOOR = np.float32(1e-3)

    def _blur(self, x):
        """K x, the one Gaussian on the detector intensity, identity if unset.

        K^T = K (see _psf_blur), so this one call serves both directions."""
        return x if self.psf_w is None else _psf_blur(x, self.psf_w)

    # ---- the misfit and its derivatives: ONE function per model -----------
    # DEBUG MODE (2026-09-15): no helper kernels, no @cp.fuse(), and the two
    # misfit models have SEPARATE functions -- *_int for intensity, *_amp for
    # amplitude.  Each of the eight spells out its whole formula in one body:
    # the mask product W, the blurs, that model's weights, the reduction.  A
    # formula can be checked by reading a single function, stepped into, and
    # called on float64 input; no body carries a model branch, and nothing in
    # the group calls anything in the group.
    #
    # F0/dF0/d2F_dF0/gF0 are one-line dispatchers on self.model and are what
    # the cascade calls -- the only four model branches in the group.  The
    # price of the split is that p = K[W g'/2] is written out six times (three
    # derivatives x two models); that is deliberate, and it is what makes each
    # function readable on its own.
    #
    # The previous structure -- eight @cp.fuse()d static kernels behind
    # _pw/_cw/_F0_p, one code path for both models -- is in git; restore it
    # before any production run.  Fused, cp.fuse folded the mask broadcast and
    # the sqrt into one elementwise kernel, which is why the amplitude forms
    # cost the same memory traffic as the intensity ones and why sqrt(d) is
    # recomputed rather than stored: these kernels are bandwidth-bound and d is
    # a full detector chunk.
    #
    # self._blur is NOT part of this group: it is K itself, shared with
    # PrbfitTerm and with the err reduction, and it stays a method.

    def F0(self, x, d):
        """In: (x0), Out: const.  Dispatch on self.model."""
        return (self.F0_amp if self.model == 'amplitude' else self.F0_int)(x, d)

    def dF0(self, x, y, d, return_x=False):
        """In: (x0,y0), Out: const.  Dispatch on self.model."""
        f = self.dF0_amp if self.model == 'amplitude' else self.dF0_int
        return f(x, y, d)

    def d2F_dF0(self, x, y, z, w, d):
        """In: (x0,y0,z0,w0), Out: const.  Dispatch on self.model."""
        f = self.d2F_dF0_amp if self.model == 'amplitude' else self.d2F_dF0_int
        return f(x, y, z, w, d)

    def gF0(self, x, y):
        """In: x, y = F0(F1(..(x)))), Out: y0.  Dispatch on self.model."""
        return (self.gF0_amp if self.model == 'amplitude' else self.gF0_int)(x, y)

    # ---- intensity model:  F0 = 1/N sum W (J - d)^2,  J = K|x|^2 ----------

    @nvtx.annotate("F0_int", color="green")
    def F0_int(self, x, d):
        """F0 = 1/N sum W (J - d)^2,   J = K|x|^2,  W = my*mx."""
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        t = J - d
        return 1 / self.data_size * cp.sum(W * t * t)

    @nvtx.annotate("dF0_int", color="green")
    def dF0_int(self, x, y, d):
        """dF0[y] = 4/N sum p Re(conj(x) y),   p = K[W (J - d)]."""
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        p = self._blur(W * (J - d))
        return 4 / self.data_size * redot(p * x, y)

    @nvtx.annotate("d2F0_dF0_int", color="purple")
    def d2F_dF0_int(self, x, y, z, w, d):
        """d2F0[y,z] = 4/N sum { 2 t Re(conj(x) y) + p Re(conj(z) y) }
                       + 4/N sum p Re(conj(x) w)        (the w slot, = dF0[w])

            p = K[W (J - d)]
            t = K[W K Re(conj(x) z)]     <- TWO nested blurs, the mask between
                                            them; the curvature weight is 1
        """
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        p = self._blur(W * (J - d))
        t = self._blur(W * self._blur(reprod(x, z)))
        v = 2 * t * reprod(x, y) + p * reprod(z, y)
        if w is not None:
            v += p * reprod(x, w)
        return 4 / self.data_size * cp.sum(v)

    @nvtx.annotate("gF0_int", color="green")
    def gF0_int(self, x, y):
        """grad F0 = (4/N) p x,   p = K[W (J - d)].  y is the data d."""
        d = y
        x = self.apply_F_from(x, 1)
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        p = self._blur(W * (J - d))
        return (np.float32(4 / self.data_size) * p) * x

    # ---- amplitude model:  F0 = 1/N sum W (a - sqrt(d))^2,  a = sqrt(K|x|^2)
    # a is floored at _AMP_FLOOR wherever it appears in a DENOMINATOR (the two
    # derivatives); F0 itself is left exact.

    @nvtx.annotate("F0_amp", color="green")
    def F0_amp(self, x, d):
        """F0 = 1/N sum W (sqrt(J) - sqrt(d))^2,   J = K|x|^2,  W = my*mx."""
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        t = cp.sqrt(J) - cp.sqrt(d)
        return 1 / self.data_size * cp.sum(W * t * t)

    @nvtx.annotate("dF0_amp", color="green")
    def dF0_amp(self, x, y, d):
        """dF0[y] = 4/N sum p Re(conj(x) y),   p = K[W (a - sqrt(d))/(2a)],
        a = max(sqrt(J), floor)."""
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        a = cp.maximum(cp.sqrt(J), self._AMP_FLOOR)
        p = self._blur(W * 0.5 * (1.0 - cp.sqrt(d) / a))
        return 4 / self.data_size * redot(p * x, y)

    @nvtx.annotate("d2F0_dF0_amp", color="purple")
    def d2F_dF0_amp(self, x, y, z, w, d):
        """d2F0[y,z] = 4/N sum { 2 t Re(conj(x) y) + p Re(conj(z) y) }
                       + 4/N sum p Re(conj(x) w)        (the w slot, = dF0[w])

            p = K[W (a - sqrt(d))/(2a)]
            t = K[W (sqrt(d)/(4 a^3)) K Re(conj(x) z)]   <- TWO nested blurs;
                                                            the curvature weight
                                                            sits BETWEEN them,
                                                            in the mask's slot
        """
        W = self._mask_y * self._mask_x
        # J is built once and kept live: the curvature weight needs it as well
        # as p does.  u is the INNER convolution of the t term.
        J = self._blur(reprod(x, x))
        u = self._blur(reprod(x, z))
        a = cp.maximum(cp.sqrt(J), self._AMP_FLOOR)
        p = self._blur(W * 0.5 * (1.0 - cp.sqrt(d) / a))
        t = self._blur(W * (0.25 * cp.sqrt(d) / (a * a * a)) * u)
        v = 2 * t * reprod(x, y) + p * reprod(z, y)
        if w is not None:
            v += p * reprod(x, w)
        return 4 / self.data_size * cp.sum(v)

    @nvtx.annotate("gF0_amp", color="green")
    def gF0_amp(self, x, y):
        """grad F0 = (4/N) p x,   p = K[W (a - sqrt(d))/(2a)].  y is the data d."""
        d = y
        x = self.apply_F_from(x, 1)
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        a = cp.maximum(cp.sqrt(J), self._AMP_FLOOR)
        p = self._blur(W * 0.5 * (1.0 - cp.sqrt(d) / a))
        return (np.float32(4 / self.data_size) * p) * x


    ####### x0 = F1(x11,x12) = D(x11\cdot x12)
    @nvtx.annotate("F1", color="green")
    def F1(self, x):
        """In: (x11,x12), Out: x0. Per-dist: arrays carry singleton dist axis; uses self._dist_idx."""

        x11, x12 = x  # x11: [nz, n], x12: [chunk, nz, n]
        return self.cl_prop.D(x11 * x12, self._dist_idx)

    @nvtx.annotate("dF1", color="green")
    def dF1(self, x, y, return_x=True):
        """In: (x11,x12),(y11,y12) Out: y0. Per-dist."""
        x11, x12 = x
        y11, y12 = y
        y0 = self.cl_prop.D(y11 * x12 + x11 * y12, self._dist_idx)
        if return_x:
            x0 = self.cl_prop.D(x11 * x12, self._dist_idx)
            return x0, y0
        return y0

    @nvtx.annotate("d2F_dF1", color="purple")
    def d2F_dF1(self, x, y, z, w):
        """In: (x11,x12),(y11,y12),(z11,z12) Out: y0. Per-dist."""
        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        w11, w12 = w

        if y12 is z12:
            y0 = 2 * y11 * y12
        else:
            y0 = y11 * z12 + z11 * y12

        if w11 is not None:
            y0 += w11 * x12
        if w12 is not None:
            y0 += x11 * w12

        return self.cl_prop.D(y0, self._dist_idx)

    @nvtx.annotate("gF1", color="green")
    def gF1(self, x, y):
        """In: x=(x01,x02,x03),(y0) Out: y11,y12. Per-dist."""
        y0 = y  # [chunk, nz, n]
        x = self.apply_F_from(x, 2)
        x11, x12 = x  # x11: [nz, n], x12: [chunk, nz, n]
        y12 = self.cl_prop.DT(y0, self._dist_idx)
        y11 = cp.sum(y12 * np.conj(x12), axis=0)  # [nz, n]
        y12 *= np.conj(x11)
        return y11, y12

    ######## (x11,x12) = F2(x21,x22) = (x21,e^{1j x22})
    @staticmethod
    @cp.fuse()
    def _F2_fused(x22):
        return cp.exp(1j * x22)

    @nvtx.annotate("F2", color="green")
    def F2(self, x):
        """In: (x21,x22) Out: (x11,x12)"""

        x21, x22 = x
        x11 = x21
        x12 = self._F2_fused(x22)
        return x11, x12

    @staticmethod
    @cp.fuse()
    def _dF2_fused(x22, y22):
        x12 = cp.exp(1j * x22)
        y12 = x12 * 1j * y22
        return x12, y12

    @nvtx.annotate("dF2", color="green")
    def dF2(self, x, y, return_x=True):
        """In: (x21,x22),(y21,y22) Out: (x11,x12),(y11,y12)"""

        x21, x22 = x
        y21, y22 = y

        x12, y12 = self._dF2_fused(x22, y22)
        x11 = x21
        y11 = y21

        return ([x11, x12], [y11, y12]) if return_x else [y11, y12]
    
    @staticmethod
    @cp.fuse()
    def _d2F_dF2_fused(x22, y22, z22, w22):
        y12 = cp.exp(1j * x22) * (-y22 * z22)
        if w22 is not None:
            y12 = y12 + cp.exp(1j * x22) * 1j * w22
        return y12

    @nvtx.annotate("d2F_dF2", color="purple")
    def d2F_dF2(self, x, y, z, w):
        """In: (x21,x22),(y21,y22),(z21,z22),(w21,w22) Out: (y11,y12)"""

        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        w21, w22 = w

        y12 = self._d2F_dF2_fused(x22, y22, z22, w22)
        y11 = w21

        return [y11, y12]
    
    @staticmethod
    @cp.fuse()
    def _gF2_fused(x22, y12):
        return (-1j) * y12 * cp.conj(cp.exp(1j * x22))

    @nvtx.annotate("gF2", color="green")
    def gF2(self, x, y):
        """In: x(x01, x02, x03) ,(y11,y12) Out: (y21,y22)"""

        y11, y12 = y

        x = self.apply_F_from(x, 3)
        x21, x22 = x

        y22 = self._gF2_fused(x22, y12)

        y21 = y11
        return [y21, y22]
    
    ####### (x21,x22) = F3(x31,x32,x33,x34) = (x31,S_{x33,x34}(x32))
    #
    # x34 is the per-(theta, axis) demagnification produced by F4, a
    # *differentiable* input. The identity d/dm_axis = -tau_axis(pixel)*d/dr_axis
    # lets the kernels fold Delta_m into a per-pixel effective Delta_r, so a
    # differentiated shift still costs one launch.
    @nvtx.annotate("F3", color="green")
    def F3(self, x):
        """In: (x31, x32, x33, x34)  Out: (x21,x22). Per-dist; uses self._dist_idx."""
        x31, x32, x33, x34 = x  # x32: [chunk, nzobj, ndobj] dist-agnostic, x33/x34: [chunk, 2]
        c   = self.cl_shift.coeff_cached(x32)
        x22 = self.cl_shift.curlySc(c, x33, x34)
        return [x31, x22]

    @nvtx.annotate("dF3", color="green")
    def dF3(self, x, y, return_x=True):
        """In: (x31..x34),(y31..y34)  Out: (y31, y22). Per-dist."""
        x31, x32, x33, x34 = x
        y31, y32, y33, y34 = y
        c   = self.cl_shift.coeff_cached(x32)
        c1  = self.cl_shift.coeff_cached(y32)
        y22 = self.cl_shift.dcurlySmc(c, x33, x34, c1, y33, y34)
        if return_x:
            x22 = self.cl_shift.curlySc(c, x33, x34)
            return [x31, x22], [y31, y22]
        return [y31, y22]

    @nvtx.annotate("d2F_dF3", color="purple")
    def d2F_dF3(self, x, y, z, w):
        """In: (x31..x34),(y31..y34),(z31..z34),(w31..w34)  Out: (w31, y22). Per-dist.

        S is linear in the coefficient field, so the second differential is
            d2S[(dc_y, dg_y), (dc_z, dg_z)]
              = d2S/dg2 [dg_y, dg_z]  +  dS/dg[dg_z](dc_y)  +  dS/dg[dg_y](dc_z)
        with g = (r, m) the geometry pair. The kernel's c1/c2 slots are each
        contracted with their OWN direction, so the two mixed terms are obtained
        by passing the coefficients CROSSED: c1 = cz against direction y, and
        c2 = cy against direction z. That keeps it to one launch and is what
        makes the off-diagonal form B(g, e) correct as well as B(g, g).
        """
        x31, x32, x33, x34 = x
        y31, y32, y33, y34 = y
        z31, z32, z33, z34 = z
        w31, w32, w33, w34 = w

        c   = self.cl_shift.coeff_cached(x32)
        cy  = self.cl_shift.coeff_cached(y32)
        cz  = self.cl_shift.coeff_cached(z32)
        y22 = self.cl_shift.d2curlySmc(c, x33, x34,
                                       cz, y33, y34,
                                       cy, z33, z34)

        if w32 is not None or w33 is not None or w34 is not None:
            cw   = (self.cl_shift.coeff_cached(w32) if w32 is not None
                    else self._zero_like_cached(c))
            w33u = w33 if w33 is not None else self._zero_like_cached(x33)
            w34u = w34 if w34 is not None else self._zero_like_cached(x34)
            y22 += self.cl_shift.dcurlySmc(c, x33, x34, cw, w33u, w34u)

        return [w31, y22]

    @nvtx.annotate("gF3", color="green")
    def gF3(self, x, y):
        """In: x(x31..x34), y(y21,y22)  Out: (y31, y32, y33, y34). Per-dist.

        Returns un-coeff'd Deltapsi in y32; caller must sum over distances and apply
        cl_shift.coeff() once after the dist loop. y34 is the gradient with
        respect to the effective demagnification, which gF4 turns into a
        gradient with respect to the shrinkage parameters.
        """
        y21, y22 = y  # y22: [chunk, nz, n]
        x = self.apply_F_from(x, 4)
        x31, x32, x33, x34 = x
        c = self.cl_shift.coeff_cached(x32)
        y32, y33, y34 = self.cl_shift.dcurlySadjmc(c, x33, x34, y22)
        return [y21, y32, y33, y34]

    ####### (x31,x32,x33,x34) = F4(x41,x42,x43,x44)
    #
    # x44 is tp = [[A_y, A_x], [B_y, B_x]] for this distance; with
    # t = theta_idx/(ntheta-1), shrink(t) = A*t + B and
    # demag = (1 + shrink)/norm_magnification. Linear in tp, so d2F4 vanishes
    # and d2F_dF4 hands the level below an all-None w.
    @nvtx.annotate("F4", color="green")
    def F4(self, x):
        """In: (x41, x42, x43, x44)  Out: (x31, x32, x33, x34). Per-dist."""
        x41, x42, x43, x44 = x
        A, B  = x44[0, :], x44[1, :]                       # (2,) each: y, x
        demag = (1.0 + (A[None, :] * self._t_chunk + B[None, :])) \
                / self.norm_magnifications_gpu[self._dist_idx]
        return [x41, x42, x43, demag]

    @nvtx.annotate("dF4", color="green")
    def dF4(self, x, y, return_x=True):
        """In: (x41..x44),(y41..y44)  Out: (y41, y42, y43, ddemag). Per-dist."""
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y
        t  = self._t_chunk
        nm = self.norm_magnifications_gpu[self._dist_idx]
        dA, dB = y44[0, :], y44[1, :]
        ddemag = (dA[None, :] * t + dB[None, :]) / nm
        if return_x:
            A, B  = x44[0, :], x44[1, :]
            demag = (1.0 + (A[None, :] * t + B[None, :])) / nm
            return [x41, x42, x43, demag], [y41, y42, y43, ddemag]
        return [y41, y42, y43, ddemag]

    @nvtx.annotate("d2F_dF4", color="purple")
    def d2F_dF4(self, x, y, z, w):
        """In: (x41..x44),(y41..y44),(z41..z44),(w41..w44)  Out: (w41..w43, d2demag).

        F4 is affine in tp, so the genuine second-order term is zero; the only
        thing to carry down is the level-above correction w, mapped through dF4.
        In the current cascade F4 is the outermost level and w is always None.
        """
        x41, x42, x43, x44 = x
        w41, w42, w43, w44 = w
        if w44 is None:
            return [w41, w42, w43, None]
        t  = self._t_chunk
        nm = self.norm_magnifications_gpu[self._dist_idx]
        dA, dB = w44[0, :], w44[1, :]
        return [w41, w42, w43, (dA[None, :] * t + dB[None, :]) / nm]

    @nvtx.annotate("gF4", color="green")
    def gF4(self, x, y):
        """In: x(x41..x44), y(y21, y32, y33, y34)  Out: (y21, y32, y33, y44).

        Adjoint of dF4: y34 is d/d(demag) over this chunk's angles, and
        demag = (1 + A*t + B)/nm, so dA picks up the t-weighted sum and dB the
        plain one. Summing over theta here is what makes tp a global variable --
        the remaining sum over chunks and ranks is done by the caller.
        """
        y21, y32, y33, y34 = y
        x  = self.apply_F_from(x, 5)          # no-op: F4 is the outermost level
        nm = self.norm_magnifications_gpu[self._dist_idx]
        g_A = cp.sum(y34 * self._t_chunk, axis=0) / nm
        g_B = cp.sum(y34,                 axis=0) / nm
        return [y21, y32, y33, cp.stack([g_A, g_B], axis=0)]

    @timer
    def min(self, prb, obj, pos, proj, tp=None):
        """Value of the functional. `tp` defaults to the current shrinkage
        parameters, so pre-shrinkage four-argument callers keep working."""
        if tp is None:
            tp = self.vars['tp']
        out = cp.zeros(1, dtype="float32")

        for k0, k1 in self._dist_groups():
            nd = k1 - k0
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
            def _min_dists(self, out, c_proj, *a):
                c_pos  = a[0 * nd:1 * nd]
                c_data = a[1 * nd:2 * nd]
                c_mask = a[2 * nd:3 * nd]
                c_t    = a[3 * nd]
                c_prb  = a[3 * nd + 1]
                c_tp   = a[3 * nd + 2]
                self.cl_shift.coeff_cache_reset()
                self._t_chunk = c_t
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._set_mask_chunk(c_mask[j])
                    self.apply_F_cache_reset()
                    x = [c_prb[j], c_proj, c_pos[j], c_tp[j]]
                    y = self.apply_F_from(x, 1)
                    out[:] += self.F0(y, c_data[j])

            ks = range(k0, k1)
            _min_dists(self, out, proj,
                       *[pos[k]          for k in ks],
                       *[self.data[k]    for k in ks],
                       *[self.mask_1d[k] for k in ks],
                       self.t_local,
                       prb[k0:k1], tp[k0:k1])

        out = out[0]
        if self.rank == 0 and hasattr(self, 'cl_prb_term'):
            out += self.cl_prb_term.energy_local(prb)
        if hasattr(self, 'cl_lap_term'):
            out += self.cl_lap_term.energy_local()
        return self.allreduce(np.array(out.get(),dtype='float32'))

    def vis_debug(self, vars, i, writer=None):
        """Per-iter checkpoint write (pos-error plot bundled in)."""
        if writer is None or not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return
        # On a resumed run, i == start_iter would overwrite the checkpoint we
        # just read from; on a fresh run (start_iter == 0) iter 0 is a genuine
        # new state and is saved as checkpoint_0000.
        if i < self.start_iter or (i == self.start_iter and self.start_iter > 0):
            return
        writer.write_checkpoint(
            vars, i, self.norm_const, pos_init=self.pos_init,
            shrink=self._tp_to_shrink_local(vars['tp']),
            shrink_init=self._tp_to_shrink_local(self.tp_init),
            shrink_gt=getattr(self, 'shrink_gt', None))
        self._log_shrink_stats(vars['tp'], i)

    def check_approximation(self, vars, etas, top, bottom, alpha, i, writer=None):
        """Compare the real functional along the descent direction with the
        quadratic model used to pick alpha:
            f_real(t)   = self.min(vars + t*etas)
            f_approx(t) = self.min(vars) - top*t + 0.5*bottom*t**2
        Sampled at npp points in [0, 2*alpha]. With a writer, saves a PNG per
        triggered iteration under writer.path_out/check_approximation/; without
        a writer, shows the plot inline (rank 0 only).
        """
        if not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return

        # lazy-allocate scratch buffers (only created on the first triggered iter)
        if not hasattr(self, '_chk_objt'):
            self._chk_objt  = make_pinned(vars['obj'].shape,  'complex64')
            self._chk_projt = make_pinned(vars['proj'].shape, 'complex64')
            self._chk_prbt  = cp.empty_like(vars['prb'])
            self._chk_post  = cp.empty_like(vars['pos'])
            self._chk_tpt   = cp.empty_like(vars['tp'])

        objt, prbt, post, projt = self._chk_objt, self._chk_prbt, self._chk_post, self._chk_projt
        tpt = self._chk_tpt

        npp = 5
        t = np.linspace(0, 2 * alpha, npp).astype('float32')
        err_real = np.zeros(npp, dtype='float32')

        for k in range(npp):
            self.linear_batch(vars['obj'],  etas['obj'],  1, t[k], out=objt)
            self.linear_batch(vars['prb'],  etas['prb'],  1, t[k], out=prbt)
            self.linear_batch(vars['pos'],  etas['pos'],  1, t[k], out=post)
            self.linear_batch(vars['proj'], etas['proj'], 1, t[k], out=projt)
            self.linear_batch(vars['tp'],   etas['tp'],   1, t[k], out=tpt)
            err_real[k] = self.min(prbt, objt, post, projt, tpt)

        f0 = self.min(vars['prb'], vars['obj'], vars['pos'], vars['proj'],
                      vars['tp'])
        err_approx = f0 - top * t + 0.5 * bottom * t * t

        if self.rank != 0:
            return

        if writer is None:
            mshow_approx(t, err_real, err_approx, True)
            return

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, err_real,   "o-", label="real")
        ax.plot(t, err_approx, "x-", label="approx")
        ax.set_xlabel("t")
        ax.set_ylabel("functional")
        ax.set_title(f"iter {i}: alpha={alpha:.3e}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        out_dir  = os.path.join(writer.path_out, "check_approximation")
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"check_approx_{i:04}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info(f"check_approximation plot → {png_path}")

    def _is_error_iter(self, i):
        """True on the iterations error_debug logs.  _iterate asks this before
        apply_step so the convergence norms are only computed when they will
        actually be written."""
        return self.error_step != -1 and i % self.error_step == 0

    def _record_dobj(self, res):
        """Turn apply_step's local (norm2 of the step, norm2 of obj) into the
        numbers logged.

        obj is z-distributed, so both sums are partial and need one allreduce.
        Inside BH obj carries a 1/norm_const (precalc divides, postcalc
        multiplies back), so the absolute number is scaled back to physical
        object units; the relative one is scale-free either way."""
        d2, o2 = self.allreduce_scalars(*res)
        self.dobj2 = float(self.norm_const) ** 2 * d2
        self.dobj_rel = float(np.sqrt(d2 / o2)) if o2 > 0 else float('nan')

    def error_debug(self, vars, i):
        """Error logging and CSV checkpoint export. i=-1 is the initial-state call from
        BH (before the loop) and is always logged regardless of error_step."""
        if i != -1 and not self._is_error_iter(i):
            return

        t_min = time.time()
        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"],
                       vars["tp"])
        min_time = time.time() - t_min
        if self.rank==0:
            if i==-1:
                logger.warning(f"Initial {err=:1.5e} ")
                # No step has been taken yet, so there is no ||obj^{n+1}-obj^n||.
                self.table.loc[len(self.table)] = [i, err, 0, np.nan, np.nan]
            else:
                ittime = time.time()-self.time_start
                # dobj2 = ||obj^{n+1} - obj^n||_2^2 in physical object units,
                # dobj_rel = ||obj^{n+1} - obj^n|| / ||obj^{n+1}||.  Set by
                # _record_dobj from the update pass just before this call; nan
                # if error_debug was reached some other way (a bare call, or a
                # trial whose apply_step was not asked for them).
                dobj2 = getattr(self, 'dobj2', np.nan)
                dobj_rel = getattr(self, 'dobj_rel', np.nan)
                logger.warning(f"iter={i}: {ittime:.4f}sec (-min={ittime - min_time:.4f}sec) {err=:1.5e} |dobj|^2={dobj2:1.5e} rel={dobj_rel:1.3e}")
                self.table.loc[len(self.table)] = [i, err, ittime, dobj2, dobj_rel]
            self.dobj2 = self.dobj_rel = np.nan
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)
                # ... and a per-level copy.  The levels of a hierarchical run
                # share path_out, so conv.csv only ever holds the last level
                # that ran; conv_bin{bin}.csv keeps every level's history.
                if hasattr(self, 'bin'):
                    self.table.to_csv(f"{self.path_out}/conv_bin{self.bin}.csv",
                                      index=False)

    def gen_data(self, vars, out):
        """Generate synthetic data. Distances run inside the theta-chunk loop,
        ndistchunk per pass, so the vars['proj'] chunk is uploaded once for all
        of them instead of once per distance.

        Writes INTENSITY, K|psi|^2 -- the same convention reader.py now uses and
        the same quantity F0 compares against, blur included.  (It wrote the
        amplitude, and was called gen_sqrt_data, while the misfit was on
        amplitudes.)"""

        self._ensure_tp(vars)
        self.eff_demag[:] = self._eff_demag_from_tp(vars['tp'])
        self._log_shrink_stats(vars['tp'], -1)
        vars["obj"] /= self.norm_const
        self.fwd_tomo(vars["obj"], out=self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])

        for k0, k1 in self._dist_groups():
            nd = k1 - k0
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=nd)
            def _gen_data_dists(self, *a):
                o      = a[0:nd]                        # nd proper outputs
                c_proj = a[nd]
                c_pos  = a[nd + 1 + 0 * nd:nd + 1 + 1 * nd]
                c_t    = a[nd + 1 + 1 * nd]
                c_prb  = a[nd + 2 + 1 * nd]
                c_tp   = a[nd + 3 + 1 * nd]
                self.cl_shift.coeff_cache_reset()
                self._t_chunk = c_t
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self.apply_F_cache_reset()
                    x = [c_prb[j], c_proj, c_pos[j], c_tp[j]]
                    y = self.apply_F_from(x, 1)
                    # K|y|^2, not |y|^2: synthetic data has to carry the same
                    # blur the model applies, or every test against it is vacuous
                    o[j][:] = self._blur(reprod(y, y))

            ks = range(k0, k1)
            _gen_data_dists(self,
                            *[out[k] for k in ks],
                            vars['proj'],
                            *[vars['pos'][k] for k in ks],
                            self.t_local,
                            vars['prb'][k0:k1], vars['tp'][k0:k1])

        vars["obj"] *= self.norm_const




