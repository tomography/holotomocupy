import numpy as np
import cupy as cp
import os
import time
import tifffile
import warnings
import pandas as pd
import nvtx
import cupy.fft

from .propagation import Propagation
from .shift import Shift
from .shift_fft import ShiftFFT
from .chunking import Chunking
from .utils import (make_pinned, mshow, mshow_polar, mshow_pos, redot,
                    reprod, timer, write_tiff)
from .mpi_functions import MPIClass
from .psf import psf_taps, psf_blur
from .logger_config import logger

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=".*peer.*")
cupy.fft.config.get_plan_cache().set_size(0)  # dont waste GPU memory


class RecNFP:
    """Near-field ptychography reconstruction (MPI-parallel over theta).

    Forward model:
        x0 = F1(F2(F3(prb, proj, pos)))
           = D( prb * exp(i * S_pos(proj)) )

    Variables: prb (nz×n, complex), proj (nzobj×nobj, real), pos (ntheta×2)
    Parallelisation: theta distributed across MPI ranks; prb/proj replicated.
    """

    # Variables driven by the BH loop. A subclass with extra variables can
    # override this and inherit BH unchanged.
    _var_names = ("prb", "proj", "pos")

    def __init__(self, args):

        for key, value in vars(args).items():
            setattr(self, key, value)

        self.shift_type = getattr(args, 'shift_type', 'cubic')

        # cascade: F0 ◦ F1 ◦ F2 ◦ F3
        self.F      = [self.F0,      self.F1,      self.F2,      self.F3]
        self.gF     = [self.gF0,     self.gF1,     self.gF2,     self.gF3]
        self.dF     = [self.dF0,     self.dF1,     self.dF2,     self.dF3]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2, self.d2F_dF3]

        multiplier   = 4
        float_item   = np.dtype("float32").itemsize
        complex_item = np.dtype("complex64").itemsize
        # double-buffered data chunks (dominant) + overhead for other proper arrays
        nbytes = int(multiplier * self.nchunk * (self.nz * self.n * float_item + self.nobj * self.nobj * complex_item))

        # MPI: distribute theta; prb/proj replicated on all ranks
        self.cl_mpi       = MPIClass(args.comm, self.nzobj, self.ntheta, self.nobj, 'complex64')
        self.local_ntheta = self.cl_mpi.local_ntheta
        self.rank         = self.cl_mpi.rank
        self.st_theta     = self.cl_mpi.st_theta
        self.end_theta    = self.cl_mpi.end_theta

        if self.rank == 0 and hasattr(self, 'path_out') and self.path_out:
            os.makedirs(os.path.join(self.path_out, 'checkpoints_tiff'), exist_ok=True)
        args.comm.Barrier()

        wavelength    = 1.24e-09 / self.energy
        z1            = self.z1
        z2            = self.focustodetectordistance - z1
        magnification = self.focustodetectordistance / z1
        distance      = z1 * z2 / self.focustodetectordistance
        voxelsize     = self.detector_pixelsize / magnification

        self.rho_sq = {
            'proj': args.rho[0]**2,
            'prb':  args.rho[1]**2,
            'pos':  args.rho[2]**2,
        }

        # args.estimate_rho asks for a coordinate search on rho[prb, pos]
        # before the real loop (estimate_rho_coord).  Defaults keep every
        # caller that predates the knob on the fixed-rho path.
        if not hasattr(self, 'estimate_rho'):
            self.estimate_rho = False
        if not hasattr(self, 'rho_estimate_niter'):
            self.rho_estimate_niter = 16
        if not hasattr(self, 'rho_trial_error_step'):
            self.rho_trial_error_step = -1

        # Detector PSF as ONE Gaussian on the detector
        # INTENSITY (see the F0 block and psf.psf_taps).  sigma is in detector
        # pixels of the grid the data is on -- for NFP that is the unbinned
        # detector, so no bin factor enters.  Absent or 0 -> psf_w is None and
        # _blur is the identity, which is the pre-PSF behaviour exactly.
        self.psf_sigma, self.psf_w = psf_taps(getattr(args, 'psf_sigma', 0.0))
        if self.psf_w is not None and self.rank == 0:
            logger.info(
                f"PSF: one Gaussian on the detector intensity, "
                f"sigma={self.psf_sigma:g} det.px ({self.psf_w.size} taps)")

        # Data-misfit model, 'intensity' (default) or 'amplitude'; orthogonal to
        # psf_sigma, both blur the model intensity with the same K.  See the F0
        # block here and config._parse_model.  Absent key -> 'intensity', so
        # every existing step0 config is unaffected.
        self.model = getattr(args, 'model', 'intensity')
        if self.model not in ('intensity', 'amplitude'):
            raise ValueError(f"model must be 'intensity' or 'amplitude', "
                             f"got {self.model!r}")
        if self.model != 'intensity' and self.rank == 0:
            logger.info(
                "misfit model: AMPLITUDE, 1/N sum (sqrt(K|psi|^2) - sqrt(d))^2"
                " -- err is ~4x smaller than under the intensity model and the"
                " two are NOT comparable")

        self.cl_chunking = Chunking(nbytes, self.nchunk)
        self.cl_prop     = Propagation(self.n, self.nz, self.nchunk, 1, wavelength, voxelsize,
                                       np.array([distance]))
        # nchunk builds the cuFFT plans up front.  The global plan cache is off
        # (set_size(0) above), so without it ShiftFFT re-plans on every call.
        if self.shift_type == 'fft':
            self.cl_shift = ShiftFFT(self.n, self.nobj, self.nz, self.nzobj, self.nchunk)
        elif self.shift_type == 'cubic':
            self.cl_shift = Shift(self.n, self.nobj, self.nz, self.nzobj, self.nchunk)
        else:
            raise ValueError(f"shift_type must be 'cubic' or 'fft', got {self.shift_type!r}")

        self.alloc_arrays()

        self.table = pd.DataFrame(columns=["iter", "err", "time"])

        self.data_size = self.ntheta * self.nz * self.n
        self.prb_size  = self.nz * self.n

        self.gpu_batch    = self.cl_chunking.gpu_batch
        self.redot_batch  = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.mulc_batch   = self.cl_chunking.mulc_batch
        self.allreduce    = self.cl_mpi.allreduce
        self.allreduce2   = self.cl_mpi.allreduce2
        self.allreduce_scalars = self.cl_mpi.allreduce_scalars

    def alloc_arrays(self):
        self.vars = {
            'prb':  cp.empty([self.nz, self.n],       dtype='complex64'),
            'proj': cp.zeros([self.nzobj, self.nobj],  dtype='complex64'),
            'pos':  cp.zeros([self.local_ntheta, 2],   dtype='float32'),
        }
        self.data = make_pinned([self.local_ntheta, self.nz, self.n], dtype='float32')
        self.grads, self.etas = {}, {}
        for ge in self.grads, self.etas:
            ge['prb']  = cp.zeros([self.nz, self.n],      dtype='complex64')
            ge['proj'] = cp.zeros([self.nzobj, self.nobj], dtype='complex64')
            ge['pos']  = cp.zeros([self.local_ntheta, 2],  dtype='float32')

    def BH(self, writer=None):
        vars  = self.vars
        grads = self.grads
        etas  = self.etas

        self.precalc(vars)
        self.error_debug(vars, -1)

        if self.estimate_rho:
            self.estimate_rho_coord(vars, grads, etas,
                                    niter_trial=self.rho_estimate_niter)

        self._iterate(vars, grads, etas, writer)
        return vars

    def _iterate(self, vars, grads, etas, writer=None):
        """Main BH iteration loop. Assumes precalc() has already run."""
        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            with nvtx.annotate(f"::BH:nfp:{i}"):
                self.compute_gradient(vars, grads)
                if getattr(self, 'fused_hessian', True):
                    alpha = self._compute_step_fused(vars, grads, etas, i)
                else:                      # the old three-sweep path
                    self.compute_beta(vars, grads, etas, i)
                    alpha = self.compute_alpha(vars, grads, etas)
                self.apply_step(vars, etas, alpha)
                self.log_iter(vars, i, writer)

    def estimate_rho_coord(self, vars, grads, etas, niter_trial=16, max_extend=8):
        """Coordinate search on rho[prb, pos], as Rec.estimate_rho_coord.

        rho here is [proj, prb, pos] with no `tp`; `proj` is the reference
        scale and is left alone, exactly as `obj` is in the 3-D search.
        Scores come from self.min(), which allreduces, so every rank walks the
        same branches and the search needs no collective of its own.
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
        self.error_step      = int(self.rho_trial_error_step)
        self.checkpoint_step = -1

        def _reset_trial():
            for k, v in vars.items():
                v[:] = snap_vars[k]
            for buf in grads.values(): buf[:] = 0
            for buf in etas.values():  buf[:] = 0
            self.table      = pd.DataFrame(columns=["iter", "err", "time"])
            self.start_iter = 0

        def _run_trial(rho_vec):
            _reset_trial()
            self.rho_sq = {'proj': rho_vec[0]**2, 'prb': rho_vec[1]**2,
                           'pos':  rho_vec[2]**2}
            try:
                self._iterate(vars, grads, etas, writer=None)
                err = float(self.min(vars['prb'], vars['proj'], vars['pos']))
                if not np.isfinite(err):
                    err = float('inf')
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f'rho trial {rho_vec} crashed '
                                   f'({type(e).__name__}: {e}) -> err=inf')
                err = float('inf')
            return err

        # Errors keyed by the full rho vector, shared across coordinates: the
        # winner of prb is exactly the centre probe of pos, so without this the
        # switch re-runs a trial it already has.
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

        base = [float(np.sqrt(self.rho_sq[k])) for k in ('proj', 'prb', 'pos')]
        if self.rank == 0:
            logger.warning(f'estimate_rho_coord: start from {base}, '
                           f'niter_trial={niter_trial}')

        # proj stays at whatever it was; prb and (when free) pos get searched.
        history = {}
        for idx, name in ((1, 'prb'), (2, 'pos')):
            if base[idx] > 0:
                base[idx], history[name] = _coord(base, idx, name, base[idx])
            else:
                history[name] = []      # frozen: nothing to scale

        # Restore state so the outer BH loop starts clean.
        _reset_trial()
        self.table           = snap_table
        self.start_iter      = snap_start_iter
        self.niter           = snap_niter
        self.error_step      = snap_error_step
        self.checkpoint_step = snap_ckpt_step

        self.rho_sq = {'proj': base[0]**2, 'prb': base[1]**2, 'pos': base[2]**2}
        self.rho    = list(base)
        if self.rank == 0:
            logger.warning(f'estimate_rho_coord: final rho = {base}')
        return history

    def precalc(self, vars):
        """One-time setup at the start of BH: snapshot initial positions."""
        self.pos_init = vars['pos'].copy()

    def compute_gradient(self, vars, grads):
        """Gradients + per-variable rho_sq scaling (NFP variant: scaling lives in BH,
        not inside gF cascade, unlike rec_mpi.py)."""
        with nvtx.annotate("gradients"):
            self.gradients(vars, grads)
        for v in self._var_names:
            self.mulc_batch(grads[v], grads[v], self.rho_sq[v])

    def compute_beta(self, vars, grads, etas, i):
        """Update etas in place: first iter is pure steepest descent (etas = -grads);
        subsequent iters apply etas = beta*etas - grads with the CG coefficient."""
        if i == self.start_iter:
            for v in self._var_names:
                self.mulc_batch(etas[v], grads[v], -1)
            return
        with nvtx.annotate(":::BH:calc beta"):
            top, bottom = self.allreduce2(
                self.hessian(vars, grads, etas),
                self.hessian(vars, etas,  etas),
            )
            beta = top / bottom
            for v in self._var_names:
                self.linear_batch(etas[v], grads[v], beta, -1)

    def compute_alpha(self, vars, grads, etas):
        """Step size: alpha = top / bottom with top = -<grad, eta>/rho_sq (probe & proj
        contributions only on rank 0 since they're replicated), bottom = <eta, H·eta>."""
        with nvtx.annotate(":::BH:calc_alpha"):
            top = -self.redot_batch(grads['pos'], etas['pos']) / self.rho_sq['pos']
            if self.rank == 0:
                for v in self._var_names:
                    if v == 'pos':
                        continue
                    top -= self.redot_batch(grads[v], etas[v]) / self.rho_sq[v]
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            return top / bottom

    def apply_step(self, vars, etas, alpha):
        """var ← var + alpha·eta for every variable."""
        for v in self._var_names:
            self.linear_batch(vars[v], etas[v], 1, alpha)

    def log_iter(self, vars, i, writer):
        """Error + checkpoint hooks for this iter."""
        with nvtx.annotate(":::BH:calc error", color='gray'):
            self.error_debug(vars, i)
        with nvtx.annotate(":::BH:vis_debug", color='gray'):
            self.vis_debug(vars, i, writer)

    def hessian(self, vars, grads, etas):
        return self.hessian_cascade(vars, grads, etas)

    @timer
    def hessian_cascade(self, vars, grads, etas):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade(
            self, out, d,
            x2, y2, z2,   # pos  — proper (theta-distributed)
            x0, y0, z0,   # prb  — non-proper gpu
            x1, y1, z1,   # proj — non-proper gpu
        ):
            self.cl_shift.coeff_cache_reset()
            x = [x0, x1, x2]
            y = [y0, y1, y2]
            z = [z0, z1, z2]
            w = [None, None, None]
            y_is_z = y[0] is z[0]

            for id in range(1, len(self.F))[::-1]:
                w = self.d2F_dF[id](x, y, z, w)
                fx, y = self.dF[id](x, y)
                if y_is_z:
                    z = y
                else:
                    z = self.dF[id](x, z, return_x=False)
                x = fx

            out[:] += self.d2F_dF[0](x, y, z, w, d)

        _hessian_cascade(
            self, out, self.data,
            vars["pos"],  grads["pos"],  etas["pos"],
            vars["prb"],  grads["prb"],  etas["prb"],
            vars["proj"], grads["proj"], etas["proj"],
        )
        return out[0].get()

    def hessian3(self, vars, grads, etas):
        """{B(g,g), B(g,e), B(e,e)} from ONE cascade sweep.

        The three bilinear forms the step needs, instead of three separate
        `hessian` calls.  The x-chain -- which streams the data and the three
        proj/prb chunks and is essentially all of the cost -- is advanced once
        and shared.  NFP has no probe-fit or Laplacian term, so this is the
        whole Hessian.
        """
        return self.hessian_cascade3(vars, grads, etas)

    @timer
    def hessian_cascade3(self, vars, grads, etas):
        out = cp.zeros(3, dtype="float32")
        # Chunking treats a non-proper output by its axis-0 length, so 3 must
        # not be the chunk's theta count.
        assert self.local_ntheta != 3, "local_ntheta==3 aliases the accumulator"

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade3(
            self, out, d,
            x2, g2, e2,   # pos  -- proper (theta-distributed)
            x0, g0, e0,   # prb  -- non-proper gpu
            x1, g1, e1,   # proj -- non-proper gpu
        ):
            self.cl_shift.coeff_cache_reset()
            x = [x0, x1, x2]
            g = [g0, g1, g2]
            e = [e0, e1, e2]
            # Passing the SAME list object on the diagonal forms keeps the
            # `y is z` fast paths in d2F_dF1/d2F_dF3 alive.
            wgg = [None, None, None]
            wge = [None, None, None]
            wee = [None, None, None]
            for id in range(1, len(self.F))[::-1]:
                # all three contractions before dF advances the chains: they
                # must see the pre-update x, g, e
                wgg = self.d2F_dF[id](x, g, g, wgg)
                wge = self.d2F_dF[id](x, g, e, wge)
                wee = self.d2F_dF[id](x, e, e, wee)
                fx, gn = self.dF[id](x, g)
                en = self.dF[id](x, e, return_x=False)
                x, g, e = fx, gn, en
            out[0:1] += self.d2F_dF[0](x, g, g, wgg, d)
            out[1:2] += self.d2F_dF[0](x, g, e, wge, d)
            out[2:3] += self.d2F_dF[0](x, e, e, wee, d)

        _hessian_cascade3(
            self, out, self.data,
            vars["pos"],  grads["pos"],  etas["pos"],
            vars["prb"],  grads["prb"],  etas["prb"],
            vars["proj"], grads["proj"], etas["proj"],
        )
        h = out.get()
        return float(h[0]), float(h[1]), float(h[2])

    def _compute_step_fused(self, vars, grads, etas, i):
        """beta, the etas update and alpha from ONE sweep instead of three.

        Same algebra as the three-sweep path: with eta_new = beta*eta - g,
            B(eta_new, eta_new) = beta^2 B(e,e) - 2 beta B(g,e) + B(g,g)
        so the two ratios come out of {Qgg, Bge, Qee}.  `check_fused_hessian`
        re-measures both the classic way and logs the difference.
        """
        check = getattr(self, 'check_fused_hessian', False)
        Qgg, Bge, Qee = self.allreduce_scalars(*self.hessian3(vars, grads, etas))

        # First iteration is steepest descent: etas is zero, so the sweep
        # measured Bge = Qee = 0 and only the ratio needs the special case.
        #
        # powell_restart floors beta at 0 (Bge < 0 means the new gradient
        # couples negatively to the old direction, so beta*e points away from
        # descent and dropping it is a valid CG restart).  OFF by default here:
        # it is a change of ALGORITHM, not of cost, and beta really does come
        # out negative on most iterations of this problem, so leaving it off
        # keeps the fused path arithmetically identical to the three-sweep one.
        beta = 0.0 if i == self.start_iter else Bge / Qee
        if getattr(self, 'powell_restart', False):
            beta = max(beta, 0.0)
        if check and i > self.start_iter:
            t, b = self.allreduce2(self.hessian(vars, grads, etas),
                                   self.hessian(vars, etas, etas))
            self._log_fused_check(i, "beta", beta, t / b)

        # etas <- beta*etas - grads, and the alpha numerator with it.  Must run
        # AFTER the sweep: it overwrites the direction the sweep just read.
        top = 0.0
        for v in self._var_names:
            self.linear_batch(etas[v], grads[v], beta, -1)
            if v == 'pos' or self.rank == 0:      # prb/proj are replicated
                top -= self.redot_batch(grads[v], etas[v]) / self.rho_sq[v]
        top, = self.allreduce_scalars(top)

        bottom = beta * beta * Qee - 2.0 * beta * Bge + Qgg
        scale = abs(beta * beta * Qee) + abs(2.0 * beta * Bge) + abs(Qgg)
        if check:
            ref = self.allreduce_scalars(self.hessian(vars, etas, etas))[0]
            self._log_fused_check(i, "bottom", bottom, ref)
        if not bottom > 1e-6 * scale:
            # Concave or flat along eta: top/bottom would be a negative alpha
            # and the update would walk uphill.  top = <-grad, eta> > 0 for any
            # descent direction, so |bottom| keeps the step downhill.
            logger.warning(
                f"iter={i}: non-convex alpha denominator bottom={bottom:.6e} "
                f"from Qgg={Qgg:.6e} Bge={Bge:.6e} Qee={Qee:.6e} beta={beta:.6e}"
                f" -- stepping along |bottom| instead of taking alpha<0")
            bottom = abs(bottom) if bottom != 0.0 else 1e-6 * scale
        return top / bottom

    def _log_fused_check(self, i, name, got, ref):
        rel = abs(got - ref) / abs(ref) if ref != 0 else abs(got)
        logger.info(f"iter={i}: fused-check {name:>6}  fused={got:+.9e}  "
                    f"measured={ref:+.9e}  rel={rel:.3e}")

    def gradients(self, vars, grads):
        self.gradients_cascade(vars, grads)
        grads['prb'][:]  = cp.array(self.allreduce(grads['prb'].get()))
        grads['proj'][:] = cp.array(self.allreduce(grads['proj'].get()))

    @timer
    def gradients_cascade(self, vars, grads):
        grads['prb'][:]  = 0
        grads['proj'][:] = 0

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _gradients_cascade(self, gradpos, gradprb, gradproj, d, pos, prb, proj):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, pos]
            y = d
            for id in range(len(self.gF)):
                y = self.gF[id](x, y)
            gradprb[:]  += y[0]
            gradproj[:] += y[1]
            gradpos[:]   = y[2]

        _gradients_cascade(
            self, grads['pos'], grads['prb'], grads['proj'],
            self.data, vars['pos'], vars['prb'], vars['proj'],
        )

    ####################### Cascade functions #######################
    # Variables: x = [prb, proj, pos]
    # F3: (prb, proj, pos) → (prb, S_pos(proj))
    # F2: (prb, shifted_proj) → (prb, exp(i·shifted_proj))
    # F1: (prb, exp_proj) → D(prb · exp_proj)
    # F0: ||K|·|² - d||²  (INTENSITY, PSF-blurred; see the F0 block)
    #################################################################

    def apply_F_from(self, x, from_level):
        """Apply F[from_level], F[from_level+1], ..., F[len(F)-1] inside-out, returning
        the partial cascade value at level (from_level-1). Used by gF* methods that need
        to fast-forward x through cascade levels above their own."""
        for k in range(from_level, len(self.F))[::-1]:
            x = self.F[k](x)
        return x

    ####### F0(x0) = 1/n \|g(K|x0|^2, d)\|_2^2
    #
    # TWO MISFIT MODELS, selected by self.model, exactly as in rec_mpi.Rec.F0
    # (read the long comment there; this is the same thing with W = 1).  With
    # I = |x|^2, J = K I the blurred MODEL intensity, a = sqrt(J) and
    # s = sqrt(d):
    #
    #   intensity   F0 = 1/N sum (J - d)^2
    #   amplitude   F0 = 1/N sum (a - s)^2
    #
    # PSF WORKS IN BOTH and sits in the same place in both: the detector PSF
    # from the finite focal spot and the detector PSF blur INTENSITY,
    # incoherently, so neither can be folded into cl_prop.D (which is coherent)
    # and their only legal home is on I.  Both are Gaussian, so they are carried
    # as ONE Gaussian K (see psf.psf_blur; it is self-adjoint, so K^T is
    # literally K), the identity when psf_sigma is unset.  The amplitude model
    # then compares sqrt(K I) against sqrt(d) -- the blur stays on intensity and
    # only the COMPARISON moves -- which at psf_sigma = 0 is exactly the misfit
    # the step0 drivers used before the move to intensity.
    #
    # d IS THE MEASURED INTENSITY IN BOTH, not its square root.  The step0.py
    # drivers used to take the sqrt on load, back when the only misfit was the
    # amplitude one; they no longer do, and gen_data generates intensity to
    # match.  Nothing squares d, and the amplitude path takes its own sqrt here.
    #
    # There is no detector mask here, unlike rec_mpi: NFP never shifts the
    # object off its own grid, so every detector pixel is in range and W = 1.
    #
    # ONE SET OF DERIVATIVES COVERS BOTH.  Writing the misfit as 1/N sum g(J),
    # with u = Re(conj(x) z), v = Re(conj(x) y) and dJ[z] = 2 K(u):
    #
    #   dF0[y]    = 4/N sum p v                          p = K[g'/2]
    #   d2F0[y,z] = 4/N sum { 2 t v + p Re(conj(z) y) }  t = K[(g''/2) K u]
    #                                 (+ 4/N sum p Re(conj(x) w))
    #
    #                g               g'/2 (_pw)        g''/2 (_cw factor)
    #   intensity    (J - d)^2       J - d              1
    #   amplitude    (a - s)^2       (a - s)/(2a)       s/(4 a^3)
    #
    # The amplitude row divides by a and a^3 -- which is why this misfit was
    # dropped in the first place -- so both denominators are floored at
    # _AMP_FLOOR; F0 itself stays exact.  The intensity row divides by nothing
    # and needs no floor.  g''/2 >= 0 for both, so the t term is a genuine
    # square in the quadratic form and all the indefiniteness is in p, which is
    # proportional to the residual and vanishes at the solution.  Both Hessian
    # terms are manifestly symmetric in y,z.  The Hessian needs TWO nested
    # convolutions; it is not a substitution into a pointwise kernel.
    #
    # err is NOT comparable across the two models (amplitude is ~4x smaller with
    # flat-field-normalised data), and neither is it comparable with runs made
    # before the move to intensity.

    # Floor on the model amplitude a = sqrt(K|x|^2) in the amplitude model's two
    # denominators; see the same constant in rec_mpi.Rec.
    _AMP_FLOOR = np.float32(1e-3)

    def _blur(self, x):
        """K x, the one Gaussian on the detector intensity, identity if unset.

        K^T = K (see psf.psf_blur), so this one call serves both directions."""
        return x if self.psf_w is None else psf_blur(x, self.psf_w)

    @staticmethod
    @cp.fuse()
    def _F0_fused(J, d):
        t = J - d
        return t * t

    @staticmethod
    @cp.fuse()
    def _F0_fused_amp(J, d):
        t = cp.sqrt(J) - cp.sqrt(d)
        return t * t

    @staticmethod
    @cp.fuse()
    def _r_fused_amp(J, d, floor):
        a = cp.maximum(cp.sqrt(J), floor)
        return 0.5 * (1.0 - cp.sqrt(d) / a)

    @staticmethod
    @cp.fuse()
    def _c_fused_amp(v, J, d, floor):
        a = cp.maximum(cp.sqrt(J), floor)
        return (0.25 * cp.sqrt(d) / (a * a * a)) * v

    def _pw(self, J, d):
        """g'(J)/2, the pre-blur residual weight: (J - d) for intensity,
        (a - sqrt(d))/(2a) for amplitude."""
        if self.model == 'amplitude':
            return self._r_fused_amp(J, d, self._AMP_FLOOR)
        return J - d

    def _cw(self, v, J, d):
        """g''(J)/2 * v: v itself for intensity, sqrt(d)/(4a^3) * v for
        amplitude."""
        if self.model == 'amplitude':
            return self._c_fused_amp(v, J, d, self._AMP_FLOOR)
        return v

    def _F0_p(self, x, d):
        """p = K[g'(K|x|^2)/2], the pointwise weight both derivatives use."""
        return self._blur(self._pw(self._blur(reprod(x, x)), d))

    def F0(self, x, d):
        J = self._blur(reprod(x, x))
        kern = (self._F0_fused_amp if self.model == 'amplitude'
                else self._F0_fused)
        return 1 / self.data_size * cp.sum(kern(J, d))

    def dF0(self, x, y, d, return_x=False):
        return 4 / self.data_size * redot(self._F0_p(x, d) * x, y)

    @staticmethod
    @cp.fuse()
    def _d2F_dF0_fused(t, p, x, y, z):
        return 2 * t * reprod(x, y) + p * reprod(z, y)

    def d2F_dF0(self, x, y, z, w, d):
        # J is kept live: the amplitude curvature weight needs it as well as p
        J = self._blur(reprod(x, x))
        p = self._blur(self._pw(J, d))
        # the two nested convolutions (K^T = K, so both are the same call)
        t = self._blur(self._cw(self._blur(reprod(x, z)), J, d))
        v = self._d2F_dF0_fused(t, p, x, y, z)
        if w is not None:
            v += p * reprod(x, w)
        return 4 / self.data_size * cp.sum(v)

    @staticmethod
    @cp.fuse()
    def _gF0_fused(x, p, scale):
        return (scale * p) * x

    def gF0(self, x, y):
        """In: x, y = the data d."""
        x = self.apply_F_from(x, 1)
        return self._gF0_fused(x, self._F0_p(x, y),
                               np.float32(4 / self.data_size))

    ####### F1: (prb, exp_proj) → D(prb · exp_proj)
    def F1(self, x):
        x11, x12 = x
        return self.cl_prop.D(x11 * x12, 0)

    def dF1(self, x, y, return_x=True):
        x11, x12 = x
        y11, y12 = y
        y0 = self.cl_prop.D(y11 * x12 + x11 * y12, 0)
        if return_x:
            return self.cl_prop.D(x11 * x12, 0), y0
        return y0

    def d2F_dF1(self, x, y, z, w):
        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        w11, w12 = w
        if y12 is z12:
            y0 = 2 * y11 * y12
        else:
            y0 = y11 * z12 + z11 * y12
        if w11 is not None:
            y0 = y0 + w11 * x12
        if w12 is not None:
            y0 = y0 + x11 * w12
        return self.cl_prop.D(y0, 0)

    def gF1(self, x, y):
        y0 = y
        x = self.apply_F_from(x, 2)
        x11, x12 = x
        y12 = self.cl_prop.DT(y0, 0)
        y11 = cp.sum(y12 * cp.conj(x12), axis=0)  # sum over theta → (nz, n)
        y12 = y12 * cp.conj(x11)
        return y11, y12

    ####### F2: (prb, shifted_proj) → (prb, exp(i·shifted_proj))
    @staticmethod
    @cp.fuse()
    def _F2_fused(x22):
        return cp.exp(1j * x22)

    def F2(self, x):
        x21, x22 = x
        return x21, self._F2_fused(x22)

    @staticmethod
    @cp.fuse()
    def _dF2_fused(x22, y22):
        e = cp.exp(1j * x22)
        return e, e * 1j * y22

    def dF2(self, x, y, return_x=True):
        x21, x22 = x
        y21, y22 = y
        x12, y12 = self._dF2_fused(x22, y22)
        return ([x21, x12], [y21, y12]) if return_x else [y21, y12]

    @staticmethod
    @cp.fuse()
    def _d2F_dF2_fused(x22, y22, z22, w22):
        e = cp.exp(1j * x22)
        r = e * (-y22 * z22)
        if w22 is not None:
            r = r + e * 1j * w22
        return r

    def d2F_dF2(self, x, y, z, w):
        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        w21, w22 = w
        return [w21, self._d2F_dF2_fused(x22, y22, z22, w22)]

    @staticmethod
    @cp.fuse()
    def _gF2_fused(x22, y12):
        return (-1j) * y12 * cp.conj(cp.exp(1j * x22))

    def gF2(self, x, y):
        y11, y12 = y
        x = self.apply_F_from(x, 3)
        x21, x22 = x
        y22 = self._gF2_fused(x22, y12)
        return [y11, y22]

    ####### F3: (prb, proj, pos) → (prb, S_pos(proj))
    # coeff(x32) is cached within a chunk; callers (gradients_cascade / hessian_cascade
    # closures) MUST invoke cl_shift.coeff_cache_reset() at chunk boundaries since
    # id(arr) values are reused once an earlier array is GC'd.
    #
    # m comes from cl_shift.unit_mag: NFP never magnifies, and the operator-owned
    # buffer lets ShiftFFT skip the device sync its magnification test would cost.

    def _tiled_coeff(self, psi, n):
        """Cached coeff(psi) broadcast to [n, ...] for the per-theta shift kernels."""
        return cp.tile(self.cl_shift.coeff_cached(psi)[None], [n, 1, 1])

    def F3(self, x):
        x31, x32, x33 = x
        n = len(x33)
        c = self._tiled_coeff(x32, n)
        m = self.cl_shift.unit_mag(n)
        return x31, self.cl_shift.curlySc(c, x33, m)

    def dF3(self, x, y, return_x=True):
        x31, x32, x33 = x
        y31, y32, y33 = y
        n  = len(x33)
        c  = self._tiled_coeff(x32, n)
        c1 = self._tiled_coeff(y32, n)
        m = self.cl_shift.unit_mag(n)
        y22 = self.cl_shift.dcurlySc(c, x33, m, c1, y33)
        if return_x:
            x22 = self.cl_shift.curlySc(c, x33, m)
            return [x31, x22], [y31, y22]
        return [y31, y22]

    def d2F_dF3(self, x, y, z, w):
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        w31, w32, w33 = w
        n  = len(x33)
        c  = self._tiled_coeff(x32, n)
        cy = self._tiled_coeff(y32, n)
        cz = self._tiled_coeff(z32, n)
        m = self.cl_shift.unit_mag(n)
        # Coefficients passed CROSSED: the fused kernel contracts each
        # coefficient with the shift in its own slot, so the mixed second
        # differential needs c_z against dr_y and c_y against dr_z. See the
        # slot-pairing note in Shift.d2curlySc; d2F_dF1 above does the same
        # crossing for the bilinear a*b. Same as Rec.d2F_dF3 in rec_mpi.py.
        y22 = self.cl_shift.d2curlySc(c, x33, m, cz, y33, cy, z33)
        if w32 is not None:
            cw = self._tiled_coeff(w32, n)
            y22 = y22 + self.cl_shift.dcurlySc(c, x33, m, cw, w33)
        return [w31, y22]

    def gF3(self, x, y):
        y21, y22 = y
        x = self.apply_F_from(x, 4)
        x31, x32, x33 = x
        n = len(x33)
        c = self._tiled_coeff(x32, n)
        m = self.cl_shift.unit_mag(n)
        Deltapsi, y33 = self.cl_shift.dcurlySadjc(c, x33, m, y22)
        y32 = cp.zeros([self.nzobj, self.nobj], dtype='complex64')
        y32[:] = cp.sum(Deltapsi, axis=0)
        y32[:] = self.cl_shift.coeff(y32)
        return [y21, y32, y33]

    @timer
    def min(self, prb, proj, pos):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, pos, data, prb, proj):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, pos]
            y = self.apply_F_from(x, 1)
            out[:] += self.F0(y, data)

        _min(self, out, pos, self.data, prb, proj)
        return float(self.allreduce(np.array([out[0].get()], dtype='float32'))[0])

    def vis_debug(self, vars, i, writer=None):
        if not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return
        if writer is not None:
            if i > self.start_iter:
                writer.write_checkpoint(vars, i)
            if self.rank == 0 and hasattr(self, 'path_out') and self.path_out:
                tiff_dir  = os.path.join(self.path_out, 'checkpoints_tiff')
                tiff_path = os.path.join(tiff_dir, f'checkpoint_{i:04}_proj_re.tiff')
                tifffile.imwrite(tiff_path, cp.asnumpy(vars['proj'].real))
                logger.info(f"NFP: proj_re TIFF saved → {tiff_path}")
        elif self.rank == 0:
            if hasattr(self, 'path_out'):
                tiff_dir = os.path.join(self.path_out, 'checkpoints_tiff')
                logger.info(f"Saving iter {i}: proj, prb to {tiff_dir}")
                write_tiff(vars['proj'].real,     f'{tiff_dir}/proj{i:04}')
                write_tiff(cp.angle(vars['prb']), f'{tiff_dir}/prb{i:04}')
                np.save(f'{tiff_dir}/prb{i:04}.npy', vars['prb'].get())
            else:
                mshow(vars['proj'].real, True)
                mshow_polar(vars['prb'], True)
                mshow_pos(vars['pos'] - self.pos_init, True)

    def error_debug(self, vars, i):
        """i=-1 is the initial-state call from BH (before the loop) and is always logged
        regardless of error_step."""
        if i != -1 and not (i % self.error_step == 0 and self.error_step != -1):
            return
        err = self.min(vars['prb'], vars['proj'], vars['pos'])

        # Gather position errors from all ranks to rank 0
        pos_err = (vars['pos'] - self.pos_init).get()   # [local_ntheta, 2]
        all_pos_err = self.cl_mpi.comm.gather(pos_err, root=0)

        if self.rank == 0:
            if i == -1:
                logger.warning(f"Initial {err=:1.5e}")
                self.table.loc[len(self.table)] = [i, err, 0]
            else:
                ittime = time.time() - self.time_start
                logger.warning(f"iter={i}: {ittime:.4f}sec {err=:1.5e}")
                self.table.loc[len(self.table)] = [i, err, ittime]
            pos_err_all = np.concatenate(all_pos_err, axis=0)
            logger.warning(f"  pos err y: {np.array2string(pos_err_all[:, 0], precision=4, separator=', ')}")
            logger.warning(f"  pos err x: {np.array2string(pos_err_all[:, 1], precision=4, separator=', ')}")
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv_nfp.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)

    def gen_data(self, vars, out):
        """Generate synthetic data.

        Writes INTENSITY, K|psi|^2 -- the same quantity F0 compares against,
        blur included, so a test against it is not vacuous.  (It wrote the
        amplitude, and was called gen_sqrt_data, while the misfit was on
        amplitudes.)"""
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, out, pos, prb, proj):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, pos]
            y = self.apply_F_from(x, 1)
            out[:] = self._blur(reprod(y, y))
        _gen_data(self, out, vars['pos'], vars['prb'], vars['proj'])
