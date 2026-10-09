import math
import cupy as cp
import cupyx.scipy.fft as cufft
from .cuda_kernels import (
    s_kernel,
    sback_kernel,
    ds_kernel,
    d2s_kernel,
    dsadj_kernel,
    dsm_kernel,
    d2sm_kernel,
    dsmadj_kernel,
)
from .utils import redot


def _ascontig(x):
    """cp.ascontiguousarray that also accepts numpy/pinned input (auto-uploads).
    Newer cupy rejects numpy inputs to ascontiguousarray; we go through cp.asarray
    first so callers can pass pinned-numpy slices (e.g. vars['pos'][:, k]) directly."""
    return cp.ascontiguousarray(cp.asarray(x))


class Shift():
    """Cubic B-spline shift operator (requires coeff() prefilter, C2 smooth)."""

    def __init__(self, n, npsi, nz, nzpsi, nchunk=None):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi

        # Forward B-spline denominator (unit magnification, k=0,1)
        x = cp.linspace(-1/2, 1/2 - 1/npsi,  npsi ).astype('float32')
        y = cp.linspace(-1/2, 1/2 - 1/nzpsi, nzpsi).astype('float32')
        divx = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * x)).astype('float32')
        divy = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * y)).astype('float32')
        self.ifB3 = 1 / cp.fft.fftshift(cp.outer(divy, divx), axes=(-1, -2))

        # Same denominator on the small (nz x n) detector grid, for coeff_back().
        xb = cp.linspace(-1/2, 1/2 - 1/n,  n ).astype('float32')
        yb = cp.linspace(-1/2, 1/2 - 1/nz, nz).astype('float32')
        divxb = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * xb)).astype('float32')
        divyb = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * yb)).astype('float32')
        self.ifB3_back = 1 / cp.fft.fftshift(cp.outer(divyb, divxb), axes=(-1, -2))

        # cuFFT C2C plans for coeff(), memoized by input shape. The global cupy
        # plan cache is disabled (rec_mpi sets its size to 0), so without this every
        # short trailing chunk would rebuild a plan + work area. The shape set is
        # tiny and fixed, and every extra plan is smaller than the nchunk one.
        self._coeff_plans = {}
        if nchunk is not None:
            self._coeff_plan((nchunk, nzpsi, npsi))   # warm up the common shape

        self._coeff_cache  = {}
        self._coeff_hits   = 0
        self._coeff_misses = 0
        self._unit_mag     = {}      # ntheta -> owned ones((ntheta, 2))

        self._sk       = s_kernel
        self._dsk      = ds_kernel
        self._d2sk     = d2s_kernel
        self._dsadjk   = dsadj_kernel
        self._dsmk     = dsm_kernel
        self._d2smk    = d2sm_kernel
        self._dsmadjk  = dsmadj_kernel

    # ------------------------------------------------------------------
    # B-spline basis
    # ------------------------------------------------------------------

    def phi(self, t):
        return (
            (-2 < t) * (t <= -1) * (t + 2)**3
            + (-1 < t) * (t <=  1) * (4 - 6*t**2 + 3*t**3 * cp.sign(t))
            + ( 1 < t) * (t <=  2) * (2 - t)**3
        )

    # ------------------------------------------------------------------
    # Internal kernel launcher — eliminates repeated if/else dispatch
    # ------------------------------------------------------------------

    def _launch(self, kernel, ntheta, args):
        grid = (math.ceil(self.n / 16), math.ceil(self.nz / 16), ntheta)
        kernel(grid, (16, 16, 1), args)

    # ------------------------------------------------------------------
    # B-spline coefficient computation
    # ------------------------------------------------------------------

    def _coeff_plan(self, shape):
        """cuFFT C2C plan for a `shape` transform over the last two axes, memoized."""
        plan = self._coeff_plans.get(shape)
        if plan is None:
            _tmp = cp.empty(shape, dtype='complex64')
            plan = cufft.get_fft_plan(_tmp, axes=(-2, -1), value_type='C2C')
            self._coeff_plans[shape] = plan
            del _tmp
        return plan

    def coeff(self, psi):
        """B-spline prefilter  ifft2(fft2(psi) * ifB3).

        One object-plane temporary instead of three: fft2 allocates it, the ifB3
        multiply is in place, and ifft2 transforms in place -- the same
        overwrite_x pattern already used in Tomo.R/RT.
        """
        with self._coeff_plan(tuple(psi.shape)):
            out = cufft.fft2(psi)
            out *= self.ifB3
            out = cufft.ifft2(out, overwrite_x=True)
        return out

    def coeff_cached(self, psi):
        """coeff(psi) memoized by id(psi). MUST be paired with explicit
        coeff_cache_reset() at safe lifetime boundaries (e.g., per chunk):
        id() values can be reused across distinct Python objects once an
        earlier one is garbage-collected. Hit/miss counters are exposed for
        verification; reset along with the cache."""
        key = id(psi)
        cached = self._coeff_cache.get(key)
        if cached is None:
            self._coeff_misses += 1
            cached = self.coeff(psi)
            self._coeff_cache[key] = cached
        else:
            self._coeff_hits += 1
        return cached

    def coeff_cache_reset(self):
        self._coeff_cache = {}

    def coeff_cache_stats(self, reset=False):
        """Return (hits, misses) accumulated since the last stats reset.
        Set reset=True to zero the counters."""
        stats = (self._coeff_hits, self._coeff_misses)
        if reset:
            self._coeff_hits = 0
            self._coeff_misses = 0
        return stats

    def unit_mag(self, ntheta):
        """Cached ones((ntheta, 2)) for callers that never magnify.

        Same API as ShiftFFT.unit_mag, so a caller can build m without knowing
        which operator it has.  Here it only saves the per-call allocation;
        this class has no magnification branch to short-circuit.
        """
        m = self._unit_mag.get(ntheta)
        if m is None:
            m = cp.ones((ntheta, 2), dtype='float32')
            self._unit_mag[ntheta] = m
        return m

    # ------------------------------------------------------------------
    # Forward / adjoint shift  S / S*
    # ------------------------------------------------------------------

    def S(self, c, r, m):
        ntheta = c.shape[0]
        # Kernel writes every (t, k, i) — no need to zero first.
        spsi = cp.empty([ntheta, self.nz, self.n], dtype='complex64')
        c = _ascontig(c)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def Sadj(self, spsi, r, m):
        ntheta = spsi.shape[0]
        c = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        spsi = _ascontig(spsi)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 1))
        return c

    # ------------------------------------------------------------------
    # Composed operator  curlyS = S ∘ coeff
    # ------------------------------------------------------------------

    def curlyS(self, psi, r, m):
        return self.S(self.coeff(psi), r, m)

    # ------------------------------------------------------------------
    # Back-projection shift for Paganin initial guess
    # ------------------------------------------------------------------

    def coeff_back(self, psi):
        """B-spline prefilter on the small (nz x n) input grid for back-interpolation.

        Mirrors coeff(): the denominator is built once in __init__ (it depends
        only on nz, n) and the transform reuses a memoized plan and a single
        temporary instead of allocating three per call.
        """
        with self._coeff_plan(tuple(psi.shape)):
            out = cufft.fft2(psi)
            out *= self.ifB3_back
            out = cufft.ifft2(out, overwrite_x=True)
        return out

    def Sback(self, c, r, m):
        """Gather-interpolate from small (n x nz) B-spline coefficients to large (npsi x nzpsi) grid."""
        ntheta = c.shape[0]
        # Kernel writes every (t, k, i) — no zero-fill needed.
        g    = cp.empty([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        c    = _ascontig(c)
        r    = _ascontig(r)
        m    = _ascontig(cp.asarray(m, dtype='float32'))
        sback_kernel(
            (math.ceil(self.npsi / 32), math.ceil(self.nzpsi / 32), ntheta),
            (32, 32, 1),
            (g, c, r, m,
             self.n, self.npsi, self.nz, self.nzpsi, ntheta),
        )
        return g

    def curlySback(self, psi, r, m):
        """Interpolate from small (n x nz) grid to large (npsi x nzpsi) with shift+magnification."""
        return self.Sback(self.coeff_back(psi), r, m)

    # ------------------------------------------------------------------
    # Optimized coefficient-space variants  (operate on pre-computed coefficients)
    # ------------------------------------------------------------------

    def curlySc(self, c, r, m):
        ntheta = c.shape[0]
        spsi = cp.empty([ntheta, self.nz, self.n], dtype='complex64')
        c = _ascontig(c)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def dcurlySc(self, c, r, m, c1, Deltar):
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], 'complex64')
        c       = _ascontig(c)
        c1      = _ascontig(c1)
        r       = _ascontig(r)
        Deltar  = _ascontig(Deltar)
        m       = _ascontig(m)

        self._launch(self._dsk, ntheta,
                     (res, c, c1, r, m, Deltar,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        
        return res
        
    def dcurlySadjc(self, c, r, m, Deltaphi):
        
        ntheta = c.shape[0]
        # out1 is an atomicAdd target -> MUST be zeroed. dt1/dt2 are written every
        # position by the kernel; out2 is overwritten by the redot() lines below.
        out1 = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        out2  = cp.empty(r.shape, dtype='float32')
        dt1  = cp.empty(Deltaphi.shape, 'complex64')
        dt2  = cp.empty(Deltaphi.shape, 'complex64')
        c        = _ascontig(c)
        r        = _ascontig(r)
        Deltaphi = _ascontig(Deltaphi)
        m        = _ascontig(m)

        self._launch(self._dsadjk, ntheta,
                     (out1, dt1, dt2, c, Deltaphi, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        out2[:, 0] = redot(Deltaphi, dt1, axis=(1, 2))
        out2[:, 1] = redot(Deltaphi, dt2, axis=(1, 2))
        
        return [out1, out2]

    def d2curlySc(self, c, r, m, c1, Deltar1, c2, Deltar2):
        """Single-pass 2nd directional derivative on (c, r), m constant.

        SLOT PAIRING -- the caller must CROSS the coefficients.

        The fused kernel pairs each coefficient with the shift in its OWN slot,
        i.e. it evaluates

            d2/dr2 (c)[Deltar1, Deltar2]  +  d/dr (c1)[Deltar1]  +  d/dr (c2)[Deltar2]

        but the second differential of S along directions y = (c_y, dr_y) and
        z = (c_z, dr_z) needs the mixed terms, each coefficient against the
        OTHER direction's shift:

            d2/dr2 (c)[dr_y, dr_z]  +  d/dr (c_y)[dr_z]  +  d/dr (c_z)[dr_y]

        so a caller wanting B(y, z) must pass

            d2curlySc(c, r, m, c_z, dr_y, c_y, dr_z)

        The unfused reference implementation writes the crossing out inside the
        function itself -- github.com/nikitinvv/holotomocupy @ 0c098b1,
        shift.py:175:

            dT(c1, r, Deltar2) + dT(c2, r, Deltar1) + d2T(c, r, Deltar1, Deltar2)

        Fusing the three launches into one moved that responsibility to the
        call site. On the diagonal (y is z) the two orders agree, so a Taylor
        or approximation test cannot see the difference -- only an off-diagonal
        check does, e.g. the polarization identity
        B(y+z, y+z) == B(y,y) + 2 B(y,z) + B(z,z).
        """
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], 'complex64')
        c       = _ascontig(c)
        c1      = _ascontig(c1)
        c2      = _ascontig(c2)
        r       = _ascontig(r)
        Deltar1 = _ascontig(Deltar1)
        Deltar2 = _ascontig(Deltar2)
        m       = _ascontig(m)

        self._launch(self._d2sk, ntheta,
                     (res, c, c1, c2, r, m, Deltar1, Deltar2,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        return res

    # ------------------------------------------------------------------
    # Magnification-aware variants: same as the *c family above, but the
    # magnification m is a differentiable input rather than a constant.
    # Used when the shrinkage (and hence the effective demagnification) is a
    # reconstruction variable -- see Rec.F4 in rec_mpi.py.
    # ------------------------------------------------------------------

    def dcurlySmc(self, c, r, m, c1, Deltar, Deltam):
        """Single-pass (c, r, m) directional derivative:
            curlySc(c1, r, m)
          + d/dr curlySc(c, r, m) * Deltar
          + d/dm curlySc(c, r, m) * Deltam

        Same signature as dcurlySc with an extra Deltam (chunk, 2). Uses
        the identity d/dm_axis = -tau_axis(pixel) * d/dr_axis to fold Deltam
        into a per-pixel effective r-direction Deltar - tau * Deltam inside
        the CUDA kernel, so this is exactly one kernel launch (same cost as
        dcurlySc) instead of dcurlySc + two extra weighted grad-r calls.
        """
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], 'complex64')
        c       = _ascontig(c)
        c1      = _ascontig(c1)
        r       = _ascontig(r)
        Deltar  = _ascontig(Deltar)
        Deltam  = _ascontig(Deltam)
        m       = _ascontig(m)

        self._launch(self._dsmk, ntheta,
                     (res, c, c1, r, m, Deltar, Deltam,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        return res

    def dcurlySadjmc(self, c, r, m, Deltaphi):
        """Adjoint of dcurlySmc. Returns [out1, out2_r, out2_m] such that
            <dcurlySmc(c, r, m, c1, Deltar, Deltam), g>
              = <c1, out1> + <Deltar, out2_r> + <Deltam, out2_m>
        for the standard inner products (complex for c1, real for the r/m
        direction vectors).

        dsmadj_kernel writes four per-pixel fields:
            dt1  = d curlySc/d r_y,  dt2  = d curlySc/d r_x
            dtm1 = d curlySc/d m_y,  dtm2 = d curlySc/d m_x
        so all four adjoint reductions are a plain redot with Deltaphi -- no
        big broadcast multiply of a (chunk, nz, n) array by a tau vector.
        """
        ntheta = c.shape[0]
        # out1 is an atomicAdd target -> MUST be zeroed. dt*/dtm* are written at
        # every position by the kernel; out2_* are overwritten by the redots below.
        out1   = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        out2_r = cp.empty(r.shape, dtype='float32')
        out2_m = cp.empty(r.shape, dtype='float32')
        dt1    = cp.empty(Deltaphi.shape, 'complex64')
        dt2    = cp.empty(Deltaphi.shape, 'complex64')
        dtm1   = cp.empty(Deltaphi.shape, 'complex64')
        dtm2   = cp.empty(Deltaphi.shape, 'complex64')
        c        = _ascontig(c)
        r        = _ascontig(r)
        Deltaphi = _ascontig(Deltaphi)
        m        = _ascontig(m)

        self._launch(self._dsmadjk, ntheta,
                     (out1, dt1, dt2, dtm1, dtm2, c, Deltaphi, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        out2_r[:, 0] = redot(Deltaphi, dt1,  axis=(1, 2))
        out2_r[:, 1] = redot(Deltaphi, dt2,  axis=(1, 2))
        out2_m[:, 0] = redot(Deltaphi, dtm1, axis=(1, 2))
        out2_m[:, 1] = redot(Deltaphi, dtm2, axis=(1, 2))

        return [out1, out2_r, out2_m]

    def d2curlySmc(self, c, r, m, c1, Deltar1, Deltam1, c2, Deltar2, Deltam2):
        """Single-pass 2nd directional derivative on (c, r, m).

        Bilinear form on directions (c1, dr1, dm1) and (c2, dr2, dm2).
        Returns (with c-linearity d2/dc2 = 0):
            d2/dr2   * dr1*dr2
          + d2/dm2   * dm1*dm2
          + d2/drdm  * (dr1*dm2 + dr2*dm1)
          + d2/dcdr  * (c1*dr2 + c2*dr1)
          + d2/dcdm  * (c1*dm2 + c2*dm1)

        Uses d/dm_axis = -tau_axis(pixel) * d/dr_axis to fold dm into a
        per-pixel effective dr = dr - tau * dm for BOTH direction slots
        inside the CUDA kernel -- one kernel launch, same cost as d2curlySc.

        SLOT PAIRING: as in d2curlySc, the kernel contracts each coefficient
        with the geometry in its own slot, so a caller wanting B(y, z) passes
        the coefficients CROSSED -- (c_z, dg_y) in slot 1 and (c_y, dg_z) in
        slot 2. See d2curlySc for the derivation and for why the diagonal
        tests cannot catch it. Rec.d2F_dF3 is the caller.
        """
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], 'complex64')
        c       = _ascontig(c)
        c1      = _ascontig(c1)
        c2      = _ascontig(c2)
        r       = _ascontig(r)
        Deltar1 = _ascontig(Deltar1)
        Deltam1 = _ascontig(Deltam1)
        Deltar2 = _ascontig(Deltar2)
        Deltam2 = _ascontig(Deltam2)
        m       = _ascontig(m)

        self._launch(self._d2smk, ntheta,
                     (res, c, c1, c2, r, m, Deltar1, Deltam1, Deltar2, Deltam2,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        return res
    