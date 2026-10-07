import math
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from .cuda_kernels import gather_kernel
from .utils import redot, logger


class Tomo:
    """Functionality for Radon transforms and exp"""

    def __init__(self, n, nz, theta, mask_r, nd=None):
        """Usfft parameters.

        `nd` is the number of detector samples per projection, i.e. the width of
        the sinogram/projection plane.  It defaults to `n` (detector pixel ==
        object pixel); the only other supported value is `2*n`, a detector twice
        as finely sampled over the *same* field of view.  The Fourier step along
        the detector stays 1/n either way, so R and RT remain an exact adjoint
        pair and the sinogram *values* are unchanged -- only sampled more densely.

        Detector bins whose Cartesian frequency (fr*cos, -fr*sin) falls outside
        the padded FFT's square -- they only exist once nd > n -- are skipped
        and read back as zero: the object lives on the n grid, so its spectrum
        is the square |kx|, |ky| <= 1/2, and the sinogram is the band-limited
        interpolation of the coarse one onto the nd grid.  The test is on the
        Cartesian pair, not on |fr|, so the square's corners -- out to
        |fr| = sqrt(2)/2 at 45 deg -- are kept.  (Letting the gather index wrap
        instead, as ~/APS_PXM/tomo_usfft does, models the object as a delta
        comb; at nd = 2n that makes theta = 0 and 90 read an n-periodic spectrum
        and come out as combs with every odd detector sample exactly zero,
        which backprojects to vertical and horizontal line artifacts.  See the
        gather kernel.)
        """
        nd = n if nd is None else int(nd)
        if nd not in (n, 2 * n):
            raise ValueError(f"nd must be n={n} or 2n={2*n}, got {nd}")
        eps = 1e-3  # accuracy of usfft
        mu = -math.log(eps) / (2 * n * n)
        m  = math.ceil(2 * n / math.pi * math.sqrt(-mu * math.log(eps) + (mu * n) ** 2 / 4))

        # interpolation kernel
        t = cp.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        dx, dy = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32")) * (1 - n % 4)

        # (+1,-1) sign arrays for fftshift-via-multiply
        c1dfftshift = (1 - 2 * ((cp.arange(1, nd + 1) % 2))).astype("int8")
        c2dtmp      = (1 - 2 * ((cp.arange(1, 2 * n + 1) % 2))).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        mua = cp.array([mu], dtype="float32")

        # Lazily-filled caches for the fbp path (filter response by name, cuFFT
        # plan by shape). rec_mpi sets the global cupy plan cache size to 0.
        self._filters   = {}
        self._fft_plans = {}

        self.n      = n
        self.nd     = nd
        self.ntheta = len(theta)
        self.theta  = cp.array(theta.astype("float32"))

        if mask_r > 0:
            t1d = np.linspace(-1, 1, self.n)
            x, y = np.meshgrid(t1d, t1d)
            circ  = (x**2 + y**2 < mask_r).astype("float32")
            g     = np.exp(-(20**2) * (x**2 + y**2))
            fcirc = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
            fg    = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))
            mask  = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fcirc * fg))).real.astype("float32")
            mask /= np.amax(mask)
        else:
            mask = 1.0

        self.mask = mask
        phi *= cp.array(mask / (n * np.sqrt(n * self.ntheta)))
        self.pars = m, mua, phi, c1dfftshift, c2dfftshift
        self._buf_fde = cp.empty([nz, 2 * n, 2 * n], dtype="complex64")

        self._nz       = nz
        self._buf_sino = cp.zeros([self.ntheta, nz, nd], dtype="complex64")
        self._plan_2d  = cufft.get_fft_plan(self._buf_fde,  axes=(-2, -1), value_type='C2C')
        self._plan_1d  = cufft.get_fft_plan(self._buf_sino, axes=(-1,),    value_type='C2C')

    def R(self, obj):
        """Radon transform"""
        nz = obj.shape[0]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP0: fill full fde buffer, zero-padding extra z-slices
        self._buf_fde.fill(0)
        cp.multiply(obj, phi, out=self._buf_fde[:nz, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2])
        # STEP1: 2D FFT on full buffer (always matches plan)
        self._buf_fde *= c2dfftshift
        with self._plan_2d:
            cufft.fft2(self._buf_fde, overwrite_x=True)
        self._buf_fde *= c2dfftshift
        # STEP2: NUFFT gather into full buf_sino (extra slices are zero, no effect).
        # No memset needed: with dir==0 the gather kernel starts each thread from
        # g0 = 0 and *assigns* g[g_ind], covering every element of _buf_sino.
        gather_kernel(
            (math.ceil(self.nd / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde, self.theta, m, mua, n, self.nd,
             self.ntheta, self._nz, 0),
        )
        # STEP3: 1D IFFT on full buf_sino
        self._buf_sino *= c1dfftshift
        with self._plan_1d:
            cufft.ifft(self._buf_sino, overwrite_x=True)
        self._buf_sino *= c1dfftshift
        # STEP4: normalization, crop.  The 1-D inverse transform divides by nd,
        # but the Fourier step is 1/n regardless of the detector sampling, so
        # nd/n puts it back: the sinogram then holds the same values on a finer
        # grid, and RT is its exact adjoint for any nd.
        result = self._buf_sino[:, :nz] * (self.nd / (4 * n))
        if obj.dtype == 'float32':
            result = result.real
        return cp.ascontiguousarray(result)

    def RT(self, data):
        """Adjoint Radon transform"""
        nz = data.shape[1]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP1: copy into full buf_sino, zero-pad, 1D FFT
        self._buf_sino[:, :nz] = (data * c1dfftshift).astype('complex64')
        self._buf_sino[:, nz:] = 0
        with self._plan_1d:
            cufft.fft(self._buf_sino, overwrite_x=True)
        self._buf_sino *= c1dfftshift
        # STEP2: NUFFT scatter from full buf_sino (extra slices are zero, contribute nothing)
        self._buf_fde.fill(0)
        gather_kernel(
            (math.ceil(self.nd / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde, self.theta, m, mua, n, self.nd,
             self.ntheta, self._nz, 1),
        )
        # STEP3: 2D IFFT on full buffer (always matches plan)
        self._buf_fde *= c2dfftshift
        with self._plan_2d:
            cufft.ifft2(self._buf_fde, overwrite_x=True)
        self._buf_fde *= c2dfftshift
        # STEP4: unpadding, crop to nz
        result = self._buf_fde[:nz, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2] * phi
        if data.dtype == 'float32':
            result = result.real
        return cp.ascontiguousarray(result)

    def _fbp_filter(self, filter_name):
        """1-D frequency response for `filter_name`, built once and cached.

        Depends only on nd, so rebuilding it per call (as this used to) is pure
        waste on the fbp path.
        """
        h = self._filters.get(filter_name)
        if h is not None:
            return h
        nd = self.nd
        f  = cp.fft.fftfreq(nd).astype('float32')  # f in [-0.5, 0.5)
        af = cp.abs(f)*4*nd

        if filter_name == 'ramp':
            # Ram-Lak: |ω|
            h = af
        elif filter_name == 'shepp':
            # Shepp-Logan: |ω| × sinc(ω)
            # cp.sinc uses normalized sinc: sinc(x) = sin(πx)/(πx), sinc(0) = 1
            h = af * cp.sinc(f)
        elif filter_name == 'parzen':
            # Parzen (B-spline order-4) window applied to the ramp.
            # u = 2|f| maps [0, 0.5] → [0, 1].  Note this is |f|, NOT af: af
            # carries the ramp's own 4*nd gain, and feeding that to the window
            # sent 2*(1-u)**3 to ~1e9 and made 'parzen' unusable at any real nd.
            u = 2 * cp.abs(f)
            w = cp.where(u <= 0.5,
                         1 - 6*u**2 + 6*u**3,   # inner region
                         2*(1 - u)**3)            # outer region (tapers to 0 at Nyquist)
            h = af * w
        else:
            raise ValueError(
                f"Unknown filter '{filter_name}'. Choose: ramp, shepp, parzen."
            )

        h = h.astype('complex64')
        self._filters[filter_name] = h
        return h

    def _filter_sino(self, data, filter_name):
        """Apply a 1-D frequency-domain filter along the detector axis (last axis).

        Parameters
        ----------
        data : cupy ndarray, shape [ntheta, nz, nd], float32 or complex64
        filter_name : str  — 'ramp', 'shepp', or 'parzen'

        Returns
        -------
        Filtered array with the same shape and dtype as `data`.
        """
        h = self._fbp_filter(filter_name)
        # One temporary instead of three: the complex64 view allocates it, the
        # filter multiply is in place, and both transforms run in place. rec_mpi
        # sets the global cuFFT plan cache to 0, so the plan is memoized here.
        d = data if data.dtype == cp.complex64 else data.astype('complex64')
        with self._fft_plan(tuple(d.shape)):
            out  = cufft.fft(d, axis=-1, overwrite_x=(d is not data))
            out *= h
            out  = cufft.ifft(out, axis=-1, overwrite_x=True)

        if cp.iscomplexobj(data):
            return out.astype(data.dtype, copy=False)
        return out.real.astype(data.dtype, copy=False)

    def _fft_plan(self, shape):
        """cuFFT C2C plan over the last axis for `shape`, memoized."""
        plan = self._fft_plans.get(shape)
        if plan is None:
            _tmp = cp.empty(shape, dtype='complex64')
            plan = cufft.get_fft_plan(_tmp, axes=(-1,), value_type='C2C')
            self._fft_plans[shape] = plan
            del _tmp
        return plan

    def fbp(self, data, filter_name='ramp'):
        """Filtered back-projection: apply a 1-D filter then RT.

        Parameters
        ----------
        data : array_like [ntheta, nz, nd], float32 or complex64
            Sinogram projections (numpy or cupy).
        filter_name : str
            'ramp'   — Ram-Lak ramp filter |ω|
            'shepp'  — Shepp-Logan:        |ω| × sinc(ω)
            'parzen' — Parzen B-spline-4:  |ω| × w_parzen(2ω)

        Returns
        -------
        Reconstruction array [nz, n, n], same dtype as `data`.
        """
        # n/nd: the forward transform inside RT sums nd detector samples where
        # the object grid has n, so RT (and hence R^T R) scales with nd/n.
        norm_const = np.float32(np.sqrt(self.n / self.ntheta) * self.n / self.nd)
        data = cp.asarray(data)
        res = self.RT(self._filter_sino(data, filter_name))
        res *= norm_const
        return  res

    def rec_tomo(self, d, niter=1):
        """Iterative CG tomography reconstruction for initial guess"""

        def minf(Ru, d):
            return np.linalg.norm(Ru - d) ** 2

        u  = cp.zeros([d.shape[1], self.n, self.n], dtype=d.dtype)
        Ru = self.R(u)
        for k in range(niter):
            if k % 32 == 0:
                logger.info(f"rec_tomo iter {k}: err={minf(Ru, d):.6e}")
            tmp   = 2 * (Ru - d)
            grad  = self.RT(tmp)
            Rgrad = self.R(grad)
            if k == 0:
                eta  = -grad
                Reta = -Rgrad
            else:
                beta = redot(Rgrad, Reta) / redot(Reta, Reta)
                eta  = beta * eta  - grad
                Reta = beta * Reta - Rgrad
            alpha = -redot(grad, eta) / (2 * redot(Reta, Reta))
            u  += alpha * eta
            Ru += alpha * Reta

        return u
