import math
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from .cuda_kernels import (gather_kernel, sample_tiles_kernel,
                           scatter_binned_kernel)
from .utils import redot, logger

# RT's largest tile, and the cap on samples per block.  Both measured on an
# A100 over n = 256..4736; 64 is also as large as 48 kB of shared memory allows.
TILE    = 64
SUBSIZE = 2048


class Tomo:
    """Functionality for Radon transforms and exp"""

    def __init__(self, n, nz, theta, mask_r, nd=None):
        """Usfft parameters.

        `nd` is the detector width, `n` (default) or `2*n` -- twice as finely
        sampled over the same field of view.  The Fourier step stays 1/n either
        way, so the sinogram values are unchanged and R/RT stay an adjoint pair.

        Detector bins falling outside the padded FFT's square, which only exist
        once nd > n, are skipped and read back as zero: the object's spectrum is
        that square, so the sinogram is the band-limited interpolation of the
        coarse one.  Letting them wrap instead gives line artifacts.
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

        self._bins = None             # RT's index, built on first use

    # ----------------------------------------------------------------- bins

    def _build_bins(self):
        """Sample lists per tile of the padded grid, for the binned scatter.

        Each sample is listed in every tile its 2m+1 stencil touches, so a
        block can clip to its own tile and write nothing outside it.  theta and
        the detector grid are fixed for a `Tomo`'s lifetime, so this is built
        once and amortised over every RT.

        Counting first and placing second recomputes the geometry instead of
        keeping the pairs: one global sort would need 3.0 GB against 276 MB of
        output, and the geometry kernel is only a few flops per sample.
        """
        n, nd, twon = self.n, self.nd, 2 * self.n
        m, subsize  = self.pars[0], SUBSIZE
        span = 2 * m + 1
        # b need not divide 2n -- the last row and column are just narrower --
        # but every tile must be at least a stencil wide, or a stencil could
        # straddle three tiles and sample_tiles only records four corners.
        # Skip b+1 divisible by 32: that is the shared row stride, and a
        # multiple of 32 puts every row on the same bank (20% on RT at n=2500).
        b = next((t for t in range(min(TILE, twon - 1), span - 1, -1)
                  if (twon % t == 0 or twon % t >= span) and (t + 1) % 32), 0)
        if b < span:
            raise ValueError(
                f"RT: no tile <= {TILE} fits n={n} (2n={twon}, stencil {span})")
        ntx   = -(-twon // b)             # ceil: the last tile may be partial
        ntile = ntx * ntx
        grid  = lambda na: (math.ceil(nd / 32), math.ceil(na / 32), 1)
        # Chunked over angles so the transient 4-per-sample candidate array
        # stays bounded; everything below is O(chunk), not O(ntheta).
        chunk = max(1, 2097152 // nd)
        spans = [(a0, min(chunk, self.ntheta - a0))
                 for a0 in range(0, self.ntheta, chunk)]

        # Pass 1: how many (sample, tile) pairs land in each tile.
        counts = cp.zeros(ntile, dtype='int64')
        for a0, na in spans:
            out = cp.empty(na * nd * 4, dtype='int32')
            sample_tiles_kernel(grid(na), (32, 32, 1),
                                (out, self.theta, m, n, nd, a0, na, b, ntx))
            counts += cp.bincount(out[out >= 0], minlength=ntile)
            del out

        # Pass 2: place each pair at its final offset.  Runs come out ordered
        # by sample index, which keeps the scatter kernel's reads of g
        # coalesced; ordering by detector bin instead measured 0.89x -> 0.82x.
        offs = cp.concatenate((cp.zeros(1, 'int64'), cp.cumsum(counts)))
        bin_samples = cp.empty(int(offs[-1]), dtype='int32')
        cursor = offs[:-1].copy()
        for a0, na in spans:
            out = cp.empty(na * nd * 4, dtype='int32')
            sample_tiles_kernel(grid(na), (32, 32, 1),
                                (out, self.theta, m, n, nd, a0, na, b, ntx))
            keep = out >= 0
            tid  = out[keep]
            sid  = cp.repeat(cp.arange(a0 * nd, (a0 + na) * nd, dtype='int32'),
                             4)[keep]
            del out, keep
            # Radix sort, so it is stable: the four slots of one sample keep
            # their order and the runs come out ordered by sample index.
            order = cp.argsort(tid)
            tid_s = tid[order]
            cnt   = cp.bincount(tid, minlength=ntile)
            del tid
            # Element i of the sorted chunk sits at chunk-local offset
            # i - local[t] within tile t, hence at global offset
            # i + (cursor[t] - local[t]).  One gather, no second sort.
            local = cp.cumsum(cnt) - cnt
            dest  = cp.arange(tid_s.size, dtype='int64') + (cursor - local)[tid_s]
            bin_samples[dest] = sid[order]
            cursor += cnt
            del order, tid_s, cnt, local, dest, sid

        # Pass 3: polar density goes as 1/r, so central tiles hold ~100x what
        # rim tiles do.  Bins longer than `subsize` are cut into subproblems
        # that combine through global atomics, or the GPU waits on a few tiles.
        counts = cp.asnumpy(counts).astype('int64')
        start  = np.concatenate(([0], np.cumsum(counts)))
        nsubs  = -(-counts // subsize)                     # ceil, 0 when empty
        tile   = np.repeat(np.arange(ntile), nsubs)
        k      = (np.arange(nsubs.sum())
                  - np.repeat(np.cumsum(nsubs) - nsubs, nsubs))
        beg    = start[tile] + k * subsize
        end    = np.minimum(beg + subsize, start[tile + 1])

        self._bins = dict(
            b=b, ntx=ntx, subsize=subsize,
            samples=bin_samples,
            tile=cp.asarray(tile, dtype='int32'),
            beg=cp.asarray(beg, dtype='int32'),
            end=cp.asarray(end, dtype='int32'),
            atomic=cp.asarray(nsubs[tile] > 1, dtype='uint8'),
            nsub=len(tile), ntile=ntile, counts=counts,
            shmem=2 * b * (b + 1) * 4,     # two padded planes, see the kernel
        )
        logger.info(f"Tomo binned scatter: {self.bin_stats()}")

    def bin_stats(self):
        """One-line summary of the binned scatter's index."""
        z = self._bins
        c = z['counts']
        nz_ = c[c > 0]
        mb = sum(a.nbytes for a in (z['samples'], z['tile'], z['beg'],
                                    z['end'], z['atomic'])) / 2**20
        return (f"tile {z['b']}x{z['b']}: {z['ntile']} tiles, "
                f"{100 * len(nz_) / z['ntile']:.0f}% non-empty, "
                f"samples/tile median {int(np.median(nz_))} max {int(nz_.max())}, "
                f"{z['nsub']} subproblems "
                f"({100 * float(cp.mean(z['atomic'])):.0f}% combining), "
                f"index {mb:.0f} MB, duplication "
                f"{c.sum() / (self.ntheta * self.nd):.2f}x")

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
        # No memset needed: the gather kernel starts each thread from g0 = 0 and
        # *assigns* g[g_ind], covering every element of _buf_sino.
        gather_kernel(
            (math.ceil(self.nd / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde, self.theta, m, mua, n, self.nd,
             self.ntheta, self._nz),
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
        # STEP2: NUFFT scatter from full buf_sino (extra slices are zero,
        # contribute nothing).  The fill is needed because a split tile
        # accumulates into f rather than storing, and empty tiles are skipped.
        self._buf_fde.fill(0)
        if self._bins is None:
            self._build_bins()
        z = self._bins
        scatter_binned_kernel(
            (z['nsub'], self._nz), (256,),
            (self._buf_fde, self._buf_sino, self.theta,
             z['samples'], z['tile'], z['beg'], z['end'], z['atomic'],
             m, mua, n, self.nd, self.ntheta, self._nz, z['b'], z['ntx']),
            shared_mem=z['shmem'],
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
