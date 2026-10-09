import math
import cupy as cp
import cupyx.scipy.fft as cufft
from .cuda_kernels import pad_fwd_kernel, pad_adj_kernel
from .logger_config import logger
try:
    from .conv2d_cufftdx import Conv2DCUFFTDX, cufftdx_available
except Exception:
    def cufftdx_available():
        return False


class Propagation:
    """Functionality for Propagation"""

    def __init__(self, n, nz, ntheta, ndist, wavelength, voxelsize, distance):
        self.n       = n
        self.nz      = nz
        self.wavelength = wavelength
        self.voxelsize = voxelsize
        self.distance = distance

        # Pre-allocated work buffer (avoid per-call allocation)
        self._buf_big = cp.empty([ntheta, 2 * nz, 2 * n], dtype="complex64")

        # Separable 1-D Fresnel phasers (persistent, KB scale):
        #   fker[j] = fker_y[j, :, None] * fker_x[j, None, :]
        #          = exp(-i·pi·λ·z[j]·(fx²+fy²)) / norm
        # 1/norm folded into fker_x. DT uses .conj() of these.
        fx = cp.fft.fftfreq(2 * n,  d=voxelsize)
        fy = cp.fft.fftfreq(2 * nz, d=voxelsize)
        z  = cp.asarray(distance)[:, None]           # [ndist, 1]
        norm = float(4 * n * nz)
        self.fker_x = (cp.exp(-1j * cp.pi * wavelength * z * fx[None, :] ** 2)
                       / norm).astype('complex64')   # [ndist, 2n]
        self.fker_y =  cp.exp(-1j * cp.pi * wavelength * z * fy[None, :] ** 2
                             ).astype('complex64')   # [ndist, 2nz]
        # 2-D assembly buffer; reassembled per D/DT call via cp.multiply(out=).
        self._fker_buf = cp.empty([2 * nz, 2 * n], dtype='complex64')

        # cuFFTDx handle (optional — falls back to cuPy if unavailable).
        # JIT compilation is expected to have been done already by rank 0 via
        # cufftdx_precompile() in rec_mpi.py before this constructor is called.
        self._use_cufftdx = cufftdx_available()
        if self._use_cufftdx:
            try:
                self._conv2d = Conv2DCUFFTDX(2 * nz, 2 * n)
            except Exception as e:
                logger.warning(f"cuFFTDx unavailable ({e}), falling back to cuPy FFT.")
                self._use_cufftdx = False
        if not self._use_cufftdx:
            self._plan_2d = cufft.get_fft_plan(self._buf_big, axes=(-2, -1), value_type='C2C')
            
    def _fwd_pad(self, f, fpad):
        """Symmetric padding: f (ntheta, nz, n) -> fpad (ntheta, 2nz, 2n)"""
        ntheta, nz, n = f.shape
        f = cp.ascontiguousarray(f)
        pad_fwd_kernel(
            (math.ceil(2 * n / 32), math.ceil(2 * nz / 32), ntheta),
            (32, 32, 1),
            (fpad, f, n, nz, ntheta),
        )

    def _adj_pad(self, fpad, f):
        """Adjoint padding: fpad (ntheta, 2nz, 2n) -> f (ntheta, nz, n)"""
        ntheta = fpad.shape[0]
        nz     = fpad.shape[1] // 2
        n      = fpad.shape[2] // 2
        fpad = cp.ascontiguousarray(fpad)
        pad_adj_kernel(
            (math.ceil(n / 32), math.ceil(nz / 32), ntheta),
            (32, 32, 1),
            (fpad, f, n, nz, ntheta),
        )

    def D(self, psi, j):
        """Forward propagator."""
        psi = cp.asarray(psi)            # no-op for cupy; H2D for pinned numpy
        added_dim = psi.ndim == 2
        if added_dim:
            psi = psi[cp.newaxis]

        ntheta = psi.shape[0]
        # No fill(0): pad_fwd_kernel writes every element of _buf_big[:ntheta, :, :],
        # and rows ≥ ntheta are decoupled (per-theta FFT) so stale data there is harmless.
        self._fwd_pad(psi, self._buf_big[:ntheta])

        cp.multiply(self.fker_y[j][:, None], self.fker_x[j][None, :], out=self._fker_buf)

        if self._use_cufftdx:
            self._conv2d.run(self._buf_big, self._fker_buf, self._buf_big)
        else:
            with self._plan_2d:
                cufft.fft2(self._buf_big, overwrite_x=True)
            self._buf_big *= self._fker_buf
            with self._plan_2d:
                cufft.ifft2(self._buf_big, overwrite_x=True, norm="forward")
        result = self._buf_big[:ntheta, self.nz // 2 : -self.nz // 2, self.n // 2 : -self.n // 2].copy()

        return result[0] if added_dim else result

    def DT(self, big_psi, j):
        """Adjoint propagator."""
        big_psi = cp.asarray(big_psi)    # no-op for cupy; H2D for pinned numpy
        added_dim = big_psi.ndim == 2
        if added_dim:
            big_psi = big_psi[cp.newaxis]

        ntheta = big_psi.shape[0]
        self._buf_big.fill(0)
        self._buf_big[:ntheta, self.nz // 2 : -self.nz // 2, self.n // 2 : -self.n // 2] = big_psi

        # Adjoint kernel = conj of forward.
        cp.multiply(self.fker_y[j][:, None].conj(),
                    self.fker_x[j][None, :].conj(), out=self._fker_buf)

        if self._use_cufftdx:
            self._conv2d.run(self._buf_big, self._fker_buf, self._buf_big)
        else:
            with self._plan_2d:
                cufft.fft2(self._buf_big, overwrite_x=True)
            self._buf_big *= self._fker_buf
            with self._plan_2d:
                cufft.ifft2(self._buf_big, overwrite_x=True, norm="forward")

        result = cp.zeros_like(big_psi)
        self._adj_pad(self._buf_big[:ntheta], result)

        return result[0] if added_dim else result
