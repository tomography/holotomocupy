import cupy as cp
import numpy as np
from .cuda_kernels import wrap_kernel, wrapadj_kernel, gather_mag_kernel
from .chunking import gpu_batch


def _init(ne):
    # usfft parameters
    eps = 1e-3  # accuracy of usfft
    mu = -cp.log(eps) / (2 * ne * ne)
    m = int(cp.ceil(2 * ne * 1 / cp.pi * cp.sqrt(-mu *
            cp.log(eps) + (mu * ne) * (mu * ne) / 4)))
    # extra arrays
    # interpolation kernel
    t = cp.linspace(-1/2, 1/2, ne, endpoint=False).astype('float32')
    [dx, dy] = cp.meshgrid(t, t)
    phi = cp.exp((mu * (ne * ne) * (dx * dx + dy * dy)
                  ).astype('float32')) * (1-ne % 4)

    # (+1,-1) arrays for fftshift
    c2dtmp = 1-2*((cp.arange(1, 2*ne+1) % 2)).astype('int8')
    c2dfftshift = cp.outer(c2dtmp, c2dtmp)
    c2dtmp = 1-2*((cp.arange(1, ne+1) % 2)).astype('int8')
    c2dfftshift0 = cp.outer(c2dtmp, c2dtmp)
    return m, mu, phi, c2dfftshift, c2dfftshift0


@gpu_batch
def M(f, magnification=1, n=None):
    """Data magnification via switching to the Fourier domain"""

    [ntheta, ne] = f.shape[:2]
    if n == None:
        n = ne
    if ne == n and (magnification-1.0) < 1e-6:
        return f.copy()

    m, mu, phi, c2dfftshift, c2dfftshift0 = _init(ne)
    # FFT2D
    fde = cp.fft.fft2(f*c2dfftshift0)*c2dfftshift0
    # adjoint USFFT2D
    fde = fde*phi

    fde = cp.pad(fde, ((0, 0), (ne//2, ne//2), (ne//2, ne//2)))
    fde = cp.fft.fft2(fde*c2dfftshift)*c2dfftshift
    fde = cp.pad(fde, ((0, 0), (m, m), (m, m)))
    wrap_kernel((int(cp.ceil((2 * ne + 2 * m)/32)),
                int(cp.ceil((2 * ne + 2 * m)/32)), ntheta), (32, 32, 1), (fde, ne, ntheta, m))
    mua = cp.array([mu], dtype='float32')
    magnificationa = cp.array([magnification], dtype='float32')
    res = cp.zeros([ntheta, n, n], dtype='complex64')
    gather_mag_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32)), ntheta),
                      (32, 32, 1), (res, fde, magnificationa, m, mua, n, ne, ntheta, 0))

    res = res/cp.float32(4*ne*ne*ne)
    return res


@gpu_batch
def MT(f, magnification=1, ne=None):
    """Adjoint data magnification via switching to the Fourier domain"""

    [ntheta, n] = f.shape[:2]
    if ne == None:
        ne = n
    if (ne == n) and (magnification-1.0) < 1e-6:
        return f.copy()

    m, mu, phi, c2dfftshift, c2dfftshift0 = _init(ne)

    mua = cp.array([mu], dtype='float32')
    magnificationa = cp.array([magnification], dtype='float32')
    fde = cp.zeros([ntheta, 2*m+2*ne, 2*m+2*ne], dtype='complex64')
    gather_mag_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32)), ntheta),
                      (32, 32, 1), (f, fde, magnificationa, m, mua, n, ne, ntheta, 1))
    wrapadj_kernel((int(cp.ceil((2 * ne + 2 * m)/32)),
                    int(cp.ceil((2 * ne + 2 * m)/32)), ntheta), (32, 32, 1), (fde, ne, ntheta, m))

    fde = fde[:, m:-m, m:-m]
    fde = cp.fft.ifft2(fde*c2dfftshift)*c2dfftshift

    fde = fde[:, ne//2:3*ne//2, ne//2:3*ne//2]*phi
    fde = cp.fft.ifft2(fde*c2dfftshift0)*c2dfftshift0

    return fde


# import tifffile
# import matplotlib.pyplot as plt

# a = tifffile.imread('../../tests/data/delta-chip-192.tiff')
# a = a-1j*a
# magnification = 1.5
# ne = 192
# n = 128

# aa = M(a,magnification,n)
# aaa = MT(aa,magnification,ne)
# print(np.sum(aa*np.conj(aa)))
# print(np.sum(a*np.conj(aaa)))
# aaaa = M(aaa, magnification,n)
# print(np.sum(aaaa*np.conj(aaaa)/np.sum(aa*np.conj(aaaa))))

# plt.figure()
# plt.imshow(aa[92].real)
# plt.colorbar()
# plt.savefig('t.png')
# plt.figure()
# plt.imshow(aaa[92].real)
# plt.colorbar()
# plt.savefig('t1.png')


# plt.figure()
# import dxchange
# for k in range(5):
#     b = M1(a, pars,k)
#     plt.plot(b[92,b.shape[1]//2,16:32].real.get(),label=f"{k}")
#     dxchange.write_tiff(b[92].real.get(),f"t/{k}.tiff",overwrite=True)
# b = M(a, pars)
# dxchange.write_tiff(b[92].real.get(),f"t/6.tiff",overwrite=True)
# plt.plot(b[92,b.shape[1]//2,16:32].real.get(),label=f"F")
# # plt.colorbar()
# plt.legend()
# plt.savefig('t1.png')

# cp.array(cp.random.random([192, 2]), dtype='float32')
# aa = S(a, pars)
# aaa = ST(aa, pars)
