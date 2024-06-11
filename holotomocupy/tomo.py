import cupy as cp
import numpy as np

from .cuda_kernels import wrap_kernel, wrapadj_kernel, gather_kernel
from .chunking import gpu_batch


def _init(nz, n):
    # usfft parameters
    eps = 1e-3  # accuracy of usfft
    mu = -cp.log(eps) / (2 * n * n)
    m = int(cp.ceil(2 * n * 1 / cp.pi * cp.sqrt(-mu *
            cp.log(eps) + (mu * n) * (mu * n) / 4)))
    # extra arrays
    # interpolation kernel
    t = cp.linspace(-1/2, 1/2, n, endpoint=False).astype('float32')
    [dx, dy] = cp.meshgrid(t, t)
    phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype('float32')) * (1-n % 4)

    # padded fft, reusable by chunks
    fde = cp.zeros([nz, 2*m+2*n, 2*m+2*n], dtype='complex64')
    # (+1,-1) arrays for fftshift
    c1dfftshift = (1-2*((cp.arange(1, n+1) % 2))).astype('int8')
    c2dtmp = 1-2*((cp.arange(1, 2*n+1) % 2)).astype('int8')
    c2dfftshift = cp.outer(c2dtmp, c2dtmp)
    return m, mu, phi, fde, c1dfftshift, c2dfftshift


@gpu_batch
def R(obj, theta, rotation_axis):
    """Radon transform for tomography projection
    Parameters
    ----------
    obj : ndarray
        Input 3D object, shape [nz,n,n]    
    theta : ndarray
        Projection angles, shape [ntheta]    
    rotation_axis : float
        Rotation axis 

    Returns
    -------
    sino : ndarray
        Output sinograms, shape [nz,ntheta,n]    
    """
    
    [nz, n, n] = obj.shape
    theta = cp.array(theta, dtype='float32')
    ntheta = len(theta)
    m, mu, phi, fde, c1dfftshift, c2dfftshift = _init(nz, n)

    sino = cp.zeros([nz, ntheta, n], dtype='complex64')

    # STEP0: multiplication by phi, padding
    fde = obj*phi
    fde = cp.pad(fde, ((0, 0), (n//2, n//2), (n//2, n//2)))

    # STEP1: fft 2d
    fde = cp.fft.fft2(fde*c2dfftshift)*c2dfftshift
    fde = cp.pad(fde, ((0, 0), (m, m), (m, m)))
    # STEP2: fft 2d
    wrap_kernel((int(cp.ceil((2 * n + 2 * m)/32)),
                int(cp.ceil((2 * n + 2 * m)/32)), nz), (32, 32, 1), (fde, n, nz, m))
    mua = cp.array([mu], dtype='float32')
    gather_kernel((int(cp.ceil(n/32)), int(cp.ceil(ntheta/32)), nz),
                  (32, 32, 1), (sino, fde, theta, m, mua, n, ntheta, nz, 0))

    # STEP3: ifft 1d
    sino = cp.fft.ifft(c1dfftshift*sino)*c1dfftshift

    # STEP4: Shift based on the rotation axis
    t = cp.fft.fftfreq(n).astype('float32')
    w = cp.exp(-2*cp.pi*1j*t*(rotation_axis + n/2))
    sino = cp.fft.ifft(w*cp.fft.fft(sino))
    # normalization for the unity test
    sino /= cp.float32(4*n)    
    return sino


@gpu_batch
def RT(sino, theta, rotation_axis):
    """Radon transform for tomography projection
    Parameters
    ----------
    obj : ndarray
        Input sinograms, shape [nz,ntheta,n]    
    theta : ndarray
        Projection angles, shape [ntheta]    
    rotation_axis : float
        Rotation axis 

    Returns
    -------
    obj : ndarray
        Output 3D object, shape [nz,n,n]
    """
    
    [nz, ntheta, n] = sino.shape
    theta = cp.array(theta, dtype='float32')

    m, mu, phi, fde, c1dfftshift, c2dfftshift = _init(nz, n)

    # STEP0: Shift based on the rotation axis
    t = cp.fft.fftfreq(n).astype('float32')
    w = cp.exp(-2*cp.pi*1j*t*(-rotation_axis + n/2))
    sino = cp.fft.ifft(w*cp.fft.fft(sino))

    # STEP1: fft 1d
    sino = cp.fft.fft(c1dfftshift*sino)*c1dfftshift

    # STEP2: interpolation (gathering) in the frequency domain
    # dont understand why RawKernel cant work with float, I have to send it as an array (TODO)
    mua = cp.array([mu], dtype='float32')
    gather_kernel((int(cp.ceil(n/32)), int(cp.ceil(ntheta/32)), nz),
                  (32, 32, 1), (sino, fde, theta, m, mua, n, ntheta, nz, 1))
    wrapadj_kernel((int(cp.ceil((2 * n + 2 * m)/32)),
                    int(cp.ceil((2 * n + 2 * m)/32)), nz), (32, 32, 1), (fde, n, nz, m))

    # STEP3: ifft 2d
    fde = fde[:, m:-m, m:-m]
    fde = cp.fft.ifft2(fde*c2dfftshift)*c2dfftshift

    # STEP4: unpadding, multiplication by phi
    fde = fde[:, n//2:3*n//2, n//2:3*n//2]*phi
    fde /= cp.float32(n)  # normalization for the unity test
    return fde

# import matplotlib.pyplot as plt
# sinogram = np.load("sinogram.npy").swapaxes(0,1)# sinogram.npy is saved as projections
# angles_rad = np.load("angles_rad.npy")[:]
# sinogram = sinogram[:,:,:]+1j*sinogram[:,:,:]

# rotation_axis = sinogram.shape[-1]/2
# obj = RT(sinogram, angles_rad, rotation_axis)
# plt.figure()
# plt.imshow(obj[0].imag, cmap='gray')
# plt.savefig('t.png')

# sino = R(obj, angles_rad, rotation_axis)

# print(np.sum(sino*np.conj(sinogram)))
# print(np.sum(obj*np.conj(obj)))


# obj2 = RT(sino, pars)
# print(cp.sum(obj*cp.conj(obj))/cp.sum(obj*cp.conj(obj2)))


# plt.figure()
# plt.imshow(sino[0].imag.get(), cmap='gray')
# plt.savefig('t1.png')
