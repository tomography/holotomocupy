import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage
from .chunking import gpu_batch


@gpu_batch
def S(psi, shift=0):
    """ 2D shift operator
    
    Parameters
    ----------
    psi : ndarray
        Input 3D array, shape [ntheta,n,n] 
    shift : ndarray
        x,y shifts of each 2D array, shape [ntheta,2]
    
    Returns
    -------
    res : ndarray
         Shifted array
    
    """
    res = psi.copy()
    if np.all(shift == 0):
        return res
    p = cp.pad(shift, (0, res.shape[0]-shift.shape[0]))
    n = psi.shape[-1]
    res = cp.pad(res, ((0, 0), (n//2, n//2), (n//2, n//2)), 'symmetric')
    x = cp.fft.fftfreq(2*n).astype('float32')
    [x, y] = cp.meshgrid(x, x)
    pp = cp.exp(-2*cp.pi*1j * (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res = cp.fft.ifft2(pp*cp.fft.fft2(res))
    res = res[:, n//2:-n//2, n//2:-n//2]
    return res


@gpu_batch
def ST(psi, shift=0):
    """ Adjoint 2D shift operator (-shift)
    
    Parameters
    ----------
    psi : ndarray
        Input 3D array, shape [ntheta,n,n] 
    shift : ndarray
        x,y shifts of each 2D array, shape [ntheta,2]
    
    Returns
    -------
    res : ndarray
         Shifted array
    
    """
    res = psi.copy()
    if np.all(shift == 0):
        return res
    p = cp.pad(shift, (0, res.shape[0]-shift.shape[0]))
    n = psi.shape[-1]
    res = cp.pad(res, ((0, 0), (n//2, n//2), (n//2, n//2)), 'symmetric')
    x = cp.fft.fftfreq(2*n).astype('float32')
    [x, y] = cp.meshgrid(x, x)
    pp = cp.exp(2*cp.pi*1j * (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res = cp.fft.ifft2(pp*cp.fft.fft2(res))
    res = res[:, n//2:-n//2, n//2:-n//2]
    return res


# @gpu_batch
# def S(psi, shift=0):
#     """Shift operator"""
#     res = psi.copy()
#     if np.all(shift==0):
#         return res
#     p = cp.pad(shift,(0,res.shape[0]-shift.shape[0]))
#     for k in range(p.shape[0]):
#         res[k] = ndimage.shift(res[k], p[k], order=2, mode='nearest', prefilter=True)
#     return res

# @gpu_batch
# def ST(psi, shift=0):
#     """Adjoint Shift operator"""
#     res = psi.copy()
#     if np.all(shift==0):
#         return res
#     p = cp.pad(-shift,(0,res.shape[0]-shift.shape[0]))
#     for k in range(p.shape[0]):
#         res[k] = ndimage.shift(res[k], p[k], order=2, mode='nearest', prefilter=True)
#     return res

def _upsampled_dft(data, ups,
                   upsample_factor=1, axis_offsets=None):

    im2pi = 1j * 2 * np.pi
    tdata = data.copy()
    kernel = (cp.tile(cp.arange(ups), (data.shape[0], 1))-axis_offsets[:, 1:2])[
        :, :, None]*cp.fft.fftfreq(data.shape[2], upsample_factor)
    kernel = cp.exp(-im2pi * kernel)
    tdata = cp.einsum('ijk,ipk->ijp', kernel, tdata)
    kernel = (cp.tile(cp.arange(ups), (data.shape[0], 1))-axis_offsets[:, 0:1])[
        :, :, None]*cp.fft.fftfreq(data.shape[1], upsample_factor)
    kernel = cp.exp(-im2pi * kernel)
    rec = cp.einsum('ijk,ipk->ijp', kernel, tdata)

    return rec

@gpu_batch
def registration_shift(src_image, target_image, upsample_factor=1, space="real"):
    """Efficient subpixel image translation registration by cross-correlation.

    Parameters
    ----------
    src_image : ndarray
        Image to register
    target_image : ndarray
        Reference image.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).
        Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data. "real" means
        data will be FFT'd to compute the correlation, while "fourier"
        data will bypass FFT of input data. Case insensitive. Not
        used if any of ``reference_mask`` or ``moving_mask`` is not
        None.
    
    Returns
    -------
    shift : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        the axis order of the input array.
    """
    
    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = cp.fft.fft2(src_image)
        target_freq = cp.fft.fft2(target_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = cp.fft.ifft2(image_product)
    A = cp.abs(cross_correlation)
    maxima = A.reshape(A.shape[0], -1).argmax(1)
    maxima = cp.column_stack(cp.unravel_index(maxima, A[0, :, :].shape))

    midpoints = cp.array([cp.fix(axis_size / 2)
                          for axis_size in shape[1:]])

    shifts = cp.array(maxima, dtype=np.float64)
    ids = cp.where(shifts[:, 0] > midpoints[0])
    shifts[ids[0], 0] -= shape[1]
    ids = cp.where(shifts[:, 1] > midpoints[1])
    shifts[ids[0], 1] -= shape[2]

    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        shifts = cp.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = cp.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = cp.fix(upsampled_region_size / 2.0)

        normalization = (src_freq[0].size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate

        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        A = cp.abs(cross_correlation)
        maxima = A.reshape(A.shape[0], -1).argmax(1)
        maxima = cp.column_stack(
            cp.unravel_index(maxima, A[0, :, :].shape))

        maxima = cp.array(maxima, dtype=np.float64) - dftshift

        shifts = shifts + maxima / upsample_factor

    return shifts

# import tifffile
# # a = tifffile.imread('../tests/data/delta-chip-192.tiff')
# # # a = tifffile.imread('/data/vnikitin/modeling/r_256_32_4/r00256.tiff')
# a = tifffile.imread('/data/vnikitin/modeling/ref_3d_ald_256_0.tiff')
# a = a-1j*a
# shift = np.array(np.random.random([a.shape[0], 2]), dtype='float32')*6
# b = S(a, shift)
# # aa = S(b, -shift)

# #b = Sa
# shift_rec = registration_shift(b,a,upsample_factor=1000)
# print(shift_rec)
# print(shift)
# import matplotlib.pyplot as plt
# plt.plot(shift_rec[:,0])
# plt.plot(shift[:,0])
# plt.show()

# exit()


# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(b[b.shape[0]//2].real,cmap='gray')
# plt.colorbar()
# plt.savefig('t.png')

# plt.figure()
# plt.imshow(a[b.shape[0]//2].real,cmap='gray')
# plt.colorbar()
# plt.savefig('t1.png')


# plt.figure()
# plt.imshow(bb[b.shape[0]//2].real.get()-b[b.shape[0]//2].real.get(),cmap='gray')
# plt.colorbar()
# plt.savefig('t2.png')

# plt.figure()
# import dxchange
# for k in range(5):
#     b = S(a, pars,k)
#     plt.plot(b[b.shape,aa.shape[1]//2,16:32].real.get(),label=f"{k}")
#     dxchange.write_tiff(b[92].real.get(),f"t/{k}.tiff",overwrite=True)
# b = S1(a, pars)
# dxchange.write_tiff(b[92].real.get(),f"t/6.tiff",overwrite=True)
# plt.plot(b[92,aa.shape[1]//2,16:32].real.get(),label=f"F")
# # plt.colorbar()
# plt.legend()
# plt.savefig('t1.png')


# plt.figure()
# plt.imshow(aa[92].real.get(),cmap='gray')
# plt.colorbar()
# plt.savefig('t1.png')


# print(cp.sum(b*cp.conj(b)))
# print(cp.sum(a*cp.conj(aa)))
# import tifffile
# a =  cp.array(tifffile.imread('../../tests/data/delta-chip-192.tiff'))
# a = a-1j*a
# aa = fwd_pad(a)
# aaa = adj_pad(aa)

# plt.figure()
# plt.imshow(aa[92].real.get())
# plt.colorbar()
# plt.savefig('t.png')

# plt.figure()
# plt.imshow(aaa[92].real.get())
# plt.colorbar()
# plt.savefig('t1.png')
