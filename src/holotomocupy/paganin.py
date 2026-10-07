"""Multi-distance Paganin phase retrieval — step 5's initial guess.

One least-squares inversion of the single-material contrast-transfer model
across all distances at once, which is what makes it multi-distance: each
plane contributes its own Taylor factor and they are solved together.
"""
import cupy as cp


def multi_paganin(data, distances, wavelength, voxelsize, delta_beta, alpha=1e-2):
    """Phase from [ndist, ny, nx] flat-field-normalised intensities.

    `distances` are the EFFECTIVE propagation distances of the planes, i.e.
    already divided by the squared magnification.  `alpha` regularises the
    inversion where the Taylor factors are small.
    """
    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    fy = cp.fft.fftfreq(data.shape[-2], d=voxelsize).astype('float32')
    fx, fy = cp.meshgrid(fx, fy)
    num = 0
    den = 0
    for j in range(data.shape[0]):
        taylor = 1 + wavelength * distances[j] * cp.pi * delta_beta * (fx**2 + fy**2)
        num += taylor * cp.fft.fft2(data[j].astype('complex64'))
        den += taylor**2
    num /= len(distances)
    den = den / len(distances) + alpha
    return cp.log(cp.real(cp.fft.ifft2(num / den))) * delta_beta * 0.5
