
import cupy as cp
from holotomo.chunking import gpu_batch

@gpu_batch
def multiPaganin(rads, dists, wavelength, voxelsize, delta_beta,  alpha):
    """
    Phase retrieval method based on the MultiPaganin method
    """
    fx = cp.fft.fftfreq(rads.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)

    numerator = 0
    denominator = 0
    for j in range(0, rads.shape[1]):
        rad_freq = cp.fft.fft2(rads[:, j])
        taylorExp = 1 + wavelength * dists[j] * cp.pi * (delta_beta) * (fx**2+fy**2)
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + taylorExp**2

    numerator = numerator / len(dists)
    denominator = (denominator / len(dists)) + alpha

    phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
    phase = (delta_beta) * 0.5 * phase

    return phase

@gpu_batch
def CTFPurePhase(rads, dists, wavelength, voxelsize, alpha):
    """
    weak phase approximation from Cloetens et al. 2002
    """
    fx = cp.fft.fftfreq(rads.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    numerator = 0
    denominator = 0
    for j in range(0, len(dists)):
        rad_freq = cp.fft.fft2(rads[:, j])
        taylorExp = cp.sin(cp.pi*wavelength*dists[j]*(fx**2+fy**2))
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + 2*taylorExp**2
    numerator = numerator / len(dists)
    denominator = (denominator / len(dists)) + alpha
    phase = cp.real(cp.fft.ifft2(numerator / denominator))
    phase = 0.5 * phase
    return phase