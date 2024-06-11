
import cupy as cp
from holotomocupy.chunking import gpu_batch

@gpu_batch
def multiPaganin(data, distances, wavelength, voxelsize, delta_beta,  alpha):
    """ Phase retrieval based on the MultiPaganin method

    Parameters
    ----------
    data : ndarray, float32
        Input data for several distances, shape (ntheta,ndist,n,n) 
    distance : ndarray
        Distances in m, shape (ndist) 
    wavelength:
        Wave length in m
    voxelsize:
        Voxel size in m        
    delta_beta:
        Ratio between the real and imag components of the refractive index (u=delta+ibeta)
    alpha:
        Constant to avoid division by zero    

    Returns
    -------
    phase : ndarray
        Recovered phase of the object, shape [ntheta,n,n]
    """
    
    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)

    numerator = 0
    denominator = 0
    for j in range(0, data.shape[1]):
        rad_freq = cp.fft.fft2(data[:, j])
        taylorExp = 1 + wavelength * distances[j] * cp.pi * (delta_beta) * (fx**2+fy**2)
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + taylorExp**2

    numerator = numerator / len(distances)
    denominator = (denominator / len(distances)) + alpha

    phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
    phase = (delta_beta) * 0.5 * phase

    return phase

@gpu_batch
def CTFPurePhase(data, distances, wavelength, voxelsize, alpha):
    """
    Weak phase approximation from Cloetens et al. 2002
    
    Parameters
    ----------
    data : ndarray, float32
        Input data for several distances, shape (ntheta,ndist,n,n) 
    distance : ndarray
        Distances in m, shape (ndist) 
    wavelength:
        Wave length in m
    voxelsize:
        Voxel size in m        
    alpha:
        Constant to avoid division by zero    

    Returns
    -------
    phase : ndarray
        Recovered phase of the object, shape [ntheta,n,n]
    """

    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    numerator = 0
    denominator = 0
    for j in range(0, len(distances)):
        rad_freq = cp.fft.fft2(data[:, j])
        taylorExp = cp.sin(cp.pi*wavelength*distances[j]*(fx**2+fy**2))
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + 2*taylorExp**2
    numerator = numerator / len(distances)
    denominator = (denominator / len(distances)) + alpha
    phase = cp.real(cp.fft.ifft2(numerator / denominator))
    phase = 0.5 * phase
    return phase