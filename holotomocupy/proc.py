
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage
from holotomocupy.chunking import gpu_batch

@gpu_batch
def remove_outliers(data, dezinger, dezinger_threshold):
    """Remove outliers (dezinger)
    Parameters
    ----------
    data : ndarray
        Input 3D array
    dezinger: int
        Radius for the median filter
    dezinger_threshold: float
        Threshold for outliers
    
    Returns
    -------
    res : ndarray
        Output array    
    """
    res = data.copy()
    if (int(dezinger) > 0):
        w = int(dezinger)
        # print(data.shape)
        fdata = ndimage.median_filter(data, [1,w, w])
        res[:] = cp.where(cp.logical_and(
            data > fdata, (data - fdata) > dezinger_threshold), fdata, data)
    return res

@gpu_batch
def linear(x,y,a,b):
    """Linear operation res = ax+by

    Parameters
    ----------
    x,y : ndarray
        Input arrays
    a,b: float
        Input constants    
    
    Returns
    -------
    res : ndarray
        Output array
    """
    return a*x+b*y

@gpu_batch
def _dai_yuan_alpha(d,grad,grad0):
    divident = cp.zeros([d.shape[0]],dtype='float32')
    divisor = cp.zeros([d.shape[0]],dtype=d.dtype)
    for k in range(d.shape[0]):
        divident[k] = cp.linalg.norm(grad[k])**2
        divisor[k] = cp.real(cp.vdot(d[k], grad[k]-grad0[k]))
            
    return [divident,divisor]

def dai_yuan(d,grad,grad0):
    """Dai-Yuan direction for the CG scheme
    
    Parameters
    ----------
    d : ndarray        
        The Dai-Yuan direction from the previous iteration
    grad : ndarray        
        Gradient on the current iteration
    grad0 : ndarray        
        Gradient on the previous iteration    
    
    Returns
    -------
    res : ndarray
        New Dai-Yuan direction
    """
    
    [divident,divisor] = _dai_yuan_alpha(d,grad,grad0)    
    alpha = np.sum(divident)/np.sum(divisor)
    res = linear(grad,d,-1,alpha)
    return res 
