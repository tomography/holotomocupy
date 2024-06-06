
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage
from holotomo.chunking import gpu_batch

@gpu_batch
def remove_outliers(data, dezinger, dezinger_threshold):
    """Remove outliers"""

    if (int(dezinger) > 0):
        w = int(dezinger)
        # print(data.shape)
        fdata = ndimage.median_filter(data, [1,w, w])
        data[:] = cp.where(cp.logical_and(
            data > fdata, (data - fdata) > dezinger_threshold), fdata, data)
    return data

@gpu_batch
def linear(x,y,a,b):
    return a*x+b*y

@gpu_batch
def _dai_yuan_alpha(d,grad,grad0):
    divident = cp.zeros([d.shape[0]],dtype='float32')
    divisor = cp.zeros([d.shape[0]],dtype=d.dtype)
    for k in range(d.shape[0]):
        divident[k] = cp.linalg.norm(grad[k])**2
        divisor[k] = cp.vdot(d[k], grad[k]-grad0[k])
            
    return [divident,divisor]

def dai_yuan(d,grad,grad0):
    [divident,divisor] = _dai_yuan_alpha(d,grad,grad0)    
    alpha = np.sum(divident)/np.sum(divisor)
    return linear(grad,d,-1,alpha)