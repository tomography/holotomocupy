
import cupy as cp
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