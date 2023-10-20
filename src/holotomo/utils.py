from scipy import ndimage
import numpy as np
import time
from itertools import islice

def tic():
    # Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        return time.time() - startTime_for_tictoc


def find_min_max(data):
    """Find min and max values according to histogram"""

    mmin = np.zeros(data.shape[0], dtype='float32')
    mmax = np.zeros(data.shape[0], dtype='float32')

    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:], 1000)
        stend = np.where(h > np.max(h)*0.005)
        st = stend[0][0]
        end = stend[0][-1]
        mmin[k] = e[st]
        mmax[k] = e[end+1]
        # mmin[k] = np.min(data[k])
        # mmax[k] = np.max(data[k])
    # print(mmin,mmax)
    return mmin, mmax


def paddata(data, ne):
    """Pad tomography projections"""
    n = data.shape[-1]
    datae = np.pad(data, ((0, 0), (0, 0), (ne//2-n//2, ne//2-n//2)), 'edge')
    return datae


def unpaddata(data, n):
    """Unpad tomography projections"""
    ne = data.shape[-1]
    return data[:, :, ne//2-n//2:ne//2+n//2]


def unpadobject(f, n):
    """Unpad 3d object"""
    ne = f.shape[-1]
    return f[:, ne//2-n//2:ne//2+n//2, ne//2-n//2:ne//2+n//2]


def interpdense(u, psi, lamd, flow):
    """Interpolate ADMM variables to a twice dense grid"""

    unew = ndimage.zoom(u, 2, order=1)
    psinew = ndimage.zoom(psi, (1, 2, 2), order=1)
    lamdnew = ndimage.zoom(lamd, (1, 2, 2), order=1)
    flownew = ndimage.zoom(flow, (1, 2, 2, 1), order=1)/2
    return unew, psinew, lamdnew, flownew

def chunk(iterable, size):
    it = iter(iterable)
    item = list(islice(it, size))
    while item:
        yield np.array(item)
        item = list(islice(it, size))