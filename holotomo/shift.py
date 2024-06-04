import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage
from .chunking import gpu_batch
from .chunking import global_chunk

# @batch
# def S(psi, st=0, end=None, shift=0):
#     """Shift operator"""
#     res = psi.copy()        
#     if end==None:
#         end = shift.shape[0]        
#     if np.all(shift==0):
#         return res
#     p = cp.array(shift)[st:end]
#     for k in range(p.shape[0]):
#         res[k] = ndimage.shift(res[k], p[k], order=2, mode='nearest', prefilter=True)
#     return res


@gpu_batch
def S(psi, shift=None):
    """Shift operator"""    
    res = psi.copy()        
    if np.all(shift==0):
        return res
    p = cp.pad(shift,(0,res.shape[0]-shift.shape[0]))
    n = psi.shape[-1]
    res = cp.pad(res,((0,0),(n//2,n//2),(n//2,n//2)),'symmetric')    
    x = cp.fft.fftfreq(2*n).astype('float32')
    [x, y] = cp.meshgrid(x, x)    
    pp = cp.exp(-2*cp.pi*1j * (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res = cp.fft.ifft2(pp*cp.fft.fft2(res))
    res = res[:,n//2:-n//2,n//2:-n//2]
    return res

@gpu_batch
def ST(psi, shift=None):
    """Shift operator"""    
    res = psi.copy()        
    if np.all(shift==0):
        return res
    p = cp.pad(shift,(0,res.shape[0]-shift.shape[0]))
    n = psi.shape[-1]
    res = cp.pad(res,((0,0),(n//2,n//2),(n//2,n//2)),'symmetric')    
    x = cp.fft.fftfreq(2*n).astype('float32')
    [x, y] = cp.meshgrid(x, x)    
    pp = cp.exp(2*cp.pi*1j * (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res = cp.fft.ifft2(pp*cp.fft.fft2(res))
    res = res[:,n//2:-n//2,n//2:-n//2]
    return res

# import tifffile
# a = tifffile.imread('../../tests/data/delta-chip-192.tiff')
# # a = tifffile.imread('/data/vnikitin/modeling/r_256_32_4/r00256.tiff')
# # a = tifffile.imread('/data/vnikitin/modeling/ref_3d_ald_256_0.tiff')
# a = a-1j*a
# shift = np.array(np.random.random([a.shape[0], 2]), dtype='float32')+3
# b = S(a, shift)
# aa = S(b, -shift)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(b[b.shape[0]//2].real,cmap='gray')
# plt.colorbar()
# plt.savefig('t.png')

# plt.figure()
# plt.imshow(aa[b.shape[0]//2].real,cmap='gray')
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
