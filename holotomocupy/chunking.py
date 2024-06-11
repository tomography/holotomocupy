import cupy as cp
import numpy as np

global_chunk = 16


def gpu_batch(func):
    """
    Decorator for processing data by chunks on GPU
    
    Parameters
    ----------
    func : function for processing on GPU. The function should have syntax:
    [out1,out2,...] = func(in1, in2,.., par1,par2..), where 
    arrays in1,in2,.., out1,out2,.. have the same shape in the first dimension 
    treated as the dimension for chunking data processing on GPU. The func() should 
    perform processing on cupy arrays and return the result as a list of ndarray 
    or 1 ndarray.
    
    Example:
    nn=180
    a = np.zeros([nn,5,3])
    b = np.zeros([nn,5,3,2])
    
    def func(in1,in2,par):
        out1 = in1+in2[:,0]*par
        out2 = in1+in2[:,1]*par
        return [out1,out2]
    """
    
    def inner(*args, **kwargs):
        nn = args[0].shape[0]
        chunk = min(global_chunk, nn)  
        nchunk = int(np.ceil(nn/chunk))

        # if array is on gpu then just run the function
        if isinstance(args[0], cp.ndarray):
            out = func(*args, **kwargs)
            return out
        
        #else do processing by chunks
        inp_gpu = []
        out = []

        # determine the number o inputs
        ninp = 0
        for k in range(0, len(args)):
            if (isinstance(args[k], np.ndarray) or isinstance(args[k], cp.ndarray)) and args[k].shape[0] == nn:
                inp_gpu.append(
                    cp.empty([chunk, *args[k].shape[1:]], dtype=args[k].dtype))
                ninp += 1
            else:
                break
        
        # run by chunks
        for k in range(nchunk):
            st, end = k*chunk, min(nn, (k+1)*chunk)
            s = end-st

            # copy to gpu
            for j in range(ninp):
                inp_gpu[j][:s].set(args[j][st:end])
            inp_gpu0 = [a for a in inp_gpu]
            
            #run function
            out_gpu = func(*inp_gpu0, *args[ninp:], **kwargs)

            if not isinstance(out_gpu, list):
                out_gpu = [out_gpu]
            
            if k == 0:  # first time we know the out shape
                nout = len(out_gpu)
                for j in range(nout):
                    out.append(
                        np.empty([nn, *out_gpu[j].shape[1:]], dtype=out_gpu[j].dtype))                                
            
            # copy from gpu            
            for j in range(nout):
                out_gpu[j][:s].get(out=out[j][st:end])  # contiguous copy, fast
                    
        if nout == 1:
            out = out[0]
        return out
    return inner




#####TO TRY WITh PINNED MEMORY

#cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
# streams for overlapping data transfers with computations
# stream1 = cp.cuda.Stream(non_blocking=False)
# stream2 = cp.cuda.Stream(non_blocking=False)
# stream3 = cp.cuda.Stream(non_blocking=False)

# def pinned_array(array):
#     """Allocate pinned memory and associate it with numpy array"""

#     mem = cp.cuda.alloc_pinned_memory(array.nbytes)
#     src = np.frombuffer(
#         mem, array.dtype, array.size).reshape(array.shape)
#     src[...] = array
#     return src


# def gpu_batch(func):
#     #cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

#     def inner(*args, **kwargs):
#         nn = args[0].shape[0]
#         chunk = min(global_chunk, nn)  # calculate based on data sizes
#         nchunk = int(np.ceil(nn/chunk))
#         if isinstance(args[0], cp.ndarray):
#             out = func(*args, **kwargs)
#             return out

#         inp_gpu = []
#         out_gpu = []
#         out = []

#         ninp = 0
#         for k in range(0, len(args)):
#             if isinstance(args[k], np.ndarray) and args[k].shape[0] == nn:
#                 inp_gpu.append(
#                     cp.empty([2, chunk, *args[k].shape[1:]], dtype=args[k].dtype))
#                 ninp += 1
#             else:
#                 break
#         for k in range(nchunk+2):
#             if (k > 0 and k < nchunk+1):
#                 with stream2:
#                     st, end = (k-1)*chunk, min(nn, k*chunk)
#                     inp_gpu0 = [a[(k-1) % 2] for a in inp_gpu]
#                     tmp = func(*inp_gpu0, *args[ninp:], **kwargs)
#                     if not isinstance(tmp, list):
#                         tmp = [tmp]
#                     if k == 1:  # first time we know the out shape
#                         nout = len(tmp)
#                         for j in range(nout):
#                             out_gpu.append(
#                                 cp.empty([2, chunk, *tmp[j].shape[1:]], dtype=tmp[j].dtype))
#                             out.append(
#                                 np.empty([nn, *tmp[j].shape[1:]], dtype=tmp[j].dtype))
#                     for j in range(nout):
#                         out_gpu[j][(k-1) % 2] = tmp[j]
#             if (k > 1):
#                 with stream3:  # gpu->cpu copy
#                     for j in range(nout):
#                         # out_gpu[j][(k-2) % 2].get(out=out_pinned[j]
#                         #                           [(k-2) % 2])  # contiguous copy, fast
#                         st, end = (k-2)*chunk, min(nn, (k-1)*chunk)
#                         s = end-st
#                         out_gpu[j][(k-2) % 2,:s].get(out=out[j][st:end])  # contiguous copy, fast

#             if (k < nchunk):
#                 with stream1:  # cpu->gpu copy
#                     st, end = k*chunk, min(nn, (k+1)*chunk)
#                     s = end-st
#                     for j in range(ninp):
#                         # inp_pinned[j][k % 2, :s] = args[j][st:end]
#                         # # contiguous copy, fast
#                         # inp_gpu[j][k % 2].set(inp_pinned[j][k % 2])
#                         #inp_pinned[j][k % 2, :s] = args[j][st:end]
#                         # contiguous copy, fast
#                         inp_gpu[j][k % 2,:s].set(args[j][st:end])

#             # stream3.synchronize()
#             # if (k > 1):
#             #     st, end = (k-2)*chunk, min(nn, (k-1)*chunk)
#             #     s = end-st
#             #     for j in range(nout):
#             #         out[j][st:end] = out_pinned[j][(k-2) % 2, :s]

#             stream1.synchronize()
#             stream2.synchronize()
#             stream3.synchronize()
#         if nout == 1:
#             out = out[0]
#         return out
#     return inner


# @gpu_batch(8)
# def S(psi, shift):
#     """Shift operator"""
#     n = psi.shape[-1]
#     p = shift.copy()#[st:end]
#     res = psi.copy()
#     # if p.shape[0]!=res.shape[0]:
#         # res = cp.tile(res,(shift.shape[0],1,1))
#     res = cp.pad(res,((0,0),(n//2,n//2),(n//2,n//2)),'symmetric')
#     x = cp.fft.fftfreq(2*n).astype('float32')
#     [x, y] = cp.meshgrid(x, x)
#     pp = cp.exp(-2*cp.pi*1j * (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
#     res = cp.fft.ifft2(pp*cp.fft.fft2(res))
#     res = res[:,n//2:-n//2,n//2:-n//2]
#     return [res,res]

# cp.random.seed(10)
# a = tifffile.imread('../../tests/data/delta-chip-192.tiff')
# a = a+1j*a/2
# b = np.empty_like(a)
# shift = np.array(np.random.random([a.shape[0], 2]), dtype='float32')+3


# [b,b0] = S(a,shift)
# [bb,bb0] = S(cp.array(a),cp.array(shift))

# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(b[19].real,cmap='gray')
# plt.colorbar()
# plt.savefig('t1.png')

# plt.figure()
# plt.imshow(bb[19].real.get(),cmap='gray')
# plt.colorbar()
# plt.savefig('t.png')

# # # print(np.linalg.norm(c))
# print(np.linalg.norm(b))
# print(cp.linalg.norm(bb))
# print(np.linalg.norm(b.real-bb.get().real))


# # print(np.linalg.norm(b-c))
