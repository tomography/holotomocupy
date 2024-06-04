import cupy as cp
import numpy as np
import tifffile

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
# streams for overlapping data transfers with computations
stream1 = cp.cuda.Stream(non_blocking=False)
stream2 = cp.cuda.Stream(non_blocking=False)
stream3 = cp.cuda.Stream(non_blocking=False)
global_chunk=16

def pinned_array(array):
    """Allocate pinned memory and associate it with numpy array"""

    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(
        mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src

# def gpu_batch(chunk=8):
#     def inner_batch(func):
#         def inner(inp, *args, **kwargs):
#             if isinstance(inp,cp.ndarray):
#                 out = func(inp, *args, st=0,end=inp.shape[0], **kwargs)
#                 return out
            
#             ntheta = inp.shape[0]
#             nchunk = int(np.ceil(ntheta/chunk))
            
#             inp_gpu = cp.empty([2,chunk,*inp.shape[1:]],dtype=inp.dtype)        
#             inp_pinned = np.empty([2,chunk,*inp.shape[1:]],dtype=inp.dtype)
                    
#             for k in range(nchunk+2):
#                 if(k > 0 and k < nchunk+1):
#                     with stream2:
#                         st, end = (k-1)*chunk, min(ntheta,k*chunk)
#                         tmp = func(inp_gpu[(k-1)%2],*args, st=st,end=end, **kwargs)
#                         if k==1:# first time we know the out shape
#                             out_gpu = cp.empty([2,chunk,*tmp.shape[1:]],dtype=tmp.dtype)
#                             out_pinned = np.empty([2,chunk,*tmp.shape[1:]],dtype=tmp.dtype)
#                             out = np.empty([inp.shape[0],*tmp.shape[1:]],dtype=out_gpu.dtype)     
#                         out_gpu[(k-1)%2] = tmp                    
#                 if(k > 1):
#                     with stream3:  # gpu->cpu copy        
#                         st, end = (k-2)*chunk, min(ntheta,(k-1)*chunk)
#                         s = end-st
#                         out_gpu[(k-2)%2].get(out=out_pinned[(k-2)%2])# contiguous copy, fast                                                            
#                         out[st:end] = out_pinned[(k-2)%2,:s]
                        
#                 if(k<nchunk):
#                     with stream1:  # cpu->gpu copy
#                         st, end = k*chunk, min(ntheta,(k+1)*chunk)
#                         s = end-st
#                         inp_pinned[k%2, :s] = inp[st:end]
#                         inp_gpu[k%2].set(inp_pinned[k%2])# contiguous copy, fast
                        
#                 stream1.synchronize()
#                 stream2.synchronize()
#                 stream3.synchronize()
#             return out
#         return inner
#     return inner_batch

# def gpu_batch2(chunk=8):
#     def inner_batch(func):
#         def inner(inp1, inp2, *args, **kwargs):
#             if isinstance(inp1,cp.ndarray):
#                 out = func(inp1, inp2, *args, st=0,end=inp1.shape[0], **kwargs)
#                 return out
            
#             ntheta = inp1.shape[0]
#             nchunk = int(np.ceil(ntheta/chunk))
            
#             inp1_gpu = cp.empty([2,chunk,*inp1.shape[1:]],dtype=inp1.dtype)        
#             inp1_pinned = np.empty([2,chunk,*inp1.shape[1:]],dtype=inp1.dtype)
#             inp2_gpu = cp.empty([2,chunk,*inp2.shape[1:]],dtype=inp2.dtype)        
#             inp2_pinned = np.empty([2,chunk,*inp2.shape[1:]],dtype=inp2.dtype)
                    
#             for k in range(nchunk+2):
#                 if(k > 0 and k < nchunk+1):
#                     with stream2:
#                         st, end = (k-1)*chunk, min(ntheta,k*chunk)
#                         tmp = func(inp1_gpu[(k-1)%2],inp2_gpu[(k-1)%2],*args, st=st,end=end, **kwargs)
#                         if k==1:# first time we know the out shape
#                             out_gpu = cp.empty([2,chunk,*tmp.shape[1:]],dtype=tmp.dtype)
#                             out_pinned = np.empty([2,chunk,*tmp.shape[1:]],dtype=tmp.dtype)
#                             out = np.empty([inp1.shape[0],*tmp.shape[1:]],dtype=out_gpu.dtype)     
#                         out_gpu[(k-1)%2] = tmp                    
#                 if(k > 1):
#                     with stream3:  # gpu->cpu copy        
#                         st, end = (k-2)*chunk, min(ntheta,(k-1)*chunk)
#                         s = end-st
#                         out_gpu[(k-2)%2].get(out=out_pinned[(k-2)%2])# contiguous copy, fast                                                            
#                         out[st:end] = out_pinned[(k-2)%2,:s]
                        
#                 if(k<nchunk):
#                     with stream1:  # cpu->gpu copy
#                         st, end = k*chunk, min(ntheta,(k+1)*chunk)
#                         s = end-st
#                         inp1_pinned[k%2, :s] = inp1[st:end]
#                         inp2_pinned[k%2, :s] = inp2[st:end]
#                         inp1_gpu[k%2].set(inp1_pinned[k%2])# contiguous copy, fast                        
#                         inp2_gpu[k%2].set(inp2_pinned[k%2])# contiguous copy, fast
                        
#                 stream1.synchronize()
#                 stream2.synchronize()
#                 stream3.synchronize()
#             return out
#         return inner
#     return inner_batch


def gpu_batch(func):
    def inner(*args, **kwargs):
        ntheta = args[0].shape[0]
        chunk = min(global_chunk,ntheta) # calculate based on data sizes
        nchunk = int(np.ceil(ntheta/chunk))
        if isinstance(args[0],cp.ndarray):
            out = func(*args, **kwargs)                
            return out
        
        inp_gpu=[]
        inp_pinned=[]
        out_gpu=[]
        out_pinned=[]
        out=[]

        ninp = 0
        for k in range(0,len(args)):
            if isinstance(args[k],np.ndarray) and args[k].shape[0]==ntheta:
                inp_gpu.append(cp.empty([2,chunk,*args[k].shape[1:]],dtype=args[k].dtype))
                inp_pinned.append(pinned_array(np.empty([2,chunk,*args[k].shape[1:]],dtype=args[k].dtype)))
                ninp+=1
            else:
                break
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with stream2:
                    st, end = (k-1)*chunk, min(ntheta,k*chunk)
                    inp_gpu0 = [a[(k-1)%2] for a in inp_gpu]
                    tmp = func(*inp_gpu0,*args[ninp:], **kwargs)
                    if not isinstance(tmp,list):                                                                              
                        tmp=[tmp]
                    if k==1:# first time we know the out shape          
                        nout = len(tmp)                            
                        for j in range(nout):
                            out_gpu.append(cp.empty([2,chunk,*tmp[j].shape[1:]],dtype=tmp[j].dtype))
                            out_pinned.append(pinned_array(np.empty([2,chunk,*tmp[j].shape[1:]],dtype=tmp[j].dtype)))
                            out.append(np.empty([ntheta,*tmp[j].shape[1:]],dtype=tmp[j].dtype))     
                    for j in range(nout):
                        out_gpu[j][(k-1)%2] = tmp[j]                    
            if(k > 1):
                with stream3:  # gpu->cpu copy        
                    for j in range(nout):
                        out_gpu[j][(k-2)%2].get(out=out_pinned[j][(k-2)%2])# contiguous copy, fast                                                            
                    
            if(k<nchunk):
                with stream1:  # cpu->gpu copy
                    st, end = k*chunk, min(ntheta,(k+1)*chunk)
                    s = end-st
                    for j in range(ninp):
                        inp_pinned[j][k%2, :s] = args[j][st:end]
                        inp_gpu[j][k%2].set(inp_pinned[j][k%2])# contiguous copy, fast                        
            
            stream3.synchronize()
            if(k > 1):
                st, end = (k-2)*chunk, min(ntheta,(k-1)*chunk)
                s = end-st
                for j in range(nout):
                    out[j][st:end] = out_pinned[j][(k-2)%2,:s]
                    
            stream1.synchronize()
            stream2.synchronize()
            stream3.synchronize()
        if nout==1:
            out=out[0]
        return out
    return inner


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
