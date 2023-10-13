"""Module for tomography."""

import cupy as cp
import numpy as np
import threading
import concurrent.futures as cf
from .radonusfft import radonusfft
from .utils import chunk
from functools import partial
import matplotlib
matplotlib.use('Agg')

class SolverTomo(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    pnz : int
        The number of pair slice partitions to process together
        simultaneously (multiple of nz)
    ngpus : int
        Number of gpus        
    """

    def __init__(self, theta, ntheta, nz, n, pnz, center, ngpus):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        # print(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus)
        if(nz % pnz > 0):
            print('Error, pnz is not a multiple of nz')
            exit()
        super().__init__(ntheta, pnz, n, center, theta.ctypes.data, ngpus)
        self.nz = nz

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo(self, u, gpu):
        """Radon transform (R)"""
        data = cp.zeros([self.ntheta, self.pnz, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        u = cp.ascontiguousarray(u)
        data = cp.ascontiguousarray(data)        
        self.fwd(data.data.ptr, u.data.ptr, gpu)
        return data

    def adj_tomo(self, data, gpu, filter=False):
        """Adjoint Radon transform (R^*)"""
        u = cp.zeros([self.pnz, self.n, self.n], dtype='complex64')
        u = cp.ascontiguousarray(u)
        data = cp.ascontiguousarray(data)
        # C++ wrapper, send pointers to GPU arrays
        self.adj(u.data.ptr, data.data.ptr, gpu, filter)
        return u

    def fwd_tomo_batch(self, u):
        """Batch of Tomography transform (R)"""
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        for ids in chunk(range(self.nz), self.pnz):
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            # Radon transform
            res_gpu = self.fwd_tomo(u_gpu, 0)
            # copy result to cpu
            res[:, ids] = res_gpu.get()
        return res

    def adj_tomo_batch(self, data):
        """Batch of adjoint Tomography transform (R*)"""
        res = np.zeros([self.nz, self.n, self.n], dtype='complex64')
        for ids in chunk(range(self.nz), self.pnz):
            # copy data part to gpu
            data_gpu = cp.array(data[:, ids])

            # Adjoint Radon transform
            res_gpu = self.adj_tomo(data_gpu, 0)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    def grad_tomo(self, xi0, K, init, titer, igpu):
        """G solver for rho||KRu-xi0||_2 by z-slice partitions"""
        # minimization functional
        def minf(KRu, gu):
            f = cp.linalg.norm(KRu-xi0)**2            
            return f
        u = init.copy()
        minf1 = 1e12
        for i in range(titer):           
            KRu = K*self.fwd_tomo(u, igpu)
            grad = self.adj_tomo(cp.conj(K)*(KRu-xi0), igpu)/(self.ntheta * self.n * 1.0)# for some reason float value is needed
            # update step
            u = u + 0.5*(-grad)
            # if(i%1==0):              
                # print('t',i,minf(KRu, -1))
            # minf0 = minf(KRu, None)                
            # if(minf1 < minf0):
            #     print('error in tomo', minf0, minf1)
            # minf1 = minf0
        return u

    def grad_tomo_multi_gpu(self, xi0, K, u, titer, lock, ids):
        """Pick GPU, copy data, run reconstruction"""
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()

        cp.cuda.Device(gpu).use()
        u_gpu = cp.array(u[ids])
        xi0_gpu = cp.array(xi0[:, ids])
        K_gpu = cp.array(K[:, ids])
        # reconstruct
        u_gpu = self.grad_tomo(xi0_gpu, K_gpu, u_gpu, titer, gpu)
        u[ids] = u_gpu.get()

        BUSYGPUS[gpu] = 0

        return u[ids]

    def grad_tomo_batch(self, xi0, K, init, titer, dbg=False):
        """CG solver for rho||KRu-xi0||_2 by z-slice partitions"""
        u = init.copy()
        ids_list = chunk(range(self.nz), self.pnz)
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for ui in e.map(partial(self.grad_tomo_multi_gpu, xi0, K, u, titer, lock), ids_list):
                u[np.arange(0, ui.shape[0])+shift] = ui
                shift += ui.shape[0]
        cp.cuda.Device(0).use()
        return u