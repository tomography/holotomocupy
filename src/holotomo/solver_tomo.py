"""Module for tomography."""

import cupy as cp
import numpy as np
import threading
import concurrent.futures as cf
from functools import partial
from .tomo import tomo
from .utils import chunk

class SolverTomo(tomo):
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
    """

    def __init__(self, theta, ntheta, nz, n, pnz, center):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        if(nz % pnz > 0):
            print('Error, pnz is not a multiple of nz')
            exit()
        super().__init__(ntheta, pnz, n, center, theta.ctypes.data, 1)
        self.nz = nz

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo(self, u, gpu=0):
        """Radon transform (R)"""
        data = cp.zeros([self.ntheta, self.pnz, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        u = cp.ascontiguousarray(u)
        data = cp.ascontiguousarray(data)        
        self.fwd(data.data.ptr, u.data.ptr, gpu)
        return data

    def adj_tomo(self, data, gpu=0, filter=False):
        """Adjoint Radon transform (R^*)"""
        u = cp.zeros([self.pnz, self.n, self.n], dtype='complex64')
        u = cp.ascontiguousarray(u)
        data = cp.ascontiguousarray(data)
        # C++ wrapper, send pointers to GPU arrays
        self.adj(u.data.ptr, data.data.ptr, gpu, filter)
        return u
    
    def paddata(self, data, ne):
        """Pad tomography projections"""
        n = data.shape[-1]
        datae = np.pad(data, ((0, 0), (0, 0), (ne//2-n//2, ne//2-n//2)), 'edge')
        return datae

    def unpaddata(self, data, n):
        """Unpad tomography projections"""
        ne = data.shape[-1]
        return data[:, :, ne//2-n//2:ne//2+n//2]


    def unpadobject(self, f, n):
        """Unpad 3d object"""
        ne = f.shape[-1]
        return f[:, ne//2-n//2:ne//2+n//2, ne//2-n//2:ne//2+n//2]

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

    def line_search(self, minf, gamma, u, fu, d, fd):
        """ Line search for the step sizes gamma"""

        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-12):
            gamma *= 0.5
        if(gamma <= 1e-12):  # direction not found
            #print('no direction')
            gamma = 0
        return gamma
    
    def cg_tomo(self, data, init, piter):
        """Conjugate gradients method for tomography"""

        # minimization functional
        def minf(u,fu):
            f = cp.linalg.norm(fu-data)**2            
            return f        
        u = init.copy()
        
        gamma = 1# init gamma as a large value
        for i in range(piter):
            fu = self.fwd_tomo(u)
            grad = self.adj_tomo(fu-data)/(self.ntheta*self.n)
            
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (1e-15+(cp.sum(cp.conj(d)*(grad-grad0))))*d
            grad0 = grad
            # line search
            fd = self.fwd_tomo(d)
            gamma = self.line_search(minf, gamma, u, fu, d, fd)
            u = u + gamma*d
            print(f'{i}) {gamma=}, err={minf(u,fu)}')
        
        return u

    def cg_tomo_batch(self, data, init, piter):
        """Batch of Holography transforms"""
        res = np.zeros([self.nz, self.n, self.n], dtype='complex64')
        for ids in chunk(range(self.nz), self.pnz):
            print(f'Processing slices: {ids}')
            # copy data part to gpu
            data_gpu = cp.ascontiguousarray(cp.array(data[:,ids]))
            init_gpu = cp.array(init[ids])
            # Radon transform
            res_gpu = self.cg_tomo(data_gpu, init_gpu, piter)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res