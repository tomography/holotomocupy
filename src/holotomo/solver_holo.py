"""Module for tomography."""

import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage
from .holo import holo
from .utils import chunk
PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]


class SolverHolo():

    def __init__(self, ntheta, nz, n, ptheta, voxelsize, energy, distances, magnification):
        self.n = n
        self.nz = nz
        self.voxelsize = voxelsize
        self.energy = energy
        self.distances = distances
        self.ntheta = ntheta
        self.ptheta = ptheta
        self.magnification = magnification
        
        # Precalculate Fresnel propagators for different distances
        self.fP = cp.zeros([len(distances),nz,n],dtype='complex64')
        for k in distances:
            fx = cp.fft.fftshift(cp.fft.fftfreq(n,d=voxelsize))
            [fx,fy] = cp.meshgrid(fx,fx)
            for i,d in enumerate(distances):
                self.fP[i] = cp.exp(-1j*cp.pi*self.wavelength()*d*(fx**2+fy**2))
        
        #CUDA C class for faster USFFT and padding
        self.cl_holo = holo(2*n,2*nz, ptheta)
                        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        #self.free()
        pass
    
    def wavelength(self):
        """Wavelength"""
        return PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy

    def exptomo(self, psi):
        """Exp representation of projections, exp(i\psi\pi/\lambda)"""
        return np.exp(1j*psi * self.voxelsize * 2*cp.pi / self.wavelength())
    
    def mlog(self,psi):
        res = psi.copy()
        res[np.abs(psi)<1e-32]=1e-32
        res = np.log(res)
        return res
    
    def logtomo(self, psi):
        """Log representation of projections, -i/\nu log(psi)"""
        return -1j * self.wavelength()/ (2*cp.pi) * self.mlog(psi) / self.voxelsize
    
    def line_search(self, minf, gamma, u, fu, d, fd):
        """ Line search for the step sizes gamma"""

        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-12):
            gamma *= 0.5
        if(gamma <= 1e-12):  # direction not found
            #print('no direction')
            gamma = 0
        return gamma
    
    def fwd_pad(self,f):
        """Data padding"""
        
        pad_width = self.n//2
        fpad = cp.zeros([self.ptheta,self.nz+2*pad_width,self.n+2*pad_width],dtype='complex64')
        # symmetric padding (CUDA C)
        self.cl_holo.fwd_padsym(fpad.data.ptr,f.data.ptr,pad_width,0)
        return fpad
    
    def adj_pad(self,fpad):
        """Adjoint operator for data padding"""
        
        pad_width = self.n//2
        f = cp.zeros([self.ptheta,self.nz, self.n],dtype='complex64')
        # Adjoint to symmetric padding
        self.cl_holo.adj_padsym(f.data.ptr,fpad.data.ptr,pad_width,0)
        return f
    
    def fwd_resample(self,f,magnification):
        """Data magnification via Fourier domain"""
        # fr = ndimage.zoom(f,(1,magnification/2,magnification/2))
        # st = (fr.shape[1])//2-self.n//2
        # end = st+self.n
        # fr = fr[:,st:end,st:end]
        
        # print(fr.shape)
        xk = -((cp.arange(-self.n/2,self.n/2)/self.n)/magnification).astype('float32')
        [xk,yk] = cp.meshgrid(xk,xk)
        fr = cp.zeros([self.ptheta,self.n,self.n],dtype='complex64') 
        f = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(f)))
        f = cp.ascontiguousarray(f)
        self.cl_holo.fwd_usfft(fr.data.ptr,f.data.ptr,xk.data.ptr,yk.data.ptr,0)
        fr = fr/self.n/self.n/4
        return fr

    def adj_resample(self,fr,magnification):
        """Adjoint to data magnification via Fourier domain"""
        # f = cp.zeros([self.ptheta,2*self.n,2*self.n],dtype='complex64')
        # fr = ndimage.zoom(fr,(1,2/magnification,2/magnification))
       
        # st = self.n-fr.shape[1]//2
        # end = self.n-fr.shape[1]//2+fr.shape[1]
        
        # f[:,st:end,st:end]=fr*(magnification/2)**2
        xk = -((cp.arange(-self.n/2,self.n/2)/self.n)/magnification).astype('float32')
        [xk,yk] = cp.meshgrid(xk,xk)
        f = cp.zeros([self.ptheta,2*self.n,2*self.n],dtype='complex64') 
        fr = cp.ascontiguousarray(fr)
        self.cl_holo.adj_usfft(f.data.ptr, fr.data.ptr,xk.data.ptr,yk.data.ptr,0)                 
        f = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(f)))
        return f
    
    def fwd_propagate(self,f,fP):
        """Fresnel transform"""
        
        ff = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(f)))
        ff = ff*fP
        ff = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(ff)))
        return ff
    
    def adj_propagate(self,ff,fP):
        """Adjoint to Fresnel transform"""
        
        f = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ff)))
        f = f*cp.conj(fP) 
        f = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(f)))
        return f
        
    def fwd_holo(self, psi, prb):
        """holography transform: padding, magnification, multiplication by probe, Fresnel transform"""
        
        data = cp.zeros([len(self.distances),self.ptheta,self.nz, self.n], dtype='complex64')
        #for itheta in range(self.ptheta):
        # print(psi.shape)
        psi_pad = self.fwd_pad(psi)        
        psi_pad = cp.ascontiguousarray(psi_pad)
        for i in range(len(self.distances)):            
            psir = self.fwd_resample(psi_pad,self.magnification[i]*2)
            psir *= prb[i]
            data[i] = self.fwd_propagate(psir,self.fP[i])                            
        
        return data
    
    # def fwd_holo(self, psi, prb):
    #     """holography transform: padding, magnification, multiplication by probe, Fresnel transform"""
        
    #     data = cp.zeros([len(self.distances),self.ptheta,self.nz, self.n], dtype='complex64')
    #     #for itheta in range(self.ptheta):
    #     # print(psi.shape)
        
    #     for i in range(len(self.distances)):                        
    #         psi *= prb[i]
    #         datar = self.fwd_propagate(psi,self.fP[i])                            
    #         datar_pad = self.fwd_pad(datar)        
    #         datar_pad = cp.ascontiguousarray(datar_pad)
    #         data[i] = self.fwd_resample(datar_pad,self.magnification[i]*2)
        
        # return data


    def adj_holo(self, data, prb):
        """Adjoint holography transform wrt object (adjoint operations in reverse order))"""
        
        psi = cp.zeros([self.ptheta,self.nz, self.n], dtype='complex64')
        for i in range(len(self.distances)):
            psir = self.adj_propagate(data[i],self.fP[i])       
            psir *= cp.conj(prb[i])
            psi_pad = self.adj_resample(psir,self.magnification[i]*2)
            psi += self.adj_pad(psi_pad)
        return psi
    
    # def adj_holo(self, data, prb):
    #     """Adjoint holography transform wrt object (adjoint operations in reverse order))"""
        
    #     psi = cp.zeros([self.ptheta,self.nz, self.n], dtype='complex64')
    #     for i in range(len(self.distances)):
    #         datar_pad = self.adj_resample(data[i],self.magnification[i]*2)
    #         datar = self.adj_pad(datar_pad)
    #         psir = self.adj_propagate(datar,self.fP[i])       
    #         psir *= cp.conj(prb[i])
    #         psi += psir
    #     return psi
    
    def cg_holo(self, data, init, prb,  piter):
        """Conjugate gradients method for holography"""

        # minimization functional
        def minf(psi,fpsi):
            f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2            
            # f = cp.linalg.norm(cp.abs(fpsi)**2-data)**2            
            return f        
        psi = init.copy()
        
        maxprb = cp.max(cp.abs(prb))
        gamma = 1# init gamma as a large value
        for i in range(piter):
            fpsi = self.fwd_holo(psi,prb)
            grad = self.adj_holo(
               fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), prb)/maxprb**2
            #grad = self.adj_holo(
                 #(cp.abs(fpsi)**2-data)*fpsi,prb)/maxprb**2
            
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (1e-30+(cp.sum(cp.conj(d)*(grad-grad0))))*d
            grad0 = grad
            # line search
            fd = self.fwd_holo(d, prb)
            gamma = self.line_search(minf, gamma, psi, fpsi, d, fd)
            psi = psi + gamma*d
            print(f'{i}) {gamma=}, err={minf(psi,fpsi)}')
        
        return psi

    def fwd_holo_batch(self, psi, prb):
        """Batch of Holography transforms"""
        res = np.zeros([len(self.distances),self.ntheta, self.nz, self.n], dtype='complex64')
        prb_gpu = cp.array(prb)
        for ids in chunk(range(self.ntheta), self.ptheta):
            # copy data part to gpu
            psi_gpu = cp.array(psi[ids])            
            # Radon transform
            res_gpu = self.fwd_holo(psi_gpu, prb_gpu)
            # copy result to cpu
            res[:,ids] = res_gpu.get()
        return res
    
    def adj_holo_batch(self, fpsi, prb):
        """Batch of Holography transforms"""
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        for ids in chunk(range(self.ntheta), self.ptheta):
            # copy data part to gpu
            fpsi_gpu = cp.array(fpsi[:,ids])
            prb_gpu = cp.array(prb)
            # Radon transform
            res_gpu = self.adj_holo(fpsi_gpu, prb_gpu)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    def cg_holo_batch(self, data, init, prb, piter):
        """Batch of Holography transforms"""
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        prb_gpu = cp.array(prb)            
        for ids in chunk(range(self.ntheta), self.ptheta):
            # print(f'Processing angles: {ids}')
            # copy data part to gpu
            data_gpu = cp.array(data[:,ids])
            init_gpu = cp.array(init[ids])
            # Radon transform
            res_gpu = self.cg_holo(data_gpu, init_gpu,prb_gpu, piter)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res


#######  WORK
    # def grad_holo(self, data, psi, prb, piter, recover_prb):
    #     def minf(fpsi, psi):
    #         f = cp.linalg.norm(cp.sqrt(cp.abs(fpsi)) - cp.sqrt(data))**2
    #         return f

    #     for i in range(piter):

    #         # 1) object retrieval subproblem with fixed prbs
    #         # sum of forward operators associated with each prb
    #         # sum of abs value of forward operators            
    #         absfpsi = data*0
    #         absfpsi += cp.abs(self.fwd_holo(psi, prb))**2                                
                        
    #         a = cp.sum(cp.sqrt(absfpsi*data))
    #         b = cp.sum(absfpsi)
    #         prb *= (a/b)
    #         absfpsi *= (a/b)**2
            
    #         gradpsi = cp.zeros(
    #             [self.ntheta, self.nz, self.n], dtype='complex64')
    #         fpsi = self.fwd_holo(psi, prb)
    #         afpsi = self.adj_holo(fpsi, prb)
                
    #         r = cp.real(cp.sum(psi*cp.conj(afpsi)) /
    #             (cp.sum(afpsi*cp.conj(afpsi))+1e-32))
                
    #         gradpsi += self.adj_holo(
    #             fpsi - cp.sqrt(data)*fpsi/(cp.sqrt(absfpsi)+1e-32), prb)                
    #         gradpsi *= r/2
    #         # update psi
    #         psi = psi + 0.5 * (-gradpsi)
    #         fpsi = self.fwd_holo(psi, prb)
 
    #         print(i, minf(fpsi, psi))
 
    #         if (recover_prb):
    #             if(i == 0):
    #                gradprb = prb*0
    #             # 2) prb retrieval subproblem with fixed object
    #             # sum of forward operators associated with each prb

    #             absfprb = cp.abs(self.fwd_holo(psi, prb))**2                        

    #             fprb = self.fwd_holo(psi, prb)
    #             afprb = self.adjprb_holo(fprb, psi)
    #             r = cp.real(
    #                 cp.sum(prb*cp.conj(afprb))/(cp.sum(afprb*cp.conj(afprb))+1e-32))
    #             #print(f'{r=}')
    #             # take gradient
    #             gradprb = self.adjprb_holo(
    #                 fprb - cp.sqrt(data) * fprb/(cp.sqrt(absfprb)+1e-32), psi)
    #             gradprb *= r/2
    #             prb = prb + 0.5 * (-gradprb)
 
    #     return psi, prb
