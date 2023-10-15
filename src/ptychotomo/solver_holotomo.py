"""Module for tomography."""

import cupy as cp
import numpy as np
import threading
import concurrent.futures as cf
from .utils import chunk
from functools import partial
from .holotomo import holotomo

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class SolverHolotomo():

    def __init__(self, ntheta, nz, n, voxelsize, energy, distances, magnification):
        self.cl_holotomo=holotomo(2*n,2*nz,1)
        self.n = n
        self.nz = nz
        self.voxelsize = voxelsize
        self.energy = energy
        self.distances = distances
        self.ntheta = ntheta
        self.magnification = magnification
        self.fP = cp.zeros([len(distances),nz,n],dtype='complex64')
        for k in distances:
            fx = cp.fft.fftshift(cp.fft.fftfreq(n,d=voxelsize))
            [fx,fy] = cp.meshgrid(fx,fx)
            for i,d in enumerate(distances):
                lamd = 2*cp.pi/self.wavenumber()
                self.fP[i] = cp.exp(-1j*cp.pi*lamd*d*(fx**2+fy**2))
        print(f'{self.wavenumber()=}')
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        #self.free()
        pass

    def mlog(self, psi):
        res = psi.copy()
        res[np.abs(psi) < 1e-32] = 1e-32
        res = np.log(res)
        return res

    def wavenumber(self):
        """Wave number index"""
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy)

    def exptomo(self, psi):
        """Exp representation of projections, exp(i\nu\psi)"""
        return np.exp(1j*psi * self.voxelsize * self.wavenumber())

    def logtomo(self, psi):
        """Log representation of projections, -i/\nu log(psi)"""
        return -1j / self.wavenumber() * self.mlog(psi) / self.voxelsize

    # Line search for the step sizes gamma
    def line_search(self, minf, gamma, u, fu, d, fd):
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-12):
            gamma *= 0.5
        if(gamma <= 1e-12):  # direction not found
            #print('no direction')
            gamma = 0
        return gamma
    
    def fwd_pad(self,f):
        
        pad_width = self.n//2
        fpad = cp.zeros([self.n+2*pad_width,self.n+2*pad_width],dtype='complex64')
        # print(cp.linalg.norm(f))
        # f = cp.ascontiguousarray(f)
        self.cl_holotomo.fwd_padsym(fpad.data.ptr,f.data.ptr,pad_width,0)
        # for k in range(self.n+2*pad_width):
        #     for j in range(self.n+2*pad_width):
        #         if (k < pad_width):
        #             kk = pad_width-k-1
        #         elif (k >= self.n + pad_width):
        #             kk = 2*self.n-k+pad_width-1            
        #         else:                
        #             kk = k-pad_width
                    
        #         if (j < pad_width):
        #             jj = pad_width-j-1            
        #         elif (j >= self.n + pad_width):
        #             jj = 2*self.n-j+pad_width-1
        #         else:
        #             jj= j-pad_width       
        #         fpad[k,j]=f[kk,jj]
            
        return fpad
    
    def adj_pad(self,fpad):
        pad_width = self.n//2
        f = cp.zeros([self.n, self.n],dtype='complex64')
        self.cl_holotomo.adj_padsym(f.data.ptr,fpad.data.ptr,pad_width,0)
        # for k in range(self.n+2*pad_width):
        #     for j in range(self.n+2*pad_width):
        #         if (k < pad_width):
        #             kk = pad_width-k-1
        #         elif (k >= self.n + pad_width):
        #             kk = 2*self.n-k+pad_width-1            
        #         else:                
        #             kk = k-pad_width
                    
        #         if (j < pad_width):
        #             jj = pad_width-j-1            
        #         elif (j >= self.n + pad_width):
        #             jj = 2*self.n-j+pad_width-1
        #         else:
        #             jj= j-pad_width       
        #         f[kk,jj]+=fpad[k,j]
        return f
    
    def fwd_resample(self,f,magnification):
        xk = -((cp.arange(-self.n/2,self.n/2)/self.n)/magnification).astype('float32')
        [xk,yk] = cp.meshgrid(xk,xk)
        fr = cp.zeros([self.n,self.n],dtype='complex64') 
        f = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(f)))
        self.cl_holotomo.fwd_usfft(fr.data.ptr,f.data.ptr,xk.data.ptr,yk.data.ptr,0)
        fr = fr/self.n/self.n/4
        return fr

    def adj_resample(self,fr,magnification):
        xk = -((cp.arange(-self.n/2,self.n/2)/self.n)/magnification).astype('float32')
        [xk,yk] = cp.meshgrid(xk,xk)
        f = cp.zeros([2*self.n,2*self.n],dtype='complex64') 
        self.cl_holotomo.adj_usfft(f.data.ptr, fr.data.ptr,xk.data.ptr,yk.data.ptr,0)                 
        f = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(f)))
        return f
    
    def fwd_propagate(self,f,fP):
        ff = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(f)))
        ff = ff*fP 
        ff = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(ff)))
        return ff
    
    def adj_propagate(self,ff,fP):
        f = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ff)))
        f = f*cp.conj(fP) 
        f = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(f)))
        return f
        
    def fwd_holotomo(self, psi, prb):
        """Holotomography transform (FQ)"""
        data = cp.zeros([self.ntheta,len(self.distances),self.nz, self.n], dtype='complex64')
        for itheta in range(self.ntheta):
            for i in range(len(self.distances)):
                psi_pad = self.fwd_pad(psi[itheta])
                psir = self.fwd_resample(psi_pad,self.magnification[i]*2)
                psir *= prb[itheta,i]
                data[itheta,i] = self.fwd_propagate(psir,self.fP[i])                
        return data

    def adj_holotomo(self, data, prb):
        """Holotomography transform (FQ)"""
        psi = cp.zeros([self.ntheta,self.nz, self.n], dtype='complex64')
        for itheta in range(self. ntheta):
            for i in range(len(self.distances)):
                psir = self.adj_propagate(data[itheta,i],self.fP[i])       
                psir *= cp.conj(prb[itheta,i])
                psi_pad = self.adj_resample(psir,self.magnification[i]*2)
                psi[itheta] += self.adj_pad(psi_pad)
        return psi

    def adjprb_holotomo(self, data, psi):
        """Holotomography transform (FQ)"""
        prb = cp.zeros([self.ntheta,self.nz, self.n], dtype='complex64')
        for itheta in range(self. ntheta):
            for i,d in enumerate(self.distances):
                fdata = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(data[itheta,i])))
                prb[itheta] += np.conj(psi[itheta])*np.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(fdata*cp.conj(self.fP[i]))))
        return prb
    
        # Conjugate gradients for ptychography

    def cg_holotomo(self, data, init, prb,  piter, model):
        # minimization functional
        def minf(psi, fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(cp.abs(fpsi)**2-2*data * self.mlog(cp.abs(fpsi)))
            return f

        psi = init.copy()
        gamma = 1#/self.n  # init gamma as a large value
        for i in range(piter):
            fpsi = self.fwd_holotomo(psi,prb)
            if model == 'gaussian':
                grad = self.adj_holotomo(
                    fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), prb)
            elif model == 'poisson':
                grad = self.adj_holotomo(fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32),prb)
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    ((cp.sum(cp.conj(d)*(grad-grad0))))*d
            grad0 = grad
            # line search
            fd = self.fwd_holotomo(d, prb)
            gamma = self.line_search(minf, gamma, psi, fpsi, d, fd)
            psi = psi + gamma*d
            print(gamma,minf(psi, fpsi))
        if(cp.amax(cp.abs(cp.angle(psi))) > 3.14):
            print('possible phase wrap, max computed angle',
                  cp.amax(cp.abs(cp.angle(psi))))

        return psi

    def grad_holotomo(self, data, psi, prb, piter, recover_prb):
        def minf(fpsi, psi):
            f = cp.linalg.norm(cp.sqrt(cp.abs(fpsi)) - cp.sqrt(data))**2
            return f

        for i in range(piter):

            # 1) object retrieval subproblem with fixed prbs
            # sum of forward operators associated with each prb
            # sum of abs value of forward operators            
            absfpsi = data*0
            absfpsi += cp.abs(self.fwd_holotomo(psi, prb))**2                                
                        
            a = cp.sum(cp.sqrt(absfpsi*data))
            b = cp.sum(absfpsi)
            prb *= (a/b)
            absfpsi *= (a/b)**2
            
            gradpsi = cp.zeros(
                [self.ntheta, self.nz, self.n], dtype='complex64')
            fpsi = self.fwd_holotomo(psi, prb)
            afpsi = self.adj_holotomo(fpsi, prb)
                
            r = cp.real(cp.sum(psi*cp.conj(afpsi)) /
                (cp.sum(afpsi*cp.conj(afpsi))+1e-32))
                
            gradpsi += self.adj_holotomo(
                fpsi - cp.sqrt(data)*fpsi/(cp.sqrt(absfpsi)+1e-32), prb)                
            gradpsi *= r/2
            # update psi
            psi = psi + 0.5 * (-gradpsi)
            fpsi = self.fwd_holotomo(psi, prb)
 
            print(i, minf(fpsi, psi))
 
            if (recover_prb):
                if(i == 0):
                   gradprb = prb*0
                # 2) prb retrieval subproblem with fixed object
                # sum of forward operators associated with each prb

                absfprb = cp.abs(self.fwd_holotomo(psi, prb))**2                        

                fprb = self.fwd_holotomo(psi, prb)
                afprb = self.adjprb_holotomo(fprb, psi)
                r = cp.real(
                    cp.sum(prb*cp.conj(afprb))/(cp.sum(afprb*cp.conj(afprb))+1e-32))
                #print(f'{r=}')
                # take gradient
                gradprb = self.adjprb_holotomo(
                    fprb - cp.sqrt(data) * fprb/(cp.sqrt(absfprb)+1e-32), psi)
                gradprb *= r/2
                prb = prb + 0.5 * (-gradprb)
 
        return psi, prb
