"""Module for tomography."""

import cupy as cp
import numpy as np
import threading
import concurrent.futures as cf
from .ptychofft import ptychofft
from .utils import chunk
from functools import partial

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class SolverPtycho(ptychofft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        Number of projections.    
    ptheta : int
        Number of projections for simultaneous processing by 1 gpu            
    nz, n : int
        The pixel height and width of the projection.
    nscan : int
        Number of scanning positions
    ndet : int
        Detector size
    nprb : int
        prb size
    nmodes : int
        Number of prb modes
    voxelsize : float
        Object voxel size
    energy : float
        X-ray energy
    ngpus : int
        Number of gpus        
    """

    def __init__(self, ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus):
        """Please see help(SolverTomo) for more info."""
        if(ntheta % ptheta > 0):
            print('Error, ptheta is not a multiple of ntheta')
            exit()

        self.ntheta = ntheta
        self.nmodes = nmodes
        self.voxelsize = voxelsize
        self.energy = energy

        super().__init__(ptheta, nz, n, nscan, ndet, nprb, ngpus)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

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

    def fwd_ptycho(self, psi, prb, scan, igpu):
        """Ptychography transform (FQ)"""
        res = cp.zeros([self.ptheta, self.nscan, self.ndet,
                        self.ndet], dtype='complex64')
        res = cp.ascontiguousarray(res)
        psi = cp.ascontiguousarray(psi)
        prb = cp.ascontiguousarray(prb)
        scan = cp.ascontiguousarray(scan)
        self.fwd(res.data.ptr, psi.data.ptr,
                 prb.data.ptr, scan.data.ptr, igpu)
        return res

    def adj_ptycho(self, data, prb, scan, igpu):
        """Adjoint ptychography transform (Q*F*)"""
        res = cp.zeros([self.ptheta, self.nz, self.n],
                       dtype='complex64')
        data = data.copy()  # avoid this todo
        res = cp.ascontiguousarray(res)
        data = cp.ascontiguousarray(data)
        prb = cp.ascontiguousarray(prb)
        scan = cp.ascontiguousarray(scan)
        self.adj(res.data.ptr, data.data.ptr,
                 prb.data.ptr, scan.data.ptr, igpu)
        return res

    def adj_ptycho_prb(self, data, psi, scan, igpu):
        """Adjoint ptychography transform wrt prb (psi*F*)"""
        res = cp.zeros([self.ptheta, self.nprb, self.nprb],
                       dtype='complex64')
        data = data.copy()  # avoid this todo
        res = cp.ascontiguousarray(res)
        data = cp.ascontiguousarray(data)
        psi = cp.ascontiguousarray(psi)
        scan = cp.ascontiguousarray(scan)
        self.adjprb(res.data.ptr, data.data.ptr,
                    psi.data.ptr, scan.data.ptr, igpu)
        return res

    def fwd_ptycho_batch(self, psi, prb, scan):
        """Batch of Ptychography transform (FQ)"""
        data = np.zeros([self.ntheta, self.nmodes, self.nscan, self.ndet,
                         self.ndet], dtype='complex64')
        for ids in chunk(range(self.ntheta), self.ptheta):
            psi_gpu = cp.array(psi[ids])
            prb_gpu = cp.array(prb[ids])
            scan_gpu = cp.array(scan[:, ids])
            for m in range(self.nmodes):
                data_gpu = self.fwd_ptycho(psi_gpu, prb_gpu[:, m], scan_gpu, 0)
                data[ids, m] = data_gpu.get()
#        data *= (self.ndet*self.ndet)  # FFT compensation
        return data

    def adj_ptycho_batch(self, data, prb, scan):
        """Batch of Ptychography transform (FQ)"""
        psi = np.zeros([self.ntheta, self.nz, self.n],
                       dtype='complex64')
        for ids in chunk(range(self.ntheta), self.ptheta):
            scan_gpu = cp.array(scan[:, ids])
            for m in range(self.nmodes):
                data_gpu = cp.array(data[ids, m])
                prb_gpu = cp.array(prb[ids, m])

                psi_gpu = self.adj_ptycho(data_gpu,
                                          prb_gpu, scan_gpu, 0)
                psi[ids] += psi_gpu.get()
        # psi *= (self.ndet*self.ndet)  # FFT compensation
        return psi

    def adj_ptycho_prb_batch(self, data, psi, scan):
        """Batch of Ptychography transform (FQ)"""
        prb = np.zeros([self.ntheta, self.nmodes, self.nprb, self.nprb],
                       dtype='complex64')
        for ids in chunk(range(self.ntheta), self.ptheta):
            psi_gpu = cp.array(psi[ids])
            scan_gpu = cp.array(scan[:, ids])
            for m in range(self.nmodes):
                data_gpu = cp.array(data[ids, m])
                prb_gpu = self.adj_ptycho_prb(
                    data_gpu, psi_gpu, scan_gpu, 0)
                prb[ids, m] = prb_gpu.get()
        # prb *= (self.ndet*self.ndet)  # FFT compensation
        return prb

    def grad_ptycho(self, data, psi, prb, scan, hlamd, rho, piter, recover_prb, igpu):
        # &\psi_1^{k+1}, (q^{k+1}) =  \argmin_{\psi_1, q} \sum_{j = 1}^{n}
        # \left\{ |\Fop\Qop_q\psi_1|_j^2-2d_j\log |\Fop\Qop_p\psi_1|_j \right\} +
        # \rho_1\|h1 -\psi_1 +\lambda_1^k /\rho_1\| _2^2,
        # h1 == \Top_{t^k} \psi_3^k
        # minimization functional
        def minf(fpsi, psi):
            f = cp.linalg.norm(cp.sqrt(cp.abs(fpsi)) - cp.sqrt(data))**2
            if(rho>0):
                f += rho*cp.linalg.norm(psi-hlamd)**2
            return f

        minf1 = 1e12
        for i in range(piter):

            # 1) object retrieval subproblem with fixed prbs
            # sum of forward operators associated with each prb
            # sum of abs value of forward operators            
            absfpsi = data*0
            for m in range(self.nmodes):
                absfpsi += cp.abs(self.fwd_ptycho(psi, prb[:, m], scan, igpu))**2                                
                        
            a = cp.sum(cp.sqrt(absfpsi*data))
            b = cp.sum(absfpsi)
            prb *= (a/b)
            absfpsi *= (a/b)**2
            
            gradpsi = cp.zeros(
                [self.ptheta, self.nz, self.n], dtype='complex64')
            for m in range(self.nmodes):
                fpsi = self.fwd_ptycho(psi, prb[:, m], scan, igpu)
                afpsi = self.adj_ptycho(fpsi, prb[:, m], scan, igpu)
                
                if(m == 0):
                    r = cp.real(cp.sum(psi*cp.conj(afpsi)) /
                            (cp.sum(afpsi*cp.conj(afpsi))+1e-32))
                
                gradpsi += self.adj_ptycho(
                    fpsi - cp.sqrt(data)*fpsi/(cp.sqrt(absfpsi)+1e-32), prb[:, m], scan, igpu)                
            if(rho>0):                                
                gradpsi += rho*(psi-hlamd)
                gradpsi *= min(1/rho, r)/2
            else:
                gradpsi *= r/2
            # update psi
            psi = psi + 0.5 * (-gradpsi)

            if (recover_prb):
                if(i == 0):
                    gradprb = prb*0
                for m in range(self.nmodes):
                    # 2) prb retrieval subproblem with fixed object
                    # sum of forward operators associated with each prb

                    absfprb = data*0
                    for mm in range(self.nmodes):
                        absfprb += cp.abs(self.fwd_ptycho(psi, prb[:, mm], scan, igpu))**2                        

                    fprb = self.fwd_ptycho(psi, prb[:, m], scan, igpu)
                    afprb = self.adj_ptycho_prb(fprb, psi, scan, igpu)
                    r = cp.real(
                        cp.sum(prb[:, m]*cp.conj(afprb))/(cp.sum(afprb*cp.conj(afprb))+1e-32))
                    
                    # take gradient
                    gradprb[:, m] = self.adj_ptycho_prb(
                        fprb - cp.sqrt(data) * fprb/(cp.sqrt(absfprb)+1e-32), psi, scan, igpu)
                    gradprb[:, m] *= r/2
                    prb[:, m] = prb[:, m] + 0.5 * (-gradprb[:, m])

            # minf0 = minf(absfpsi, psi)
            # if(minf0 > minf1):
            #     print('error inptycho', minf0, minf1)
            # minf1 = minf0

        return psi, prb


    def grad_ptycho_multi_gpu(self, data, psi, prb, scan, hlamd, rho, piter, recover_prb, lock,  ids):
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
        data_gpu = cp.array(data[ids], order='C')
        psi_gpu = cp.array(psi[ids])
        hlamd_gpu = cp.array(hlamd[ids])
        prb_gpu = cp.array(prb[ids])
        scan_gpu = cp.array(scan[:, ids])

        psi_gpu, prb_gpu = self.grad_ptycho(
            data_gpu, psi_gpu, prb_gpu, scan_gpu, hlamd_gpu, rho, piter, recover_prb, gpu)

        psi[ids] = psi_gpu.get()
        prb[ids] = prb_gpu.get()

        BUSYGPUS[gpu] = 0

        return psi[ids], prb[ids]

    def grad_ptycho_batch(self, data, psiinit, prbinit, scan, hlamd, rho, piter, recover_prb):

        psi = psiinit.copy()
        prb = prbinit.copy()
        ids_list = chunk(range(self.ntheta), self.ptheta)

        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for psii, prbi in e.map(partial(self.grad_ptycho_multi_gpu, data, psi, prb, scan, hlamd, rho, piter, recover_prb, lock), ids_list):
                psi[shift:shift+psii.shape[0]] = psii
                prb[shift:shift+psii.shape[0]] = prbi
                shift += psii.shape[0]
        cp.cuda.Device(0).use()
        return psi, prb        


    # xi0,K, and K for linearization of the tomography problem
    def takexi(self, psi3, lamd3, rho3):
        # bg subtraction parameters
        r = self.nprb//2
        m1 = np.mean(
            np.angle(psi3[:, :, r:2*r]))
        m2 = np.mean(np.angle(
            psi3[:, :, psi3.shape[2]-2*r:psi3.shape[2]-r]))
        pshift = (m1+m2)/2
        
        t = psi3-lamd3/rho3
        t *= np.exp(-1j*pshift)
        logt = self.mlog(t)

        # K, xi0, xi1
        K = 1j*self.voxelsize * self.wavenumber()*t
        K = K/np.amax(np.abs(K))  # normalization
        xi0 = K*(-1j*(logt) /
                 (self.voxelsize * self.wavenumber()))        
        return xi0, K, pshift

    def take_error(self, data, psi1, prb,scan):
        err = 0
        for ids in chunk(range(self.ntheta), self.ptheta):            
            data_gpu = cp.array(data[ids])
            # normalized data
            absfprb = cp.zeros_like(data_gpu)
            psi1_gpu = cp.array(psi1[ids])
            prb_gpu = cp.array(prb[ids])
            scan_gpu = cp.array(scan[:, ids])
            for m in range(self.nmodes):
                absfprb += cp.abs(self.fwd_ptycho(psi1_gpu, prb_gpu[:,m], scan_gpu, 0))**2                 
            err += cp.linalg.norm(cp.sqrt(absfprb)-cp.sqrt(data_gpu))**2
        return err
