import os
import signal
import sys
import h5py
import cupy as cp
import dxchange
import numpy as np
import matplotlib.pyplot as plt
import ptychotomo as pt
from random import sample
import scipy.ndimage as ndimage

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'


if __name__ == "__main__":

    ngpus = 4
    ntheta = 174
    nscan = int(sys.argv[1])
    shift_type = str(sys.argv[2])

    n = 384
    nz = 320
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    recover_prb = True

    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 32  # ptychography iterations
    titer = 32  # tomography iterations
    diter = 8
    niter = 300  # ADMM iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography
    pnz = 16  # number of slice partitions for simultaneous processing in tomography
    center = 192

    nmodes = 4

    theta = np.zeros(ntheta, dtype='float32')
    data = np.zeros([ntheta, nscan, det[0], det[1]], dtype='float32')
    scan = np.zeros([2, ntheta, nscan], dtype='float32')-1

    
    for k in range(ntheta):
        print('read angle', k)
        scan0 = np.load(data_prefix+'datanpy/scan128sorted_'+str(k)+'.npy')        
        if(shift_type=='stxm'):
            scan0[0] += np.load('sx_stxm_new2.npy')[k]
            scan0[1] += np.load('sy_stxm_new2.npy')[k]
        else:
            shifts = np.load(data_prefix+'/datanpy/shifts.npy')[k]
            #shiftspart = np.load(data_prefix+'/datanpy/shiftscrop.npy')[k]
            shiftssum = np.load(data_prefix+'/datanpy/shiftssum.npy')[k]
            print(np.round(shifts[0]+shiftssum[0]),np.round(shifts[1]+shiftssum[1]))
            scan0[0] -= np.round(shifts[0]+shiftssum[0])
            scan0[1] -= np.round(shifts[1]+shiftssum[1])            
        scan0[1] -= 64+30
        scan0[0] -= 160

        ids = np.where((scan0[1, 0] < n-nprb)*(scan0[0, 0] <
                                               nz-nprb)*(scan0[0, 0] >= 0)*(scan0[1, 0] >= 0))[0]
        print(len(ids))                                               
        ids = ids[sample(range(len(ids)), min(len(ids),nscan))]
        
        scan[0, k, :len(ids)] = scan0[1, 0, ids]
        scan[1, k, :len(ids)] = scan0[0, 0, ids]
        scan[0, k, len(ids):] = -1
        scan[1, k, len(ids):] = -1
        data[k, :len(ids)] = np.load(
            data_prefix+'datanpy/data128sorted_'+str(k)+'.npy')[ids]
        theta[k] = np.load(data_prefix+'datanpy/theta128sorted_'+str(k)+'.npy')
        print("%d) %02f" % (k,theta[k]/np.pi*180))
    print(theta/np.pi*180)
    # Load a 3D object
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64', order='C')
    prb[:] = np.load(data_prefix+'datanpy/prb128.npy')

    # Initial guess
    h1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1  # +1e-10j
    h2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    h3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1  # +1e-10j

    psi1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1  # +1e-10j

    # psi1abs = dxchange.read_tiff_stack(data_prefix+'recTrueTrue47000psi1iterabs512/0_00000.tiff',ind=range(0,ntheta))
    # psi1angle = dxchange.read_tiff_stack(data_prefix+'recTrueTrue47000psi1iter512/0_00000.tiff',ind=range(0,ntheta))
    # psi1 = psi1abs*np.exp(1j*psi1angle)
    psi2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    psi3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1  # +1e-10j
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    lamd2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    u = np.zeros([nz, n, n], dtype='complex64', order='C')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32', order='C')

    data = np.fft.fftshift(data, axes=(2, 3))
    # Class gpu solver
    slv = pt.Solver(nscan, theta, center, det, voxelsize,
                    energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus)
    name = data_prefix+'rec_' + \
        str(shift_type)+str(recover_prb)+str(align)+str(nmodes)+str(nscan)

    # ADMM
    u, psi3, psi2, psi1, flow, prb = slv.admm(
        data, psi3, psi2, psi1, flow, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, u, alpha, piter, titer, diter, niter, model, recover_prb, align, name)
