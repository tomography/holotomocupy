import numpy as np
import dxchange
import ptychotomo
from random import sample
import matplotlib.pyplot as plt
import sys
#data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'

if __name__ == "__main__":    

    # read object
    n = 384  # object size n x,y
    nz = 320  # object size in z
    ntheta = 174  # number of angles
    pnz = 8  # partial size for nz
    ptheta = 1  # partial size for ntheta
    voxelsize = 18.03*1e-7  # object voxel size
    energy = 12.4  # xray energy
    ndet = 128  # detector size
    nprb = 128  # probe size
    nmodes = 4  # number of probe modes
    ngpus = 8  # number of GPUs

    data_prefix = sys.argv[1]
    nscan = int(sys.argv[2])
    center = float(sys.argv[3])
    align = int(sys.argv[4])
    step = int(sys.argv[5])
    

    # reconstruction paramters
    recover_prb = True  # recover probe or not
    piter = 32  # ptycho iterations
    titer = 32 # tomo iterations
    diter = 32  # deform iterations
    niter = 129  # admm iterations

#    piter = 256  # ptycho iterations
#    titer = 256 # tomo iterations
#    diter = 8  # deform iterations
#    niter = 1  # admm iterations

    dbg_step = 10000
    step_flow = 2    
    start_win = 256+32
        
    # Load probe
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb[:] = np.load(data_prefix+'datanpy/prb128.npy')[:nmodes]

    theta = np.zeros(ntheta, dtype='float32')
    data = np.zeros([ntheta, nscan, ndet, ndet], dtype='float32')
    scan = -np.ones([2, ntheta, nscan], dtype='float32')

    for k in range(ntheta):
        print('read angle', k)
        data0 = np.load(data_prefix+'datanpy/data128sorted_'+str(k)+'.npy').reshape(81,169,128,128)              
        scan0 = np.load(data_prefix+'datanpy/scan128sorted_'+str(k)+'.npy').reshape(2,1,81,169)          
        ids = np.arange(k%step,169,step)
        scan0 = scan0[:,:,:,ids].reshape(2,1,len(ids)*81)
        data0 = data0[:,ids].reshape(len(ids)*81,128,128)
        
        shifts1 = np.load(data_prefix+'/datanpy/shifts1_'+str(2000)+'.npy')[k]
        shifts2 = np.load(data_prefix+'/datanpy/shifts2_'+str(2700)+'.npy')[k]
        
        scan0[1] -= (shifts1[1]+shifts2[1])
        scan0[0] -= (shifts1[0]+shifts2[0])
        scan0[1] -= (64+29)
        scan0[0] -= (160)
        # ignore position out of field of view            
        ids = np.where((scan0[0,0]<nz-nprb)*(scan0[1,0]<n-nprb)*(scan0[0,0]>=0)*(scan0[1,0]>=0))[0]

        print(f'{len(ids)}')
        scan[:,k,:min(len(ids),nscan)] = scan0[:, 0, ids]
        data[k,:min(len(ids),nscan)] = data0[ids]    
        print(nscan)
        # data[k, :len(ids)] = data0[ids]
        theta[k] = np.load(data_prefix+'datanpy/theta128sorted_'+str(k*174//ntheta)+'.npy')

    # normaliza data to the detector size and fftshift it
    data /= (ndet*ndet)
    data = np.fft.fftshift(data, axes=(2, 3))
    # Initial guess
    # variable index: 1 - ptycho problem, 2 - regularization (not implemented), 3 - tomo
    # used as in the pdf documents
    h1 = np.ones([ntheta, nz, n], dtype='complex64')
    h3 = np.ones([ntheta, nz, n], dtype='complex64')
    psi1 = np.ones([ntheta, nz, n], dtype='complex64')
    psi3 = np.ones([ntheta, nz, n], dtype='complex64')
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64')
    lamd3 = np.zeros([ntheta, nz, n], dtype='complex64')
    u = np.zeros([nz, n, n], dtype='complex64')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')

    data_prefix += 'rec/'+str(nscan)+'align'+str(align)+str(center)+'/'
    with ptychotomo.SolverAdmm(nscan, theta, center, ndet, voxelsize, energy,
                               ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus) as aslv:
        u, psi1, psi3, flow, prb = aslv.admm(
            data, psi1, psi3, flow, prb, scan,
            h1, h3, lamd1, lamd3,
            u, piter, titer, diter, niter, recover_prb, align, start_win=start_win,
            step_flow=step_flow, name=data_prefix+'tmp/', dbg_step=dbg_step)

    dxchange.write_tiff_stack(
        np.angle(psi1), data_prefix+'rec_admm/psiangle/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.abs(psi1), data_prefix+'rec_admm/psiamp/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.angle(psi3), data_prefix+'rec_admm/psi3angle/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.abs(psi3), data_prefix+'rec_admm/psi3amp/p', overwrite=True)

    dxchange.write_tiff_stack(u.real,data_prefix+'rec_admm/ure/u', overwrite=True)
    dxchange.write_tiff_stack(u.imag,data_prefix+'rec_admm/uim/u', overwrite=True)
