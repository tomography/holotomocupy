import numpy as np
import dxchange
import ptychotomo
from random import sample
import matplotlib.pyplot as plt
import sys
data_prefix = '/data/staff/tomograms/vviknik/nanomax/'

if __name__ == "__main__":    

    # read object
    n = 384  # object size n x,y
    nz = 320  # object size in z
    ntheta = 166  # number of angles
    pnz = 8  # partial size for nz
    ptheta = 1  # partial size for ntheta
    voxelsize = 18.03*1e-7  # object voxel size
    energy = 12.4  # xray energy
    ndet = 128  # detector size
    nprb = 128  # probe size
    nmodes = 4  # number of probe modes
    ngpus = 4  # number of GPUs

    nscan = int(sys.argv[1])
    shift_type = str(sys.argv[2])
    center = float(sys.argv[3])
    align = int(sys.argv[4])
    

    # reconstruction paramters
    recover_prb = True  # recover probe or not
    piter = 32  # ptycho iterations
    titer = 32 # tomo iterations
    diter = 32  # deform iterations
    niter = 129   # admm iterations
    dbg_step = 4
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
        scan0 = np.load(data_prefix+'datanpy/scan128sorted_'+str(k*166//ntheta)+'.npy')        
        if(shift_type=='stxm'):
            shifts = np.load(data_prefix+'/datanpy/shiftsazat.npy')[k*166//ntheta]
            sx = shifts[0]
            sy = shifts[1]
        else:
            shifts = np.load(data_prefix+'/datanpy/shifts.npy')[k*166//ntheta]
            #shiftspart = np.load(data_prefix+'/datanpy/shiftscrop.npy')[k]
            shiftscrop = np.load(data_prefix+'/datanpy/shiftscrop.npy')[k*166//ntheta]
            sx = shifts[0]+shiftscrop[0]
            sy = shifts[1]+shiftscrop[1]
        print(sx,sy)
        scan0[0] -= sx
        scan0[1] -= sy                                     
        scan0[1] -= 64+30
        scan0[0] -= 160

        ids = np.where((scan0[1, 0] < n-nprb)*(scan0[0, 0] <
                                               nz-nprb)*(scan0[0, 0] >= 0)*(scan0[1, 0] >= 0))[0]
        print(f'{len(ids)}')  
        ids = ids[sample(range(len(ids)), min(len(ids),nscan))]
        
        # switch x and y
        scan[:, k, :len(ids)] = scan0[:, 0, ids]        
        # plt.plot(scan[0,k], scan[1,k], 'r.')
        # plt.savefig(f'{data_prefix}/png/scan{k:03}.png')
        # plt.clf()
        
        data[k, :len(ids)] = np.load(
            data_prefix+'datanpy/data128sorted_'+str(k*166//ntheta)+'.npy')[ids]
        # data[k, :len(ids)] = data0[ids]
        theta[k] = np.load(data_prefix+'datanpy/theta128sorted_'+str(k*166//ntheta)+'.npy')

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

    data_prefix += 'rec/'+str(nscan)+shift_type+'align'+str(align)+str(center)+'/'
    with ptychotomo.SolverAdmm(nscan, theta, center, ndet, voxelsize, energy,
                               ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus) as aslv:
        u, psi1, psi3, flow, prb = aslv.admm(
            data, psi1, psi3, flow, prb, scan,
            h1, h3, lamd1, lamd3,
            u, piter, titer, diter, niter, recover_prb, align, start_win=start_win,
            step_flow=step_flow, name=data_prefix+'tmp/', dbg_step=dbg_step)

    dxchange.write_tiff_stack(
        np.angle(psi1), data_prefix+'recadmm/psiangle/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.abs(psi1), data_prefix+'recadmm/psiamp/p', overwrite=True)
    dxchange.write_tiff_stack(u.real, data_prefix+'recadmm/ure/u', overwrite=True)
    dxchange.write_tiff_stack(u.imag, data_prefix+'recadmm/uim/u', overwrite=True)
