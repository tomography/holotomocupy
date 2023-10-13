import numpy as np
import dxchange
import ptychotomo
from random import sample
import matplotlib.pyplot as plt

data_prefix = './tmp/'

if __name__ == "__main__":    

    # read object
    n = 256  # object size n x,y
    nz = 256  # object size in z
    ntheta = 1#  # number of angles
    pnz = 32 # partial size for nz
    ptheta = 1  # partial size for ntheta
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    ndet = 128  # detector size
    nprb = 128  # probe size
    nmodes = 1  # number of probe modes
    nscan = 300  # number of scan positions per each angle
    ngpus = 1  # number of GPUs
    center = n/2
    # reconstruction paramters
    recover_prb = False  # recover probe or not
    piter = 128  # ptycho iterations
    # Load a 3D object
    delta = dxchange.read_tiff('data_lego/delta-lego-256.tiff')
    beta = dxchange.read_tiff('data_lego/beta-lego-256.tiff')
    u = delta+1j*beta

    # Load probe
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb_amp = dxchange.read_tiff('data/probes_amp.tiff')[:nmodes]
    prb_ang = dxchange.read_tiff('data/probes_ang.tiff')[:nmodes]
    prb[:] = prb_amp*np.exp(1j*prb_ang)

    # Load scan positions
    scan = np.zeros([2, ntheta, nscan], dtype='float32') - 1  # -1 to be skipped in computations
    scan_all = np.load('data_lego/scan.npy')
    
    for k in range(ntheta):
        scan0 = scan_all[:, k]
        ids = np.where((scan0[0] < n-nprb)*(scan0[1] < nz-nprb)*(scan0[1] >= 0)*(scan0[0] >= 0))[0]
        ids = ids[sample(range(len(ids)), min(len(ids), nscan))]
        scan[:, k, :len(ids)] = scan0[:, ids]
        plt.plot(scan[0], scan[1], 'r.')
        plt.savefig(f'data_lego/scan{k:03}.png')
        plt.clf()

    # init rotation angles
    theta = np.linspace(0, np.pi, ntheta).astype('float32')

    # Form data
    with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        psi = tslv.fwd_tomo_batch(u)
    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi = pslv.exptomo(psi)
        data = pslv.fwd_ptycho_batch(psi, prb, scan)
        data = np.sum(np.abs(data)**2, axis=1)

    # Initial guess
    # variable index: 1 - ptycho problem, 2 - regularization (not implemented), 3 - tomo
    # used as in the pdf documents
    psi = np.ones([ntheta, nz, n], dtype='complex64')
    
    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi, prb = pslv.grad_ptycho_batch(data, psi, prb, scan, psi*0, -1, piter, recover_prb)                

    dxchange.write_tiff(np.angle(psi[0]),'data_lego/rec_ptycho/psiangle',overwrite=True)    
    dxchange.write_tiff(np.abs(psi[0]),'data_lego/rec_ptycho/psiamp',overwrite=True)    
    dxchange.write_tiff_stack(np.angle(prb[0]),'data_lego/rec_ptycho/prbangle',overwrite=True)    
    dxchange.write_tiff_stack(np.abs(prb[0]),'data_lego/rec_ptycho/prbamp',overwrite=True)    
    
    
