import numpy as np
import dxchange
import ptychotomo
from random import sample
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # read object
    n = 256  # object size n x,y
    nz = 256  # object size in z
    pnz = 32
    ntheta = 174  # number of angles (rotations)
    ptheta = 2
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    ndet = 128
    nprb = 128
    center = n/2
    nmodes = 4
    nscan = 300
    ngpus = 1

    theta = np.linspace(0, np.pi, ntheta).astype('float32')

    # Load a 3D object
    beta = dxchange.read_tiff('data/beta-chip-256.tiff')
    delta = -dxchange.read_tiff('data/delta-chip-256.tiff')
    u = delta+1j*beta

    # Load probe
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb_amp = dxchange.read_tiff('data/probes_amp.tiff')[:nmodes]
    prb_ang = dxchange.read_tiff('data/probes_ang.tiff')[:nmodes]
    prb[:] = prb_amp*np.exp(1j*prb_ang)

    # Load scan positions
    scan = np.zeros([2, ntheta, nscan], dtype='float32') - 1
    for k in range(ntheta):
        scan0 = np.load(f'data/scan/scan128sorted_{k}.npy')
        ids = np.where((scan0[1, 0] < n-nprb)*(scan0[0, 0] <
                                               nz-nprb)*(scan0[0, 0] >= 0)*(scan0[1, 0] >= 0))[0]
        ids = ids[sample(range(len(ids)), min(len(ids), nscan))]
        scan[0, k, :len(ids)] = scan0[1, 0, ids]
        scan[1, k, :len(ids)] = scan0[0, 0, ids]
        #plt.plot(scan[0], scan[1], 'r.')
        # plt.savefig(f'data/scan{k:03}.png')

    with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        psi = tslv.fwd_tomo_batch(u)
        print(f'{np.linalg.norm(psi) = }')
    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi = pslv.exptomo(psi)
        data = pslv.fwd_ptycho_batch(psi, prb, scan)
        psia = pslv.adj_ptycho_batch(data, prb, scan)
        prba = pslv.adj_ptycho_prb_batch(data, psi, scan)
        print(f'{psi.shape = }')
        print(f'{psia.shape = }')
        print(f'{prb.shape = }')
        print(f'{prba.shape = }')
        print(f'{np.linalg.norm(data) = }')
        print(f'{np.linalg.norm(psia) = }')
        print(f'{np.linalg.norm(prba) = }')
    
    print(
        f'<psi,Q*F*FQpsi>=<FQpsi,FQpsi>: {np.sum(psi*np.conj(psia)):e} ? {np.sum(data*np.conj(data)):e}')
    print(
        f'<prb,P*F*FPprb>=<FPprb,FPprb>: {np.sum(prb*np.conj(prba)):e} ? {np.sum(data*np.conj(data)):e}')
