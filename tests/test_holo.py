import numpy as np
import dxchange
import holotomo
import h5py

if __name__ == "__main__":
    # read object
    n = 192  # object size in each dimension
    ntheta = 2  # number of angles (rotations)

    pn = 192  # tomography chunk size for GPU processing
    ptheta = 2  # holography chunk size for GPU processing

    center = n/2  # rotation axis
    theta = np.linspace(0, np.pi, ntheta).astype('float32')  # projection angles

    # ID16a setup
    voxelsize = 10e-9*2048/n  # object voxel size
    energy = 33.35  # [keV] xray energy
    focusToDetectorDistance = 1.28
    sx0 = 3.7e-4
    ndist = 2
    z1 = np.array([4.584e-3, 4.765e-3, 5.488e-3, 6.9895e-3])[:ndist]-sx0
    z2 = focusToDetectorDistance-z1
    distances = (z1*z2)/focusToDetectorDistance
    magnifications = focusToDetectorDistance/z1
    norm_magnifications = magnifications/magnifications[0]
    # scaled propagation distances due to magnified probes
    distances = distances*norm_magnifications**2

    z1p = z1[0]  # positions of the probe for reconstruction
    z2p = z1-np.tile(z1p, len(z1))
    # magnification when propagating from the probe plane to the detector
    magnifications2 = (z1p+z2p)/z1p
    # propagation distances after switching from the point source wave to plane wave,
    distances2 = (z1p*z2p)/(z1p+z2p)
    norm_magnifications2 = magnifications2 / \
        magnifications2[0]  # normalized magnifications
    # scaled propagation distances due to magnified probes
    distances2 = distances2*norm_magnifications2**2
    distances2 = distances2*(z1p/z1)**2

    print(f'{voxelsize=}')
    print(f'{energy=}')
    print(f'{focusToDetectorDistance=}')
    print(f'{sx0=}')
    print(f'{z1=}')
    print(f'{z2=}')
    print(f'{distances=}')
    print(f'{distances2=}')
    print(f'{magnifications=}')

    print(f'normalized magnifications= {norm_magnifications}')

    # Load a 3D object
    beta0 = dxchange.read_tiff('data/beta-chip-192.tiff')
    delta0 = dxchange.read_tiff('data/delta-chip-192.tiff')

    # pad with zeros
    beta = np.zeros([2*n, 2*n, 2*n], dtype='float32')
    delta = np.zeros([2*n, 2*n, 2*n], dtype='float32')
    delta[n-96:n+96, n-96:n+96, n-96:n+96] = delta0
    beta[n-96:n+96, n-96:n+96, n-96:n+96] = beta0

    u = delta+1j*beta
    u = u.astype('complex64')

    # load probe from ID16a recovered by NFP for the first distance
    prb_abs = dxchange.read_tiff(f'data/prb_id16a/prb_abs_{n}.tiff')[0:1]
    prb_phase = dxchange.read_tiff(f'data/prb_id16a/prb_phase_{n}.tiff')[0:1]
    prb = prb_abs*np.exp(1j*prb_phase)*0+1

    # compute tomographic projections
    with holotomo.SolverTomo(theta, ntheta, 2*n, 2*n, 2*pn, 2*center) as tslv:
        proj = tslv.fwd_tomo_batch(u)

    # shifts of motors between different planes
    shifts = (np.random.random([ndist, ntheta, 2]).astype('float32')-0.5)*10

    # propagate projections with holography
    with holotomo.SolverHolo(ntheta, n, ptheta, voxelsize, energy, distances, norm_magnifications, distances2) as pslv:
        # transmission function
        psi = pslv.exptomo(proj)

        # compute forward holography operator for all projections
        fpsi = pslv.fwd_holo_batch(psi, prb, shifts)
        # data on the detector
        data = np.abs(fpsi)**2
        # save generated data
        for k in range(ntheta):
            for j in range(len(distances)):
                dxchange.write_tiff(
                    data[j, k], f'data/test_holo_operators/data/a{k}_d{j}', overwrite=True)
        for k in range(ntheta):
            dxchange.write_tiff(
                np.abs(proj[0]), f'data/test_holo_operators/obj/abs_a{k}', overwrite=True)
            dxchange.write_tiff(
                np.angle(proj[0]), f'data/test_holo_operators/obj/phase_a{k}', overwrite=True)
        dxchange.write_tiff(
            np.abs(prb[0]), f'data/test_holo_operators/prb/abs', overwrite=True)
        dxchange.write_tiff(
            np.angle(prb[0]), f'data/test_holo_operators/prb/phase', overwrite=True)

        print('generated data saved to ./data/test_holo_operators/')

        # compute adjoint transform
        psi0 = pslv.adj_holo_batch(fpsi, prb, shifts)
        prb0 = pslv.adj_holo_prb_batch(fpsi, psi, shifts)
        
        print(
            f'Adjoint test for object: {np.sum(psi*np.conj(psi0))} ? {np.sum(fpsi*np.conj(fpsi))}')
        print(
            f'Adjoint test for probe: {np.sum(prb*np.conj(prb0))} ? {np.sum(fpsi*np.conj(fpsi))}')


        # fpsi0 = pslv.fwd_holo_batch(psi0, prb, shifts)
        # fprb0 = pslv.fwd_holo_batch(psi, prb0, shifts)
        
        # print(np.sum(fpsi0*np.conj(fpsi))/np.sum(fpsi0*np.conj(fpsi0)))
        # print(np.sum(psi0*np.conj(psi))/np.sum(psi0*np.conj(psi0)))
        
        # print(np.sum(fprb0*np.conj(fpsi))/np.sum(fprb0*np.conj(fprb0)))
        # print(np.sum(prb0*np.conj(prb))/np.sum(prb0*np.conj(prb0)))
        
        # print(
        #     f'Norm test for probe: {np.sum(prb*np.conj(prb0))} ? {np.sum(prb*np.conj(prb))}')
        # 

