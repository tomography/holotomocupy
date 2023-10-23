import numpy as np
import dxchange
import holotomo

if __name__ == "__main__":
    # read object
    n = 256  # object size n x,y
    nz = 256 # object size in z    
    ntheta = 1  # number of angles (rotations)

    pnz = 256 # tomography chunk size for GPU processing
    ptheta = 1 # holography chunk size for GPU processing
    
    center = n/2 # rotation axis
    theta = np.linspace(0, np.pi, ntheta).astype('float32') # projection angles
    
    # ID16a setup
    voxelsize = 1e-6*2048/n#object voxel size
    energy = 33.35  # [keV] xray energy    
    focusToDetectorDistance = 128
    sx0 = 0.037
    z1 = np.array([0.4584,0.4765,0.5488,0.69895])-sx0
    z2 = focusToDetectorDistance-z1
    distances = (z1*z2)/focusToDetectorDistance
    magnifications = focusToDetectorDistance/z1    
    norm_magnifications = magnifications/magnifications[0]
    
    print(f'{voxelsize=}')
    print(f'{energy=}') 
    print(f'{focusToDetectorDistance=}')
    print(f'{sx0=}')
    print(f'{z1=}')
    print(f'{z2=}')
    print(f'{distances=}')
    print(f'{magnifications=}')    
    print(f'normalized magnifications= {norm_magnifications}')
    
    # Load a 3D object and probe
    beta = dxchange.read_tiff('data/beta-chip-256.tiff')
    delta = dxchange.read_tiff('data/delta-chip-256.tiff')
    
    v = np.arange(-n//2,n//2)/n
    [vx,vy] = np.meshgrid(v,v)
    v=np.exp(-10*(vx**2+vy**2))*10
    delta = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(delta))*v)))).real
    beta = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beta))*v)))).real
    delta[delta>0]=0
    beta[beta<0]=0
    
    delta/=6# /6 to avoid phase wrapping
    u = delta+1j*beta
    u = u.astype('complex64')
    
    # load probes from ID16a recovered by NFP for 4 distances, crop it to the object size
    prb = np.zeros([len(distances),nz,n],dtype='complex64')    
    prb_abs = dxchange.read_tiff_stack('data/prb_id16a/prb_abs_00000.tiff',ind=range(len(distances)))[:,1024-n//2:1024+n//2,1024-n//2:1024+n//2]
    prb_phase = dxchange.read_tiff_stack('data/prb_id16a/prb_phase_00000.tiff',ind=range(len(distances)))[:,1024-n//2:1024+n//2,1024-n//2:1024+n//2]
    prb[:] = prb_abs*np.exp(1j*prb_phase)   
    # compute tomographic projections
    with holotomo.SolverTomo(theta, ntheta, nz, n, pnz, center) as tslv:
        proj = tslv.fwd_tomo_batch(u) 
    
    # propagate projections with holography
    with holotomo.SolverHolo(ntheta, nz, n, ptheta, voxelsize, energy, distances, norm_magnifications)  as pslv:
        # transmission function
        psi = pslv.exptomo(proj)
        
        # compute forward holography operator for all projections
        fpsi = pslv.fwd_holo_batch(psi,prb)
        # data on the detector
        data = np.abs(fpsi)**2
        
        # save generated data
        for k in range(ntheta):
            for j in range(len(distances)):
                dxchange.write_tiff(data[j,k],f'data/test_holo_operators/data/a{k}_d{j}',overwrite=True)        
        for j in range(len(distances)):        
            dxchange.write_tiff(np.abs(prb[j]),f'data/test_holo_operators/prb/abs_d{j}',overwrite=True)
            dxchange.write_tiff(np.angle(prb[j]),f'data/test_holo_operators/prb/phase_d{j}',overwrite=True)
        for k in range(ntheta):        
            dxchange.write_tiff(np.abs(psi[k]),f'data/test_holo_operators/obj/abs_a{k}',overwrite=True)
            dxchange.write_tiff(np.angle(psi[k]),f'data/test_holo_operators/obj/phase_a{k}',overwrite=True)
        
        print('generated data saved to ./data/test_holo_operators/')
        
        #compute adjoint transform
        psi0 = pslv.adj_holo_batch(fpsi, prb)
    print(f'Adjoint test: {np.sum(psi*np.conj(psi0))} ? {np.sum(fpsi*np.conj(fpsi))}')        
    # exit()
    print('Test reconstruction by CG')
    piter = 64
    init = psi*0+1
    with holotomo.SolverHolo(ntheta, nz, n, ptheta, voxelsize, energy, distances, norm_magnifications)  as pslv:
        rec = pslv.cg_holo_batch(data,init,prb,piter)
        for k in range(ntheta):        
            dxchange.write_tiff(np.abs(rec[k]),f'data/test_holo_operators/rec/abs_a{k}',overwrite=True)
            dxchange.write_tiff(np.angle(rec[k]),f'data/test_holo_operators/rec/phase_a{k}',overwrite=True)