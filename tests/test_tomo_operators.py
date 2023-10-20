import numpy as np
import dxchange
import holotomo

if __name__ == "__main__":
    
   # read object
    n = 256  # object size n x,y
    nz = 256  # object size in z    
    ntheta = 128  # number of angles (rotations)
    
    pnz = 256 # tomography chunk size for GPU processing
    
    center = n/2 # rotation axis
    theta = np.linspace(0, np.pi, ntheta).astype('float32') # projection angles
    # Load a 3D object 
    beta = dxchange.read_tiff('data/beta-chip-256.tiff')[128-nz//2:128+nz//2,128-n//2:128+n//2,128-n//2:128+n//2]
    delta = dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2,128-n//2:128+n//2,128-n//2:128+n//2]
    u = delta+1j*beta
    
    # simulate data
    with holotomo.SolverTomo(theta, ntheta, nz, n, pnz, center) as tslv:
        data = tslv.fwd_tomo_batch(u)
        u0 = tslv.adj_tomo_batch(data)
    print(f'Adjoint test: {np.sum(u*np.conj(u0))} ? {np.sum(data*np.conj(data))}')        
    
    # save generated data
    for k in range(ntheta):
        dxchange.write_tiff(data[k].real,f'data/test_tomo_operators/data_real/a{k}',overwrite=True)
        dxchange.write_tiff(data[k].imag,f'data/test_tomo_operators/data_imag/a{k}',overwrite=True)
    
    print('generated data saved to ./data/test_tomo_operators/')    
    # test with padding (needed for the ADMM scheme)
    ne = 3*n//2
    # adjoint test with data padding
    with holotomo.SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2) as tslv:
        data = tslv.paddata(data, ne)
        u0 = tslv.adj_tomo_batch(data)
        u0 = tslv.unpadobject(u0, n)

    print(f'Adjoint test for padded data: {np.sum(u*np.conj(u0))} ? {np.sum(data*np.conj(data))}')     
  
    
    # Conjugate gradient solver    
    with holotomo.SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2) as tslv:
        # simulate data
        data = tslv.paddata(data, ne)
        # cg solver
        init = np.zeros([nz,ne,ne],dtype='complex64')
        piter = 32
        u = tslv.cg_tomo_batch(data,init,piter)
        u = tslv.unpadobject(u, n)
    # save generated data
    dxchange.write_tiff_stack(u.real,f'data/test_tomo_operators/rec_real/r',overwrite=True)
    dxchange.write_tiff_stack(u.imag,f'data/test_tomo_operators/rec_imag/r',overwrite=True)
        
    print('reconstructed object save to ./data/test_tomo_operators/rec_real{imag}')    
    