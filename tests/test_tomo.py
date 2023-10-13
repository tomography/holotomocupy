import numpy as np
import dxchange
import ptychotomo

if __name__ == "__main__":
    
    # read object
    u = dxchange.read_tiff('data/init_object.tiff')
    u = u+1j*u/2

    nz, n, _ = u.shape

    # parameters
    center = n/2
    ntheta = 384
    ne = 3*n//2
    ngpus = 1
    pnz = nz//2
    theta = np.linspace(0, 4*np.pi, ntheta).astype('float32')

    # simulate data
    with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        data = tslv.fwd_tomo_batch(u)

    # adjoint test with data padding
    with ptychotomo.SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus) as tslv:
        data = ptychotomo.utils.paddata(data, ne)
        ua = tslv.adj_tomo_batch(data)
        ua = ptychotomo.utils.unpadobject(ua, n)

    print(f'norm data = {np.linalg.norm(data)}')
    print(f'norm object = {np.linalg.norm(ua)}')
    print(
        f'<u,R*Ru>=<Ru,Ru>: {np.sum(u*np.conj(ua)):e} ? {np.sum(data*np.conj(data)):e}')
