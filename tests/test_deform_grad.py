import numpy as np
import dxchange
import ptychotomo
from ptychotomo.utils import tic, toc

if __name__ == "__main__":

    # read object
    data = dxchange.read_tiff('data/deformed_data.tiff')+0j
    ntheta, nz, n = data.shape

    # parameters
    pprot = ntheta//2
    pars = [0.5, 3, 32, 16, 5, 1.1, 4]
    ngpus = 1
    ptheta = 16
    # simulate data
    diter = 16
    mmin, mmax = ptychotomo.utils.find_min_max(data.real)
    with ptychotomo.SolverDeform(ntheta, nz, n, ptheta, ngpus) as dslv:
        # register flow
        print(np.tile(data[pprot:2*pprot],[2,1,1]).shape)
        flow = dslv.registration_flow_batch(
            np.tile(data[pprot:2*pprot].real,[2,1,1]), np.tile(data[:pprot].real,[2,1,1]), mmin, mmax, None, pars)
        res = dslv.grad_deform_gpu_batch(data, data*0, flow, diter, xi1=0, rho=-1, dbg=False)


    dxchange.write_tiff(data[:pprot].real, 'data/deform_grad/data0', overwrite=True)
    dxchange.write_tiff(res.real, 'data/deform_grad/data1', overwrite=True)
    dxchange.write_tiff(data[pprot:2*pprot].real, 'data/deform_grad/data2', overwrite=True)
    
    