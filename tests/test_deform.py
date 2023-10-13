import numpy as np
import dxchange
import ptychotomo
from ptychotomo.utils import tic, toc

if __name__ == "__main__":

    # read object
    data = dxchange.read_tiff('data/deformed_data.tiff')
    ntheta, nz, n = data.shape

    # parameters
    pprot = ntheta//2
    pars = [0.5, 3, 12, 16, 5, 1.1, 4]
    ngpus = 1
    ptheta = 16
    # simulate data
    mmin, mmax = ptychotomo.utils.find_min_max(data)
    with ptychotomo.SolverDeform(pprot, nz, n, ptheta, ngpus) as dslv:
        # register flow
        tic()
        flow = None
        for k in range(4):
            flow = dslv.registration_flow_batch(
                data[pprot:2*pprot], data[:pprot], mmin[:pprot], mmax[:pprot], flow, pars)
        print(f'registration time: {toc()}')
        # adjoint test for apply flow
        data0 = data[:pprot]
        data1 = data[pprot:2*pprot]
        tic()
        data1_unwrap = dslv.apply_flow_gpu_batch(data1, flow)
        print(f'apply flow time: {toc()}')
        data2 = dslv.apply_flow_gpu_batch(data1_unwrap, -flow)

    print(f'data0-data1={np.linalg.norm(data0-data1)}')
    print(f'data0-data1_unwrap={np.linalg.norm(data0-data1_unwrap)}')
    print(f'norm flow = {np.linalg.norm(flow)}')
    print(
        f'<data,D*Ddata>=<Ddata,Ddata>: {np.sum(data1*np.conj(data2)):e} ? {np.sum(data1_unwrap*np.conj(data1_unwrap)):e}')
    dxchange.write_tiff(data0, 'data/deform/data0', overwrite=True)
    dxchange.write_tiff(data1, 'data/deform/data1', overwrite=True)
    dxchange.write_tiff(
        data1_unwrap, 'data/deform/data1_unwrap', overwrite=True)
    dxchange.write_tiff(data0-data1_unwrap,
                        'data/deform/difunwrap', overwrite=True)
    dxchange.write_tiff(data0-data1, 'data/deform/dif', overwrite=True)
