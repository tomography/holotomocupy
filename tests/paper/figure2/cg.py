import cupy as cp
import numpy as np
import holotomo

def line_search(minf, gamma, u, fu, d, fd):
    """ Line search for the step sizes gamma"""
    while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-12):
        gamma *= 0.5
    if(gamma <= 1e-12):  # direction not found
        #print('no direction')
        gamma = 0
    return gamma
    
def cg_holo(pslv, data, init, prb,  piter, nerr_th, codes=None):
    """Conjugate gradients method for holography"""

    # minimization functional
    def minf(psi,fpsi):
        f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2            
        # f = cp.linalg.norm(cp.abs(fpsi)**2-data)**2            
        return f        
    psi = init.copy()
    norm_data = np.linalg.norm(data)
    maxprb = cp.max(cp.abs(prb))
    conv = np.zeros(piter)
    for i in range(piter):
        fpsi = pslv.fwd_holo(psi,prb,codes)
        grad = pslv.adj_holo(
           fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), prb, codes)/maxprb**2
        #grad = pslv.adj_holo(
             #(cp.abs(fpsi)**2-data)*fpsi,prb)/maxprb**2
        
        # Dai-Yuan direction
        if i == 0:
            d = -grad
        else:
            d = -grad+cp.linalg.norm(grad)**2 / \
                ((cp.sum(cp.conj(d)*(grad-grad0))))*d
        grad0 = grad
        # line search
        fd = pslv.fwd_holo(d, prb,codes)
        gamma = line_search(minf, 1, psi, fpsi, d, fd)
        psi = psi + gamma*d
        err=minf(psi,fpsi)
        nerr = err/norm_data
        conv[i] = err
        print(f'{i}) {gamma=}, {err=:1.2e}, {nerr=:1.2e}')    
        if nerr<nerr_th:            
            print(f'stopped at {i}')                        
            print(f'{i}) {gamma=}, {err=:1.2e}, {nerr=:1.2e}')    
            break
            
            
    
    return psi, conv

def cg_holo_batch(pslv, data, init, prb, piter, nerr_th, codes=None):
    """Batch of CG solvers"""
    
    res = np.zeros([pslv.ntheta, pslv.nz, pslv.n], dtype='complex64')
    prb_gpu = cp.array(prb)            
    if codes is not None:
        codes_gpu = cp.array(codes)
    else:
        codes_gpu=None          
    for ids in holotomo.utils.chunk(range(pslv.ntheta), pslv.ptheta):
        # copy data part to gpu
        data_gpu = cp.array(data[:,ids])
        init_gpu = cp.array(init[ids])
        
        # Radon transform
        res_gpu,conv = cg_holo(pslv, data_gpu, init_gpu,prb_gpu, piter, nerr_th, codes_gpu)
        # copy result to cpu
        res[ids] = res_gpu.get()
    return res,conv

