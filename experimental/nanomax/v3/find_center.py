import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import ptychotomo
import sys

# data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'
if __name__ == "__main__":

    n = 512-128
    nz = 1    
    niter = 128
    ngpus = 1
    ntheta = 174
    pnz = 1
    nmodes =4
    data_prefix = sys.argv[1]
    nscan = int(sys.argv[2])
    
    theta = np.zeros([ntheta],dtype='float32')
    
    psi = np.zeros([ntheta, 1, n], dtype='complex64', order='C')
    #psiangle0 = dxchange.read_tiff_stack(data_prefix+'reccrop_align_sum/psiangle'+str(nmodes)+str(nscan)+'/r_00000.tiff', ind=np.arange(0,173))
    for k in range(ntheta):        
        psiangle = dxchange.read_tiff(data_prefix+'rec_crop2/psiangle'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff')
        # psiamp = dxchange.read_tiff(f'{data_prefix}rec_full_sorted_aligned_check_new/psiamp{nmodes}{nscan}r_{k:05}.tif')  
        
        print(np.linalg.norm(psiangle))
        # psiangle = dxchange.read_tiff(data_prefix+'rec_crop_sum_check/psiangle'+str(nmodes)+str(nscan)+'/r%d.tiff' % k)
        # psiamp = dxchange.read_tiff(data_prefix+'rec_crop_sum_check/psiamp'+str(nmodes)+str(nscan)+'/r%d.tiff' % k)
        print(psiangle.shape)
        psi[k] = psiangle[:,psiangle.shape[1]//3]
        theta[k] = np.load(data_prefix+'datanpy/theta128sorted_'+str(k)+'.npy')
    
    psi += 1j*0
    for center in range(n//2-10,n//2+40):
        print(center)
        u = np.zeros([1,n,n],dtype='complex64')
        with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
            u = tslv.grad_tomo_batch(psi, psi*0+1, u, niter)
            
            dxchange.write_tiff(u[0].real,data_prefix+'test_center_'+str(nscan)+'/r'+str(center)+'.tiff',overwrite=True)
            
    
    