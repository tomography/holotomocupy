import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import ptychotomo

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'
if __name__ == "__main__":

    n = 512
    nz = 1
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    niter = 32
    nmodes = 4
    ngpus = 1
    nscan = 13689
    ntheta = 166
    pnz = 1
    ptheta = 1
    
    theta = np.zeros([ntheta],dtype='float32')
    
    psi = np.zeros([ntheta, 1, n], dtype='complex64', order='C')
    #psiangle0 = dxchange.read_tiff_stack(data_prefix+'reccrop_align_sum/psiangle'+str(nmodes)+str(nscan)+'/r_00000.tiff', ind=np.arange(0,173))
    for k in range(ntheta):
        print(f'{data_prefix}recfull_sorted_aligned_new_check/psiangle{nmodes}{nscan}/r_{k:05}.tiff')
        psiangle = dxchange.read_tiff(f'{data_prefix}recfull_sorted_aligned_new_check/psiangle{nmodes}{nscan}/r_{k:05}.tiff')  
        # psiamp = dxchange.read_tiff(f'{data_prefix}rec_full_sorted_aligned_check_new/psiamp{nmodes}{nscan}r_{k:05}.tif')  
        
        print(np.linalg.norm(psiangle))
        # psiangle = dxchange.read_tiff(data_prefix+'rec_crop_sum_check/psiangle'+str(nmodes)+str(nscan)+'/r%d.tiff' % k)
        # psiamp = dxchange.read_tiff(data_prefix+'rec_crop_sum_check/psiamp'+str(nmodes)+str(nscan)+'/r%d.tiff' % k)

        psi[k] = psiangle[250:251]
        theta[k] = np.load(data_prefix+'datanpy/theta128sorted_'+str(k)+'.npy')
    
    psi+=1j*0
    for center in range(250,300):
        print(center)
        u = np.zeros([1,n,n],dtype='complex64')
        with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
            u = tslv.cg_tomo_batch(psi, u, niter)
            
            dxchange.write_tiff(u[0].real,data_prefix+'test_center_admm/r'+str(center)+'.tiff',overwrite=True)
            dxchange.write_tiff(u[0].imag,data_prefix+'test_center_admm/i'+str(center)+'.tiff',overwrite=True)
    