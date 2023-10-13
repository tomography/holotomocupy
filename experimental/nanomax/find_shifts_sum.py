import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign
from tomoalign.utils import find_min_max
from skimage.feature import register_translation
import matplotlib.pyplot as plt
data_prefix = '/data/staff/tomograms/vviknik/nanomax/'
if __name__ == "__main__":

    # Model parameters
    n = 512-128
    nz = 512-192
    nmodes = 4
    nscan = 13689
    ntheta = 166  # number of angles (rotations)
    shift = np.zeros((ntheta,2),dtype='float32')
    ssum = np.zeros([ntheta,nz-200],dtype='float32')
    cm = np.zeros(ntheta,dtype='float32')
    psiangle = np.zeros([ntheta,nz,n],dtype='float32')       
    psiamp = np.zeros([ntheta,nz,n],dtype='float32')       

    for k in range(ntheta):        
        
        # psiangle[k] =  dxchange.read_tiff(data_prefix+'reccrop_sift_check/psiangle/r'+str(k)+'.tiff')  
        # psiamp[k] = dxchange.read_tiff(data_prefix+'reccrop_sift_check/psiamp/r'+str(k)+'.tiff')          
        # print(f'{data_prefix}recfull_sorted_aligned_new/r_{k:04}.tif')
        psiangle[k] = dxchange.read_tiff(f'{data_prefix}reccrop_sift_check/psiangle/r_{k:05}.tiff')  
        #psiamp[k] = dxchange.read_tiff(f'{data_prefix}recfull_sorted_aligned_new//r_{k:04}.tiff')          
        # print(psiangle[k,30:420].shape)
        # exit()
        psiangle0 = psiangle[k,100:-100].copy()
        psiangle0[psiangle0<0.0] = 0
        ssum[k] = np.sum(psiangle0,axis=1)
    for k in range(0,ntheta):

        # print(ssum[k].shape)
        # print(ssum[0].shape)
        # shift0, error, diffphase = register_translation(ssum[k,np.newaxis], ssum[0,np.newaxis], upsample_factor=10)
        # plt.clf()
        # plt.plot(ssum[0],'r')
        # plt.plot(ssum[k],'b')
        # plt.plot(np.roll(ssum[k],-int(shift0[1])),'g')
        # plt.savefig(f'{data_prefix}/tmp/sum{k:03}')
        # print(shift0)

        # psiangle[k] = np.roll(psiangle[k],-int(shift0[1]),axis=0)    
        
        # print(f'{data_prefix}reccrop_sift_check/r_{k:05}.tiff')
        psiangle0 = psiangle[k,100:-100].copy()
        psiangle0[psiangle0<0.0] = 0
        cm = ndimage.measurements.center_of_mass(psiangle0)
        # print(cm)
        psiangle[k] = np.roll(psiangle[k],-int(cm[1]-n//2),axis=1)    
        # #psiamp[k] = np.roll(psiamp[k],-int(cm[1]-n//2),axis=1)    
        
        print(k,cm[1]-n//2)
        # shift[k,0] = shift0
        # shift[k,1] = cm[1]-n//2

    # np.save(data_prefix+'datanpy/shiftssum_new',shift)        
    dxchange.write_tiff_stack(psiangle,data_prefix+'reccrop_sift_sum/psiangle/r.tiff',overwrite=True)                
    # dxchange.write_tiff_stack(psiamp,data_prefix+'recfull_sorted_aligned_new_check/psiamp'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    
