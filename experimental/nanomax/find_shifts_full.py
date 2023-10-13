import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign
from tomoalign.utils import find_min_max
from skimage.feature import register_translation

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'
if __name__ == "__main__":

    # Model parameters
    n = 512  # object size n x,y
    nz = 512  # object size in z
    nmodes = 4
    nscan = 13689
    ntheta = 166  # number of angles (rotations)
    data = dxchange.read_tiff_stack(data_prefix+'recfull_sorted/psiangle'+str(nmodes)+str(nscan)+'/r_00000.tiff',ind=np.arange(0,ntheta)).astype('float32')       
    dataabs = dxchange.read_tiff_stack(data_prefix+'recfull_sorted/psiamp'+str(nmodes)+str(nscan)+'/r_00000.tiff',ind=np.arange(0,ntheta)).astype('float32')       
    data_sift = dxchange.read_tiff_stack(data_prefix+'recfull_sorted_sift/r_00000.tiff', ind=np.arange(0,ntheta)).astype('float32')    
    shift = np.zeros((ntheta,2),dtype='float32')
    for k in range(ntheta):
        shift0, error, diffphase = register_translation(data[k], data_sift[k], upsample_factor=1)
        print(shift0)
        data[k]=np.roll(data[k],(-int(shift0[0]),-int(shift0[1])),axis=(0,1))
        dataabs[k]=np.roll(dataabs[k],(-int(shift0[0]),-int(shift0[1])),axis=(0,1))
        shift[k]=shift0
    dxchange.write_tiff_stack(data,data_prefix+'recfull_sorted_sift_check/psiangle'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    dxchange.write_tiff_stack(dataabs,data_prefix+'recfull_sorted_sift_check/psiamp'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    np.save(data_prefix+'/datanpy/shifts.npy',shift)

