#Initialization of the sift object is time consuming: it compiles all the code.
import os
import numpy as np
import dxchange
from skimage.registration import phase_cross_correlation
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from numpy import median

#set to 1 to see the compilation going on
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"
#switch to "GPU" to "CPU" to enable fail-save version.
devicetype="CPU"
from silx.image import sift




data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'

    
n = 512-128
nz = 512-192
ntheta = 174
img0 = np.zeros([ntheta,nz,n], dtype='float32')
shift = np.zeros([ntheta,2], dtype='float32')
for k in range(ntheta):
    img0[k] = dxchange.read_tiff(data_prefix+'pgm2_crop/'+str(k)+'.tiff')

shifts = np.zeros([ntheta,2],dtype='float32')
imgres = img0.copy()
for k in range(0,ntheta):
    print(k)
    img = img0[k].copy()
    img[img<0.0] = 0
    cm = ndimage.measurements.center_of_mass(img)
    
    # shifts[k,0] = shifts[max(k-1,0),0]+shifty
    shifts[k,1] = cm[1]-n//2
    # shifts[k,1] = int(cm[1]-256)
    print(shifts[k])
    imgres[k] = np.roll(img0[k],-int(shifts[k,1]),axis=1)    
    imgres[k] = np.roll(imgres[k],-int(shifts[k,0]),axis=0)    
dxchange.write_tiff_stack(imgres.astype('float32'),data_prefix+'prealign_crop_cm/r.tiff',overwrite=True)
print(shifts)
np.save(data_prefix+'/datanpy/shifts_crop_cm.npy',shifts)