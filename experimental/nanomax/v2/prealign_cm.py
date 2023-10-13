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

n = 512
ntheta = 174
img0 = np.zeros([ntheta,n,n], dtype='float32')
shift = np.zeros([ntheta,2], dtype='float32')
for k in range(ntheta):
    img0[k] = dxchange.read_tiff(data_prefix+'pgm3/'+str(k)+'.tiff')

sift_ocl = sift.SiftPlan(template=img0[0], devicetype=devicetype)
keypoints = sift_ocl(img0[0])
print("Number of keypoints: %s"%len(keypoints))
print("Keypoint content:")
print(keypoints.dtype)
print("x: %.3f \t y: %.3f \t sigma: %.3f \t angle: %.3f" %
      (keypoints[-1].x,keypoints[-1].y,keypoints[-1].scale,keypoints[-1].angle))
print("descriptor:")
print(keypoints[-1].desc)

plt.imshow(img0[0])
plt.plot(keypoints[:].x, keypoints[:].y,".g")
plt.savefig(data_prefix+'tmp/r1')


shifts = np.zeros([ntheta,2],dtype='float32')
imgres = img0.copy()

for k in range(0,ntheta):
    print(k)
    keypoints = sift_ocl(img0[max(0,k-1)])
    keypoints_shifted = sift_ocl(img0[k])
# plt.imshow(img0[1])
# plt.plot(keypoints_shifted[:].x, keypoints_shifted[:].y,".g")
# plt.savefig(data_prefix+'tmp/r2')
    mp = sift.MatchPlan()
    match = mp(keypoints, keypoints_shifted)
    print("Number of Keypoints with for image 1 : %i, For image 2 : %i, Matching keypoints: %i" % (keypoints.size, keypoints_shifted.size, match.shape[0]))

    shiftx = median(match[:,1].x-match[:,0].x)
    shifty = median(match[:,1].y-match[:,0].y)
    print("Measured offsets dx: %.3f, dy: %.3f" % (shiftx, shifty))    
    
    img = img0[k,150:300].copy()
    img[img<0.0] = 0
    cm = ndimage.measurements.center_of_mass(img)
    
    shifts[k,0] = shifts[max(k-1,0),0]+shifty
    # shifts[k,1] = shifts[max(k-1,0),1]+shiftx       
    shifts[k,1] = int(cm[1]-256)

    imgres[k] = np.roll(img0[k],-int(shifts[k,1]),axis=1)    
    imgres[k] = np.roll(imgres[k],-int(shifts[k,0]),axis=0)    

dxchange.write_tiff_stack(imgres.astype('float32'),data_prefix+'prealign_cm/r.tiff',overwrite=True)
print(shifts)
np.save(data_prefix+'/datanpy/shifts3.npy',shifts)