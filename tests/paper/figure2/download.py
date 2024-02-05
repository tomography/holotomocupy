# pip install numpy tifffile cloud-volume
import numpy as np
import tifffile
from cloudvolume import CloudVolume

vol = CloudVolume(
    "s3://open-neurodata/bloss/bloss18/image", mip=0, use_https=True
)

# load data into numpy array
cutout = np.transpose(vol[5632+4096:5632+4096*2, 3584+4096:3584+4096*2, 215:216])[0,0].astype('float32')
print(cutout.shape)
cutout = 0.5*(cutout[::2]+cutout[1::2])
cutout = 0.5*(cutout[:,::2]+cutout[:,1::2])/255.0
# save cutout as TIFF
tifffile.imwrite("/data/holo/data.tiff", data=cutout)