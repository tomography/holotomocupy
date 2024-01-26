import numpy as np
import dxchange
import holotomo
import matplotlib.pyplot as plt
import xraylib
import cv2
import tifffile
import sys
from skimage.metrics import structural_similarity

PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]

n = 1024  # object size in x,y
nz = 1024  # object size in z    
ntheta = 1  # number of angles (rotations)

pnz = nz # tomography chunk size for GPU processing 
ptheta = ntheta # holography chunk size for GPU processing

center = n/2 # rotation axis
theta = np.linspace(0, np.pi, ntheta).astype('float32') # projection angles

# ID16a setup
voxelsize = 10e-9*2048/n # [m] object voxel size 
energy = 33.35  # [keV] x-ray energy    
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy # [m]
focusToDetectorDistance = 1.28 # [m]
sx0 = 3.7e-4 # [m] motor offset from the focal spot
ndistances = 8
z1 = np.tile(np.array([4.584e-3]),ndistances)-sx0 # distances between planes and the focal spot
z1p = z1.copy()#[:]#np.tile(np.array([focusToDetectorDistance/16]),ndistances)

import random
import scipy.ndimage as ndimage

ill_feature_size = 1e-6#/magnifications[0]

nill = 2**14

# nill = int(np.round(n*voxelsize*magnifications[0]//ill_feature_size))
ill = np.zeros([8,nill,nill],dtype='bool')
for k in  range(ill.shape[0]):
    ill0 = np.zeros([nill*nill],dtype='bool')
    ill_ids = random.sample(range(0, nill*nill), nill*nill//2)
    ill0[ill_ids] = 1
    ill[k] = ill0.reshape(nill,nill)


for k in range(ndistances):    
    name = f'code_{k}'
    np.save(f'codes/{name}.npy',ill[k])
