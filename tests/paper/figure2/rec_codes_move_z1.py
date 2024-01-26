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
# z1 = np.array([4.584,4.765,5.488,6.9895])*1e-3-sx0 # distances between planes and the focal spot
# z1 = np.array([4.584,5.488,5.488,6.9895])*1e-3-sx0 # distances between planes and the focal spot
ndistances = 4
z1 = np.tile(np.array([4.584e-3]),ndistances)-sx0 # distances between planes and the focal spot
z1p = z1.copy()#[:]#np.tile(np.array([focusToDetectorDistance/16]),ndistances)
z1p[:] = float(sys.argv[1])
z2 = z1p-z1 # distances between planes and the detector
z2p = focusToDetectorDistance-z1p
magnifications = (z1+z2)/z1 # actual magnifications
norm_magnifications = magnifications/magnifications[0] # normalized magnifications
distances = (z1*z2)/(z1+z2) # propagation distances after switching from the point source wave to plane wave,
distances = distances*norm_magnifications**2 # scaled propagation distances due to magnified probes

magnifications2 = (z1p+z2p)/z1p
distances2 = (z1p*z2p)/(z1p+z2p) # propagation distances after switching from the point source wave to plane wave,
norm_magnifications2 = magnifications2/magnifications2[0] # normalized magnifications
distances2 = distances2*norm_magnifications**2 # scaled propagation distances due to magnified probes
distances2 = distances2*(z1/z1p)**2
img = np.zeros((n, n, 3), np.uint8)
triangle = np.array([(n//16, n//2-n//32), (n//16, n//2+n//32), (n//2-n//128, n//2)], np.float32)
star = img[:,:,0]*0
for i in range(0, 360, 15):
    img = np.zeros((n, n, 3), np.uint8)
    degree = i
    theta = degree * np.pi / 180
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]], np.float32)    
    rotated = cv2.gemm(triangle-n//2, rot_mat, 1, None, 1, flags=cv2.GEMM_2_T)+n//2
    cv2.fillPoly(img, [np.int32(rotated)], (255, 0, 0))
    star+=img[:,:,0]
[x,y] = np.meshgrid(np.arange(-n//2,n//2),np.arange(-n//2,n//2))
x = x/n*2
y = y/n*2
# add holes in triangles
circ = (x**2+y**2>0.385)+(x**2+y**2<0.365)
circ *= (x**2+y**2>0.053)+(x**2+y**2<0.05)
circ *= (x**2+y**2>0.0085)+(x**2+y**2<0.008)
star = star*circ/255
v = np.arange(-n//2,n//2)/n
[vx,vy] = np.meshgrid(v,v)
v = np.exp(-5*(vx**2+vy**2))
fu = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(star)))
star = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(fu*v))).real

delta = 1-xraylib.Refractive_Index_Re('Au',energy,19.3)
beta = xraylib.Refractive_Index_Im('Au',energy,19.3)

thickness = 600e-9/voxelsize # siemens star thickness in pixels
#print(f'thckness in pixels {thickness}')
# form Transmittance function
u = star*(-delta+1j*beta) # note -delta
Ru = u*thickness 
psi = np.exp(1j * Ru * voxelsize * 2 * np.pi / wavelength)[np.newaxis].astype('complex64')
tifffile.imwrite('data/psi_amp.tiff',np.abs(psi),)
tifffile.imwrite('data/psi_angle.tiff',np.angle(psi))



pslv = holotomo.SolverHolo(ntheta, nz, n, ptheta, voxelsize, energy, distances, norm_magnifications,distances2)
prb = np.ones([len(distances),nz,n],dtype='complex64')
prb = tifffile.imread(f'data/prb_abs_{n}.tiff')*np.exp(1j* tifffile.imread(f'data/prb_phase_{n}.tiff'))
prb[:] = prb[0]*0+1

import random
import scipy.ndimage as ndimage

ill_feature_size = 2e-6#/magnifications[0]
# print(f'{ill_feature_size=}')
nill = int(np.round(n*voxelsize*magnifications[0]//ill_feature_size))
# print(f'{nill=}')
ill = np.zeros([ndistances,nill,nill],dtype=np.int32)
for k in  range(ill.shape[0]):
    ill0 = np.zeros([nill*nill],dtype=np.int32)
    ill_ids = random.sample(range(0, nill*nill), nill*nill//2)
    ill0[ill_ids] = 1
    ill[k] = ill0.reshape(nill,nill)
ill = ndimage.zoom(ill,[1,n/nill,n/nill],order=0)
delta = 1-xraylib.Refractive_Index_Re('Au',energy,19.3)
beta = xraylib.Refractive_Index_Im('Au',energy,19.3)
# delta = 1-xraylib.Refractive_Index_Re('(C5O2H8)n',energy,1.18)
# beta = xraylib.Refractive_Index_Im('(C5O2H8)n',energy,1.18)
thickness = 3*1e-6/voxelsize # thickness in pixels

# form Transmittance function
Rill = ill*(-delta+1j*beta)*thickness 
psiill = np.exp(1j * Rill * voxelsize * 2 * np.pi / pslv.wavelength()).astype('complex64')

# psiill=psiill*0+1######################
# psiill=None
# prb*=psiill

fpsi = pslv.fwd_holo_batch(psi,prb,psiill)
data = np.abs(fpsi)**2

for k in range(len(distances)):    
    tifffile.imwrite(f'data/data_{k}_{z1p[0]}_{ill_feature_size}.tiff',data[k])
    #print(np.linalg.norm(data[k]))
data0 = data.copy()
prb0 = prb.copy()
distances0 = distances.copy()
distances20 = distances2.copy()

from cg import cg_holo_batch    
err = np.zeros(4)
for k in range(2,5):
    rec_distances = k
    prb = prb0[:rec_distances]
    data = data0[:rec_distances]
    distances = distances0[:rec_distances]
    distances2 = distances20[:rec_distances]
    pslv = holotomo.SolverHolo(ntheta, nz, n, ptheta, voxelsize, energy, distances, norm_magnifications,distances2)
    piter = 1000000#512*(k+1) # number of CG iters
    nerr_th = 5e-4
    init = np.ones([ntheta,nz,n],dtype='complex64')  # initial guess
    rec = cg_holo_batch(pslv, data, init, prb,  piter,nerr_th,psiill)
    tifffile.imwrite(f'data/rec_amp_{k}codes{z1p[0]}.tiff',np.abs(rec))
    tifffile.imwrite(f'data/rec_angle_{k}codes{z1p[0]}.tiff',np.angle(rec))
    a = np.angle(rec[0])
    b = np.angle(psi[0])
    a-=np.mean(a)
    b-=np.mean(b)
    data_range=np.amax(b)-np.amin(b)
    (score, diff) = structural_similarity(a,b, full=True,data_range=data_range)
    print(f'{ill_feature_size=}, ncodes={k} error={np.linalg.norm(rec-psi)} ssim {score} data/rec_angle_{k}codes{z1p[0]}_{ill_feature_size}.tiff')
