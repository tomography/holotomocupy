import numpy as np
import dxchange
import holotomo
import xraylib
import cv2
import tifffile
import sys
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from cg import cg_holo_batch   
 
import random
import scipy.ndimage as ndimage

imsize = int(sys.argv[1])
code_position = float(sys.argv[2]) #mm
code_size = float(sys.argv[3]) #um
ncodes = int(sys.argv[4]) 
use_prb = sys.argv[5]
use_codes = sys.argv[6]
nerr_th = float(sys.argv[7])
PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]

n = imsize  # object size in x,y
nz = imsize  # object size in z    

pnz = nz # tomography chunk size for GPU processing 

center = n/2 # rotation axis

# ID16a setup
voxelsize = 10e-9*2048/n # [m] object voxel size 
energy = 33.35  # [keV] x-ray energy    
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy # [m]
focusToDetectorDistance = 1.28 # [m]
sx0 = 3.7e-4 # [m] motor offset from the focal spot
z1 = np.tile(np.array([4.584e-3]),ncodes)-sx0 # distances between planes and the focal spot
z1p = z1.copy()#[:]#np.tile(np.array([focusToDetectorDistance/16]),ncodes)
z1p[:] = code_position*1e-3
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
# img = np.zeros((n, n, 3), np.uint8)
# triangle = np.array([(n//16, n//2-n//32), (n//16, n//2+n//32), (n//2-n//128, n//2)], np.float32)
# star = img[:,:,0]*0
# for i in range(0, 360, 15):
#     img = np.zeros((n, n, 3), np.uint8)
#     degree = i
#     theta = degree * np.pi / 180
#     rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
#                         [np.sin(theta), np.cos(theta)]], np.float32)    
#     rotated = cv2.gemm(triangle-n//2, rot_mat, 1, None, 1, flags=cv2.GEMM_2_T)+n//2
#     cv2.fillPoly(img, [np.int32(rotated)], (255, 0, 0))
#     star+=img[:,:,0]
# [x,y] = np.meshgrid(np.arange(-n//2,n//2),np.arange(-n//2,n//2))
# x = x/n*2
# y = y/n*2
# # add holes in triangles
# circ = (x**2+y**2>0.385)+(x**2+y**2<0.365)
# circ *= (x**2+y**2>0.053)+(x**2+y**2<0.05)
# circ *= (x**2+y**2>0.0085)+(x**2+y**2<0.008)
# star = star*circ/255

# v = np.arange(-n//2,n//2)/n
# [vx,vy] = np.meshgrid(v,v)
# v = np.exp(-5*(vx**2+vy**2))
# fu = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(star)))
# star = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(fu*v))).real

# np.save('star.npy',star)

# star = np.load('star.npy')
star = tifffile.imread('/data/holo/data.tiff')#*0+np.random.random(star.shape)



# exit()
# delta = 1-xraylib.Refractive_Index_Re('Au',energy,19.3)
# beta = xraylib.Refractive_Index_Im('Au',energy,19.3)
delta = 1-xraylib.Refractive_Index_Re('(C5O2H8)n',energy,1.18)
beta = xraylib.Refractive_Index_Im('(C5O2H8)n',energy,1.18)
thickness = 10*600e-9/voxelsize # siemens star thickness in pixels
# form Transmittance function
u = star*(-delta+1j*beta) # note -delta
Ru = u*thickness 
psi = np.exp(1j * Ru * voxelsize * 2 * np.pi / wavelength)[np.newaxis].astype('complex64')

pslv = holotomo.SolverHolo(1, nz, n, 1, voxelsize, energy, distances, norm_magnifications,distances2)
prb = np.ones([len(distances),nz,n],dtype='complex64')
prb[:] = (tifffile.imread(f'data/prb_abs_{n}.tiff')*np.exp(1j* tifffile.imread(f'data/prb_phase_{n}.tiff')))[0]
if use_prb=='False':
    prb[:] = prb[0]*0+1

ill_feature_size = code_size*1e-6

nill = int(n*voxelsize*magnifications[0]//(ill_feature_size*2))*2
ill = np.zeros([ncodes,nill,nill],dtype='bool')
for k in  range(ill.shape[0]):
    ill0 = np.load(f'codes/code_{k}.npy')
    ill[k] = ill0[ill0.shape[0]//2-nill//2:ill0.shape[0]//2+(nill)//2,ill0.shape[1]//2-nill//2:ill0.shape[1]//2+(nill)//2]#.reshape(nill,nill)
ill = ndimage.zoom(ill,[1,n/nill,n/nill],order=0,grid_mode=True,mode='grid-wrap')

v = np.arange(-n//2,n//2)/n
[vx,vy] = np.meshgrid(v,v)
v = np.exp(-50*(vx**2+vy**2))
fill = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(ill)))
ill = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(fill*v))).real.astype('float32')
tifffile.imwrite(f'/data/holo/brain_code{imsize}_{code_position}mm_{code_size}um_{ncodes}_{use_prb}_{use_codes}.tiff',ill[0])
# print(np.amax(np.abs(psiill)))
exit()



delta = 1-xraylib.Refractive_Index_Re('Au',energy,19.3)
beta = xraylib.Refractive_Index_Im('Au',energy,19.3)

thickness = 3*1e-6/voxelsize # thickness in pixels

# form Transmittance function
Rill = ill*(-delta+1j*beta)*thickness 
psiill = np.exp(1j * Rill * voxelsize * 2 * np.pi / pslv.wavelength()).astype('complex64')

if use_codes=='False':
    psiill[:] = 1

fpsi = pslv.fwd_holo_batch(psi,prb,psiill)
data = np.abs(fpsi)**2

prb = prb[:ncodes]
data = data[:ncodes]
distances = distances[:ncodes]
distances2 = distances2[:ncodes]
pslv = holotomo.SolverHolo(1, nz, n, 1, voxelsize, energy, distances, norm_magnifications,distances2)
piter = 4000

init = np.ones([1,nz,n],dtype='complex64')  # initial guess
rec = cg_holo_batch(pslv, data, init, prb,  piter,nerr_th,psiill)
a = np.angle(rec[0])
b = np.angle(psi[0])
a-=np.mean(a)
b-=np.mean(b)
data_range=np.amax(b)-np.amin(b)
(ssim, diff) = structural_similarity(a,b, full=True,data_range=data_range)
psnr = np.linalg.norm(rec-psi)


print(f'{imsize}_{code_position}mm_{code_size}um_{ncodes}_{use_prb}_{use_codes} {psnr=} {ssim=}')
tifffile.imwrite(f'/data/holo/brain_data{imsize}_{code_position}mm_{code_size}um_{ncodes}_{use_prb}_{use_codes}.tiff',data[0])
tifffile.imwrite(f'/data/holo/brain_rec{imsize}_{code_position}mm_{code_size}um_{ncodes}_{use_prb}_{use_codes}.tiff',np.angle(rec))
np.save(f'res_numpy/brain_pnsr{imsize}_{code_position}mm_{code_size}um_{ncodes}_prb{use_prb}_code{use_codes}',psnr)
np.save(f'res_numpy/brain_ssim{imsize}_{code_position}mm_{code_size}um_{ncodes}_prb{use_prb}_code{use_codes}',ssim)
