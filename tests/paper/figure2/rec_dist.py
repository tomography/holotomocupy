import numpy as np
import dxchange
import holotomo
import matplotlib.pyplot as plt
import xraylib
import cv2
import tifffile
from skimage.metrics import structural_similarity

PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]

n = 2048  # object size in x,y
nz = 2048  # object size in z    
ntheta = 1  # number of angles (rotations)
noise='True'
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
z1 = np.array([4.584,4.765,5.488,6.9895])*1e-3-sx0 # distances between planes and the focal spot
z2 = focusToDetectorDistance-z1 # distances between planes and the detector
magnifications = focusToDetectorDistance/z1 # actual magnifications
norm_magnifications = magnifications/magnifications[0] # normalized magnifications
distances = (z1*z2)/focusToDetectorDistance # propagation distances after switching from the point source wave to plane wave,
distances = distances*norm_magnifications**2 # scaled propagation distances due to magnified probes

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
print(f'thckness in pixels {thickness}')
# form Transmittance function
u = star*(-delta+1j*beta) # note -delta
Ru = u*thickness 
psi = np.exp(1j * Ru * voxelsize * 2 * np.pi / wavelength)[np.newaxis].astype('complex64')
# tifffile.imwrite('data/psi_amp.tiff',np.abs(psi),)
tifffile.imwrite('data/psi_angle.tiff',np.angle(psi))



pslv = holotomo.SolverHolo(ntheta, nz, n, ptheta, voxelsize, energy, distances, norm_magnifications)
prb = np.ones([len(distances),nz,n],dtype='complex64')
prb = tifffile.imread(f'data/prb_abs_{n}.tiff')*np.exp(1j* tifffile.imread(f'data/prb_phase_{n}.tiff'))
prb[:] = 1
fpsi = pslv.fwd_holo_batch(psi,prb)
data = np.abs(fpsi)**2

if noise:
    data = np.random.poisson(data*50).astype('float32')/50
data =data.astype('float32')

for k in range(len(distances)):    
    tifffile.imwrite(f'data/data_{k}.tiff',data[k])

data0 = data.copy()
prb0 = prb.copy()
distances0 = distances.copy()

from cg import cg_holo_batch    
err = np.zeros(4)
for k in range(4,5):
    rec_distances = k
    prb = prb0[:rec_distances]
    data = data0[:rec_distances]
    distances = distances0[:rec_distances]
    pslv = holotomo.SolverHolo(ntheta, nz, n, ptheta, voxelsize, energy, distances, norm_magnifications)
    piter = 128#512*(k+1) # number of CG iters
    nerr_th = 1e-10
    init = np.ones([ntheta,nz,n],dtype='complex64')  # initial guess
    rec, conv = cg_holo_batch(pslv, data, init, prb,  piter,nerr_th)
    tifffile.imwrite(f'/data/holo/rec_amp_{k}dist{piter}.tiff',np.abs(rec))
    tifffile.imwrite(f'/data/holo/rec_angle_noise{noise}_{k}dist{piter}.tiff',np.angle(rec))
    a = np.angle(rec[0])
    b = np.angle(psi[0])
    a-=np.mean(a)
    b-=np.mean(b)
    data_range=np.amax(b)-np.amin(b)
    (ssim, diff) = structural_similarity(a,b, full=True,data_range=data_range)
    psnr = np.linalg.norm(rec-psi)
    print(f'ndist={k} error={np.linalg.norm(rec-psi)} ssim {ssim} data/rec_angle_{k}dist.tiff')
    np.save(f'res_numpy/pnsrnoise{noise}{k}dist{piter}',psnr)
    np.save(f'res_numpy/ssimnoise{noise}{k}dist{piter}',ssim)
    np.save(f'res_numpy/convnoise{noise}{k}dist{piter}',conv)