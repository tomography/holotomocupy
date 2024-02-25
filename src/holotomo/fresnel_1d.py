import cupy as cp
PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]


n = 2048
voxelsize = 10e-9 # object voxel size
energy = 33.35  # [keV] xray energy
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy
focusToDetectorDistance = 1.28 # [m] distance between focus and detector

# Sample to detector propagation
z1 = cp.array([4.584e-3, 4.765e-3, 5.488e-3, 6.9895e-3])-3.7e-4 # [m] distances between focus and sample planes
z2 = focusToDetectorDistance-z1 # [m] distances between sample planes and detector
distances = (z1*z2)/focusToDetectorDistance # [m] propagation distances after switching from the point source wave to plane wave
magnifications = focusToDetectorDistance/z1 # magnification when propagating from the sample plane to the detector
norm_magnifications = magnifications/magnifications[0] # normalized magnifications
distances = distances*norm_magnifications**2 # scaled propagation distances due to magnified probes

# Assuming the probe is given at the first plane, propagate it to the sample planes
z1_p = z1[0]  # position of the probe for reconstruction
z2_p = z1-cp.tile(z1_p, len(z1)) # distance between the probe and sample
magnifications_p = (z1_p+z2_p)/z1_p # magnification when propagating from the probe plane to the detector
distances_p = (z1_p*z2_p)/(z1_p+z2_p) # propagation distances after switching from the point source wave to plane wave,
norm_magnifications_p = magnifications_p/magnifications_p[0]  # normalized magnifications
distances_p = distances_p*norm_magnifications_p**2 # scaled propagation distances due to magnified probes
distances_p = distances_p*(z1_p/z1)**2 # scaling for double propagation 

# Fresnel kernels
fP = cp.zeros([len(distances), 2*n], dtype='complex64')
fP_p = cp.zeros([len(distances_p), 2*n], dtype='complex64')
fx = cp.fft.fftshift(cp.fft.fftfreq(2*n, d=voxelsize))
for i, d in enumerate(distances):
    fP[i] = cp.exp(-1j*cp.pi*wavelength*d*fx**2)/2
for i, d in enumerate(distances_p):
    fP_p[i] = cp.exp(-1j*cp.pi *wavelength*d*fx**2)/2

def propagate(f,fP):
    '''Propagate with Fresnel propagator fP'''
    ff = cp.pad(f,(n//2,n//2))
    ff = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ff)))
    ff = ff*fP
    ff = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(ff)))
    ff = ff[n//2:-n//2,n//2:-n//2]
    return ff

f = cp.ones(n,dtype='complex64')
print(f.shape)
ff = propagate(f, fP_p[0])
fff = propagate(ff, fP[0])

print(fff)





