import numpy as np

PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]

n = 2048 # 1d signal sizie
ns = 128 # number of signals
voxelsize = 10e-9 # object voxel size
energy = 33.35  # [keV] xray energy
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy
focusToDetectorDistance = 1.28 # [m] distance between focus and detector

# Sample to detector propagation
z1 = np.array([4.584e-3, 4.765e-3, 5.488e-3, 6.9895e-3])-3.7e-4 # [m] distances between focus and sample planes
z2 = focusToDetectorDistance-z1 # [m] distances between `sample planes and detector
distances = (z1*z2)/focusToDetectorDistance # [m] propagation distances after switching from the point source wave to plane wave
magnifications = focusToDetectorDistance/z1 # magnification when propagating from the sample plane to the detector
norm_magnifications = magnifications/magnifications[0] # normalized magnifications
distances = distances*norm_magnifications**2 # scaled propagation distances due to magnified probes

# Assuming the probe is given at the first plane, propagate it to the sample planes
z1_p = z1[0]/2  # position of the probe for reconstruction
z2_p = z1-np.tile(z1_p, len(z1)) # distance between the probe and sample
magnifications_p = (z1_p+z2_p)/z1_p # magnification when propagating from the probe plane to the detector
distances_p = (z1_p*z2_p)/(z1_p+z2_p) # propagation distances after switching from the point source wave to plane wave,
norm_magnifications_p = magnifications_p/magnifications_p[0]  # normalized magnifications
distances_p = distances_p*norm_magnifications_p**2 # scaled propagation distances due to magnified probes
distances_p = distances_p*(z1_p/z1)**2 # scaling for double propagation 

# Fresnel kernels
fP = np.zeros([len(distances), 2*n], dtype='complex64')
fP_p = np.zeros([len(distances_p), 2*n], dtype='complex64')
fx = np.fft.fftfreq(2*n, d=voxelsize)
for i, d in enumerate(distances):
    fP[i] = np.exp(-1j*np.pi*wavelength*d*fx**2)/2
for i, d in enumerate(distances_p):
    fP_p[i] = np.exp(-1j*np.pi *wavelength*d*fx**2)/2

def propagate(f,fP):
    '''Propagate ns signals with Fresnel propagator fP'''
    ff = np.pad(f,((0,0),(n//2,n//2)),'symmetric')
    ff = np.fft.ifft(np.fft.fft(ff)*fP)
    ff = ff[:,n//2:-n//2]
    return ff

# Double propagation test:
f= (np.random.random([ns,n])+1j*np.random.random([ns,n])).astype('complex64')
g = np.zeros([len(distances),ns,n],dtype='complex64')
for k in range(len(distances)):
    # double propagation over different planes 
    g[k] = propagate(propagate(f, fP_p[k]),fP[k])

# data after double propagation should be the same for all distances
print(g)
print(np.linalg.norm(g-g[0],axis=(1,2))/np.linalg.norm(g[0]))


