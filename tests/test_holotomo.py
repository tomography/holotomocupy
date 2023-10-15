import numpy as np
import dxchange
import ptychotomo
from random import sample
import matplotlib.pyplot as plt
import dxchange
import cupy as cp
import scipy.ndimage as ndimage
import h5py
if __name__ == "__main__":

    # read object
    n = 384  # object size n x,y
    nz = 384  # object size in z
    pnz = 4
    ntheta = 4  # number of angles (rotations)
    ptheta = 4
    voxelsize = 1e-6*2048/n#object voxel size
    energy = 33.35  # xray energy
    center = n/2
    ngpus = 1
    theta = np.linspace(0, np.pi, ntheta).astype('float32')
    focusToDetectorDistance = 128
    sx0 = 0.037
    z1 = np.array([0.4584,0.4765,0.5488,0.69895])-sx0
    z2 = focusToDetectorDistance-z1#np.array([100,90,80,70])/20000 #z2
    distances = (z1*z2)/focusToDetectorDistance
    print(f'{voxelsize=}')
    print(f'{energy=}') 
    print(f'{focusToDetectorDistance=}')
    print(f'{sx0=}')
    print(f'{z1=}')
    print(f'{z2=}')
    print(f'{distances=}')
    magnifications = focusToDetectorDistance/z1#(fdistance-z2)
    print(f'{magnifications=}')
    magnifications/=magnifications[0]
    print(f'normalized magnifications= {magnifications}')
    # Load a 3D object
    beta = np.pad(dxchange.read_tiff('data/beta-chip-256.tiff'),((n//2-128,n//2-128),(n//2-128,n//2-128),(n//2-128,n//2-128)))
    delta = np.pad(dxchange.read_tiff('data/delta-chip-256.tiff'),((n//2-128,n//2-128),(n//2-128,n//2-128),(n//2-128,n//2-128)))
    prb = np.zeros([ntheta,len(distances),nz,n],dtype='complex64')
    #with h5py.File(f'/data/viktor/CP1_P16_530hr_time_test_005nm_01/recon/CP1_P16_530hr_time_test_005nm_01_run20.cxi','r') as fid:
     #   prb[:]=fid['/entry_1/probe/data'][0,1024-n//2:1024+n//2,1024-n//2:1024+n//2]
        #dxchange.write_tiff(prb.real,'prb_esrf/r',overwrite=True)
        #dxchange.write_tiff(prb.imag,'prb_esrf/i',overwrite=True)
    for k in range(4):
        with h5py.File(f'/data/viktor/SiemensLH_33keV_010nm_holoNfpScan_0{k+1}/recon/SiemensLH_33keV_010nm_holoNfpScan_0{k+1}_run01.cxi','r') as fid:
            prb[:,k]=fid['/entry_1/probe/data'][0,1024-n//2:1024+n//2,1024-n//2:1024+n//2]
 
   #smooth
    v = np.arange(-n//2,n//2)/n
    [vx,vy] = np.meshgrid(v,v)
    v=np.exp(-10*(vx**2+vy**2))*10
    delta = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(delta))*v)))).real
    beta = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beta))*v)))).real
    delta[delta>0]=0
    beta[beta<0]=0
    
    delta/=6
    u=delta+1j*beta
    u=u.astype('complex64')

    with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        psi = tslv.fwd_tomo_batch(u) 
   # with ptychotomo.SolverHolotomo(ntheta, nz, n, voxelsize, energy,distances_prb, 1+0*magnifications)  as pslv:
   #     prb = cp.array(prb)        
   #     prb=pslv.fwd_holotomo(cp.array(psi)*0+1,prb)
    with ptychotomo.SolverHolotomo(ntheta, nz, n, voxelsize, energy, distances, magnifications)  as pslv:
        psi = pslv.exptomo(psi)
        psi = cp.array(psi)
        prb = cp.array(prb)
        dxchange.write_tiff_stack(np.abs(psi.get()).astype('float32'),'initabs/r',overwrite=True)
        dxchange.write_tiff_stack(np.angle(psi.get()).astype('float32'),'initangle/r',overwrite=True)
        dxchange.write_tiff_stack(np.abs(prb[0].get()).astype('float32'),'initprbabs/r',overwrite=True)
        dxchange.write_tiff_stack(np.angle(prb[0].get()).astype('float32'),'initprbangle/r',overwrite=True)
        # prb=prb*0+1
        fpsi = pslv.fwd_holotomo(psi,prb)
        data = cp.abs(fpsi)**2
        dxchange.write_tiff_stack(data[0].get().astype('float32'),'datam2/d',overwrite=True)
        # fprb = pslv.fwd_holotomo(psi*0+1,prb)
        # data_prb = cp.abs(fprb)**2
        # data/=data_prb
        # dxchange.write_tiff_stack(data[0].get().astype('float32'),'datam3/d',overwrite=True)
        # exit()
       # for k in range(4):
       #     r = int(magnifications[k]*n)
       #     pad_width = n//2-r//2
       #     print(pad_width)
     #       data[:,k] = cp.pad(data[:,k,n//2-r//2:n//2+r//2,n//2-r//2:n//2+r//2],((0,0),(n//2-r//2,n//2-r//2),(n//2-r//2,n//2-r//2)),'edge')
      #      data_prb[:,k] = cp.pad(data_prb[:,k,n//2-r//2:n//2+r//2,n//2-r//2:n//2+r//2],((0,0),(n//2-r//2,n//2-r//2),(n//2-r//2,n//2-r//2)),'edge')
        #data/=data_prb 
        #dxchange.write_tiff_stack(data[0].get().astype('float32'),'datam2/d',overwrite=True)
        #dxchange.write_tiff_stack(data_prb[0].get().astype('float32'),'dataprb/d',overwrite=True)
        #exit()
        #print(np.max(data),np.min(data))
        psi0 = pslv.adj_holotomo(fpsi, prb)
        print(cp.sum(psi*np.conj(psi0)))
        print(cp.sum(fpsi*np.conj(fpsi)))
        dxchange.write_tiff_stack(np.abs(psi0.get()).astype('float32'),'init0abs/r',overwrite=True)
        dxchange.write_tiff_stack(np.angle(psi0.get()).astype('float32'),'init0angle/r',overwrite=True)
        
        # exit()
        psi0 = psi*0+1
        # prb=prb*0+1
        psi=pslv.cg_holotomo(data, psi0, prb,  64, 'gaussian')
        #prb0 = np.roll(prb0,(0,5,0))
        #psi,prb=pslv.grad_holotomo(data, psi0, prb0,  128, False)
 
    dxchange.write_tiff_stack(data[0].get().astype('float32'),'datam/d',overwrite=True)
    dxchange.write_tiff_stack(np.abs(psi.get()).astype('float32'),'recabs/r',overwrite=True)
    dxchange.write_tiff_stack(np.angle(psi.get()).astype('float32'),'recangle/r',overwrite=True)
    dxchange.write_tiff_stack(np.abs(prb.get()).astype('float32'),'recprbabs/r',overwrite=True)
    dxchange.write_tiff_stack(np.angle(prb.get()).astype('float32'),'recprbangle/r',overwrite=True)




