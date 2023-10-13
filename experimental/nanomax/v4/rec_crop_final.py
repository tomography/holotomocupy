import dxchange
import numpy as np
import sys
import ptychotomo 
from random import sample
import matplotlib.pyplot as plt

# data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'

if __name__ == "__main__":
    n = 512-128
    nz = 512-192
    ndet = 128
    ntheta = 1
    ptheta = 1 
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    recover_prb = True
    
    # Reconstrucion parameters
    piter = 16  # ptychography iterations
    nmodes = 4
    ngpus = 1
    
    data_prefix = sys.argv[1]
    id_theta = int(sys.argv[2])
    nscan = int(sys.argv[3])
    step = int(sys.argv[4])    
    piter = int(sys.argv[5])
    
    
    data = np.zeros([1, nscan, ndet, ndet], dtype='float32')
    scan = np.zeros([2, 1, nscan], dtype='float32')-1
    theta = np.zeros([1],dtype='float32')
        
    data0 = np.load(data_prefix+'datanpy/data128sorted_'+str(id_theta)+'.npy').reshape(81,169,128,128)              
    scan0 = np.load(data_prefix+'datanpy/scan128sorted_'+str(id_theta)+'.npy').reshape(2,1,81,169)          
    ids = np.arange(id_theta%step,169,step)
    scan0 = scan0[:,:,:,ids].reshape(2,1,len(ids)*81)
    data0 = data0[:,ids].reshape(len(ids)*81,128,128)
    
    shifts1 = np.load(data_prefix+'/datanpy/shifts1_'+str(2000)+'.npy')[id_theta]
    shifts2 = np.load(data_prefix+'/datanpy/shifts2_'+str(2700)+'.npy')[id_theta]
    
    scan0[1] -= (shifts1[1]+shifts2[1])
    scan0[0] -= (shifts1[0]+shifts2[0])
    scan0[1] -= (64+29)
    scan0[0] -= (160)
    # ignore position out of field of view            
    ids = np.where((scan0[0,0]<nz-nprb)*(scan0[1,0]<n-nprb)*(scan0[0,0]>=0)*(scan0[1,0]>=0))[0]

    print(f'{len(ids)}')
    scan[:,:,:min(len(ids),nscan)] = scan0[:, :, ids]
    data[0,:min(len(ids),nscan)] = data0[ids]    
    print(nscan)
    # init probes
    prb = np.zeros([1, nmodes, nprb, nprb], dtype='complex64')
    prb[:] = np.load(data_prefix+'datanpy/prb128.npy')
    
    # Initial guess
    psi = np.ones([1, nz, n], dtype='complex64')    

    # data sh
    data = np.fft.fftshift(data, axes=(2, 3))/ndet/ndet
    
    
    name = data_prefix+'rec'+str(recover_prb)+str(nmodes)+str(scan.shape[2])

    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi, prb = pslv.grad_ptycho_batch(
            data, psi, prb, scan, psi*0, -1, piter, recover_prb)   
    
    # Save result
    dxchange.write_tiff(np.angle(psi),  data_prefix+'rec_crop_final/psiangle'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    dxchange.write_tiff(np.abs(psi),   data_prefix+'rec_crop_final/psiamp'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    for m in range(nmodes):
        dxchange.write_tiff(np.angle(prb[:,m]),   data_prefix+'rec_crop_final/prbangle/r'+str(id_theta), overwrite=True)
        dxchange.write_tiff(np.abs(prb[:,m]),   data_prefix+'rec_crop_final/prbamp/r'+str(id_theta), overwrite=True)
        