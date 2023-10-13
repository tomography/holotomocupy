import dxchange
import numpy as np
import sys
import ptychotomo 
from random import sample

data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'

if __name__ == "__main__":
    
    n = 512
    nz = 512
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
    nscan = 2000
    
    id_theta = int(sys.argv[1])
    data = np.zeros([1, nscan, ndet, ndet], dtype='float32')
    scan = np.zeros([2, 1, nscan], dtype='float32')-1
    theta = np.zeros([1],dtype='float32')
        
    data0 = np.load(data_prefix+'datanpy/data128_'+str(id_theta)+'.npy')        
    scan0 = np.load(data_prefix+'datanpy/scan128_'+str(id_theta)+'.npy')     
    shifts1 = np.load(data_prefix+'/datanpy/shifts1.npy')[id_theta]
    shifts2 = np.load(data_prefix+'/datanpy/shifts2.npy')[id_theta]
    shifts3 = np.load(data_prefix+'/datanpy/shifts3.npy')[id_theta]
    
    # scan0[1] -= (shifts1[1]+shifts2[1]+shifts3[1])
    # scan0[0] -= (shifts1[0]+shifts2[0]+shifts3[0])
      
    # ignore position out of field of view            
    ids = np.where((scan0[0,0]<nz-nprb)*(scan0[1,0]<n-nprb)*(scan0[0,0]>=0)*(scan0[1,0]>=0))[0]

    print(f'{len(ids)}')
    ids = ids[sample(range(len(ids)), min(len(ids),nscan))]
    
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
    dxchange.write_tiff(np.angle(psi),  data_prefix+'rec_full_na/psiangle'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    dxchange.write_tiff(np.abs(psi),   data_prefix+'rec_full_na/psiamp'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    for m in range(nmodes):
        dxchange.write_tiff(np.angle(prb[:,m]),   data_prefix+'rec_full/prbangle/r'+str(id_theta), overwrite=True)
        dxchange.write_tiff(np.abs(prb[:,m]),   data_prefix+'rec_full/prbamp/r'+str(id_theta), overwrite=True)
        