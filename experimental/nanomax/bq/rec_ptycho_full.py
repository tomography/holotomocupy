import dxchange
import numpy as np
import sys
import ptychotomo 

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'

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
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 128  # ptychography iterations
    nmodes = 4
    ngpus = 1
    nscan = 13689
    
    id_theta = int(sys.argv[1])
    data = np.zeros([1, nscan, ndet, ndet], dtype='float32')
    scan = np.zeros([2, 1, nscan], dtype='float32')-1
    theta = np.zeros([1],dtype='float32')
        
    data[0] = np.load(data_prefix+'datanpy/data128_'+str(id_theta)+'.npy')        
    scan0 = np.load(data_prefix+'datanpy/scan128_'+str(id_theta)+'.npy')        
    scan[0] = scan0[1]
    scan[1] = scan0[0]
        
    # ignore position out of field of view        
    ids = np.where((scan[0,0]>n-1-nprb)+(scan[1,0]>nz-1-nprb)+(scan[0,0]<0)+(scan[1,0]<0))[0]
    scan[0,0,ids]=-1
    scan[1,0,ids]=-1
    data[0,ids] = 0     

    # init probes
    prb = np.zeros([1, nmodes, nprb, nprb], dtype='complex64')
    prb[:] = np.load(data_prefix+'datanpy/prb128.npy')
    
    # Initial guess
    psi = np.ones([1, nz, n], dtype='complex64')    

    # data sh
    data = np.fft.fftshift(data, axes=(2, 3))/ndet/ndet
    
    name = data_prefix+'rec'+str(recover_prb)+str(nmodes)+str(scan.shape[2])

    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi, prb = pslv.cg_ptycho_batch(data, psi, prb, scan, piter, recover_prb)
    
    # Save result
    dxchange.write_tiff(np.angle(psi),  data_prefix+'recfull/psiangle'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    dxchange.write_tiff(np.abs(psi),   data_prefix+'recfull/psiamp'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    for m in range(nmodes):
        dxchange.write_tiff(np.angle(prb[:,m]),   data_prefix+'recfull/prbangle/r'+str(id_theta), overwrite=True)
        dxchange.write_tiff(np.abs(prb[:,m]),   data_prefix+'recfull/prbamp/r'+str(id_theta), overwrite=True)
        