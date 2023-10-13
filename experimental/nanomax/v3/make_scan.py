import dxchange
import numpy as np
import sys
import ptychotomo 
from random import sample
import matplotlib.pyplot as plt
# data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'

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
    
    nmodes = 4
    ngpus = 1
    data_prefix = sys.argv[1]
    id_theta = int(sys.argv[2])
        
    scan0 = np.load(data_prefix+'datanpy/scan128sorted_'+str(id_theta)+'.npy').reshape(2,1,81,169)     

    ids = sample(range(169), 16)
    ids = np.arange(id_theta%16,169,16)
    print(ids)
    print(scan0.shape)
    
    scan0 = scan0[:,:,:,ids]
    print(scan0.shape)
    
    plt.plot(scan0[1,0],scan0[0,0],'r.')
    plt.xlim([-2,512])
    plt.ylim([-2,512])
    
    plt.savefig(data_prefix+'tmp/fig'+str(id_theta)+'.png')
    exit()
    # ignore position out of field of view            
    ids = np.where((scan0[0,0]<nz-nprb)*(scan0[1,0]<n-nprb)*(scan0[0,0]>=0)*(scan0[1,0]>=0))[0]


    print(f'{len(ids)}')
    ids = ids[sample(range(len(ids)), min(len(ids),nscan))]
    
    scan[:,:,:min(len(ids),nscan)] = scan0[:, :, ids]
    
    