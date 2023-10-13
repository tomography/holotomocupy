import matplotlib.pyplot as plt
import numpy as np
import dxchange
import os

data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'
if __name__ == "__main__":

    # Model parameters
    n = 512  # object size n x,y
    nz = 512  # object size in z
    nmodes = 4
    nscan = 13689
    ntheta = 174  # number of angles (rotations)
    ndet = 128
    theta = np.zeros(ntheta, dtype='float32')
    scan = np.zeros([2,ntheta,nscan], dtype='float32')
    
    for k in range(0,ntheta):        
        # Load a 3D object
        print(k)
        theta[k] = np.load(data_prefix+'datanpy/theta128_'+str(k)+'.npy')
        scan[:,k:k+1] = np.load(data_prefix+'datanpy/scan128_'+str(k)+'.npy')
    
    ids = np.argsort(theta)
    theta = theta[ids]
    scan = scan[:,ids]
    
    
    for k in range(0,ntheta):        
        print(k)
        
        data = np.load(data_prefix+'datanpy/data128_'+str(ids[k])+'.npy')        
        np.save(data_prefix+'/datanpy/theta128sorted_'+str(k)+'.npy',theta[k])    
        np.save(data_prefix+'/datanpy/data128sorted_'+str(k)+'.npy',data)
        np.save(data_prefix+'/datanpy/scan128sorted_'+str(k)+'.npy',scan[:,k:k+1])
        