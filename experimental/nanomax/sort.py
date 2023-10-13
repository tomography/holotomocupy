import matplotlib.pyplot as plt
import numpy as np
import dxchange

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'
if __name__ == "__main__":

    # Model parameters
    n = 512  # object size n x,y
    nz = 512  # object size in z
    nmodes = 4
    nscan = 13689
    ntheta = 166  # number of angles (rotations)
    ndet = 128
    rec = np.zeros([ntheta,nz,n],dtype='float32')
    recamp = np.zeros([ntheta,nz,n],dtype='float32')
    theta = np.zeros(ntheta, dtype='float32')
    scan = np.zeros([2,ntheta,nscan], dtype='float32')
    #data = np.zeros([ntheta,nscan,ndet,ndet],dtype='float32')

    for k in range(0,ntheta):        
        # Load a 3D object
        print(k)
        rec[k] = dxchange.read_tiff(data_prefix+'recfull/psiangle'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff')              
        recamp[k] = dxchange.read_tiff(data_prefix+'recfull/psiamp'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff')              
        theta[k] = np.load(data_prefix+'datanpy/theta128_'+str(k)+'.npy')
        scan[:,k:k+1] = np.load(data_prefix+'datanpy/scan128_'+str(k)+'.npy')
        #data[k] = np.load(data_prefix+'datanpy/data128_'+str(k)+'.npy')

    ids = np.argsort(theta)
    theta = theta[ids]
    #data = data[ids]    
    scan = scan[:,ids]
    rec = rec[ids]
    
    for k in range(0,ntheta):        
        print(k)
        data = np.load(data_prefix+'datanpy/data128_'+str(ids[k])+'.npy')        
        np.save(data_prefix+'/datanpy/theta128sorted_'+str(k)+'.npy',theta[k])    
        np.save(data_prefix+'/datanpy/data128sorted_'+str(k)+'.npy',data)
        np.save(data_prefix+'/datanpy/scan128sorted_'+str(k)+'.npy',scan[:,k:k+1])
        dxchange.write_tiff(data, data_prefix+'datafull_sorted/data'+str(k)+'.tiff',overwrite=True)            
    dxchange.write_tiff_stack(rec, data_prefix+'recfull_sorted/psiangle'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)            
    dxchange.write_tiff_stack(rec, data_prefix+'recfull_sorted/psiamp'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)            
