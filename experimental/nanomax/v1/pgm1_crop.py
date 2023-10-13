import dxchange
import numpy as np
import sys
from scipy.interpolate import griddata

data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'

if __name__ == "__main__":
    
    n = 512-128
    nz = 512-192
    ndet = 128
    ntheta = 1
    nprb = 128  # probe size
    
    # Reconstrucion parameters
    nscan = 13689
    
    id_theta = int(sys.argv[1])
    data = np.zeros([1, nscan, ndet, ndet], dtype='float32')
    scan = np.zeros([2, 1, nscan], dtype='float32')-1
    theta = np.zeros([1],dtype='float32')
        
    data[0] = np.load(data_prefix+'datanpy/data128sorted_'+str(id_theta)+'.npy')        
    scan = np.load(data_prefix+'datanpy/scan128sorted_'+str(id_theta)+'.npy')        
    shifts = np.load(data_prefix+'/datanpy/shifts.npy')[id_theta]
    scan[1] -= np.round(shifts[1])
    scan[0] -= np.round(shifts[0])
    scan[1] -= 64
    scan[0] -= 160

    # ignore position out of field of view            
    ids = np.where((scan[0,0]<nz-nprb)*(scan[1,0]<n-nprb)*(scan[0,0]>=0)*(scan[1,0]>=0))[0]

    print(f'{len(ids)}')
    scan = scan[:,0,ids]+64
    data = data[0,ids]
    a = np.sum(data[:,0:data.shape[-2]//2,:],axis=(1,2))
    b = np.sum(data[:,data.shape[-1]//2::,:],axis=(1,2))
    c = np.sum(data[:, :,0:data.shape[-2]//2],axis=(1,2))
    d = np.sum(data[:, :,data.shape[-1]//2::],axis=(1,2))
    grad_x = (a-b)/(a+b+1e-15)
    grad_y = (c-d)/(c+d+1e-15)

    x, y = np.mgrid[0:nz, 0:n]
    print(x.shape,grad_x.shape,scan.shape)
    # grad_x[0,:] =0
    # grad_x[-1,:] =0
    # grad_x[:,0] =0
    # grad_y[:,-1] =0
    # grad_y[0,:] =0
    # grad_y[-1,:] =0
    # grad_y[:,0] =0
    # grad_y[:,-1] =0
    dpc_x = griddata(scan.T, grad_x, (x,y), method='cubic', fill_value=0).astype('float32')
    dpc_y = griddata(scan.T, grad_y, (x,y), method='cubic', fill_value=0).astype('float32')
    dpc = np.sqrt((dpc_x-np.mean(dpc_x))**2+(dpc_y-np.mean(dpc_y))**2)
    
    dpc -= dpc[0,0]
    # dpc -=np.mean(dpc)
    a = np.mean(dpc[220:240,265:285])
    dpc[dpc!=0]-=a
    
    # dpc 
    # dpc=dpc_x.astype('float32')
    
    # f = griddata(scan.swapaxes(0,1), sdata, (x,y), method='nearest').astype('float32')
    
    name = data_prefix+'pgm1_crop/' + str(id_theta) + '.tiff'
    dxchange.write_tiff(dpc,name,overwrite=True)
    
