import h5py
import numpy as np
import matplotlib.pyplot as plt

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'

def read_data(id_data):
    try:
        h5file = h5py.File(
            data_prefix+'/data/scan_000'+str(id_data)+'.h5', 'r')
        data = h5file['measured/diffraction_patterns'][:].astype('float32')
        positions = h5file['measured/positions_um'][:].astype('float32')
        mask = h5file['measured/mask'][:].astype('float32')
        data *= mask
        data = np.roll(data,(2,1),axis=(1,2))    
        theta = h5file['measured/angle_deg'][()].astype('float32')/180*np.pi
        scan = ((-positions)*1e3+4.02*1e3)/18.03
        scan = scan[np.newaxis, ...].astype(
            'float32').swapaxes(0, 1).swapaxes(0, 2)
    except:
        scan = None
        theta = None
        data = None
    return data, scan, theta

def read_rec(id_data):
    h5file = h5py.File(
            data_prefix+'/data/scan'+str(id_data)+'_DM_1000.ptyr', 'r')
    psi = h5file['content/obj/Sscan00G00/data'][:]
    probe = h5file['content/probe/Sscan00G00/data'][:]
    positions = h5file['content/positions/Sscan00G00'][:]
    
    scan = ((positions)*1e9+4.02*1e3)/18.03    
    scan = scan[np.newaxis, ...].astype(
            'float32').swapaxes(0, 1).swapaxes(0, 2)
        
    return psi, probe, scan

if __name__ == "__main__":   
    kk = 0
    for k in range(134, 424, 1):
        print(kk, k)
        data, scan, theta = read_data(k)
        if(scan is not None):            
            plt.clf()
            plt.plot(scan[1],scan[0],'r.',markersize=1)            
            plt.axis('equal')
            plt.savefig(data_prefix+'/scan_pos/'+str(kk)+'.png',dpi=450)
            print('theta',theta)
            print('scan',scan.shape,np.min(scan[0]),np.min(scan[1]),np.max(scan[0]),np.max(scan[1]))
            print('data',data.shape,np.linalg.norm(data))
            np.save(data_prefix+'datanpy/theta128_'+str(kk),theta)
            np.save(data_prefix+'datanpy/scan128_'+str(kk),scan)
            np.save(data_prefix+'datanpy/data128_'+str(kk),data)            
            kk += 1                
    psirec,prbrec,scanrec = read_rec(210)    
    # Load a 3D object
    np.save(data_prefix+'datanpy/prb128',prbrec)
    