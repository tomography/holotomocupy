import numpy as np
import matplotlib.pyplot as plt
shiftssift = np.load('/data/staff/tomograms/vviknik/nanomax/datanpy/shifts.npy') + \
    np.load('/data/staff/tomograms/vviknik/nanomax/datanpy/shiftscrop.npy')
thetaazat = np.load('/data/staff/tomograms/vviknik/nanomax/datanpy/theta2.npy')
shiftsazat = shiftssift.copy()

sx = -np.load('/data/staff/tomograms/vviknik/nanomax/datanpy/sx_txm2.npy')
sy = -np.load('/data/staff/tomograms/vviknik/nanomax/datanpy/sy_txm2.npy')
theta = np.zeros(166,dtype='float32')
for k in range(0,166):        
    # Load a 3D object
    print(k)         
    theta[k] = np.load('/data/staff/tomograms/vviknik/nanomax/datanpy/theta128sorted_'+str(k)+'.npy')
th1  = np.int32(np.round(theta/np.pi*180))
th2 = np.int32(np.round(thetaazat))

sazat = shiftssift.copy()
kk=0
for k in range(len(theta)):
    while(th1[k]!=th2[kk]):
        kk+=1
    sazat[k,0] = sx[kk]
    sazat[k,1] = sy[kk]
    # print(shiftssift.shape)
    print(th1[k],th2[kk],sazat[k],shiftssift[k])
    kk+=1
np.save('/data/staff/tomograms/vviknik/nanomax/datanpy/shiftsazat.npy',sazat)  

plt.plot(shiftssift[:,0],'b.')
plt.plot(shiftsazat[:,0],'r.')
plt.plot(shiftssift[:,0]-shiftsazat[:,0],'g.')
#plt.plot(shiftsazat2[:,0],'g.')
plt.savefig('/home/vviknik/figx.png')
plt.clf()
plt.plot(shiftssift[:,1],'b.')
plt.plot(shiftsazat[:,1],'r.')
plt.plot(shiftssift[:,1]-shiftsazat[:,1],'g.')
plt.savefig('/home/vviknik/figy.png')
# exit()
