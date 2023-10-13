from pystackreg import StackReg
import numpy as np
import dxchange

img0 = dxchange.read_tiff('data_lego/delta-lego-256.tiff')[:,128].astype('float32')
n = 7
img0 = np.tile(img0,[n,1,1]).astype('float32')
sx = np.int32((np.random.rand(n)-0.5)*10)
sy = np.int32((np.random.rand(n)-0.5)*10)
sx[0] = 0
sy[0] = 0
for k in range(1,n):
    img0[k] = np.roll(img0[k],sx[k],axis=1)
    img0[k] = np.roll(img0[k],sy[k],axis=0)
    
sr = StackReg(StackReg.TRANSLATION)

# register to mean image
sres = sr.register_stack(img0, reference='first')
sresx = sres[:,0,2]
sresy = sres[:,1,2]
out_mean = sr.transform_stack(img0, tmats = sres)
dxchange.write_tiff_stack(out_mean.astype('float32'),'tmp/reg.tiff',overwrite=True)

print('init shifts={sy},{sx}')
print('res shifts={sresy},{sresx}')
print(f'errors={np.linalg.norm(sresx-sx)},{np.linalg.norm(sresy-sy)}')



