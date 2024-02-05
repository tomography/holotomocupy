import numpy as np
import matplotlib.pyplot as plt
import tifffile
imsize = 2048
ncodes = 3
use_prb = 'False'
use_codes = 'True'
code_size = 1
code_position = 8.0
psnr = np.load(f'res_numpy/pnsr{imsize}_{code_position:.1f}mm_{code_size:.1f}um_{ncodes}_prb{use_prb}_code{use_codes}.npy')
ssim = np.load(f'res_numpy/ssim{imsize}_{code_position:.1f}mm_{code_size:.1f}um_{ncodes}_prb{use_prb}_code{use_codes}.npy')
conv = np.load(f'res_numpy/conv{imsize}_{code_position:.1f}mm_{code_size:.1f}um_{ncodes}_prb{use_prb}_code{use_codes}.npy')

piter = 128
init = 'data/psi_angle.tiff'
datacnoise = tifffile.imread(f'/data/holo/datanoise{imsize}_{code_position:.1f}mm_{code_size:.1f}um_{ncodes}_{use_prb}_{use_codes}.tiff')

# # exit()
# # fig = plt.figure(figsize=(5,5))
# # plt.imshow(datacnoise[0],aspect=1,cmap='gray',vmax=3)
# # plt.colorbar()
# # plt.savefig(f'figs/datacnoise.png',dpi=300,bbox_inches="tight")
# # plt.show()
# recc = tifffile.imread(f'/data/holo/rec{imsize}_{code_position:.1f}mm_{code_size:.1f}um_{ncodes}_{use_prb}_{use_codes}.tiff')
# recc-=0.0877#np.mean(recc[0,16:32,16:32])
# recd3 = tifffile.imread(f'/data/holo/rec_angle_3dist{piter}.tiff')
# recd4 = tifffile.imread(f'/data/holo/rec_angle_4dist{piter}.tiff')
# recd3-=0.0877
# recd4-=0.0877
# datad = tifffile.imread(f'data/data_0.tiff')
# convd3 = np.load(f'res_numpy/conv3dist{piter}.npy')# in readlity 3 dists
# convd4 = np.load(f'res_numpy/conv4dist{piter}.npy')# in readlity 3 dists
    
    
    
# arr = np.arange(1,piter+1)
# fig = plt.figure(figsize=(4,4))
# plt.plot(arr,conv,'r-',label = '3 codes')
# plt.plot(arr,convd3,'b-',label = '3 distances')
# plt.plot(arr,convd4,'g-',label = '4 distances')
# plt.yscale('log')
# plt.xlabel('CG iterations')
# plt.ylabel('error')
# plt.legend()
# plt.savefig(f'figs/codes.png',dpi=300,bbox_inches="tight")
# plt.show()
# # exit()
# fig = plt.figure(figsize=(5,5))
# plt.imshow(datac[0],aspect=1,cmap='gray',vmax=3)
# plt.colorbar()
# plt.savefig(f'figs/datac.png',dpi=300,bbox_inches="tight")

# # fig = plt.figure(figsize=(5,5))
# # plt.imshow(datad[0],aspect=1,cmap='gray')#,vmax=4)
# # plt.colorbar()
# # plt.savefig(f'figs/datad.png',dpi=300,bbox_inches="tight")
vmin = -0.3
vmax = 0.1
# recd3[0,0,0] = vmin
# recd3[0,-1,-1] = vmax
# fig = plt.figure(figsize=(5,5))
# plt.imshow(recd3[0],aspect=1,cmap='gray',vmin=vmin,vmax=vmax)#,vmax=4)
# plt.colorbar()
# plt.savefig(f'figs/recd3{piter}.png',dpi=300,bbox_inches="tight")
# plt.show()

# recd4[0,0,0] = vmin
# recd4[0,-1,-1] = vmax
# fig = plt.figure(figsize=(5,5))
# plt.imshow(recd4[0],aspect=1,cmap='gray',vmin=vmin,vmax=vmax)#,vmax=4)
# plt.colorbar()
# plt.savefig(f'figs/recd4{piter}.png',dpi=300,bbox_inches="tight")
# recc[0,0,0] = vmin
# recc[0,-1,-1] = vmax
# fig = plt.figure(figsize=(5,5))
# plt.imshow(recc[0],aspect=1,cmap='gray',vmin=vmin,vmax=vmax)#,vmax=4)
# plt.colorbar()
# plt.savefig(f'figs/recc.png',dpi=300,bbox_inches="tight")

# # fig = plt.figure(figsize=(5,5))
# # plt.imshow(recc[0,1024-64:1024+64,1024-64:1024+64],aspect=1,cmap='gray')#,vmax=4)
# # plt.axis('off')
# # plt.savefig(f'figs/reccz.png',dpi=300,bbox_inches="tight")
# # fig = plt.figure(figsize=(5,5))
# # plt.imshow(recd3[0,1024-64:1024+64,1024-64:1024+64],aspect=1,cmap='gray')#,vmax=4)
# # plt.axis('off')
# # plt.savefig(f'figs/recd3z.png',dpi=300,bbox_inches="tight")

# # code_abs = tifffile.imread(f'/data/holo/code_amp.tiff')
# # code_angle = tifffile.imread(f'/data/holo/code_angle.tiff')

# # fig = plt.figure(figsize=(5,5))
# # plt.imshow(code_abs[0],aspect=1,cmap='gray')#,vmax=4)
# # plt.colorbar()
# # plt.savefig(f'figs/code_abs.png',dpi=300,bbox_inches="tight")

# # # plt.show()
# # fig = plt.figure(figsize=(5,5))
# # plt.imshow(code_angle[0],aspect=1,cmap='gray')#,vmax=4)
# # plt.colorbar()
# # plt.savefig(f'figs/code_angle.png',dpi=300,bbox_inches="tight")

# psi_angle = tifffile.imread(f'data/psi_angle.tiff')
# psi_angle[0,0,0] = vmin
# psi_angle[0,-1,-1] = vmax

# fig = plt.figure(figsize=(5,5))
# plt.imshow(psi_angle[0],aspect=1,cmap='gray',vmin=vmin,vmax=vmax)#,vmax=4)
# plt.colorbar()
# plt.savefig(f'figs/init.png',dpi=300,bbox_inches="tight")


# fig = plt.figure(figsize=(5,5))
# plt.imshow(psi_angle[0]-recc[0],aspect=1,cmap='gray')#,vmin=vmin,vmax=vmax)#,vmax=4)
# plt.colorbar()
# plt.savefig(f'figs/errc.png',dpi=300,bbox_inches="tight")

# fig = plt.figure(figsize=(5,5))
# plt.imshow(psi_angle[0]-recd3[0],aspect=1,cmap='gray')#,vmin=vmin,vmax=vmax)#,vmax=4)
# plt.colorbar()
# plt.savefig(f'figs/errd3{piter}.png',dpi=300,bbox_inches="tight")
# fig = plt.figure(figsize=(5,5))
# plt.imshow(psi_angle[0]-recd4[0],aspect=1,cmap='gray')#,vmin=vmin,vmax=vmax)#,vmax=4)
# plt.colorbar()
# plt.savefig(f'figs/errd4{piter}.png',dpi=300,bbox_inches="tight")


noise=True
recc = tifffile.imread(f'/data/holo/recnoise{noise}{imsize}_{code_position:.1f}mm_{code_size:.1f}um_{ncodes}_{use_prb}_{use_codes}.tiff')
recc-=0.0877

recd = tifffile.imread(f'/data/holo/rec_angle_noise{noise}_4dist{piter}.tiff')
recd-=0.0877

recc[0,0,0] = vmin
recc[0,-1,-1] = vmax
fig = plt.figure(figsize=(5,5))
plt.imshow(recc[0],aspect=1,cmap='gray',vmin=vmin,vmax=vmax)#,vmax=4)
plt.colorbar()
plt.savefig(f'figs/reccnoise.png',dpi=300,bbox_inches="tight")

recd[0,0,0] = vmin
recd[0,-1,-1] = vmax
fig = plt.figure(figsize=(5,5))
plt.imshow(recd[0],aspect=1,cmap='gray',vmin=vmin,vmax=vmax)#,vmax=4)
plt.colorbar()
plt.savefig(f'figs/recdnoise.png',dpi=300,bbox_inches="tight")

fig = plt.figure(figsize=(5,5))
plt.imshow(recc[0,1024-64:1024+64,1024-64:1024+64],aspect=1,cmap='gray')#,vmax=4)
plt.axis('off')
plt.savefig(f'figs/reccznoise.png',dpi=300,bbox_inches="tight")
fig = plt.figure(figsize=(5,5))
plt.imshow(recd[0,1024-64:1024+64,1024-64:1024+64],aspect=1,cmap='gray')#,vmax=4)
plt.axis('off')
plt.savefig(f'figs/recdnoisez.png',dpi=300,bbox_inches="tight")
