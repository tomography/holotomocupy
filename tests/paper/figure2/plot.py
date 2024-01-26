import numpy as np
import matplotlib.pyplot as plt

imsize = 2048
ncodes = 3
use_prb = 'True'
use_codes = 'True'
code_size = 1
r = 0
arr=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
psnr = np.zeros(len(arr))
psnr[:]=240
ssim = np.zeros(len(arr))
for k in range(r,len(arr)):    
    code_position = arr[k]
    psnr[k] = np.load(f'res_numpy/pnsr{imsize}_{code_position:.1f}mm_{code_size:.1f}um_{ncodes}_prb{use_prb}_code{use_codes}.npy')
    ssim[k] = np.load(f'res_numpy/ssim{imsize}_{code_position:.1f}mm_{code_size:.1f}um_{ncodes}_prb{use_prb}_code{use_codes}.npy')
    
# plt.figure(figsize=(16,5))
fig, ax = plt.subplots(1,2,figsize=(24,5))
fig.suptitle(f'{imsize}_{code_size:.1f}um_{ncodes}_prb{use_prb}_code{use_codes}')
ax[0].set_title('PSNR')
ax[0].plot(arr,psnr,'-')
ax[0].plot(arr[:r],psnr[:r],'rx')
ax[0].set_xticks(arr)
ax[0].set_xticklabels(arr, fontsize=9,rotation=45)  
ax[1].plot(arr,ssim,'-')
ax[1].plot(arr[:r],ssim[:r],'rx')
ax[1].set_xticks(arr)
ax[1].set_xticklabels(arr, fontsize=9,rotation=45)  
ax[1].set_title('SSIM')
# ax[1].set_ylim([0.7, 1.05])
plt.savefig(f'figs/{imsize}_{code_size:.1f}um_{ncodes}_prb{use_prb}_code{use_codes}.png',dpi=300)
plt.show()