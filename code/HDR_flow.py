''' HDR flow '''
import matplotlib.pyplot as plt
import cv2
import numpy as np
from functools import partial

from HDR_functions import CameraResponseCalibration, WhiteBalance, \
                          GlobalTM, LocalTM, histogram_equalization, GaussianFilter, BilateralFilter, \
                          SaveImg


##### Test image: memorial #####
TestImage = 'memorial'
#TestImage = 'taipei'
#TestImage = 'living_room2'
print(f'---------- Test Image is {TestImage} ----------')
### Whole HDR flow ### 
print('Start to process HDR flow...')
# Camera response calibration
radiance = CameraResponseCalibration(f'../TestImage/{TestImage}', lambda_=50)
print('--Camera response calibration done')
# Display Radiance map with pseudo-color image (log value)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i in range(3):
    axs[i].imshow(radiance[:, :, i], cmap='jet')
    axs[i].set_title(f'Channel {i}')
    axs[i].set_axis_off()
    fig.colorbar(axs[i].imshow(np.log2(radiance[:, :, i]), cmap='jet'), ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
print('done')

# White balance
ktbw = (419, 443), (389, 401)
radiance_wb = WhiteBalance(radiance, *ktbw)
print('--White balance done')
print('--Tone mapping')
# Global tone mapping
gtm_no_wb = GlobalTM(radiance, scale=1)  # without white balance
gtm = GlobalTM(radiance_wb, scale=1)     # with white balance
Reinhard = histogram_equalization(gtm)
print('    Global tone mapping done')
# Local tone mapping with gaussian filter
ltm_filter = partial(GaussianFilter, N=15, sigma_s=100)
ltm_gaussian = LocalTM(radiance_wb, ltm_filter, scale=7)
print('    Local tone mapping with gaussian filter done')
# Local tone mapping with bilateral filter
ltm_filter = partial(BilateralFilter, N=15, sigma_s=100, sigma_r=0.8)
ltm_bilateral = LocalTM(radiance_wb, ltm_filter, scale=7)
print('    Local tone mapping with bilateral filter done')
print('Whole process done\n')

### Save result ###
print('Saving results...')
SaveImg(gtm_no_wb, f'../Result/{TestImage}_gtm_no_wb.png')
SaveImg(gtm, f'../Result/{TestImage}_gtm.png')
SaveImg(ltm_gaussian, f'../Result/{TestImage}_ltm_gau.png')
SaveImg(ltm_bilateral, f'../Result/{TestImage}_ltm_bil.png')
SaveImg(Reinhard, f'../Result/{TestImage}_ltm_rei.png')
print('All results are saved\n')
