# %%
import os
import numpy as np
from skimage import io
from skimage.color import rgb2ycbcr,ycbcr2rgb,rgb2gray
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from utils import noise_quality_measure

from utils import scsr,backprojection
import pickle
# %%
# choose parameters
class args(object):
    lr_dir = 'data/set5_lr'
    sr_dir = 'result/set5_sr'
    hr_dir = 'data/set5_hr'
    
    # choose a dictionary for SR
    dic_upscale_factor = 2
    dic_lambda = 0.1
    dic_size = 1024
    dic_patch_size = 3
    
    # sparse SR factor
    lambda_factor = 0.3
    overlap = 2
    upscale_factor = 2
    max_iteration = 1000
    nu = 1
    beta = 1 # also c/gamma in paper
    color_space = 'ycbcr' # 'bw'
    
    # True for validation, False for prediction
    val_flag = True
    
para = args()
# %%
# load dictionary
dict_name = str(para.dic_size) + '_US' + str(para.dic_upscale_factor) + '_L' + str(para.dic_lambda) + '_PS' + str(para.dic_patch_size)

with open('dictionary/Dh_' + dict_name + '.pkl', 'rb') as f:
    Dh = pickle.load(f)
with open('dictionary/Dl_' + dict_name + '.pkl', 'rb') as f:
    Dl = pickle.load(f)
# %%
# super resolution img dir
if not os.path.exists(para.sr_dir):
    os.makedirs(para.sr_dir)
# %%
img_lr_file = os.listdir(para.lr_dir)
# %%
for i in range(len(img_lr_file)):
    img_name = img_lr_file[i]
    img_lr = io.imread(f"{para.lr_dir}/{img_name}")
    img_hr = io.imread(f"{para.hr_dir}/{img_name}")
    
    if para.color_space == 'ycbcr':
        img_hr_y = rgb2ycbcr(img_hr)[:,:,0]
        
        # change color space
        img_lr_ycbcr = rgb2ycbcr(img_lr)
        img_lr_y = img_lr_ycbcr[:,:,0]
        img_lr_cb = img_lr_ycbcr[:,:,1]
        img_lr_cr = img_lr_ycbcr[:,:,2]
        
        # upscale chrominance to color SR images
        # nearest neighbor interpolation
        img_sr_cb = resize(img_lr_cb, (img_hr.shape[0], img_hr.shape[1]), order=0)
        img_sr_cr = resize(img_lr_cr, (img_hr.shape[0], img_hr.shape[1]), order=0)
        
    elif para.color_space == 'bw':
        img_hr_y = rgb2gray(img_hr)
        img_lr_y = rgb2gray(img_lr)
    
    else:
        raise ValueError("Invalid color space!")
        
    # super resolution via sparse representation
    # TODO ScSR, backprojection
    img_sr_y = scsr(img_lr_y, para.upscale_factor, Dh, Dl, para.lambda_factor, para.overlap)
    img_sr_y = backprojection(img_sr_y, img_lr_y, para.max_iteration, para.nu, para.beta)
    
    # reconstructed color images
    if para.color_space == 'ycbcr':
        img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
        img_sr = ycbcr2rgb(img_sr)
        
    elif para.color_space == 'bw':
        img_sr = img_sr_y
        
    else:
        raise ValueError("Invalid color space!")

    # signal normalization
    for channel in range(img_sr.shape[2]):
        ratio = np.mean(img_lr[:, :, channel]) / (np.mean(img_sr[:, :, channel]) * 255)
        img_sr[:, :, channel] = img_sr[:, :, channel] * ratio

    # pixel intensity normalization
    img_sr = img_sr.clip(0,1)*255
    img_sr = img_sr.astype(np.uint8)
    
    # bicubic interpolation for reference
    img_bc = resize(img_lr,(img_hr.shape[0],img_hr.shape[1])).clip(0,1)*255
    img_bc = img_bc.astype(np.uint8)
    
    img_bc_y = rgb2ycbcr(img_bc)[:, :, 0]
    
    # calculate PSNR, SSIM and MSE for the luminance
    # for bicubic interpolation
    psnr_bc_hr = peak_signal_noise_ratio(img_hr_y,img_bc_y)
    ssim_bc_hr = structural_similarity(img_hr_y,img_bc_y)
    mse_bc_hr = mean_squared_error(img_hr_y,img_bc_y)
    nqm_bc_hr = noise_quality_measure(img_hr_y,img_bc_y)
    # for sparse representation SR
    psnr_sr_hr = peak_signal_noise_ratio(img_hr_y,img_sr_y)
    ssim_sr_hr = structural_similarity(img_hr_y,img_sr_y)
    mse_sr_hr = mean_squared_error(img_hr_y,img_sr_y)
    nqm_sr_hr = noise_quality_measure(img_hr_y,img_sr_y)
    
    