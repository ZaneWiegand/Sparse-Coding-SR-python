# %%
import os
import numpy as np
from skimage import io
from skimage.color import rgb2ycbcr,ycbcr2rgb,rgb2gray
import pickle
from sklearn.preprocessing import normalize
from tqdm import tqdm
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
    
    # Sparse SR factor
    lambda_factor = 0.3
    overlap = 1
    upscale_factor = 2
    max_iteration = 1000
    color_space = 'ycbcr' # 'bw'
    
    # True for validation, False for prediction
    val_flag = True
    
para = args()
# %%
# load dictionary
dict_name = str(para.dic_size) + '_US' + str(para.dic_upscale_factor) + '_L' + str(para.dic_lambda) + '_PS' + str(para.dic_patch_size)

with open('dictionary/Dh_' + dict_name + '.pkl', 'rb') as f:
    Dh = pickle.load(f)
Dh = normalize(Dh)
with open('dictionary/Dl_' + dict_name + '.pkl', 'rb') as f:
    Dl = pickle.load(f)
Dl = normalize(Dl)
# %%
# super resolution img dir
if not os.path.exists(para.sr_dir):
    os.makedirs(para.sr_dir)
# %%
img_lr_file = os.listdir(para.lr_dir)
# %%
for i in tqdm(range(len(img_lr_file))):
    img_name = img_lr_file[i]
    img_lr = io.imread(f"{para.lr_dir}/{img_name}")
    img_hr = io.imread(f"{para.hr_dir}/{img_name}")
    
    if para.color_space == 'ycbcr':
        img_hr_y = rgb2ycbcr(img_hr)[:,:,0]
        
        # change color space
        
# %%
