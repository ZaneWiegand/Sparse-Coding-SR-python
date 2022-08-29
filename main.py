# %%
import pickle
from sklearn.preprocessing import normalize
# %%
class args(object):
    lr_dir = 'data/set5_lr'
    sr_dir = 'result/set5_sr'
    hr_dir = 'data/set5_hr'
    
    # choose a dictionary for SR
    upscale_factor = 2
    lambda_factor = 0.1
    dictionary_size = 1024
    patch_size = 3
    
    val_flag = False
    
para = args()
# %%
dict_name = str(para.dictionary_size) + '_US' + str(para.upscale_factor) + '_L' + str(para.lambda_factor) + '_PS' + str(para.patch_size)

with open('dictionary/Dh_' + dict_name + '.pkl', 'rb') as f:
    Dh = pickle.load(f)
Dh = normalize(Dh)
with open('dictionary/Dl_' + dict_name + '.pkl', 'rb') as f:
    Dl = pickle.load(f)
Dl = normalize(Dl)
# %%
