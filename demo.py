# %%
import numpy as np
from skimage import io
from skimage.transform import resize
from utils import noise_quality_measure
# %%
HR = io.imread('./data/set5_hr/baby.png',as_gray=True)
# %%
LR = io.imread('./data/set5_lr/baby.png',as_gray=True)
# %%
