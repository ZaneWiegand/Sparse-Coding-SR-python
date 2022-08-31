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
img_bc = resize(LR,(HR.shape[0],HR.shape[1]))
# %%
noise_quality_measure(HR,img_bc)
# %%
io.imshow(img_bc)
# %%
io.imshow(HR)
# %%
h,w = LR.shape
# %%
h
# %%
w
# %%
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
# %%
B = np.array([1,2,3])
# %%
A.shape
# %%
B.shape
# %%
np.dot(A, B)+B.T
# %%
B.squeeze()
# %%
list(np.multiply(2,B.shape))
# %%
B.shape
# %%
from skimage.transform import resize
# %%
io.imshow(resize(LR,np.multiply(2,LR.shape)))
# %%
io.imshow(LR)
# %%
A = np.array([[1,2,3,4,5],[6,7,8,9,10]])
# %%
from sklearn.preprocessing import normalize
# %%
normalize(A,axis=0)
# %%
A = np.array([[1,2,3,4,5],[6,7,8,9,10]])
# %%
A
# %%
B = np.array([[1,1,1,2,1],[1,2,1,1,1]])
# %%
B
# %%
A/B
# %%
A = np.array([[1, 0, -2, 0, 1], ] * 5)
# %%
A
# %%
A.T
# %%
