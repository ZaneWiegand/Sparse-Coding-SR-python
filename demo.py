# %%
import numpy as np
from skimage import io
from skimage.transform import resize
from utils import noise_quality_measure
# %%
HR = io.imread('./data/set5_hr/baby.png', as_gray=True)
# %%
LR = io.imread('./data/set5_lr/baby.png', as_gray=True)
# %%
A = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
# %%
ma = np.max(A)
# %%
mi = np.argmax(A)
# %%
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# %%
A
# %%
x = np.array([1, 0, 1])
# %%
a = np.where(x != 0)[0]
a
# %%
A
# %%
a
# %%
a.shape
# %%
len(a)
# %%
np.repeat(a, len(a))
# %%
np.tile(a, len(a))
# %%
A[np.repeat(a, len(a)), np.tile(a, len(a))].reshape(len(a), len(a))
# %%
