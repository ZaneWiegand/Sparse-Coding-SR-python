# %%
from os import listdir
from copy import deepcopy
import numpy as np
from numpy.fft import fftshift, fft2, ifft2
from sklearn.preprocessing import normalize
from scipy.signal import convolve2d
from tqdm import tqdm
from skimage.transform import resize,rescale
from skimage.color import rgb2gray
import skimage.io as io
# %%
def noise_quality_measure(hr, sr, VA=np.pi/3):
    # metrics: noise quality measure 

    def ctf(f_r):
        y = 1/(200*2.6*(0.0192+0.114*f_r)*np.exp(-(0.114*f_r)**1.1))
        return y

    def cmaskn(c, ci, a, ai, i):
        cx = deepcopy(c)
        cix = deepcopy(ci)
        cix[np.abs(cix) > 1] = 1
        ct = ctf(i)
        T = ct*(.86*((cx/ct)-1)+.3)
        ai[(abs(cix-cx)-T) < 0] = a[(abs(cix-cx)-T) < 0]
        return ai

    def gthresh(x, T, z):
        z[np.abs(x) < T] = 0
        return z

    row, col = sr.shape
    X = np.linspace(-row/2+0.5, row/2-0.5, row)
    Y = np.linspace(-col/2+0.5, col/2-0.5, col)
    x, y = np.meshgrid(X, Y)
    plane = (x+1j*y)
    r = np.abs(plane)

    pi = np.pi
    
    G_0 = 0.5*(1+np.cos(pi*np.log2((r+2)*(r+2 >= 1) *(r+2 <= 4)+4*(~((r+2 <= 4)*(r+2 >= 1))))-pi))

    G_1 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 1)*(r <= 4))+4*(~((r >= 1)*(r <= 4))))-pi))

    G_2 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 2)*(r <= 8))+.5*(~((r >= 2) * (r <= 8))))))

    G_3 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 4)*(r <= 16))+4*(~((r >= 4)*(r <= 16))))-pi))

    G_4 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 8)*(r <= 32))+.5*(~((r >= 8) * (r <= 32))))))

    G_5 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 16)*(r <= 64))+4*(~((r >= 16)*(r <= 64))))-pi))
    
    GS_0 = fftshift(G_0)
    GS_1 = fftshift(G_1)
    GS_2 = fftshift(G_2)
    GS_3 = fftshift(G_3)
    GS_4 = fftshift(G_4)
    GS_5 = fftshift(G_5)

    FO = fft2(sr).T
    FI = fft2(hr).T

    L_0 = GS_0*FO
    LI_0 = GS_0*FI

    l_0 = np.real(ifft2(L_0))
    li_0 = np.real(ifft2(LI_0))

    A_1 = GS_1*FO
    AI_1 = GS_1*FI

    a_1 = np.real(ifft2(A_1))
    ai_1 = np.real(ifft2(AI_1))

    A_2 = GS_2*FO
    AI_2 = GS_2*FI

    a_2 = np.real(ifft2(A_2))
    ai_2 = np.real(ifft2(AI_2))

    A_3 = GS_3*FO
    AI_3 = GS_3*FI

    a_3 = np.real(ifft2(A_3))
    ai_3 = np.real(ifft2(AI_3))

    A_4 = GS_4*FO
    AI_4 = GS_4*FI

    a_4 = np.real(ifft2(A_4))
    ai_4 = np.real(ifft2(AI_4))

    A_5 = GS_5*FO
    AI_5 = GS_5*FI

    a_5 = np.real(ifft2(A_5))
    ai_5 = np.real(ifft2(AI_5))

    c1 = a_1/l_0
    c2 = a_2/(l_0+a_1)
    c3 = a_3/(l_0+a_1+a_2)
    c4 = a_4/(l_0+a_1+a_2+a_3)
    c5 = a_5/(l_0+a_1+a_2+a_3+a_4)

    ci1 = ai_1/li_0
    ci2 = ai_2/(li_0+ai_1)
    ci3 = ai_3/(li_0+ai_1+ai_2)
    ci4 = ai_4/(li_0+ai_1+ai_2+ai_3)
    ci5 = ai_5/(li_0+ai_1+ai_2+ai_3+ai_4)

    d1 = ctf(2/VA)
    d2 = ctf(4/VA)
    d3 = ctf(8/VA)
    d4 = ctf(16/VA)
    d5 = ctf(32/VA)

    ai_1 = cmaskn(c1, ci1, a_1, ai_1, 1)
    ai_2 = cmaskn(c2, ci2, a_2, ai_2, 2)
    ai_3 = cmaskn(c3, ci3, a_3, ai_3, 3)
    ai_4 = cmaskn(c4, ci4, a_4, ai_4, 4)
    ai_5 = cmaskn(c5, ci5, a_5, ai_5, 5)

    l0 = l_0
    li0 = li_0
    a1 = gthresh(c1, d1, a_1)
    ai1 = gthresh(ci1, d1, ai_1)
    a2 = gthresh(c2, d2, a_2)
    ai2 = gthresh(ci2, d2, ai_2)
    a3 = gthresh(c3, d3, a_3)
    ai3 = gthresh(ci3, d3, ai_3)
    a4 = gthresh(c4, d4, a_4)
    ai4 = gthresh(ci4, d4, ai_4)
    a5 = gthresh(c5, d5, a_5)
    ai5 = gthresh(ci5, d5, ai_5)

    Os = l0+a1+a2+a3+a4+a5
    Is = li0+ai1+ai2+ai3+ai4+ai5

    A = np.sum(Os**2)
    square_err = (Os-Is)*(Os-Is)
    B = np.sum(square_err)
    nqm_value = 10*np.log10(A/B)
    return nqm_value

# %%
def extract_feature(img):
    row, col = img.shape
    img_feature = np.zeros([row, col, 4])
    
    # first order gradient filters
    hf1 = np.array([[-1, 0, 1]])
    vf1 = hf1.T
    img_feature[:,:,0] = convolve2d(img, hf1, 'same')
    img_feature[:,:,1] = convolve2d(img, vf1, 'same')
    
    hf2 = np.array([[1, 0, -2, 0, 1]])
    vf2 = hf2.T
    img_feature[:,:,2] = convolve2d(img, hf2, 'same')
    img_feature[:,:,3] = convolve2d(img, vf2, 'same')
    
    return img_feature

def lin_scale(h_img, l_norm):
    h_norm = np.sqrt(np.sum(h_img*h_img))
    if h_norm>0:
        s = 1.2*l_norm/h_norm #? s = 1.2*l_norm/h_norm
        h_img = h_img*s
    return h_img

# todo
def sparse_solution(lmbd, A, b, maxiter):
    
    eps = 1e-9
    x = np.zeros((A.shape[0], 1))
    
    grad = np.dot(A,x)+b
    ma = np.max(np.abs(grad))
    mi = np.argmax(np.abs(grad))
    cnt_1 = 0
    cnt_2 = 0
    while True:
        if grad[mi]>lmbd+eps:
            x[mi] = (lmbd-grad[mi])/A[mi,mi]
        elif grad[mi]<-lmbd-eps:
            x[mi] = (-lmbd-grad[mi])/A[mi,mi]
        else:
            if np.all(x == 0):
                break
        while True:
            a = np.where(x != 0)[0] # active set
            Aa = A[np.repeat(a,len(a)),np.tile(a,len(a))].reshape(len(a),len(a))
            ba = b[a]
            xa = x[a]
            
            # new b based on unchanged sign
            vect = -lmbd*np.sign(xa)-ba
            if Aa.shape[0]==1:
                x_new = vect/Aa
            else:
                x_new = np.dot(np.linalg.inv(Aa),vect)
            idx = np.where(x_new != 0)[0]
            o_new = np.dot((vect[idx] / 2 + ba[idx]).T, x_new[idx]) + lmbd * np.sum(np.abs(x_new[idx]))
            
            # cost based on changing sign
            s = np.where(xa*x_new <= 0)[0]
            
            cnt_2 += 1
            if np.all(s == 0) or cnt_2>maxiter:
                x[a] = x_new
                cnt_2 = 0
                break
            x_min = x_new
            o_min = o_new
            d = x_new - xa
            
            t = d/xa
            for zd in s.T:
                x_s = xa - d / t[zd]
                x_s[zd] = 0
                idx = np.where(x_s != 0)[0]
                o_s = np.dot((np.dot(Aa[idx, idx], x_s[idx]) / 2 + ba[idx]).T, x_s[idx]) + lmbd * np.sum(np.abs(x_s[idx]))
                if o_s < o_min:
                    x_min = x_s
                    o_min = o_s
            
            x[a] = x_min
            
        grad = np.dot(A, x) + b
            
        temp = np.abs(grad)*(x == 0)
        ma = np.max(np.abs(temp))
        mi = np.argmax(np.abs(temp))
        
        cnt_1 += 1
        if ma<=lmbd+eps or cnt_1>maxiter:
            break
    return x

def scsr(img_lr_y, upscale_factor, Dh, Dl, lmbd, overlap, maxiter):
    # sparse coding super resolution
    
    # normalize the dictionary
    Dl = normalize(Dl,axis=0) #? normalize?  
    patch_size = int(np.sqrt(Dh.shape[0]))
       
    # bicubic interpolation of the lr image
    img_lr_y_upscale = resize(img_lr_y, np.multiply(upscale_factor, img_lr_y.shape), 3, preserve_range = True)
    
    img_sr_y_height,img_sr_y_width = img_lr_y_upscale.shape
    img_sr_y = np.zeros(img_lr_y_upscale.shape)
    cnt_matrix = np.zeros(img_lr_y_upscale.shape)
    
    # extract lr image features
    img_lr_y_feature = extract_feature(img_lr_y_upscale)
    
    # patch indexes for sparse recovery
    # drop 2 pixels at the boundary
    gridx = np.arange(3,img_sr_y_width-patch_size-2,patch_size-overlap)
    gridx = np.append(gridx,img_sr_y_width-patch_size-2)
    gridy = np.arange(3,img_sr_y_height-patch_size-2,patch_size-overlap)
    gridy = np.append(gridy,img_sr_y_height-patch_size-2)
    
    A = np.dot(Dl.T,Dl)

    # loop to recover each low-resolution patch
    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            xx = int(gridx[m])
            yy = int(gridy[n])
            
            patch = img_lr_y_upscale[yy:yy+patch_size, xx:xx+patch_size]
            patch_mean = np.mean(patch)
            patch = np.ravel(patch,'F') - patch_mean
            patch_norm = np.sqrt(np.sum(patch*patch))
            
            feature = img_lr_y_feature[yy:yy+patch_size, xx:xx+patch_size, :]
            feature = np.ravel(feature,'F')
            feature_norm = np.sqrt(np.sum(feature*feature))
            
            if feature_norm>1:
                feature = feature/feature_norm
            
            y = feature
                
            b = np.zeros([1,Dl.shape[1]])-np.dot(Dl.T,y)
            b = b.T

            # sparse recovery
            w = sparse_solution(lmbd, A, b, maxiter)
            
            # generate hr patch and scale the contrast
            h_patch = np.dot(Dh,w)
            
            # h_patch = np.zeros(patch.shape)
            h_patch = lin_scale(h_patch, patch_norm)
            
            h_patch = np.reshape(h_patch,[patch_size,patch_size])
            h_patch = h_patch+patch_mean
            
            img_sr_y[yy:yy+patch_size, xx:xx+patch_size] += h_patch
            cnt_matrix[yy:yy+patch_size, xx:xx+patch_size] += 1

    idx = np.where(cnt_matrix < 1)[0]
    img_sr_y[idx] = img_lr_y_upscale[idx]

    cnt_matrix[idx] = 1
    img_sr_y = img_sr_y/cnt_matrix
    
    return img_sr_y

# %%
def gauss2D(size,sigma):
    x, y = np.mgrid[-size/2 + 0.5:size/2 + 0.5, -size/2 + 0.5:size/2 + 0.5]
    z = np.exp(-(x*x+y*y)/(2*sigma**2))/(2*np.pi*sigma)
    z = z/np.sum(z)
    return z

def backprojection(sr, lr, iters, nu, c):
    p = gauss2D(5,1)
    p = p*p
    p = p/np.sum(p)
    
    sr = sr.astype(np.float64)
    lr = lr.astype(np.float64)
    sr_0 = sr
    
    for i in range(iters):
        #sr_blur = convolve2d(sr, p, 'same')
        sr_downscale = resize(sr, lr.shape)
        diff = lr - sr_downscale

        diff_upscale = resize(diff, sr_0.shape)
        diff_blur = convolve2d(diff_upscale, p, 'same')
        
        sr = sr + nu*(diff_blur + c*(sr_0-sr))
        
    return sr

# %%
def random_sample_patch(img_path, patch_size, num_patch, upscale_factor):
    img_dir = listdir(img_path)

    img_num = len(img_dir)
    nper_img = np.zeros((img_num, 1))

    for i in tqdm(range(img_num)):
        img = io.imread('{}{}'.format(img_path, img_dir[i]))
        nper_img[i] = img.shape[0] * img.shape[1]
    
    nper_img = np.floor(nper_img * num_patch/np.sum(nper_img))
    
    for i in tqdm(range(img_num)):
        patch_num = int(nper_img[i])
        img = io.imread('{}{}'.format(img_path, img_dir[i]))
        H, L = sample_patches(img, patch_size, patch_num, upscale_factor)
        if i == 0:
            Xh = H
            Xl = L
        else:
            Xh = np.concatenate((Xh, H), axis=1)
            Xl = np.concatenate((Xl, L), axis=1)
    return Xh, Xl

def sample_patches(img, patch_size, patch_num, upscale_factor):
    if len(img.shape) == 3:
        hIm = (rgb2gray(img)*255).astype(np.uint8)
    else:
        hIm = img

    # Generate low resolution patches
    lIm = rescale(hIm, 1/upscale_factor, 3, preserve_range = True)
    lIm = resize(lIm, hIm.shape, 3, preserve_range = True)
    # lIm = lIm.astype(np.uint8)
    nrow, ncol = hIm.shape
    
    x = np.random.permutation(range(nrow - 2*patch_size)) + patch_size
    y = np.random.permutation(range(ncol - 2*patch_size)) + patch_size
    
    X,Y = np.meshgrid(x,y)
    xrow = np.ravel(X,'F')
    ycol = np.ravel(Y,'F')
    
    if patch_num<len(xrow):
        xrow = xrow[0:patch_num]
        ycol = ycol[0:patch_num]
        
    patch_num = len(xrow)
    HP = np.zeros([patch_size**2, patch_num])
    LP = np.zeros([4*patch_size**2, patch_num])
    
    hf1 = np.array([[-1, 0, 1]])
    vf1 = hf1.T
    lImG11 = convolve2d(lIm, hf1, 'same')
    lImG12 = convolve2d(lIm, vf1, 'same')
    
    hf2 = np.array([[1, 0, -2, 0, 1]])
    vf2 = hf2.T
    lImG21 = convolve2d(lIm, hf2, 'same')
    lImG22 = convolve2d(lIm, vf2, 'same')
    
    for i in tqdm(range(patch_num)):
        row = xrow[i]
        col = ycol[i]
        
        Hpatch = np.ravel(hIm[row : row + patch_size, col : col + patch_size],'F')
        
        Lpatch1 = np.ravel(lImG11[row : row + patch_size, col : col + patch_size],'F')
        Lpatch1 = np.reshape(Lpatch1, (Lpatch1.shape[0], 1))
        Lpatch2 = np.ravel(lImG12[row : row + patch_size, col : col + patch_size],'F')
        Lpatch2 = np.reshape(Lpatch2, (Lpatch2.shape[0], 1))
        Lpatch3 = np.ravel(lImG21[row : row + patch_size, col : col + patch_size],'F')
        Lpatch3 = np.reshape(Lpatch3, (Lpatch3.shape[0], 1))
        Lpatch4 = np.ravel(lImG22[row : row + patch_size, col : col + patch_size],'F')
        Lpatch4 = np.reshape(Lpatch4, (Lpatch4.shape[0], 1))
        
        Lpatch = np.concatenate((Lpatch1,Lpatch2,Lpatch3,Lpatch4), axis = 1)
        Lpatch = np.ravel(Lpatch,'F')
        
        HP[:,i] = Hpatch - np.mean(Hpatch)
        LP[:,i] = Lpatch
        
    return HP, LP
# %%
def patch_pruning(Xh, Xl, per):
    pvars = np.var(Xh, axis=0)
    threshold = np.percentile(pvars, per)
    idx = pvars > threshold
    # print(pvars)
    Xh = Xh[:, idx]
    Xl = Xl[:, idx]
    return Xh, Xl
