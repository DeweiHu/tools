# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:09:05 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import pickle, random
import numpy as np
import matplotlib.pyplot as plt

'''
Re_Arrange reshape the volume from [nFrame*nBscan,H,W] -> [nFrame,nBscan,H,W]
'''
def Re_Arrange(volume):
    global nFrame
    n,H,W = volume.shape
    opt = np.zeros([nFrame,int(n/nFrame),H,W],dtype=np.float32)
    for i in range(n):
        idx = i % nFrame
        opt[idx,int(i/nFrame),:,:] = volume[i,:,:]
    return opt

global nFrame, radius
nFrame = 5
radius = 3

vol = util.nii_loader('E:\\Retina2 Fovea_SNR 101_reg.nii')
vol = Re_Arrange(np.transpose(vol,(2,0,1)))
vol_var = np.var(vol,axis=0)
vol = vol[0,:,:,:]

vol_proj = vol_var
H,slc,W = vol_proj.shape
sf_proj = ()
show = random.randint(radius,slc-radius)

#%%
for i in range(radius,slc-radius):
#    stack = np.transpose(vol_proj[:,i-radius:i+radius+1,:],(1,0,2))
    stack = vol_proj[i-radius:i+radius+1,:,:]
    sf_proj = sf_proj+(stack,)
    
    if i == show:
        im_grand = np.concatenate((stack[0,:,:],stack[-1,:,:]),axis=1)
        plt.figure(figsize=(12,8))
        plt.imshow(im_grand,cmap='gray')
        plt.axis('off')
        plt.show()

with open('E:\\sf_var.pickle','wb') as func:
    pickle.dump(sf_proj,func)

