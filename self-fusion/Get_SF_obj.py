# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:09:05 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import pickle, random, os
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
dataroot = 'E:\\OCTA\\data\\'

#%%
for file in os.listdir(dataroot):
    if file.endswith('svd.nii'):
        
        print('volume: {}'.format(file))
        vol = util.nii_loader(dataroot+file)
        vol = np.transpose(vol,(2,1,0))
        
        H,slc,W = vol.shape
        sf_proj = ()
        show = random.randint(radius,slc-radius)
        
        for i in range(radius,slc-radius):
            stack = vol[:,i-radius:i+radius+1,:]
            sf_proj = sf_proj+(stack,)
            
#            if i == show:
#                im_grand = np.concatenate((stack[:,0,:],stack[:,-1,:]),axis=1)
#                plt.figure(figsize=(12,8))
#                plt.imshow(im_grand,cmap='gray')
#                plt.axis('off')
#                plt.show()
        
        with open(dataroot+'SF({}).pickle'.format(file[:-4]),'wb') as func:
            pickle.dump(sf_proj,func)

