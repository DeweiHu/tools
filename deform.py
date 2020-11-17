# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 01:00:17 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import pickle, random, os, time
import numpy as np
import matplotlib.pyplot as plt
import pyelastix

dataroot = 'E:\\OCTA\\data\\'
vol = util.nii_loader(dataroot+'fovea_svd.nii')
vol = np.transpose(vol,(2,1,0))
r = 3

def deform(im_fix,im_mov):

    im_fix = np.ascontiguousarray(im_fix)
    im_mov = np.ascontiguousarray(im_mov)
    
    # Get params and change a few values
    params = pyelastix.get_default_params(type='BSPLINE')
    params.NumberOfResolutions = 4
    params.MaximumNumberOfIterations = 500
    params.FinalGridSpacingInVoxels = 10
    
    # Apply the registration (im1 and im2 can be 2D or 3D)
    im_deformed, field = pyelastix.register(im_mov, im_fix, params, verbose=0)

    return im_deformed, field

#%%
H, slc, W = vol.shape
sf_proj = ()

t1 = time.time()
for i in range(r,slc-r):
    opt = np.zeros([H,2*r+1,W],dtype=np.float32)
    stack = vol[:,i-r:i+r+1,:]
    im_fix = stack[:,r,:]
    for j in range(2*r+1):
        im_mov = stack[:,j,:]
        opt[:,j,:],_ = deform(im_fix,im_mov)
    sf_proj = sf_proj + (opt,)
t2 = time.time()

print('time consumed: {}'.format((t2-t1)/60))

with open(dataroot+'fovea_reg.pickle','wb') as func:
    pickle.dump(sf_proj,func)