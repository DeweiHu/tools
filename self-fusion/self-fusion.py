#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:15:13 2020

@author: dewei
"""

# -*- coding: utf-8 -*-
"""
The self fusion function takes a pickle input with a tuple
(stack1,stack2,...), each stack is a self-fusion target
stack: [slc,H,W]
The middle slice in stack is the fixed image
 
"""


import sys
sys.path.insert(0,'/home/dewei/Desktop/OCTA/')
import util
import os, pickle, random, subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

dataroot = '/home/dewei/Desktop/OCTA/data/'

temp = '/home/dewei/Desktop/slc/'
radius = 3

for file in os.listdir(dataroot):
    
    if file.endswith('.pickle') and file.startswith('SF'):
        print('volume:{}'.format(file))
        
        with open(dataroot+file,'rb') as func:
            data = pickle.load(func)
        
        if not os.path.exists(dataroot+file[3:-8]):
            os.makedirs(dataroot+file[3:-8])
        saveroot = dataroot+file[3:-8]    
        
        depth = len(data)
        H,d,W = data[0].shape
        opt = np.zeros([H,depth,W],dtype=np.float32)
        org = np.zeros([H,depth,W],dtype=np.float32)
        
        if not d == 2*radius+1:
            raise ValueError('radius not matching')
        
        for i in range(depth):
            # clean up the temp directory for the next input
            for weightmap in os.listdir(temp):
                os.remove(temp+weightmap)
                
            stack = data[i]
            im_fix = Image.fromarray(stack[:,radius,:])
            im_fix.save(temp + 'fix_img.tif')
            org[:,i,:] = stack[:,radius,:]
            
            # create atlases
            for j in range(2*radius+1):
                im_mov = Image.fromarray(stack[:,j,:])
                im_mov.save(temp + 'atlas{}.tif'.format(j))
            
            # call the self-fusion function & take the result
            subprocess.call("/home/dewei/self_fusion.sh")
            opt[:,i,:] = io.imread(temp+'synthResult.tif')
        
        util.nii_saver(np.transpose(opt,(2,1,0)),saveroot,'{}.nii.gz'.format(file[:-8]))
        util.nii_saver(np.transpose(org,(2,1,0)),saveroot,'{}.nii.gz'.format(file[3:-8]))