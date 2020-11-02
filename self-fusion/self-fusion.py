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
filename = 'sf_proj.pickle'
temp = '/home/dewei/Desktop/slc/'
radius = 2

with open(dataroot+filename,'rb') as func:
    data = pickle.load(func)

depth = len(data)
d,H,W = data[0].shape
opt = np.zeros([depth,H,W],dtype=np.float32)
show = random.randint(0,depth-1)

if not d == 2*radius+1:
    raise ValueError('radius not matching')

for i in range(depth):
    # clean up the temp directory for the next input
    for file in os.listdir(temp):
        os.remove(temp+file)
        
    stack = data[i]
    im_fix = Image.fromarray(stack[radius,:,:])
    im_fix.save(temp + 'fix_img.tif')
    
    # create atlases
    for j in range(2*radius+1):
        im_mov = Image.fromarray(stack[j,:,:])
        im_mov.save(temp + 'atlas{}.tif'.format(j))
    
    # call the self-fusion function & take the result
    subprocess.call("/home/dewei/self_fusion.sh")
    opt[i,:,:] = io.imread(temp+'synthResult.tif')
    
    # display an example
    if i == show:
        plt.figure(figsize=(12,6))
        plt.axis('off')
        plt.imshow(np.concatenate((stack[radius,:,:],opt[i,:,:]),axis=1),cmap='gray')
        plt.show()

util.nii_saver(opt,dataroot,'SF(proj_shadow).nii.gz')
    
        
    