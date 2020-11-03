# -*- coding: utf-8 -*-
"""

Use: the pre-processed data in shape [nFrame,nBscan,H,W]
Do : 1. take the first frame 
     2. stack slices within radius 
     3. rigid registration
     4. save is pickle tuple (stack1,stack2,....)
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import MotionCorrection as MC
import os, pickle, random
import numpy as np
import matplotlib.pyplot as plt

global nFrame, radius
nFrame = 5
radius = 3
dataroot = 'E:\\HumanData\\'
fovea_list = []
vm_train = ()

for file in os.listdir(dataroot):
    if file.startswith('HN_Fovea') and file.endswith('1.nii.gz'):
        fovea_list.append(file)

for vol in range(len(fovea_list)):
    print('volume :{}'.format(fovea_list[vol]))
    v = util.nii_loader(dataroot+fovea_list[vol])
    v = v[0,:,:,:]
    slc, H, W = v.shape
    show = random.randint(0,slc-1)
    
    for i in range(radius,slc-radius):
        stack = v[i-radius:i+radius+1,:,:]
        opt = np.zeros([2*radius+1,H,H],dtype=np.float32)
        
        im_fix = np.ascontiguousarray(v[i,:,:])
        for j in range(2*radius+1):
            im_mov = np.ascontiguousarray(stack[j,:,:])
            opt[j,:,:W] = MC.MotionCorrect(im_fix,im_mov)
        vm_train = vm_train+(opt,)
        
        if i == show:
            top = np.concatenate((stack[0,:,:W],stack[-1,:,:W]),axis=1)
            bot = np.concatenate((opt[0,:,:W],opt[-1,:,:W]),axis=1)
            
            plt.figure(figsize=(12,12))
            plt.axis('off')
            plt.title('slc: {}'.format(show),fontsize=15)
            plt.imshow(np.concatenate((top,bot),axis=0),cmap='gray')
            plt.show()

with open('E:\\vm_train.pickle','wb') as func:
    pickle.dump(vm_train,func)