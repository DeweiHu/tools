'''
The sequential-greedy is used for fast deformable registration
of a stack of consecutive Bscans to the center slice

input data: stack with shape [H,slc,W]
input temp: dir that saves the temporary warp and nifti files
output: registered stack with same dimension
'''


import sys
sys.path.insert(0,'/home/dewei/Desktop/octa/')
import util
import os, subprocess
import numpy as np
import matplotlib.pyplot as plt

def greedy(stack,temp):
	# clean the temp directory
	for file in os.listdir(temp):
		os.remove(temp+file)

	h,slc,w = stack.shape

	if slc % 2 == 1:
		r = int((slc-1) / 2)
	else:
		raise ValueError("Slice Number as to be odd.")

	stack_opt = np.zeros(stack.shape,dtype=np.float32)

	# non-stepwise deformable registration
	im_fix = stack[:,r,:]
	util.nii_saver(im_fix,temp,'im_fix.nii')

	for i in range(slc):
		im_mov = stack[:,i,:]
		util.nii_saver(im_mov,temp,'im_mov.nii')
		subprocess.call("/home/dewei/tool/greedy.sh")
		stack_opt[:,i,:] = util.nii_loader(temp+'warped.nii')

	return stack_opt

if __name__=="__main__":
	dataroot = '/home/dewei/Desktop/octa/data/'
	temp = '/home/dewei/Desktop/octa/temp/'

	vol = util.nii_loader(dataroot+'orig_fovea.nii.gz')
	stack = vol[:,90:99,:]
	stack_opt = greedy(stack,temp)

	plt.figure(figsize=(12,12))
	plt.axis('off')
	plt.title('deformed vs. original')
	plt.imshow(np.concatenate([stack_opt[:,0,:],stack[:,0,:]],axis=1),
		cmap='gray')
	plt.show()

	plt.figure(figsize=(12,12))
	plt.axis('off')
	plt.title('moved vs. fixed')
	plt.imshow(np.concatenate([stack_opt[:,0,:],stack[:,5,:]],axis=1),
		cmap='gray')
	plt.show()

	print('Execution finished.')