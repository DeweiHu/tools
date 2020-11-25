
import sys
sys.path.insert(0,'/home/dewei/Desktop/octa/')
import util
import os, subprocess
import numpy as np
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt

'''
The sequential-greedy is used for fast deformable registration
of a stack of consecutive Bscans to the center slice

input data: stack with shape [H,slc,W]
input temp: dir that saves the temporary warp and nifti files
output: registered stack with same dimension
'''
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

'''
self-fusion
input: stack of image with size [H,slc,W], the middle slc is
	   regarded as the template and rest are atlases
temp: the temporal directory to save weightmap

'''
def sf(stack,temp):
	for file in os.listdir(temp):
		os.remove(temp+file)
	H,slc,W = stack.shape
	if slc % 2 == 1:
		r = int((slc-1) / 2)
	else:
		raise ValueError("Slice Number as to be odd.")

	# save the template as tiff
	im_fix = Image.fromarray(np.float32(stack[:,r,:]))
	im_fix.save(temp + 'fix_img.tif')

	for i in range(slc):
		im_mov = Image.fromarray(np.float32(stack[:,i,:]))
		im_mov.save(temp + 'atlas{}.tif'.format(i))

	subprocess.call("/home/dewei/tool/self_fusion.sh")
	im_sf = io.imread(temp+'synthResult.tif')

	return im_sf


if __name__=="__main__":
	dataroot = '/home/dewei/Desktop/octa/data/'
	temp = '/home/dewei/Desktop/octa/temp/'

	vol = util.nii_loader(dataroot+'orig_fovea.nii.gz')
	stack = vol[:,90:99,:]
	stack_opt = greedy(stack,temp)

	im_sf = sf(stack_opt,temp)

	plt.figure(figsize=(12,12))
	plt.axis('off')
	plt.title('sf vs. local-proj')
	plt.imshow(np.concatenate((im_sf,np.mean(stack,axis=1)),axis=1),
		cmap='gray')
	plt.show()

	print('Execution finished.')