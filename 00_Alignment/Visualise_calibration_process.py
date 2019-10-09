# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 08:59:26 2019

@author: karim
"""



import numpy as np
import cv2
from utils import read_pts, get_im_num, increment_im, get_next_image, undistort_pts, get_line_mask, undistort_image, read_transformations
import matplotlib.pyplot as plt
from skimage import transform
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec


Image_path = './camera_calibration/test/dji_0691.jpg'

optical = cv2.imread(Image_path)
optical_lines = get_line_mask(optical)
thermal = cv2.imread(get_next_image(Image_path))
thermal_undistorted = undistort_image(thermal)

#transforms = read_transformations('./Newest_data/transformations_corrected_affine.txt')
#t_avg = np.median(transforms, axis=0)
#print(t_avg)
t_avg = np.load('./Newest_data/the_transform.npy')
print(t_avg)
thermal_undistorted_transformed = transform.warp(thermal_undistorted, transform.AffineTransform(t_avg), output_shape=optical.shape)


#fig = plt.figure(figsize=(7,14))
#ax = fig.add_subplot(111)
#ax.set_ylabel('o')
#ax.xaxis.label.set_color('red')
#ax.imshow(thermal)

plt.figure(figsize=(7,14))
plt.imshow(thermal)

plt.figure(figsize=(7,14))
plt.imshow(thermal_undistorted)

plt.figure(figsize=(7,14))
plt.imshow(thermal_undistorted_transformed)

### PROCESS
#plt.figure(figsize=(20,10))
#plt.subplot(1, 3, 1)
#plt.title('1. Original Thermal Image')
#plt.imshow(thermal)
#plt.subplot(1, 3, 2)
#plt.title('2. Correct Lens Distortion')
#plt.imshow(thermal_undistorted)
#plt.subplot(1, 3, 3)
#plt.title('3. Transform to Optical Coordinate System')
#plt.imshow(thermal_undistorted_transformed)



#fig = plt.figure(constrained_layout=True, figsize=(10, 10))
#
#gs = GridSpec(3, 2, figure=fig)
#ax1 = fig.add_subplot(gs[0, 0])
## identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
#ax2 = fig.add_subplot(gs[0, -1:])
#ax3 = fig.add_subplot(gs[1:, :])
#
#
#ax1.imshow(optical, alpha=0.8)
#ax1.imshow(optical_lines, cmap=cm.jet, interpolation='none')
#ax1.set_title('Optical Image with Detected Lines')
#ax1.axis('off')
#
#ax2.imshow(thermal)
#ax2.set_title('Thermal Image')
#ax2.axis('off')
#
#ax3.imshow(thermal_undistorted_transformed, alpha = 0.9)
#ax3.imshow(optical_lines, cmap=cm.jet, interpolation='none')
#ax3.set_title('Transformed Thermal Image Masked By lines From Optical Image')
#ax3.axis('off')
#
#plt.show()

### RESULT
#plt.figure(figsize=(12, 10))
#plt.subplot(2, 2, 1)
#plt.imshow(optical, alpha=0.8)
#plt.imshow(optical_lines, cmap=cm.jet, interpolation='none')
#plt.axis('off')
#plt.subplot(2, 2, 2)
#plt.imshow(thermal)
#plt.axis('off')
##plt.subplot(2, 2, 3)
##plt.imshow(optical)
##plt.imshow(thermal_undistorted_transformed, alpha=0.8)
##plt.axis('off')
#plt.subplot(2, 2, 4)
#plt.imshow(thermal_undistorted_transformed, alpha = 0.9)
#plt.imshow(optical_lines, cmap=cm.jet, interpolation='none')
#plt.axis('off')


