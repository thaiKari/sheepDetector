# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:26:51 2019

@author: karim
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from utils import select_coordinates_from_image, resize_by_scale, get_metadata, write_filename, write_metadata, write_pts, write_transformation, get_next_image, get_im_num, increment_im, read_pts, undistort_image, undistort_pts
import os
import cv2


image_dir = './camera_calibration/fakkel_DD'
#optical_im_paths = [ "DJI_0638.jpg", "DJI_0662.jpg", "DJI_0718.jpg", "DJI_0726.jpg", "DJI_0742.jpg", "DJI_0798.jpg", "DJI_0862.jpg", "DJI_0966.jpg"]
#thermal_im_path = [ "DJI_0639.jpg", "DJI_0663.jpg", "DJI_0719.jpg", "DJI_0727.jpg", "DJI_0743.jpg", "DJI_0799.jpg", "DJI_0863.jpg", "DJI_0967.jpg"]
first_im = 'DJI_0651.jpg'
last_im_optical='DJI_0692.jpg'
cur_im = first_im


#for i in range(len(optical_im_paths)):
while get_im_num(cur_im) <= get_im_num(last_im_optical):
    print(cur_im)
    optical = imageio.imread(os.path.join(image_dir, cur_im))#optical_im_paths[i] ))
    thermal = imageio.imread(os.path.join(image_dir, get_next_image(cur_im)))#thermal_im_path[i] ))

    SCALE = 1/4 #Use scale to make entire image fit on screen at same time.
    optical_pts= np.asarray(select_coordinates_from_image(resize_by_scale(SCALE, optical))) / SCALE
    thermal_pts = np.asarray(select_coordinates_from_image(thermal))
#    print(optical_pts.shape)
    
    while(optical_pts.shape != thermal_pts.shape):
        print('must be the same # of points. Try again')
        optical_pts= np.asarray(select_coordinates_from_image(resize_by_scale(SCALE, optical))) / SCALE
        thermal_pts = np.asarray(select_coordinates_from_image(thermal))


    thermal_pts_undistorted = undistort_pts(thermal_pts)
    tform = transform.ProjectiveTransform() #Or AffineTransform
    tform.estimate(optical_pts, thermal_pts_undistorted)
    
    
#    warped = transform.warp(thermal, tform, output_shape=optical.shape)
#    
#    plt.figure(figsize=(20, 10))
#    plt.imshow(optical)
#    plt.imshow(warped, cmap='jet', alpha=0.7)
    

#    
#    write_filename('./Newest_data/image_list.txt', os.path.join(image_dir, cur_im))#optical_im_paths[i])
#    
    write_pts('./camera_calibration/fakkel_DD/optical_key_pts_fakkel_DD.txt', optical_pts)
    write_pts('./camera_calibration/fakkel_DD/thermal_key_pts_fakkel_DD.txt', thermal_pts)
    #write_transformation('./Newest_data/transformations_corrected.txt', tform)
    #write_metadata('metadata.txt', get_metadata(os.path.join(image_dir, cur_im)))#optical_im_paths[i]) ))

    cur_im = increment_im(cur_im, 2)


### FOR KNOWN PTS
#optical_pts_list = read_pts('./Newest_data/optical_key_pts_fakkel.txt')
#thermal_pts_list = read_pts('./Newest_data/thermal_key_pts_fakkel.txt')
#
##for i in range(len(optical_pts_list)):
##    optical_pts = optical_pts_list[i]
##    thermal_pts = thermal_pts_list[i]
##    
##    thermal_pts_undistorted = undistort_pts(thermal_pts)
##    tform = transform.AffineTransform() #Or AffineTransform
##    tform.estimate(optical_pts, thermal_pts_undistorted)
##    print(tform.params)
##    #write_transformation('./Newest_data/transformations_corrected_affine.txt', tform)
##    
# 
#
#all_optical_pts = []
#all_thermal_pts = []
#
#for i in range(len(optical_pts_list)):
#    all_optical_pts = [*all_optical_pts, *optical_pts_list[i]]
#    all_thermal_pts = [*all_thermal_pts, *undistort_pts(thermal_pts_list[i])]
#
##tform = transform.AffineTransform()
#tform = transform.estimate_transform('affine',np.asarray(all_optical_pts), np.asarray(all_thermal_pts))
#print(tform.params)
#np.save('./Newest_data/the_transform', tform.params)