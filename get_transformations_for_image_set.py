# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:26:51 2019

@author: karim
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from utils import select_coordinates_from_image, resize_by_scale, get_metadata, write_filename, write_metadata, write_pts, write_transformation, get_next_image, get_im_num, increment_im, read_pts
import os


image_dir = 'E:/SAU/Bilder Felles/Sorterte/Flyving_dd_06 09 2019'
#optical_im_paths = [ "DJI_0638.jpg", "DJI_0662.jpg", "DJI_0718.jpg", "DJI_0726.jpg", "DJI_0742.jpg", "DJI_0798.jpg", "DJI_0862.jpg", "DJI_0966.jpg"]
#thermal_im_path = [ "DJI_0639.jpg", "DJI_0663.jpg", "DJI_0719.jpg", "DJI_0727.jpg", "DJI_0743.jpg", "DJI_0799.jpg", "DJI_0863.jpg", "DJI_0967.jpg"]
first_im = 'DJI_0651.jpg'
last_im_optical='DJI_0691.jpg'
cur_im = first_im


#for i in range(len(optical_im_paths)):
while get_im_num(cur_im) <= get_im_num(last_im_optical):
    optical = imageio.imread(os.path.join(image_dir, cur_im))#optical_im_paths[i] ))
    thermal = imageio.imread(os.path.join(image_dir, get_next_image(cur_im)))#thermal_im_path[i] ))

    SCALE = 1/4 #Use scale to make entire image fit on screen at same time.
    optical_pts= np.asarray(select_coordinates_from_image(resize_by_scale(SCALE, optical))) / SCALE
    thermal_pts = np.asarray(select_coordinates_from_image(thermal))
    print(optical_pts.shape)
    
    while(optical_pts.shape != thermal_pts.shape):
        print('must be the same # of points. Try again')
        optical_pts= np.asarray(select_coordinates_from_image(resize_by_scale(SCALE, optical))) / SCALE
        thermal_pts = np.asarray(select_coordinates_from_image(thermal))


    tform = transform.ProjectiveTransform() #Or AffineTransform
    tform.estimate(optical_pts, thermal_pts)
    
    
    warped = transform.warp(thermal, tform, output_shape=optical.shape)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(optical)
    plt.imshow(warped, cmap='jet', alpha=0.7)
    

    
    write_filename('image_list_corrected.txt', os.path.join(image_dir, cur_im))#optical_im_paths[i])
    
    write_pts('optical_key_pts_corrected.txt', optical_pts)
    write_pts('thermal_key_pts_corrected.txt', thermal_pts)
    write_transformation('transformations_corrected.txt', tform)
    #write_metadata('metadata.txt', get_metadata(os.path.join(image_dir, cur_im)))#optical_im_paths[i]) ))

    cur_im = increment_im(cur_im, 2)




    