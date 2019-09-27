# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:19:22 2019

@author: karim
"""

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
import glob


image_dir = './camera_calibration/moreThermal/*.jpg'
images = glob.glob(image_dir)


#for i in range(len(optical_im_paths)):
for fname in images:
    print(fname)
    im = imageio.imread(fname)

    pts = np.asarray(select_coordinates_from_image(im))
    print(pts.shape)
    
    while( pts.shape[0] != 54):
        print('uh oh', pts.shape, fname[-12:])
        pts = np.asarray(select_coordinates_from_image(im))

  
    write_filename('./Newest_data/image_list_moreThermal_checker_pts.txt', fname)#optical_im_paths[i])    
    write_pts('./Newest_data/moreThermal_checker_pts.txt', pts)




    
