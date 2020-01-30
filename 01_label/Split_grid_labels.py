# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:20:55 2020

@author: karim
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:10:53 2019

@author: karim
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from label_utils import read_grid_labels, write_grid_labels_to_txt
import json
import math
import cv2


w = 3200
h = 2400
grid_shape = (3,4)
grid_w = math.floor(w / grid_shape[1])
grid_h = math.floor(h / grid_shape[0])

grid_shape_new = (3,3)
path = 'G:/SAU/Labeled/Train2020/4d_vis_plus_ir2'

label_map = read_grid_labels(os.path.join(path, '00_grid_3_4.txt'), grid_shape)

folder_name = 'Crop_hw_%d_%d'%(grid_shape_new)
dst = os.path.join(path, folder_name)
label_map_crop = {}
failed_im_names = []


for im_name in os.listdir(path):
    
#    if '.JPG' in im_name:
    if 'MEDIA_DJI' in im_name: #4d
        key = im_name[:-4]
        print(key)

#        im = cv2.imread(os.path.join(path, im_name))
        im = np.load(os.path.join(path, im_name ))
        label = label_map[key]
        print(label)

        for x_grid_number in range(grid_shape[1] - grid_shape_new[1] + 1):
            for y_grid_number in range(grid_shape[0] - grid_shape_new[0] + 1):
                print(x_grid_number, y_grid_number)
                y_min_grid_split = y_grid_number
                y_max_grid_split = y_grid_number + grid_shape_new[0]
                x_min_grid_split = x_grid_number
                x_max_grid_split = x_grid_number + grid_shape_new[1]
                
                label_split = label[y_min_grid_split: y_max_grid_split , x_min_grid_split: x_max_grid_split]
                im_split = im[y_min_grid_split*grid_h: y_max_grid_split*grid_h, x_min_grid_split*grid_w: x_max_grid_split*grid_w, : ]
                print(im_split.shape)
                print(label_split)
                
                key_split = key + '_yx_{}_{}'.format(y_grid_number, x_grid_number)
                print(key_split)
                print()
                
                label_map_crop[key_split] = label_split
                np.save(os.path.join(dst, key_split + '.npy'), im_split)

write_grid_labels_to_txt( os.path.join(dst, '00_grid_%d_%d.txt'%(grid_shape_new)), label_map_crop )
            
#
#np.save(os.path.join(dst, '00_labels.npy'), label_map_crop)
#np.save(os.path.join(path, '00_labels.npy'), label_map_sub)
#
