# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:39:51 2020

@author: karim
"""
import sys
import os
sys.path.append(os.path.abspath('../01_label'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from label_utils import map_to_yolo, show_im_with_boxes_not_saved, show_im_with_bbox_and_grid, grid_cell_has_label, map_to_coco, label_center_in_grid_cell
import math
import json
import cv2


import numpy as np
import os
import math
import matplotlib.pyplot as plt
from shutil import copyfile



path = 'G:/SAU/Labeled/00_Val2020_2/01_infrared_align'
w = 3200
h = 2400

#
labels = np.load( os.path.join(path,'00_labels.npy') , allow_pickle=True).item()
im_name = 'sep19_102MEDIA_DJI_0423.JPG'
##fixed_grid_bbox_labels = {}
##
##im = np.load(os.path.join(path, im_name))
im = cv2.imread(os.path.join(path, im_name))
#
#yolo_str = map_to_yolo(labels, im.shape[1], im.shape[0])
#
##with open(os.path.join(path,"00_labels_yolo.txt"), "w") as text_file:
##    text_file.write(yolo_str)
#
show_im_with_bbox_and_grid(im[:,:,:3], labels[im_name]['labels'],grid_shape=(6,8), cmap='gray', bbox_edge_color='red')
#
#print grid to txt file
grid_shape = (6,8)
grid_label_map = {}
fixed_grid_bbox_labels = {}

label_filename = '00_grid_3_4.txt'

#with open(os.path.join( path, label_filename), "w") as file:
#    file.write('')

#n=0
#for key in os.listdir(path):#labels.keys():
#    if('MEDIA_DJI' in key) :
#        n= n+1
#        print(key)
###        with open(os.path.join( path, label_filename), "a") as file:
###            file.write(key[:-4] + ' ')
#        
#        if n %3 == 0:
#            im_labels = labels[key[:-4] + '.JPG']['labels']
#            im = np.load(os.path.join(path,key))
#            show_im_with_bbox_and_grid(im[:,:,-1], im_labels , grid_shape=grid_shape, cmap='gray', bbox_edge_color='red')

#        grid = np.zeros(grid_shape, np.uint8)
#        
#        grid_h = math.floor(h/ grid_shape[0])
#        grid_w = math.floor(w/ grid_shape[1])
##        print(grid_h, grid_w)
#        
##        im_fixed_grid_bbox_labels = []
#        i = 0
#        for x in range(grid_shape[1]):
#            for y in range(grid_shape[0]):
#                i = i + 1
#                print(i)
#                grid_minx = x*grid_w
#                grid_miny =  y*grid_h
#                grid_maxx = (x+1)*grid_w
#                grid_maxy = (y+1)*grid_h
#                if grid_cell_has_label(grid_minx,grid_miny, grid_maxx, grid_maxy, im_labels ):
#                    grid[y,x] = 1
#                    
#        
        
#        with open(os.path.join( path, label_filename), "a") as file:
#            file.write(str(grid.flatten())[1:-1] + '\n')
#                    im_fixed_grid_bbox_labels.append({'geometry':[[grid_minx, grid_miny],[grid_maxx, grid_maxy]]})
#                    print(im_fixed_grid_bbox_labels)
#                
#    grid_label_map[key] = grid.tolist()
#    fixed_grid_bbox_labels[key] = { 'labels': im_fixed_grid_bbox_labels}
#
#np.save(os.path.join(path, '00_labels_fixed_grid_bbox.npy'), fixed_grid_bbox_labels)
#coco_fixed_grid_bbox = map_to_coco(fixed_grid_bbox_labels, w0=im.shape[1], h0=im.shape[0])
#
#with open(os.path.join(path, '00_labels_fixed_grid_bbox_coco') + '.json', 'w') as fp:
#    json.dump(coco_fixed_grid_bbox, fp)

#np.save(os.path.join(path, '00_labels_grid.npy'), grid_label_map)
#coco_grid = map_to_coco(grid_label_map, w0=im.shape[1], h0=im.shape[0], grid_type=True)

#with open(os.path.join(path, '00_labels_grid_coco') + '.json', 'w') as fp:
#    json.dump(coco_grid, fp)