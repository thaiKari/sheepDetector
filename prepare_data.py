# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:47:06 2020

@author: karim
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:15:38 2020

@author: karim
"""

import sys
import os
sys.path.append(os.path.abspath('../01_label'))


import shutil
import cv2
import numpy as np
import matplotlib.patches as patches
from transformations import transform_IR_im_to_vis_coordinate_system, transform_vis_im_to_IR_coordinate_system, transform_vis_pt_list_to_IR_coordinate_system
import matplotlib.pyplot as plt
import json

#from label_utils import get_labels_from_crop, show_im_with_boxes_not_saved, map_to_coco_simple

path = 'G:/SAU/Labeled/00_Train2020_2'
IR_path = os.path.join(path, '00_infrared')
rgb_path = os.path.join(path, '00_rgb')
labels = np.load('G:/SAU/Labeled/00_annotations/00_all_labels_20200107.npy', allow_pickle=True).item()

### ALIGN IR images and extract suitable crop from both IR and rgb ##
IR_align_path = os.path.join(path, '01_infrared_align')
rgb_align_path = os.path.join(path, '01_rgb_align')

w = 4056
h = 3040
xmin = 475
ymin = 250
w_new = 3200
h_new = 2400
#
#labels_split = {} 
#labels_aligned = {}
#
#for file in os.listdir(IR_path):
#    file_label = labels[file] 
#   
#    print(file)
#    if '.JPG' in file:
#        labels_split[file] = file_label
#        im_ir = cv2.imread(os.path.join(IR_path, file)) 
#        #im_ir= cv2.cvtColor(im_ir, cv2.COLOR_BGR2RGB).astype(int)
#        
#        im_rgb = cv2.imread(os.path.join(rgb_path, file))
#        #im_rgb= cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB).astype(int)
#        
#        im_ir_transformed = (transform_IR_im_to_vis_coordinate_system(im_ir)*255).astype(int)
#        
#
#
### visualize cropped area        
##        fig,ax = plt.subplots(1, figsize=(20, 10))
##        ax.imshow(im_rgb)
##        ax.imshow(im_ir_transformed, cmap='gray', alpha = 0.7)
##        rect = patches.Rectangle((xmin,ymin), w_new, h_new ,linewidth=2,edgecolor='r',facecolor='none')
##        ax.add_patch(rect)
#        
#        
#        xmax = xmin + w_new
#        ymax = ymin + h_new
#        im_rgb_cropped = im_rgb[ymin:ymax, xmin:xmax,:]
#        im_ir_cropped = im_ir_transformed[ymin:ymax, xmin:xmax,:]
#        
#        labels_new = get_labels_from_crop(xmin, ymin , w_new, h_new, file_label['labels'])
##        show_im_with_boxes_not_saved(im_ir_cropped, labels_new)
#        
#        new_im_labels = file_label.copy() #Also keep the color of the box. just new coordinates.
#        new_im_labels['labels'] = labels_new
#        labels_aligned[file] = new_im_labels
#        
#        cv2.imwrite(os.path.join(rgb_align_path,file), im_rgb_cropped )
#        cv2.imwrite(os.path.join(IR_align_path,file), im_ir_cropped )
#        
#np.save(os.path.join(IR_path,'00_labels.npy'), labels_split)
#np.save(os.path.join(rgb_path,'00_labels.npy'), labels_split)
#np.save(os.path.join(IR_align_path,'00_labels.npy'), labels_aligned)
#np.save(os.path.join(rgb_align_path,'00_labels.npy'), labels_aligned)

###To coco
#label_map = np.load(os.path.join(rgb_align_path,'00_labels.npy'), allow_pickle=True).item()
#np.save(os.path.join(path,'00_labels.npy'), label_map)
#coco_map = map_to_coco_simple(label_map, w0=w_new, h0=h_new)
#
#with open(path + '/00_labels_simple.json', 'w') as fp:
#    json.dump(coco_map, fp)


## FIND mean and standard deviation of data.


means_b = []
means_g = []
means_r = []

stds_b = []
stds_g = []
stds_r = []

for file in os.listdir(rgb_path):
    if '.JPG' in file:
        print(file)
        im = cv2.imread( os.path.join(rgb_path, file) )
    
        means_b = [ *means_b, im[:,:,0].flatten().mean()]
        means_g = [ *means_g, im[:,:,1].flatten().mean()]
        means_r = [ *means_r, im[:,:,1].flatten().mean()]
        
        stds_b = [ *stds_b, im[:,:,0].flatten().std()]
        stds_g = [ *stds_g, im[:,:,1].flatten().std()]
        stds_r = [ *stds_r, im[:,:,1].flatten().std()]


means_b = np.array(means_b)
means_g = np.array(means_g)
means_r = np.array(means_r)

stds_b = np.array(stds_b)
stds_g = np.array(stds_g)
stds_r = np.array(stds_r)

mean = (means_r.mean(), means_g.mean(), means_b.mean() )
std = (stds_r.mean(), stds_g.mean(), stds_b.mean() )


mean_IR = (45.328, 45.328, 43.398)
std_IR = (12.743, 12.743, 12.816)

#mean_rgb = (123.530, 123.530, 97.391)
#std_rgb = (45.491, 45.491, 48.320)