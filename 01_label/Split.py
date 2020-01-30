# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:10:53 2019

@author: karim
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from label_utils import show_im_with_boxes_not_saved, split,map_to_coco,get_split_map, get_crop, show_im_with_boxes_not_saved, rotate_crop_30, rotate__centercrop, rotate_crop
import json
import math
import cv2

#w = 4056
#h = 3040
#w = 3200
#h = 2336
w = 3200
h = 2400
path = 'G:/SAU/Labeled/00_Train2020_2/01_infrared_align'
label_map = np.load(os.path.join( os.path.join(path,'00_labels.npy')), allow_pickle = True).item()

w_new = h_new = 1200
folder_name = 'Crop%d'%(w_new)
dst = os.path.join(path, folder_name)
label_map_crop = {}
label_map_sub={}
failed_im_names = []

#50% overlap
pos = 0
minxs = []
while pos < w - w_new:
    minxs.append(int(pos))
    pos = pos + w_new/2
minxs.append(int(w-w_new))
#50% overlap
pos = 0
minys = []
while pos < h - h_new:
    minys.append(int(pos))
    pos = pos + h_new/2
minys.append(int(h-h_new))


for im_name in os.listdir(path):
    
    if '.JPG' in im_name:
#    if '.npy' in im_name and not '00_labels' in im_name: #4d
        key = im_name[:-4] + '.JPG'
#        label_map_sub[key] = label_map[key]

        im = cv2.imread(os.path.join(path, im_name))
#        im = np.load(os.path.join(path, im_name))
        label = label_map[key]

        for minx in minxs:
            for miny in minys:
                im_c, label_c = get_crop(im, label, minx, miny, w_new, h_new)
                im_name_c = im_name[:-4] + '_CROPPED_[' + str(minx) + ']['  + str(miny) + ']['  + str(w_new)+ '][' + str(h_new) + '].jpg'
                cv2.imwrite(os.path.join(dst, im_name_c), im_c)
#                np.save(os.path.join(dst, im_name_c), im_c)
                label_map_crop[im_name_c] = label_c
                    
            

np.save(os.path.join(dst, '00_labels.npy'), label_map_crop)
#np.save(os.path.join(path, '00_labels.npy'), label_map_sub)
#
#labels_test = np.load(os.path.join(dst, '00_labels.npy'), allow_pickle = True).item()
#im_name = 'sep19_102MEDIA_DJI_0423_CROPPED_[600][1200][1200][1200].jpg'
#im_test = cv2.imread(os.path.join(dst, im_name))
#show_im_with_boxes_not_saved(im_test, labels_test[im_name]['labels'])