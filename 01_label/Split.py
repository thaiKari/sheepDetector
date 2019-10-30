# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:10:53 2019

@author: karim
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from label_utils import split,map_to_coco,get_split_map, get_crop, show_im_with_boxes_not_saved, rotate_crop_30, rotate_crop_60
import json
import math
import cv2

w = 4056
h = 3040
path = 'G:/SAU/Labeled/Train'
label_map = np.load(os.path.join(path, '00_train_labels.npy'), allow_pickle = True).item()


##max size to fit in gpu:
#w_new = math.floor(w*7/12) 
#h_new = math.floor(h*7/12) 
#folder_name = 'Crop_%d_%d'%(w_new,h_new)
#dst = os.path.join(path, folder_name)
#label_map_crop = {}
#
#for im_name in os.listdir(path):
#    if '.JPG' in im_name:
#
#        im = cv2.imread(os.path.join(path, im_name))
#        label = label_map[im_name]
#        #Make 5 different crops
#        minxys = [(0, 0),(w - w_new, 0),(0, h - h_new),(w - w_new, h - h_new),(int(w/2) - int(w_new/2), int(h/2)- int(h_new/2))]
#        
#        for minxy in minxys:
#            im_c, label_c = get_crop(im, label, *minxy, w_new, h_new)
#            im_name_c = im_name[:-4] + '_CROPPED_[' + str(minxy[0]) + ']['  + str(minxy[1]) + ']['  + str(w_new)+ '][' + str(h_new) + '].jpg'
#            if(len(label_c['labels']) >0):
#                cv2.imwrite(os.path.join(dst, im_name_c), im_c)
#                label_map_crop[im_name_c] = label_c
#
#np.save(os.path.join(dst, '00_labels.npy'), label_map_crop)
#
#folder_name = 'Rotations_30_60'
#dst = os.path.join(path, folder_name)
#label_map_rotations = {}
#
#for im_name in os.listdir(path):
#    if '.JPG' in im_name:
#
#        im = cv2.imread(os.path.join(path, im_name))
#        label = label_map[im_name]
#
#        im_rot30, label_rot30 = rotate_crop_30(im, label)
#        im_name_rot30 = im_name[:-4] + '_ROT30.jpg'
#
#        im_rot60, label_rot60 = rotate_crop_60(im, label)
#        im_name_rot60 = im_name[:-4] + '_ROT60.jpg'
#
#        if(len(label_rot30['labels']) >0):
#            cv2.imwrite(os.path.join(dst, im_name_rot30), im_rot30)
#            label_map_rotations[im_name_rot30] = label_rot30
#            
#        if(len(label_rot60['labels']) >0):
#            cv2.imwrite(os.path.join(dst, im_name_rot60), im_rot60)
#            label_map_rotations[im_name_rot60] = label_rot60
#
#np.save(os.path.join(dst, '00_labels.npy'), label_map_rotations)
#
#label_map_orig_crop_rotate ={**label_map,**label_map_crop, **label_map_rotations }
#coco_json = map_to_coco(label_map)

label_map = np.load(os.path.join('G:/SAU/Labeled/Train', '00_train_labels.npy'), allow_pickle = True).item()
#label_map_crop = np.load(os.path.join('G:/SAU/Labeled/Train/Crop_2366_1773', '00_labels.npy'), allow_pickle = True).item()
#label_map_rotations = np.load(os.path.join('G:/SAU/Labeled/Train/Rotations_30_60', '00_labels.npy'), allow_pickle = True).item()
#label_map_orig_crop_rotate ={**label_map,**label_map_crop, **label_map_rotations }
coco_json = map_to_coco(label_map)


with open('G:/SAU/Labeled/00_annotations' + '/labels_vis.json', 'w') as fp:
    json.dump(coco_json, fp)

#coco_json = map_to_coco(label_map,w,h)

#
#dim = 1024
#dst =  'G:/SAU/Labeled/Val_g5_9/Split_' + str(dim)
#step = 680
#label_map_1024 = get_split_map(label_map, path, w, h, step, dim)               
#np.save(os.path.join(dst, '00_labels.npy'), label_map_1024)
#
#
#dim = 512
#dst = 'G:/SAU/Labeled/Val_g5_9/Split_' + str(dim)
#step = 340
#label_map_512 = get_split_map(label_map, path, w, h, step, dim)
#np.save(os.path.join(dst, '00_labels.npy'), label_map_512)
#
#dim = 256
#dst = 'G:/SAU/Labeled/Val_g5_9/Split_' + str(dim)
#step = 170
#label_map_256 = get_split_map(label_map, path, w, h, step, dim)
#np.save(os.path.join(dst, '00_labels.npy'), label_map_256)
#
#dim = 128
#dst = 'G:/SAU/Labeled/Val_g5_9/Split_' + str(dim)
#step = 85
#label_map_128 = get_split_map(label_map, path, w, h, step, dim)
#np.save(os.path.join(dst, '00_labels.npy'), label_map_128)

#label_map_1024 = np.load(os.path.join(path + '/Split_1024', '00_labels.npy'), allow_pickle = True).item()
##coco_1024 = map_to_coco(label_map_1024,1024,1024, label_n=2194, im_start_id=446)
#label_map_512 = np.load(os.path.join(path  + '/Split_512', '00_labels.npy'), allow_pickle = True).item()
##coco_512 = map_to_coco(label_map_512,512,512, label_n=6931, im_start_id=2220)
#label_map_256 = np.load(os.path.join(path  + '/Split_256', '00_labels.npy'), allow_pickle = True).item()
##coco_256 = map_to_coco(label_map_256,256,256, label_n=11975, im_start_id=3267)
#label_map_128 = np.load(os.path.join(path + '/Split_128', '00_labels.npy'), allow_pickle = True).item()
##coco_128 = map_to_coco(label_map_128,128,128, label_n=17311, im_start_id=4194)

#label_map_all = {**label_map, **label_map_1024, **label_map_512, **label_map_256, **label_map_128 }
##coco_all = {'type':'instances', 'images':[], 'categories':coco_json['categories'], 'annotations':[*coco_json['annotations'], *coco_1024['annotations'], *coco_512['annotations'],*coco_256['annotations'], *coco_128['annotations']]}
#coco_all = map_to_coco(label_map_all)
#
#
#dst = 'G:/SAU/Labeled/Val_scales'
#with open(dst + '/coco_instances.json', 'w') as fp:
#    json.dump(coco_all, fp)