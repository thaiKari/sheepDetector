# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:10:53 2019

@author: karim
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from label_utils import split,map_to_coco,get_split_map, get_crop, show_im_with_boxes_not_saved, rotate_crop_30, rotate__centercrop, rotate_crop
import json
import math
import cv2

#w = 4056
#h = 3040
#path = 'G:/SAU/Labeled/Train'
#label_map = np.load(os.path.join( 'G:/SAU/Labeled/00_annotations/00_all_labels_20191210.npy'), allow_pickle = True).item()
#
##
##max size to fit in gpu:
#w_new = 1024
#h_new = 1024
#folder_name = 'Crop_%d_%d'%(w_new,h_new)
#dst = os.path.join(path, folder_name)
#label_map_crop = {}
#label_map_sub={}
#failed_im_names = []
#
##50% overlap
#pos = 0
#minxs = []
#while pos < w - w_new:
#    minxs.append(int(pos))
#    pos = pos + w_new/2
#minxs.append(int(w-w_new))
##50% overlap
#pos = 0
#minys = []
#while pos < h - h_new:
#    minys.append(int(pos))
#    pos = pos + h_new/2
#minys.append(int(h-h_new))
#    
#
#for im_name in os.listdir(path):
#    if '.JPG' in im_name:
#        label_map_sub[im_name] = label_map[im_name]
#        try:
#            im = cv2.imread(os.path.join(path, im_name))
#            label = label_map[im_name]
#    
#            for minx in minxs:
#                for miny in minys:
#                    im_c, label_c = get_crop(im, label, minx, miny, w_new, h_new)
#                    im_name_c = im_name[:-4] + '_CROPPED_[' + str(minx) + ']['  + str(miny) + ']['  + str(w_new)+ '][' + str(h_new) + '].jpg'
#                    cv2.imwrite(os.path.join(dst, im_name_c), im_c)
#                    label_map_crop[im_name_c] = label_c
#                    
#        except:
#            print('uhoh', im_name)
#            failed_im_names.append(im_name)
#            
#
#np.save(os.path.join(dst, '00_labels.npy'), label_map_crop)
#np.save(os.path.join(path, '00_labels.npy'), label_map_sub)
#
#w_new = 3040
#h_new = 3040
#folder_name = 'Crop_%d_%d'%(w_new,h_new)
#dst = os.path.join(path, folder_name)
#label_map_crop = {}
#label_map_sub={}
#failed_im_names = []
#
##50% overlap
#pos = 0
#minxs = []
#while pos < w - w_new:
#    minxs.append(int(pos))
#    pos = pos + w_new/2
#minxs.append(int(w-w_new))
##50% overlap
#pos = 0
#minys = []
#while pos < h - h_new:
#    minys.append(int(pos))
#    pos = pos + h_new/2
#minys.append(int(h-h_new))
#    
#
#for im_name in os.listdir(path):
#    if '.JPG' in im_name:
#        label_map_sub[im_name] = label_map[im_name]
#        try:
#            im = cv2.imread(os.path.join(path, im_name))
#            label = label_map[im_name]
#    
#            for minx in minxs:
#                for miny in minys:
#                    im_c, label_c = get_crop(im, label, minx, miny, w_new, h_new)
#                    im_name_c = im_name[:-4] + '_CROPPED_[' + str(minx) + ']['  + str(miny) + ']['  + str(w_new)+ '][' + str(h_new) + '].jpg'
#                    cv2.imwrite(os.path.join(dst, im_name_c), im_c)
#                    label_map_crop[im_name_c] = label_c
#                    
#        except:
#            print('uhoh', im_name)
#            failed_im_names.append(im_name)
#            
#
#np.save(os.path.join(dst, '00_labels.npy'), label_map_crop)
#
#
#folder_name = 'Rotations_45'
#dst = os.path.join(path, folder_name)
#print(dst)
#label_map_rotations = {}
#minxs =[0]
#minys =[2028]
#w_new = 2128
#h_new = 2128
#deg = 45
#
#for im_name in os.listdir(path):
#    if '.JPG' in im_name:
##        try:
#        im = cv2.imread(os.path.join(path, im_name))
#        label = label_map[im_name]
#
#        for minx in minxs:
#            for miny in minys:
##                print(minx, miny)
#                im_c, label_c = rotate__centercrop(im, deg, w_new, h_new, label)
#                im_name_c = im_name[:-4] + '_CROPPED_[' + str(h_new) + ']_Rot45.jpg'
#                
#                cv2.imwrite(os.path.join(dst, im_name_c), im_c)
#                label_map_crop[im_name_c] = label_c
#                
##        except:
##        print('uhoh', im_name)
#        #failed_im_names.append(im_name)
#        
#
#np.save(os.path.join(dst, '00_labels.npy'), label_map_crop)
##
#CROPS OF ROTATED CROP
w = w_new
h = h_new
path = 'G:/SAU/Labeled/Train/Rotations_45'
label_map = np.load(os.path.join( path, '00_labels.npy'), allow_pickle = True).item()


#max size to fit in gpu:
w_new = 1024
h_new = 1024
folder_name = 'Crop_%d_%d'%(w_new,h_new)
dst = os.path.join(path, folder_name)
label_map_crop = {}
label_map_sub={}
failed_im_names = []

#50% overlap
pos = 0
minxs = []
while pos < w - w_new:
    minxs.append(int(pos))
    pos = pos + w_new / 2
minxs.append(int(w-w_new))
#50% overlap
pos = 0
minys = []
while pos < h - h_new:
    minys.append(int(pos))
    pos = pos + h_new / 2
minys.append(int(h-h_new))
    

for im_name in os.listdir(path):
    if '.jpg' in im_name:
        label_map_sub[im_name] = label_map[im_name]
        try:
            im = cv2.imread(os.path.join(path, im_name))
            label = label_map[im_name]
    
            for minx in minxs:
                for miny in minys:
                    im_c, label_c = get_crop(im, label, minx, miny, w_new, h_new)
                    im_name_c = im_name[:-4] + '_CROPPED_[' + str(minx) + ']['  + str(miny) + ']['  + str(w_new)+ '][' + str(h_new) + '].jpg'

                    cv2.imwrite(os.path.join(dst, im_name_c), im_c)
                    label_map_crop[im_name_c] = label_c
                    
        except:
            print('uhoh', im_name)
            failed_im_names.append(im_name)
            

np.save(os.path.join(dst, '00_labels.npy'), label_map_crop)
#

#folder_name = 'Rotations_30_60'
#dst = os.path.join(path, folder_name)
#label_map_rotations = {}
#
#for im_name in os.listdir(path):
#    if '.JPG' in im_name:
#        try:
#            im = cv2.imread(os.path.join(path, im_name))
#            label = label_map[im_name]
#    
#            im_rot30, label_rot30 = rotate_crop_30(im, label)
#            im_name_rot30 = im_name[:-4] + '_ROT30.jpg'
#    
#            im_rot60, label_rot60 = rotate_crop_60(im, label)
#            im_name_rot60 = im_name[:-4] + '_ROT60.jpg'
#    
#            if(len(label_rot30['labels']) >0):
#                #cv2.imwrite(os.path.join(dst, im_name_rot30), im_rot30)
#                label_map_rotations[im_name_rot30] = label_rot30
#                
#            if(len(label_rot60['labels']) >0):
#                #cv2.imwrite(os.path.join(dst, im_name_rot60), im_rot60)
#                label_map_rotations[im_name_rot60] = label_rot60
#        except:
#            print('uhoh', im_name)
#            failed_im_names.append(im_name)
#
#np.save(os.path.join(dst, '00_labels.npy'), label_map_rotations)
#
#label_map_orig_crop_rotate ={**label_map_sub,**label_map_crop, **label_map_rotations }
#coco_json = map_to_coco(label_map)

#label_map = np.load(os.path.join(path, '00_labels.npy'), allow_pickle = True).item()
#label_map_crop = np.load(os.path.join(path + '/Crop_2366_1773', '00_labels.npy'), allow_pickle = True).item()
#label_map_rotations = np.load(os.path.join(path + '/Rotations_30_60', '00_labels.npy'), allow_pickle = True).item()
#label_map_orig_crop_rotate ={**label_map,**label_map_crop, **label_map_rotations }
#coco_json = map_to_coco(label_map_orig_crop_rotate)

#
#with open('G:/SAU/Labeled/00_annotations' + '/Train_crop_rot_instances.json', 'w') as fp:
#    json.dump(coco_json, fp)
#    
#np.save('G:/SAU/Labeled/00_annotations/Train_crop_rot_instances_label_map.npy', label_map_orig_crop_rotate)

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