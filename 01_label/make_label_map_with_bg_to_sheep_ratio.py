# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:42:50 2019

@author: karim
"""
import numpy as np
import os
from label_utils import  map_to_coco
import json

#path = 'G:/SAU/Labeled/Train_201912_labels'
#
#label_map = {}
#
#for m in os.listdir(path):
#    if '.npy' in m:
#        label_map_n = np.load( os.path.join(path, m), allow_pickle=True).item()
#        label_map = {**label_map, **label_map_n}
#
#dst = 'G:/SAU/Labeled/00_annotations'
#filename = 'Train_201912.npy'
#np.save(os.path.join(dst, filename), label_map)
#
##labels = np.load(os.path.join(dst, filename), allow_pickle=True).item()
#coco_label =  map_to_coco(label_map)
#
#with open(dst + '/Train_201912.json', 'w') as fp:
#    json.dump(coco_label, fp)





import random

path = 'G:/SAU/Labeled/Train2020/4d_vis_plus_ir/Crop1024'
label_name = '00_labels.npy'
label_map = np.load(os.path.join(path, label_name), allow_pickle = True).item()


sheep_count = 0
bg_count = 0
bg_keys = []

for k in label_map.keys():
    labels = label_map[k]['labels']
    if(len(labels)<1):
        bg_count = bg_count +1
        bg_keys.append(k)
    else:
        sheep_count = sheep_count +1
        
print('sheep_count', sheep_count )
print('bg_count', bg_count )

bg_ratio = 1/2
to_delete = random.sample(bg_keys, k = bg_count-int(sheep_count * bg_ratio)) 

sheep_count = 0
bg_count = 0
new_labelmap = {}
for k in label_map.keys():
    if k not in to_delete:
        new_labelmap[k] = label_map[k]
        
for k in new_labelmap.keys():
    labels = label_map[k]['labels']
    if(len(labels)<1):
        bg_count = bg_count +1
    else:
        sheep_count = sheep_count +1
        
print('sheep_count', sheep_count )
print('bg_count', bg_count )

np.save(os.path.join(path, '00_labels_bgtosheep_1to2.npy'), new_labelmap)

#np.save('G:/SAU/Labeled/00_annotations/deleted2_Trainbg_samples.npy', to_delete)
#np.save('G:/SAU/Labeled/00_annotations/Train_201912.npy', new_labelmap)


filename ='train2020_4d_crop1024_bgtosheep_1to2'
path = 'G:/SAU/Labeled/00_annotations/croped_for_alignment'
labels = np.load(os.path.join(path, filename) + '.npy', allow_pickle=True).item()

new_label_map = {}

for k in labels.keys():
    k_new = k[-4:] + '.npy'
    new_label_map[k_new]= labels[k]v
    
np.save(os.path.join(path, filename) + '.npy', new_label_map)    

coco_label =  map_to_coco(labels)

with open(os.path.join(path, filename) + '.json', 'w') as fp:
    json.dump(coco_label, fp)

w = 640
h = 480
labels = np.load('G:/SAU/Labeled/Val2020/4d_IR_coord_system/00_labels.npy', allow_pickle=True).item()
coco_label =  map_to_coco(labels, w0=w, h0=h)
with open('G:/SAU/Labeled/00_annotations/val2020_IR_coordinate_system' + '.json', 'w') as fp:
    json.dump(coco_label, fp)