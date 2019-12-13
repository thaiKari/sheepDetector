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

path = 'G:/SAU/Labeled/00_annotations/Train_201912.npy'
label_map = np.load(path, allow_pickle = True).item()


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

#Have equal number of sheep and bg.
to_delete = random.sample(bg_keys, k = bg_count-int(sheep_count/2)) 

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

np.save('G:/SAU/Labeled/00_annotations/deleted2_Trainbg_samples.npy', to_delete)
np.save('G:/SAU/Labeled/00_annotations/Train_201912.npy', new_labelmap)
coco_label =  map_to_coco(new_labelmap)

with open('G:/SAU/Labeled/00_annotations' + '/Train_201912.json', 'w') as fp:
    json.dump(coco_label, fp)
