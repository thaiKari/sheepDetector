# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:02:19 2019

@author: karim
"""

import json
import os
from shutil import copyfile
import numpy as np


#with open('labels.json') as f:
#    data = json.load(f)


    
#has_labels = []
#    
#for d in data:
#    if not d['Label'] == 'Skip':
#        has_labels.append( d['External ID'])
#   
#path = 'G:/SAU/Fra Jens/Datasett1 - Copy - Copy'
#has_path = 'G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label'
#no_label_path = 'G:/SAU/Fra Jens/Datasett1 - Copy - Copy/no_label'
     
#for l in has_labels:
#    src = os.path.join(path, l)
#    dst = os.path.join(dst_path, l)
#    copyfile(src, dst)
#
#for im in os.listdir(path):
#    src = os.path.join(path, im)
#    if im in has_labels:
#        dst = os.path.join(has_path, im)
#    else:
#        dst = os.path.join(no_label_path, im)
#    copyfile(src, dst)

#get label_map for Train images:
label_map = np.load('im_label_map_new_names.npy', allow_pickle = True).item()

path = 'G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label_grouped/Train'
train_labels = {}

for im in os.listdir(path):
    k = im.replace(' - Copy','')
#    print(k)
    try:
        train_labels[im] = label_map[k]
    except:
        print(im)
    
np.save(os.path.join(path, 'labels.npy'), train_labels)



#get label_map for Val images:
path = 'G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label_grouped/Val_g5_9'
val_labels = {}

for im in os.listdir(path):
    k = im.replace(' - Copy','')
    try:
        val_labels[im] = label_map[k]
    except:
        print(im)
    
np.save(os.path.join(path, 'labels.npy'), val_labels)

    