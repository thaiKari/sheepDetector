# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:24:05 2019

@author: karim
"""
import numpy as np
from transformations_utils import transform_vis_pt_list_to_IR_coordinate_system

im_label_map = np.load('G:/SAU/Labeled/00_annotations/00_all_labels.npy',  allow_pickle=True).item() 
im_label_map_THERMAL={}

for k in im_label_map.keys():
    print(im_label_map[k])