# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:31:00 2020

@author: karim
"""
import os
import numpy as np

root_path = 'G:/SAU/Labeled/00_Val2020_3/'
file_name = 'ir_aligned_labels'

bbox_map = np.load(os.path.join(root_path,file_name + '.npy' ), allow_pickle = True).item()

with open(os.path.join(root_path ,file_name +'.txt'), "w") as file:
                file.write("")
                
for key in bbox_map.keys():
    bboxes = bbox_map[]