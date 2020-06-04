# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:31:00 2020

@author: karim
"""
import os
import numpy as np
import json

root_path = 'G:/SAU/Labeled/00_annotations/'
file_name = 'all_labelled_20205013_infrared'

bbox_map = np.load(os.path.join(root_path,file_name + '.npy' ), allow_pickle = True).item()

def convert(o):
    if isinstance(o, np.int32): return int(o)  
    raise TypeError

with open( os.path.join(root_path, file_name + '.json'), 'w'  ) as file:
    json.dump(bbox_map, file, default=convert, indent=4)