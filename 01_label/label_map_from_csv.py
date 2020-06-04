# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:10:48 2019

@author: karim
"""
import pandas as pd
from label_utils import csv_to_label_map
import numpy as np
import os

labels = pd.read_csv("G:/SAU/Labeled/00_annotations/all_labelled_20205013.csv")
im_label_map  = csv_to_label_map(labels)
np.save("G:/SAU/Labeled/00_annotations/all_labelled_20205013.npy", im_label_map)

#all_labels = np.load("G:/SAU/Labeled/00_annotations/all_labelled_20205013.npy", allow_pickle=True).item()
#im_list = os.listdir('G:/SAU/Labeled/00_Test2020/00_rgb')
#new_labels = {}
#
#for k in all_labels.keys():
#    if k in im_list:
#        new_labels[k] = all_labels[k]
#        
#np.save('G:/SAU/Labeled/00_Test2020/00_labels.npy', new_labels)

