# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:45:33 2019

@author: karim
"""

import numpy as np


Val  = np.load('G:/SAU/Labeled/Val/00_labels.npy', allow_pickle=True).item()
Train  = np.load('G:/SAU/Labeled/Train/00_labels.npy', allow_pickle=True).item()

Val_count = {
        'total':0,
        'white':0,
        'grey':0,
        'black':0,
        'brown':0
        }


Train_count = {
        'total':0,
        'white':0,
        'grey':0,
        'black':0,
        'brown':0
        }

for item in Val.values():        
    for l in item['labels']:
        Val_count['total'] = Val_count['total'] + 1
        Val_count[l['sheep_color']] = Val_count[l['sheep_color']]+ 1
        
for item in Train.values():
    for l in item['labels']:
        Train_count['total'] = Train_count['total'] + 1
        Train_count[l['sheep_color']] = Train_count[l['sheep_color']]+ 1
