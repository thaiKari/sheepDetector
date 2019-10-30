# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:10:48 2019

@author: karim
"""
import pandas as pd
from label_utils import csv_to_label_map
import numpy as np

labels = pd.read_csv("G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label_grouped/Train/labels_REVIEWED.csv")
im_label_map  = csv_to_label_map(labels)
np.save("G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label_grouped/Train/labels_REVIEWED.npy", im_label_map)

labels = pd.read_csv("G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label_grouped/Val_g5_9/labels_REVIEWED.csv")
im_label_map  = csv_to_label_map(labels)
np.save("G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label_grouped/Val_g5_9/labels_REVIEWED.npy", im_label_map)