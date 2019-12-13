# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:10:48 2019

@author: karim
"""
import pandas as pd
from label_utils import csv_to_label_map
import numpy as np

labels = pd.read_csv("G:/SAU/Labeled/00_annotations/all_labels-2019-12-10T09_58_16.513Z.csv")
im_label_map  = csv_to_label_map(labels)
np.save("G:/SAU/Labeled/00_annotations/00_all_labels_20191210.npy", im_label_map)
