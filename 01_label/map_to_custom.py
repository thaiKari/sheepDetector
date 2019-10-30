# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:14:19 2019

@author: karim
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:51:11 2019

@author: karim
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:55:24 2019

@author: karim
"""

import json
import numpy as np


path = 'G:/SAU/Labeled/Train/Split_512/'
data = np.load(path +'00_labels.npy',  allow_pickle=True).item()
dim = 512 

custom_labels = []
#
#for k in data.keys():
#    d = data[k]
#    
#    if not d['labels'] == 'Skip' and d['labels'] :
#
#        for i in range(len(d['labels'])):
#            geom = d['labels'][i]['geometry']
#            xmin, ymin = geom[0]
#            xmax, ymax = geom[1]
#            xmin = int(xmin)
#            ymin = int(ymin)
#            xmax = int(xmax)
#            ymax = int(ymax)
#            w = xmax-xmin
#            h = ymax-ymin
#            
#            lbl_id = k + str(i)
#            
#            if w >= 1 and h >=1:
#                custom_labels.append({
#                    'filename':k,
#                    'bboxes': [[xmin, ymin, w, h]],
#                    'labels': [0],
#                    'id': lbl_id,
#                    'area':w*h,
#                    'segmentation':[],
#                    'iscrowd':0})
#        
#with open(path + '00_coco_train_512.json', 'w') as fp:
#    json.dump({'categories':categories, 'annotations':annotations, 'images':images}, fp)
