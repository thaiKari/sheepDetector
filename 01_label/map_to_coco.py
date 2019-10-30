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
#
#with open('G:/SAU/eksempel_data/annotations/instances_val2014.json') as f:
#    data_sample = json.load(f)
#    

path = 'G:/SAU/Labeled/Train/Split_100/'
data = np.load(path +'00_labels.npy',  allow_pickle=True).item()
dim = 100 

categories =[{
            'supercategory':'none',
            'name':'sheep',
             'id':0
             }]



annotations = []
images = []
label_n = -1

for im_n in range(len(data.keys())):
    k = list(data.keys())[im_n]
    d = data[k]
    if not d['labels'] == 'Skip' and d['labels'] :
        images.append({'file_name':k,
                       'height': dim,
                       'width':dim,
                       'id': im_n})
        for i in range(len(d['labels'])):
            label_n = label_n +1
            geom = d['labels'][i]['geometry']
            xmin, ymin = geom[0]
            xmax, ymax = geom[1]
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            w = xmax-xmin
            h = ymax-ymin
            
            lbl_id = k + str(i)
            
            if w >= 1 and h >=1:
                annotations.append({
                        'id': label_n,
                        'bbox': [xmin, ymin, w, h],
                        'image_id':im_n,
                        'segmentation':[],
                        'ignore':0,
                        'area':w*h,                
                        'iscrowd':0,
                        'category_id':0
                        })
    

      
with open(path + '00_coco_train_100.json', 'w') as fp:
    json.dump({'type':'instances', 'images':images, 'categories':categories, 'annotations':annotations}, fp)
