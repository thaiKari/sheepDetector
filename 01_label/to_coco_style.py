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
    

path = 'G:/SAU/Labeled/Train/Split/'
with open(path + 'lblbx-2019-10-17T13_12_49.237Z.json') as f:
    data = json.load(f)
    
## data['annotations']:
#    {'image_id': xx
#      'bbox': [xmin, ymin, w, h]
#       'category_id': n
#        'id': dasfxx}
    
## data['images']:
#    {'file_name': 'xxx.jpg',
#     'height': n,
#     'width': n,
#     'id':xxx}
    
## data['categories']
#    {'name':'Sheep',
#     'id':1
#     }

categories =[{'name':'Sheep',
             'id':1
             }]

annotations = []
images = []

for d in data:
    if not d['Label'] == 'Skip' and d['Label'] :
        im_id = d['DataRow ID']
        images.append({'file_name':d['External ID'],
                       'height': 1024,
                       'width':1024,
                       'id': im_id})
        for i in range(len(d['Label']['Sheep'])):
            l = d['Label']['Sheep'][i]
            xys = np.array(list(map( lambda xy: [xy['x'],xy['y']], l['geometry'])))
            xmin = int(np.min(xys[:,0]))
            xmax = int(np.max(xys[:,0]))
            ymin = int(np.min(xys[:,1]))
            ymax = int(np.max(xys[:,1]))
            w = xmax-xmin
            h = ymax-ymin      
            
            if w >= 1 and h >=1:
                annotations.append({'image_id':im_id,
               'bbox': [xmin, ymin, w, h],
               'category_id': 1,
                'id': d['ID'] + str(i),
                'area':w*h,
                'segmentation':[],
                'iscrowd':0})
        
with open(path + 'coco_train.json', 'w') as fp:
    json.dump({'categories':categories, 'annotations':annotations, 'images':images}, fp)
