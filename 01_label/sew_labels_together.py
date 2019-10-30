# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:18:58 2019

@author: karim
"""

import pandas as pd 
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from label_utils import get_im_details, is_partial_sheep, is_edge_sheep, iou, remove_duplicate_labels, build_label_map, count_labels, label_dict_to_json, show_im_with_boxes



dim = 1024;
labels = pd.read_csv("labels_split.csv")
im_label_map, im_label_map_needs_check  = build_label_map(labels, dim)

# count number of labels
n1 = count_labels(im_label_map)
print(n1)

##CHECK THE UNSURE ONES
new_name_map = np.load('new_name_map.npy',  allow_pickle=True).item()
#No need to check those labels that have been checked once before
keep_list = list(np.load('keep_list.npy'))
discard_list = list(np.load('discard_list.npy'))
    
im_dir = 'G:/SAU/Fra Jens/Datasett1 - Copy - Copy/'
for k in im_label_map_needs_check.keys():
    if im_label_map_needs_check[k]:
        
        for l_check in im_label_map_needs_check[k]['labels']:
            label_str = json.dumps(l_check)
            
            if label_str in keep_list:
                im_label_map[k]['labels'].append(l_check)
                continue
            if label_str in discard_list:
                continue
            
            fig,ax = plt.subplots(1, figsize=(20, 20))
            geom = l_check['geometry']
            xmin, ymin = geom[0]
            xmax, ymax = geom[1]
            w = xmax - xmin
            h = ymax - ymin
            rect_check = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect_check)
            im = cv2.imread(im_dir + new_name_map[ k +'.JPG' ])  
            
            ax.imshow(im)
            
            if im_label_map[k]['labels'] is not None:
                for l in im_label_map[k]['labels']:
                    geom = l['geometry']
                    
                    xmin, ymin = geom[0]
                    xmax, ymax = geom[1]
                    w = xmax - xmin
                    h = ymax - ymin
                    rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='r',facecolor='none')
                    ax.add_patch(rect)
                
                plt.show()
                keep = input('keep ? [y/n]: ')
                if(keep == 'y'):
                    im_label_map[k]['labels'].append(l_check)
                    keep_list.append(label_str)
                if(keep =='n'):
                    discard_list.append(label_str)
                    
np.save('keep_list.npy', keep_list)
np.save('discard_list.npy', discard_list)

#remove_duplicates (overlaping rectangles):
for k in im_label_map.keys():  
    im_data = im_label_map[k]

    if 'labels' in im_data:
        im_data['labels'] = remove_duplicate_labels(im_data['labels'])

json_data = label_dict_to_json(im_label_map)
with open('full_label.json', 'w') as fp:
    json.dump(json_data, fp)

# count number of labels
n2 = count_labels(im_label_map)
print(n2)

#make map with new better names
im_label_map_new = {}

for k in im_label_map.keys():
    im_label_map_new[new_name_map[k +'.JPG'] ] = im_label_map[k]

np.save('im_label_map_new_names.npy', im_label_map_new)


#visualize
im_name = 'aug19_102MEDIA_DJI_0144.JPG'
im_dir = 'G:/SAU/Fra Jens/Datasett1 - Copy - Copy/'
show_im_with_boxes(im_name, im_dir, im_label_map_new)





    
    

