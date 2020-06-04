# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:24:05 2019

@author: karim
"""
import numpy as np
import cv2
from transformations import transform_vis_pt_list_to_IR_coordinate_system
import matplotlib.pyplot as plt
import matplotlib.patches as patches

im_label_map = np.load('G:/SAU/Labeled/00_annotations/all_labelled_20205013.npy',  allow_pickle=True).item() 

new_im_label_map = {}
##TRANSFORM ALL LABELS TO IR COORDSYSTEM
for k in im_label_map.keys():
    new_im_label_map[k] = {}
    if 'labels' in im_label_map[k]:
        labels = []
        for l in im_label_map[k]['labels']:
            xmin, ymin = l['geometry'][0]
            xmax, ymax =  l['geometry'][1]
            corners = [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]
            transformed_corners = transform_vis_pt_list_to_IR_coordinate_system(np.asarray(corners, dtype = np.uint))
            xmin = np.min(transformed_corners[:,0])
            xmax = np.max(transformed_corners[:,0])
            ymin = np.min(transformed_corners[:,1])
            ymax = np.max(transformed_corners[:,1])
            transformed_pts = [[ int(xmin), int(ymin)], [int(xmax), int(ymax)]]

            w = xmax - xmin
            h = ymax - ymin
            
            #Label on visual image is outside scope of IR image
            if( w > 0 and h >0):
                l['geometry'] = transformed_pts
                labels.append(l)

        new_im_label_map[k] = {'labels': labels}
        
np.save('G:/SAU/Labeled/00_annotations/all_labelled_20205013_infrared.npy', new_im_label_map)

#visualize
im = cv2.imread('G:/SAU/Labeled/00_Test2020/00_infrared/may20_100MEDIA_DJI_0518.JPG')

# Create figure and axes
fig,ax = plt.subplots(1, figsize=(10, 20))

# Display the image
ax.imshow(im)

for l in new_im_label_map['may20_100MEDIA_DJI_0518.JPG']['labels']:
    geom = l['geometry']
    
    xmin, ymin = geom[0]
    xmax, ymax = geom[1]
    w = xmax - xmin
    h = ymax - ymin
    rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
#
#
