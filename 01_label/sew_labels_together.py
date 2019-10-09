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
import json


def get_im_details(full_id):
    split = full_id.split('.')
    data  = {}
    data['im_id'] = split[0]
    tmp, x, y = split[1].split('[')
    data['x']= int(x[:-1])
    data['y']= int(y[:-1])
    return data


#Sheep is too close to edge
def is_partial_sheep(xmin, xmax, ymin, ymax):
    D = 25
    
    if(xmax < D):
        return True
    if(ymax < D):
        return True
    if(dim - xmin < D):
        return True
    if(dim - ymin < D):
        return True
    
    return False

def iou(geom1, geom2):
    xmin1, ymin1 = geom1[0]
    xmax1, ymax1 = geom1[1]
    xmin2, ymin2 = geom2[0]
    xmax2, ymax2 = geom2[1]
    
    A1 = (xmax1 - xmin1)*(ymax1-ymin1)
    A2 = (xmax2 - xmin2)*(ymax2-ymin2)
    
    dx_inter = min(xmax1,xmax2) - max(xmin1, xmin2)
    dy_inter = min(ymax1, ymax2) - max(ymin1, ymin2)
    A_inter=0
    if (dx_inter > 0) and (dy_inter > 0 ):
        A_inter = dx_inter*dy_inter
    
    A_union = A1 + A2 - A_inter
    return A_inter / A_union
    


#Look for boxes with IOU > 0.55. replace with mean
def remove_duplicate_labels(labels):
    
    if(len(labels) <= 1):
        return labels
    
    ok_labels = []
    skip_indices = set()
    
    for i in range(len(labels) - 1):
        if not i in skip_indices:
            geom1 = labels[i]['geometry']
            duplicate_indices = [i]
            for j in range(i +1 , len(labels)):
                if not j in skip_indices:
                    geom2 = labels[j]['geometry']
                    if iou(geom1, geom2) > 0.5:

                        duplicate_indices.append(j)
                        skip_indices.update([i,j])

            if(len(duplicate_indices) > 1):
                geoms = list(map( lambda i: labels[i]['geometry'] , duplicate_indices))
                geoms = np.asarray(geoms)
                new_g = np.mean(geoms, axis = 0)
                labels[i]['geometry'] = new_g
                
            ok_labels.append(labels[i])
    
    if(not len(labels) - 1 in skip_indices):
        ok_labels.append(labels[len(labels) - 1])
        
    return ok_labels;
    

#groups labels from same image in dictionary
def build_label_map(labels):
    im_label_map  = {}
    for i in range(len(labels.index)):

        label = labels.iloc[i]
        label_meta  = get_im_details(label['External ID'])
        im_obj ={}
        
        if( label_meta['im_id'] in im_label_map):
            im_obj = im_label_map[label_meta['im_id'] ]
        #    print(label_meta['im_id'] )
        if(label['Label'] != 'Skip'):
            boxes = json.loads(label['Label'])
            for b in boxes['Sheep']:
                geom = list(b['geometry'])
                xs = list(map( lambda g: g['x'], geom))
                ys = list(map( lambda g: g['y'], geom))
                xmin = np.min(xs)
                xmax = np.max(xs)
                ymin = np.min(ys)
                ymax = np.max(ys)
                
                if(not is_partial_sheep(xmin, xmax, ymin, ymax )):
                    new_xmin = xmin - dim/2 + label_meta['x']
                    new_xmax = xmax - dim/2 + label_meta['x']
                    new_ymin = ymin - dim/2 + label_meta['y']
                    new_ymax = ymax - dim/2 + label_meta['y']
                    
                    label_obj = {'sheep_color':b['sheep_color'],
                                 'geometry': [ [new_xmin, new_ymin], [new_xmax, new_ymax]  ]}
                    if( 'labels' not in im_obj):
                        im_obj['labels'] = [label_obj]
                    else:
                        im_obj['labels'].append(label_obj)
                    
                else:
                    print('partial:', geom)
                
        im_label_map[label_meta['im_id'] ] = im_obj
    return im_label_map

dim = 1024;
labels = pd.read_csv("labels_split.csv")
im_label_map  = build_label_map(labels)

# count number of labels
n1 = 0
for k in im_label_map.keys(): 
    im_label_map[k]
    im_data = im_label_map[k]

    if 'labels' in im_data:
        for l in im_data['labels']:

            n1 = n1 + len(l['geometry'])

print(n1)

#remove_duplicates (overlaping rectangles):
for k in im_label_map.keys():  
    im_data = im_label_map[k]

    if 'labels' in im_data:
        im_data['labels'] = remove_duplicate_labels(im_data['labels'])




def label_dict_to_json(label_dict):
    
    def get_geometry(geom):
        minx, miny = geom[0]
        maxx, maxy = geom[1]
        return[{
                "x": minx,
                "y": miny
                },
                {
                "x": maxx,
                "y": miny
                },
                {
                "x": maxx,
                "y": maxy
                },
                {
                "x": minx,
                "y": maxy
                }
            ]
    
    def get_label(label):
        if('labels' in label):
            return {"Sheep": list(map( lambda l: {"sheep_color": l['sheep_color'],
                              "geometry":get_geometry(l['geometry'])}  , label['labels']))}
        
        
        else: return "Skip"
        
        
    data = list(map( lambda k: {"Label": get_label(label_dict[k]), "External ID": k + '.jpg'} , label_dict.keys()))
    return data
    

json_data = label_dict_to_json(im_label_map)
with open('full_label.json', 'w') as fp:
    json.dump(json_data, fp)

# count number of labels
n2 = 0
for k in im_label_map.keys(): 
    im_label_map[k]
    im_data = im_label_map[k]

    if 'labels' in im_data:
        for l in im_data['labels']:

            n2 = n2 + len(l['geometry'])

print(n2)


#visualize
im = cv2.imread('G:/SAU/Fra Jens/Datasett1/DJI_0611 (2).jpg')

# Create figure and axes
fig,ax = plt.subplots(1, figsize=(20, 20))

# Display the image
ax.imshow(im)

for l in im_label_map['DJI_0611 (2)']['labels']:
    geom = l['geometry']
    
    xmin, ymin = geom[0]
    xmax, ymax = geom[1]
    w = xmax - xmin
    h = ymax - ymin
    rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)






    
    

