# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:32:05 2019

@author: karim
"""

#from utils import get_metadata
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from shutil import copyfile

annotation_path = 'G:/SAU/Labeled/00_Test2020/01_infrared_align.npy'
annotation_path = 'G:/SAU/Labeled/00_annotations/'

# avg sheep diameter
labels_train = np.load( os.path.join(annotation_path, 'train.npy'), allow_pickle = True).item()
labels_val = np.load( os.path.join(annotation_path, 'val.npy'), allow_pickle = True).item()
labels_test = np.load( os.path.join(annotation_path, 'test.npy'), allow_pickle = True).item()
labels = {**labels_train, **labels_val, **labels_test}
#labels = labels_test
#labels = np.load(annotation_path, allow_pickle=True).item()

#annotation_path = 'G:/SAU/Labeled/00_annotations/02_old'
#labels = np.load( os.path.join(annotation_path, 'all_labelled_20205013.npy'), allow_pickle = True).item()

august_labels = {}
septeber_labels = {}
october_labels = {}
klaebu_labels = {}
orkanger_labels = {}

for key in labels.keys():
    label = labels[key]
    if 'aug' in key:
        august_labels[key] = label
    if 'sep' in key:
        septeber_labels[key] = label
    if 'oct' in key:
        october_labels[key] = label
    if 'may' in key:
        im_num = int(key[-8:-4])
        if im_num < 100 or im_num > 689:
            orkanger_labels[key] = label
        else:
            klaebu_labels[key] = label
            


print('num_ims august: ', len(august_labels.keys()))
print('num_ims septeber: ', len(septeber_labels.keys()))
print('num_ims october: ', len(october_labels.keys()))
print('num_ims klaebu: ', len(klaebu_labels.keys()))
print('num_ims orkanger: ', len(orkanger_labels.keys()))
print('num_ims total: ', len(labels.keys()))



def count_sheep_size(labels):
    lamb_count = 0
    xs_count = 0
    s_count = 0
    m_count = 0
    l_count = 0
    xl_count = 0
    
    for key in labels.keys():
    
        label = labels[key]
        diams = []

        if 'labels' in label.keys():
            for l in label['labels']:
                minx, miny = l['geometry'][0]
                maxx, maxy = l['geometry'][1]
                w = maxx - minx
                h = maxy - miny
                diam = math.sqrt( w**2 + h**2)
                
                if 'is_lamb' in l.keys():
                    
                    if l['is_lamb']:
                        lamb_count = lamb_count +1
                    else:
                        diams.append(diam)
                else:
                    diams.append(diam)
            
#            if len(diams) > 0:
#                median_diam = np.median(diams)    
#                
#                if median_diam < 50:
#                    xs_count = xs_count + 1
#                elif median_diam < 100:                    
#                    s_count = s_count + 1
#                elif median_diam < 300:
#                    m_count = m_count + 1
#                else:
#                    print(key, median_diam)
#                    l_count = l_count + 1
                    
            if len(diams) > 0:
                median_diam = np.median(diams)    
                
                if median_diam < 50:
                    xs_count = xs_count + 1
                elif median_diam < 100:
                    s_count = s_count + 1
                elif median_diam < 150: 
                    
                    m_count = m_count + 1
                elif median_diam < 300:
                    l_count = l_count + 1
#                    if 'may' in key:
#                        print(key)
                else:
                    xl_count = xl_count + 1
#                    print(key, median_diam)
#                    l_count = l_count + 1
        
    
    print('xS: ', xs_count) 
    print('Small: ', s_count)
    print('Medium: ', m_count)
    print('Large: ', l_count)
    print('xL: ', xl_count)
    print('lamb_count', lamb_count)
    
    print('{}&{}&{}&{}&{}'.format(xs_count, s_count, m_count, l_count, xl_count))
#    print('Mean_diag', np.mean(diams))
#    print('Median diag', np.median(diams))

print()
print('=========Image Median Sheep Size ==========')
print()
print('AUGUST')
count_sheep_size(august_labels)
print()
print('SEPTEMBER')
count_sheep_size(septeber_labels)
print()
print('OCTOBER')
count_sheep_size(october_labels)
print()
print('KLAEBU')
count_sheep_size(klaebu_labels)
print()
print('ORKANGER')
count_sheep_size(orkanger_labels)
print()
print('ALL')
count_sheep_size(labels)
    
## Sheep color

def count_sheep_color(labels):
    white_count = 0
    grey_count = 0
    black_count = 0
    brown_count = 0
    
    for key in labels.keys():
    
        label = labels[key]
        if 'labels' in label.keys():
            for l in label['labels']:
                if l['sheep_color']=='white':
                    white_count = white_count +1
                if l['sheep_color']=='grey':
                    grey_count = grey_count +1
                if l['sheep_color']=='black':
                    black_count = black_count +1
                if l['sheep_color']=='brown':
                    brown_count = brown_count +1

                
    print('white_count: ', white_count) 
    print('grey_count: ', grey_count)
    print('black_count: ', black_count)
    print('brown_count: ', brown_count)

print()    
print('=========Sheep Colour ==========')
print()
print('AUGUST')
count_sheep_color(august_labels)
print()
print('SEPTEMBER')
count_sheep_color(septeber_labels)
print()
print('OCTOBER')
count_sheep_color(october_labels)
print()
print('KLAEBU')
count_sheep_color(klaebu_labels)
print()
print('ORKANGER')
count_sheep_color(orkanger_labels)
print()
print('ALL')
count_sheep_color(labels)