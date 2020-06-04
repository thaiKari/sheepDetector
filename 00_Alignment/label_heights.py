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

im_path ='G:/SAU/Labeled/AUG_labeled_has_IR'
annotation_path = 'G:/SAU/Labeled/00_annotations'
ims = os.listdir(im_path)
#
#sheep_resolutions = np.load(os.path.join(im_path, 'sheep_resolution.npy'), allow_pickle = True).item()
#
#for k in sheep_resolutions.keys():
#    resolution = sheep_resolutions[k]['median_diagonal']
#    if resolution <=300 and resolution>50:
#        copyfile(im_path + '/' + k, im_path_new + '/' + k)

# avg sheep diameter
labels = np.load( os.path.join(annotation_path, '00_all_labels_20200107.npy'), allow_pickle = True).item()
sheep_diam={}
xS = []
Small = []
Medium = []
Large = []

for k in ims:
    if '.JPG' in k and not '(2)' in k:
#        print(k)
        item = labels[k]
        diams = []    
        for l in item['labels']:
            minx, miny = l['geometry'][0]
            maxx, maxy = l['geometry'][1]
            w = maxx - minx
            h = maxy - miny
            diam = math.sqrt( w**2 + h**2)
            diams.append(diam)
        median_diam = np.median(diams)
        
        if median_diam >300 or median_diam < 50:
            os.remove(os.path.join(im_path, k))
            os.remove(os.path.join(im_path, k[:-4] + ' (2).JPG'))
        
        
#        if median_diam < 50:
#            xS.append(k)
#        elif median_diam < 100:
#            Small.append(k)
#        elif median_diam < 300:
#            Medium.append(k)
#        else:
#            Large.append(k)
#        
#        if(median_diam > 149 and median_diam < 151):
#            print(k, median_diam)
#        sheep_diam[k]= {
#                'diagonals':diams,
#                'median_diagonal': median_diam
#                }
#
#print('xS: ', len(xS)) 
#print('Small: ', len(Small))
#print('Medium: ', len(Medium))
#print('Large: ', len(Large))
#
#
#np.save(os.path.join(im_path, 'sheep_resolution.npy'), sheep_diam)
#all_median_diagonals = list(map( lambda item: item['median_diagonal'], sheep_diam.values()))
#all_median_diagonals = list(sorted(all_median_diagonals))
#np.save(os.path.join(im_path, 'xS.npy'), xS)
#np.save(os.path.join(im_path, 'Small.npy'), Small)
#np.save(os.path.join(im_path, 'Medium.npy'), Medium)
#np.save(os.path.join(im_path, 'Large.npy'), Large)



#l_0_50 = 0
#l_50_100 = 0
#l_100_150 = 0
#l_150_200 = 0
#l_200_250 = 0
#l_250_300 = 0
#l_300_350 = 0
#l_350_400 = 0
#l_400_inf = 0

#bucket_vals = [150,  350, 100000]
#bucket_count = np.zeros(len(bucket_vals))
#
#for diag in all_median_diagonals:
#    for i in range(len(bucket_vals)):
#        if diag < bucket_vals[i]:
#            bucket_count[i] = bucket_count[i] +1
#            break
#            
#        
#
#


    
        

### Use log data...
#ims = list(filter( lambda n: '.JPG' in n ,ims ))
#
#heights_barom = {}
#
#prev_im = ims[0]
#
#for im in ims:
#    if not im in heights.keys():
#        try:
#            h = get_metadata( os.path.join(im_path, im ) )['height_MGL']
#            heights[im] = h
#            print(h)
#        except:
#            print('error', im)
#            heights[im] = heights[prev_im]
#        
#        prev_im = im
#            
#    
#
#np.save( os.path.join( im_path, '00_heights'), heights )