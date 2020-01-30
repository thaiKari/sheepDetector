# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:33:59 2019

@author: karim
"""
import numpy as np
import os


label_map = np.load('G:/SAU/Labeled/Val2020/4d_vis_plus_ir/00_labels.npy', allow_pickle=True).item()
#sheep_resolution = np.load(im_path + '/sheep_resolution.npy', allow_pickle=True).item()
#labels = list(np.load('G:/SAU/Labeled/Train/00_labels.npy'))


gt_dst = './Object-Detection-Metrics/groundtruths/Val2020'
for k in label_map.keys():
    labels = label_map[k]['labels']
    string = ''

    for l in labels:
#        if(l['sheep_color'] == 'brown'):
#            print(l)    
        s = 'sheep {} {} {} {} \n'.format(*l['geometry'][0], *l['geometry'][1])
        print(s)
        string = string + s

    with open( gt_dst + '/' + k[:-4] + ".txt", "w") as text_file:
        text_file.write(string)
        
confidence_T = 0.0
##  
#group = list(np.load('G:/SAU/Labeled/Val/00_Medium.npy'))   
#pred_dst = 'C:/Users/karim/Projects/SAU/02_metrics/Object-Detection-Metrics/detections/val_M'#conf' + str(confidence_T)
#pth = 'G:/SAU/Labeled/Train/pred_201912'
#label_map_path = os.listdir(pth)
#label_map = {}
#
#for l in label_map_path:
#    if '.npy' in l:
#        label_map_n = np.load( os.path.join(pth,l), allow_pickle=True).item()
#        label_map = {**label_map, **label_map_n}
#Val_ims = os.listdir('G:/SAU/Labeled/Val')
#


label_map_path = 'G:/SAU/Labeled/Val2020/4d_vis_plus_ir/03_libra20200110_2_epoch4_pred.npy'
label_map = np.load( label_map_path, allow_pickle=True).item()


pred_dst = './Object-Detection-Metrics/detections/03_libra20200110_2_epoch4_pred'
for k in label_map.keys():
    labels = label_map[k]['labels']
    string = ''
    
    for l in labels:
        if(l['confidence'] > confidence_T):
            s = 'sheep {} {} {} {} {} \n'.format( l['confidence'], *l['geometry'][0], *l['geometry'][1])
            print(s)
            string = string + s
    
    with open( pred_dst + '/' + k[:-4] + ".txt", "w") as text_file:
        text_file.write(string)
## 
#0.05: (same as 0.1)
# 'total positives': 573,
# 'total TP': 465.0,
# 'total FP': 87.0}
#
#PRESISION
#metrics['total TP'] /metrics['total TP'] + metrics['total FP']
#Out[28]: 88.0
#
#RECALL
#metrics['total TP'] /metrics['total positives']
#Out[29]: 0.8115183246073299