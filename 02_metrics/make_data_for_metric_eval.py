# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:33:59 2019

@author: karim
"""
import numpy as np
import os

#gt_dst = './Object-Detection-Metrics/groundtruths/val_20191101'
#label_map_path = 'G:/SAU/Labeled/Val/00_labels.npy'
#label_map = np.load(label_map_path, allow_pickle=True).item()
#
#for k in label_map.keys():
#    labels = label_map[k]['labels']
#    string = ''
#    for l in labels:
#
#        s = 'sheep {} {} {} {} \n'.format(*l['geometry'][0], *l['geometry'][1])
#        print(s)
#        string = string + s
#    
#    with open( gt_dst + '/' + k[:-4] + ".txt", "w") as text_file:
#        text_file.write(string)
confidence_T = 0.0
     
pred_dst = 'C:/Users/karim/Projects/SAU/02_metrics/Object-Detection-Metrics/detections/00_xlibra201912_epoch8_nosplit'#_conf' + str(confidence_T)
label_map_path = 'G:/SAU/Labeled/Val/00_nosplit_libra_20191212_epoch_8.npy'
#label_map_path = 'G:/SAU/Labeled/Val/04_pred_split_libra_20191205_0_epoch_6.npy'
label_map = np.load(label_map_path, allow_pickle=True).item()
Val_ims = os.listdir('G:/SAU/Labeled/Val')


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