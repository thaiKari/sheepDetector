# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:43:39 2019

@author: karim
"""

import sys
import matplotlib.pyplot as plt

sys.path.insert(1, './Object-Detection-Metrics/')
from pascalvoc_funcs import get_precision_recals



iouThreshold = 0.5
gtFolder = './groundtruths/Val_201912'
#detFolder = './detections/val_20191101 '
detFolder = './detections/00_xlibra201912_epoch8_nosplit'

#metrics = get_precision_recals(iouThreshold, gtFolder, detFolder )[0]


iouThresholds =[ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
iouThresholds =[ 0.5 ]
precisions = []
recalls = []
APs = []

for T in iouThresholds:
    #print(T)
    metrics = get_precision_recals(T, gtFolder, detFolder )[0]
    precisions.append(metrics['precision'])
    recalls.append(metrics['recall'])
    APs.append(metrics['AP'])
    print('{0:.3f}'.format(metrics['AP']))


#plt.rcParams.update({'font.size': 14, 'font.family' : 'normal', 'font.weight': 'normal'})
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)
#plt.figure(figsize =(15,10))
##plt.plot(recalls, precisions)  
#for i in range(len(precisions)):
#    plt.plot(recalls[i], precisions[i], label= 'T: ' + str(iouThresholds[i]) + ', AP: ' + str(APs[i]))

#plt.title('Precision Recall curve for various IOU Thresholds')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.xlim([0.1, 1])
#plt.ylim([0.7, 1])
#plt.legend()

print('PRESISION {0:.3f}'.format( metrics['total TP'] / (metrics['total TP'] + metrics['total FP'])))
print('RECALL {0:.3f}'.format(  metrics['total TP'] /metrics['total positives']))



# C 0.05, 0.809 PRESISION 0.7754, RECALL 0.8254
# C 0.1, 0.797, PRESISION 0.84, RECALL 0.8115
# C 0.3, 0.767,PRESISION 0.923, RECALL 0.7783
# C 0.5, 0.731, PRESISION 0.9592, RECALL 0.7399650959860384

## 01_libra_split
# C 0.3 0.887, PRESISION 0.424, RECALL 0.956
# C 0.5 0.875, PRESISION 0.586, RECALL 0.934
# C 0.75, 0.850, PRESISION 0.738, RECALL 0.895
# C 0.9, 0.814, PRESISION 0.865, RECALL 0.852
# C 0.925, 0.792, PRESISION 0.898, RECALL 0.827
# C 0.93, 0.786, PRESISION 0.906, RECALL 0.820
# C 0.95, 0.740, PRESISION 0.925, RECALL 0.770

## 02_libra_split (with bg_only + more augmentation):0.935 mAP
# C 0.3 0.930, PRESISION 0.397, RECALL 0.976
# C 0.5 0.919, PRESISION 0.608, RECALL 0.955
# C 0.75 0.890, PRESISION 0.823, RECALL 0.916
# C 0.8 0.868, PRESISION 0.869, RECALL 0.890
# C 0.85 0.817, PRESISION 0.914, RECALL 0.831

## 04_libra_split: 0.910 mAP
# C0.85 0.891, PRESISION 0.695, RECALL 0.943
# C0.95 0.888, PRESISION 0.760, RECALL 0.925

## 00_libra20191212: 0.912 mAP
# C0.9 PRESISION 0.760, RECALL 0.925
# C0.95 PRESISION 0.886, RECALL 0.838
# C0.96 PRESISION 0.919, RECALL 0.788
# C0.97 PRESISION 0.933, RECALL 0.677

## 00_libra20191212_epoch8: 0.933 mAP
# C0.85, PRESISION 0.882, RECALL 0.875
# C0.875, PRESISION 0.904, RECALL 0.840



## 01_libra_nosplit
#C 0.5, 0.6940, PRESISION 0.849, RECALL 0.721
#C 0.75, 0.625, PRESISION 0.920, RECALL 0.644

## 02_libra_nosplit
# C 0.3 0.702, PRESISION 0.851, RECALL 0.728