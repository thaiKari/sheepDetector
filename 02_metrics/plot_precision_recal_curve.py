# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:43:39 2019

@author: karim
"""

import sys
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import numpy as np

sys.path.insert(1, './Object-Detection-Metrics/')
from pascalvoc_funcs import get_precision_recals


iou_threshold = 0.5

gtFolder = './groundtruths/val2020'
#detFolder = './detections/val_20191101 '
detFolder = './detections/03_libra20200110_2_epoch4_pred'

print(0)
metrics = get_precision_recals(iou_threshold, gtFolder, detFolder )[0]
print('{0:.3f}'.format(metrics['AP']))





#precisions = []
#recalls = []
#APs = [metrics['AP']]
#
#resolutions = [50, 75, 100, 125, 150, 175, 200, 250, 300]#, 350]
#
#for resolution in resolutions:
#    print(resolution)
#    gt_Folder = 'groundtruths/Train_201912_sheep_resolution_'+ str(resolution)+'+'
#    detFolder = 'detections/Train_201912_sheep_resolution_'+ str(resolution)+'+'
#    
#    metrics = get_precision_recals(iou_threshold, gt_Folder, detFolder )[0]
##    precisions.append(metrics['precision'])
##    recalls.append(metrics['recall'])
#    APs.append(metrics['AP'])
#    print('{0:.3f}'.format(metrics['AP']))
#    
#plt.figure()
#plt.plot(APs)


#plt.rcParams.update({'font.size': 20})

#plt.figure(figsize=(15, 10))
#plt.title('Precision x Recall curve /n Class: sheep AP: 93.21 %', fontsize=24)
#plt.box(False)
#plt.ylabel('Precision')
#plt.xlabel('Recall')
#plt.grid()
#plt.plot(metrics['recall'], metrics['precision'], label= 'AP: ' + str(round(metrics['AP'], 3)))
##plt.legend()

#FP_dets = []
#labels = np.load('G:/SAU/Labeled/Val/00_labels.npy', allow_pickle=True).item()
#
#for i in range( len(metrics['detections'])):
#    if metrics['FP'][i]:
#        FP_dets.append(metrics['detections'][i])
#        try:
#            b=100
#            fig,ax = plt.subplots(1, figsize=(20,20))
#            im_name =  metrics['detections'][i][0] + '.JPG'
#            im = cv2.imread('G:/SAU/Labeled/Val/' + im_name)
#            minx, miny, maxx, maxy = metrics['detections'][i][3]
##            im = im[int(miny)-b:int(maxy)+b, int(minx)-b:int(maxx)+b]
#            
##            rect = patches.Rectangle((b,b),int(maxx)-int(minx),int(maxy)-int(miny),linewidth=1,edgecolor='red',facecolor='none')      
#            ax.imshow(im)
#            
#            label = labels[im_name]['labels']
#            
#            for l in label:
#                geom = l['geometry']
#        
#                xmin, ymin = geom[0]
#                xmax, ymax = geom[1]
#                w = xmax - xmin
#                h = ymax - ymin
#                rect = patches.Rectangle((xmin,ymin),w,h,linewidth=2,edgecolor='#92d050',facecolor='none')      
#                ax.add_patch(rect)
#            
#            rect = patches.Rectangle((int(minx),int(miny)),int(maxx)-int(minx),int(maxy)-int(miny),linewidth=2,edgecolor='red',facecolor='none')      
#            ax.add_patch(rect)
#            
#            
#            plt.savefig('C:/Users/karim/OneDrive - NTNU/Documents/00 I og IKT/36 Prosjektoppgave/Figures/FPS/' +metrics['detections'][i][0])
#
#            
#        except:
#            print(metrics['detections'][i])
        



#plt.rcParams.update({'font.size': 14, 'font.family' : 'normal', 'font.weight': 'normal'})
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)
#plt.figure(figsize =(15,10))
#plt.plot(recalls, precisions)  
#for i in range(len(precisions)):
#    plt.plot(recalls[i], precisions[i], label= 'T: ' + str(iouThresholds[i]) + ', AP: ' + str(APs[i]))

#plt.title('Precision Recall curve for various IOU Thresholds')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.xlim([0.1, 1])
#plt.ylim([0.7, 1])
#plt.legend()
#
#print('PRESISION {0:.3f}'.format( metrics['total TP'] / (metrics['total TP'] + metrics['total FP'])))
#print('RECALL {0:.3f}'.format(  metrics['total TP'] /metrics['total positives']))



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
#(no_split: mAP 0.764)

## 00_libra201913_epoch2: 0.932 mAP
# C0.5 PRESISION 0.572, RECALL 0.958
# C0.75 PRESISION 0.800, RECALL 0.917
# C0.85 PRESISION 0.891, RECALL 0.871
# C0.86 PRESISION 0.907, RECALL 0.861


## 01_libra_nosplit
#C 0.5, 0.6940, PRESISION 0.849, RECALL 0.721
#C 0.75, 0.625, PRESISION 0.920, RECALL 0.644

## 02_libra_nosplit
# C 0.3 0.702, PRESISION 0.851, RECALL 0.728