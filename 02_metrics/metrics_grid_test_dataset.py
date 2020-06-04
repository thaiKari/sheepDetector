# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:39:12 2020

@author: karim
"""
import os
import numpy as np
import sys
import cv2
import math
import matplotlib.pyplot as plt
import json
import matplotlib as mpl

sys.path.append(os.path.abspath('../01_label'))

from label_utils import show_im_with_bbox_and_grid, show_im_with_bbox_and_pred_grid
from sklearn.metrics import average_precision_score, precision_recall_curve

im_shape = (2400, 3200)
grid_shape = (6, 8)

#All preds on one line
def fix_file(pred_root_path, pred_file_path, grid_shape=(6,8)):
    with open(os.path.join(pred_root_path, pred_file_path)) as f:
        lines = f.readlines()
        
    s=''
    
    for line in lines:
        line=line.strip('\n')
        if '.JPG' in line:
            s = s + '\n' + line
        else:
            s = s + line
    
    with open(os.path.join(pred_root_path, pred_file_path), "w") as file:
                file.write(s)

def read_pred_file(pred_root_path, pred_file_path, grid_shape=(6,8)):
    pred_grid_map = {}
    
    with open(os.path.join(pred_root_path, pred_file_path)) as f:
        content = f.readlines()
        

    for item in content:
        item_list = item.strip().split()       
        
        if len(item_list)>0:
            if '.JPG' in item_list[0]:
                pred_grid_map[item_list[0]] = np.array(item_list[1:]).reshape(grid_shape).astype(np.float)

    return pred_grid_map



def intersection_degree(bbox, box):
    xmin1, ymin1 = bbox[0]
    xmax1, ymax1 = bbox[1]
    xmin2, ymin2 = box[0]
    xmax2, ymax2 = box[1]
    
    A1 = (xmax1 - xmin1)*(ymax1-ymin1)
    A2 = (xmax2 - xmin2)*(ymax2-ymin2)
    
    dx_inter = min(xmax1,xmax2) - max(xmin1, xmin2)
    dy_inter = min(ymax1, ymax2) - max(ymin1, ymin2)
    A_inter=0
    if (dx_inter > 0) and (dy_inter > 0 ):
        A_inter = dx_inter*dy_inter
        
    return A_inter / A1


def sheep_diag(bbox):
    minx, miny = bbox[0]
    maxx, maxy = bbox[1]
    w = maxx - minx
    h = maxy - miny
    return math.sqrt( w**2 + h**2)
    

'''
Returns value 0, 1 or -1
0: No sheep in grid
1: Sheep in grid
-1: only small part of sheep in grid. Ignore for training and evaluation
'''
def calculate_grid_value(grid_xmin, grid_ymin, grid_xmax, grid_ymax, labels):

    grid_geom = [ [grid_xmin, grid_ymin ], [grid_xmax, grid_ymax ] ]
    
    has_partial_sheep = False
    
    for minx, miny, w, h in labels:
        label_geom = [[ minx, miny ], [ minx + w, miny + w ]]
        
        label_intersection_degree = intersection_degree(label_geom, grid_geom)
        
        if label_intersection_degree > 0.2:
            return 1
        elif label_intersection_degree > 0:
            has_partial_sheep = True
        
    if has_partial_sheep:
        return -1 #Ignore this grid when calculating loss and precision recall
    return 0

def get_grid(bboxes, im_shape, grid_shape):

    grid_h = math.floor(im_shape[0]/ grid_shape[0])
    grid_w = math.floor(im_shape[1]/ grid_shape[1])
    
    grid = np.zeros(grid_shape)
    
    for x in range(grid_shape[1]):
        for y in range(grid_shape[0]):
            #if grid_cell_has_label(x*grid_w, y*grid_h, (x+1)*grid_w,(y+1)*grid_h, bboxes ):
            grid[y,x] = calculate_grid_value(x*grid_w, y*grid_h, (x+1)*grid_w,(y+1)*grid_h, bboxes )
    
    return grid

def label_xyxy_to_xywh(label):
    xmin, ymin = label[0]
    xmax, ymax = label[1]
    w = xmax - xmin
    h = ymax - ymin
    
    return [xmin, ymin,w, h]

def count_TPs_FPs_TNs_FNs(all_pred, all_gt, T=0.5):

    TPs  = 0
    FPs = 0
    TNs = 0
    FNs = 0


    for i in range(len(all_gt)):
        pred = all_pred[i]
        if pred >= T:
            pred = 1
        else:
            pred = 0
        gt = all_gt[i]

        if pred == gt:
            if pred == 1:
                TPs = TPs +1

            if pred == 0:
                TNs = TNs +1
        else:
            if pred ==1:
                FPs = FPs +1
            if pred == 0:
                FNs = FNs +1

    precision = TPs / (TPs + FPs)
    recall = TPs / (TPs + FNs)
        
    return {
        'TPs': TPs,
        'FPs': FPs,
        'TNs': TNs,
        'FNs': FNs,
        'precision': precision,
        'recall': recall,
    }

root_path = 'G:/SAU/Labeled/00_annotations'
im_path =  'G:/SAU/Labeled/00_Test2020' 
pred_file_path = 'G:/SAU/Labeled/Predictions2020/test_predictions_ensemble_20200416_2142.txt'


#fix_file(root_path, pred_file_path, grid_shape=(6,8))
pred_grid_map = read_pred_file('', pred_file_path, grid_shape=(6,8))
bbox_map = np.load(os.path.join(root_path, 'T2_aligned.npy'), allow_pickle = True).item()
#with open(os.path.join(root_path, 'ir_aligned_labels_detailed.json'), 'r') as fp:
#    bbox_map = json.load(fp)

#key = 'aug19_103MEDIA_DJI_0099.JPG'
Threshold = 0.5


#CALCULATE RECALL. SHEEP IS DETECTED IF BBOX OVERLAPS WITH ANY OF THE PREDICTED GRID CELLS.

TPs = 0 #sheep found
TPs_white = 0
TPs_grey = 0
TPs_black = 0
TPs_brown = 0
TPs_adult = 0
TPs_lamb = 0
TPs_sep = 0
FNs = 0 #sheep not found
FNs_white = 0
FNs_grey = 0
FNs_black = 0
FNs_brown = 0
FNs_adult = 0
FNs_lamb = 0


all_gt = []
all_pred = []

all_gt_s = []
all_gt_m = []
all_gt_l = []

all_pred_s = []
all_pred_m = []
all_pred_l = []


for key in bbox_map.keys():
    if '.JPG' in key:
#        im_rgb = cv2.imread( os.path.join(root_path, '01_rgb_align', key ))
#        im_rgb=cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
#        plt.figure()
#        plt.axis('off')
#        plt.imshow(im_rgb)
#        plt.draw()
#        plt.show()
        
#        print(key)
        im = cv2.imread( os.path.join(im_path, '01_rgb_align', key ))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        labels = bbox_map[key]['labels']
        labels_xywh = list(map( lambda l: label_xyxy_to_xywh(l['geometry']), labels))
        
        pred_grid = pred_grid_map[key]
        gt_grid = get_grid(labels_xywh, im_shape, grid_shape)
        
        all_pred = [*all_pred, *list(pred_grid.flatten())]
        all_gt = [*all_gt, *list(gt_grid.flatten())]


        sheep_sizes = []
        #Check if each sheep was found
        for l in labels:

            l_xywh = label_xyxy_to_xywh(l['geometry'])
            l_color = l['sheep_color']
            
            if 'is_lamb' in l.keys():
                if not l['is_lamb']:
                    sheep_sizes.append(sheep_diag(l['geometry']))
            else:
                sheep_sizes.append(sheep_diag(l['geometry']))
            
    
            #if any of the overlapping grids is predicted positive
            overlapping_gridcells = get_grid([l_xywh],im_shape, grid_shape )
            sheep_found = False
            
            for x in range(grid_shape[1]):
                for y in range(grid_shape[0]):
                    if overlapping_gridcells[y,x] != 0 and pred_grid[y,x] > Threshold:
                        sheep_found= True
                        break
                if sheep_found:
                    break
            
            if sheep_found:
                TPs = TPs +1
                
                TPs_white = TPs_white + int(l_color == 'white')
                TPs_grey =  TPs_grey + int(l_color == 'grey')
                TPs_black = TPs_black + int(l_color == 'black')
                TPs_brown = TPs_brown + int(l_color == 'brown')
                if 'is_lamb' in l.keys():
                    TPs_adult = TPs_adult + int(not  l['is_lamb'])
                    TPs_lamb = TPs_lamb + int( l['is_lamb'])


            else:
                FNs = FNs +1    
                
                FNs_white = FNs_white + int(l_color == 'white')
                FNs_grey =  FNs_grey + int(l_color == 'grey')
                FNs_black = FNs_black + int(l_color == 'black')
                FNs_brown = FNs_brown + int(l_color == 'brown')
                if 'is_lamb' in l.keys():
                    FNs_adult = FNs_adult + int(not  l['is_lamb'])
                    FNs_lamb = FNs_lamb + int( l['is_lamb'])
        
        if(len(sheep_sizes) > 0):
            median_diam = np.median(sheep_sizes)    
                    
#            if median_diam < 100:
#                all_pred_s = [*all_pred_s , *list(pred_grid.flatten())]
#                all_gt_s  = [*all_gt_s , *list(gt_grid.flatten())]
#            else:
#                all_pred_m = [*all_pred_m , *list(pred_grid.flatten())]
#                all_gt_m  = [*all_gt_m , *list(gt_grid.flatten())]

            
            if median_diam < 100:
                all_pred_s = [*all_pred_s , *list(pred_grid.flatten())]
                all_gt_s  = [*all_gt_s , *list(gt_grid.flatten())]
            elif  median_diam < 150:
                all_pred_m = [*all_pred_m , *list(pred_grid.flatten())]
                all_gt_m  = [*all_gt_m , *list(gt_grid.flatten())]
            else:
                all_pred_l = [*all_pred_l , *list(pred_grid.flatten())]
                all_gt_l  = [*all_gt_l , *list(gt_grid.flatten())]

        
        #show_im_with_bbox_and_pred_grid(im, labels, pred_grid, Threshold, bbox_edge_color='red', save_to_filename='G:/SAU/Labeled/Predictions2020/Images_ResNeXt50/Fusion/' + key)
    #show_im_with_bbox_and_pred_grid(im, labels, gt_grid, Threshold, bbox_edge_color='red')

print()
print('Threshold', Threshold)
print()

print('========== SHEEP RECALL ========')
print('TPs', TPs)    
print('FNs', FNs)
print('Recall Sheep', TPs/(TPs + FNs))
print('Recall white', (TPs_white/(TPs_white + FNs_white)) if (TPs_white + FNs_white) > 0 else '-'  )
print('Recall grey', (TPs_grey/(TPs_grey + FNs_grey)) if (TPs_grey + FNs_grey) > 0 else '-' )
print('Recall black', (TPs_black/(TPs_black + FNs_black)) if (TPs_black + FNs_black) > 0 else '-'  )
print('Recall brown', (TPs_brown/(TPs_brown + FNs_brown)) if (TPs_brown + FNs_brown) > 0 else '-' )
print('Recall adult', (TPs_adult/(TPs_adult + FNs_adult)) if (TPs_adult + FNs_adult) > 0 else '-' )
print('Recall lamb', (TPs_lamb/(TPs_lamb + FNs_lamb)) if (TPs_lamb + FNs_lamb) > 0 else '-' )
print()
print(str(TPs/(TPs + FNs)) + ' ' +
      str((TPs_white/(TPs_white + FNs_white)) if (TPs_white + FNs_white) > 0 else '-' ) + ' ' +
      str((TPs_grey/(TPs_grey + FNs_grey)) if (TPs_grey + FNs_grey) > 0 else '-' ) + ' ' +
      str((TPs_black/(TPs_black + FNs_black)) if (TPs_black + FNs_black) > 0 else '-' ) + ' ' +
      str((TPs_brown/(TPs_brown + FNs_brown)) if (TPs_brown + FNs_brown) > 0 else '-')  + ' ' + 
      str( (TPs_adult/(TPs_adult + FNs_adult)) if (TPs_adult + FNs_adult) > 0 else '-')  + ' ' + 
      str((TPs_lamb/(TPs_lamb + FNs_lamb)) if (TPs_lamb + FNs_lamb) > 0 else '-')
      )
print()
print('========== GRID STATS ========')


def filter_fuzzy_grid_cells(gt, pred):
    mask = np.array(gt)!= -1
    gt_filtered = np.array(gt.copy())[mask]
    pred_filtered = np.array(pred.copy())[mask]
    
    return gt_filtered, pred_filtered

gt_s_filtered, pred_s_filtered = filter_fuzzy_grid_cells(all_gt_s, all_pred_s)
gt_m_filtered, pred_m_filtered = filter_fuzzy_grid_cells(all_gt_m, all_pred_m)
gt_l_filtered, pred_l_filtered = filter_fuzzy_grid_cells(all_gt_l, all_pred_l)

print('AP S: {0:0.3f}'.format(
average_precision_score(np.array(gt_s_filtered).astype(int), pred_s_filtered)))
print('AP M: {0:0.3f}'.format(
average_precision_score(np.array(gt_m_filtered).astype(int), pred_m_filtered)))
print('AP L: {0:0.3f}'.format(
average_precision_score(np.array(gt_l_filtered).astype(int), pred_l_filtered)))
print()

all_gt_filtered, all_pred_filtered = filter_fuzzy_grid_cells(all_gt, all_pred)
                                                         
res = count_TPs_FPs_TNs_FNs(all_pred_filtered, all_gt_filtered, T = Threshold)
print('Precision Grid: ', res['precision'] )
print('Recall Grid: ', res['recall'] )



precision, recall, _ = precision_recall_curve(np.array(all_gt_filtered).astype(int), all_pred_filtered)


mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False
plt.figure(figsize=(8.4,5.6))
plt.grid()
plt.xlabel('Recall', fontsize=14, alpha=0.6)
plt.ylabel('Precision', fontsize=14, alpha=0.6)

plt.plot(recall, precision)

##TODO: results by month
#
average_precision_filtered = average_precision_score(np.array(all_gt_filtered).astype(int), all_pred_filtered)
print('Average precision-recall score FILTERED: {0:0.3f}'.format(
average_precision_filtered))

print()
print(
      str(average_precision_filtered) + ' ' +
      str(average_precision_score(np.array(gt_s_filtered).astype(int), pred_s_filtered)) + ' ' +
      str(average_precision_score(np.array(gt_m_filtered).astype(int), pred_m_filtered)) + ' ' +
      str(average_precision_score(np.array(gt_l_filtered).astype(int), pred_l_filtered)) + ' ' +
      str(res['precision']) + ' ' +
      str(res['recall'])
      )