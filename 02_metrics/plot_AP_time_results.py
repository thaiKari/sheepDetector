# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:20:21 2020

@author: karim
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from metrics_grid import read_pred_file, get_grid, label_xyxy_to_xywh, filter_fuzzy_grid_cells
import os
from sklearn.metrics import average_precision_score, precision_recall_curve



mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False

results_file = open("G:/SAU/Results2020_3.txt", "r")
time_stamps = []
img_types = []
fuse_depths = []
network_depths = []
Xs = []
APs_val = []
APs_train = []
rgb_resize_shapes = []
infrared_resize_shapes = []
time_stamps_rgb = []
time_stamps_infrared = []
times_per_im_val =[]
times_per_im_train =[]
colors = []
rgb_resolutions=[]
column_names=[]
APs_test=[]
times_per_im_test=[]
APs_t1 = []
APs_t2 = []
im_shape = (2400, 3200)
grid_shape = (6, 8)


color_map = {
        'rgb':'#ED7D31',
        'infrared': '#4472C4',
        'ensemble': 'green'
        }

bbox_map = np.load(os.path.join('G:/SAU/Labeled/00_Test2020', '01_infrared_align.npy'), allow_pickle = True).item()
result_map = {}

i = 0
for result in results_file:
    result_array = result.strip().split()

    
    if i == 0:
        column_names=result_array
    
    
    elif len(result_array) > 0:
        time_stamps.append(result_array[0])
        img_types.append(result_array[1])
        fuse_depths.append( (int(result_array[2])+1) if  result_array[1]=='ensemble' else '-'  )
        network_depths.append(int(result_array[3]))
        Xs.append(result_array[4])
        APs_val.append(float(result_array[5]))
        APs_train.append(float(result_array[6]))
        rgb_resize_shapes.append(result_array[7])
        infrared_resize_shapes.append(result_array[8])
        time_stamps_rgb.append(result_array[9])
        time_stamps_infrared.append(result_array[10])
        times_per_im_val.append(float(result_array[11]))
        times_per_im_train.append(float(result_array[12]))
        

        
        #if len(result_array) > 13:
        #    APs_test.append(float(result_array[13]))
            
        if len(result_array) > 14:
            times_per_im_test.append(float(result_array[14]))
            
        pred_file_path = 'test_predictions_'+ img_types[-1] + '_' +time_stamps[-1]+'.txt'
        pred_grid_map = read_pred_file('G:/SAU/Labeled/Predictions2020', pred_file_path, grid_shape=(6,8))
        
        #Calculate AP for split test set
        all_pred_t1 = []
        all_gt_t1 = []
        all_pred_t2 = []
        all_gt_t2 = []
        all_pred = []
        all_gt = []
        for key in bbox_map.keys():
            if '.JPG' in key:
                labels = bbox_map[key]['labels']
                labels_xywh = list(map( lambda l: label_xyxy_to_xywh(l['geometry']), labels))
                pred_grid = pred_grid_map[key]
                gt_grid = get_grid(labels_xywh, im_shape, grid_shape)
                if 'may' in key:
                    all_pred = [*all_pred, *list(pred_grid.flatten())]
                    all_gt = [*all_gt, *list(gt_grid.flatten())]                    
                    im_num = int(key[-8:-4])
                    if im_num > 100 and im_num < 689:
                        all_pred_t1 = [*all_pred_t1, *list(pred_grid.flatten())]
                        all_gt_t1 = [*all_gt_t1, *list(gt_grid.flatten())]
                    else:
                        all_pred_t2 = [*all_pred_t2, *list(pred_grid.flatten())]
                        all_gt_t2 = [*all_gt_t2, *list(gt_grid.flatten())]
        
        all_gt_filtered_t1, all_pred_filtered_t1 = filter_fuzzy_grid_cells(all_gt_t1, all_pred_t1)        
        average_precision_filtered_t1 = average_precision_score(np.array(all_gt_filtered_t1).astype(int), all_pred_filtered_t1)
        all_gt_filtered_t2, all_pred_filtered_t2 = filter_fuzzy_grid_cells(all_gt_t2, all_pred_t2)        
        average_precision_filtered_t2 = average_precision_score(np.array(all_gt_filtered_t2).astype(int), all_pred_filtered_t2)
        all_gt_filtered, all_pred_filtered = filter_fuzzy_grid_cells(all_gt, all_pred)        
        average_precision_filtered = average_precision_score(np.array(all_gt_filtered).astype(int), all_pred_filtered)
        
        APs_t1.append(average_precision_filtered_t1)
        APs_t2.append(average_precision_filtered_t2)
        APs_test.append(average_precision_filtered)
        
        result_map[time_stamps[-1]]={
            'img_type':img_types[-1],
            'fuse_depth':fuse_depths[-1],
            'network_depth':network_depths[-1],
            'X':Xs[-1],
            'AP_val':APs_val[-1],
            'AP_train':APs_train[-1],
            'rgb_resize_shape':rgb_resize_shapes[-1],
            'infrared_resize_shape':infrared_resize_shapes[-1],
            'time_stamp_rgb':time_stamps_rgb[-1],
            'time_stamp_infrared':time_stamps_infrared[-1],
            'time_per_im_val':times_per_im_val[-1],
            'times_per_im_train':times_per_im_train[-1],
            'AP_t1': APs_t1[-1],
            'AP_t2':APs_t2[-1],
            'AP_test':APs_test[-1]
        }
        
#        with open('G:/SAU/Results_clean.txt', 'a') as out:
#            s = time_stamps[-1] + ' '
#            if img_types[-1] == 'rgb':
#                s = s + 'RGB_'
#            elif img_types[-1] == 'infrared':
#                s = s + 'I_'
#            else:
#                s = s + 'RGB+I_'
#            
#            s = s + ('rx' if Xs[-1] == 'TRUE' else 'r')
#            s = s + str(network_depths[-1])
#            
#            s = s + ' ' + ((str(int(fuse_depths[-1])+1)) if img_types[-1] == 'ensemble' else '-')
#            s = s + ' ' + ( rgb_resize_shapes[-1] if img_types[-1] != 'infrared' else '-' )
#            s = s + ' ' + ( infrared_resize_shapes[-1] if img_types[-1] != 'rgb' else '-' )
#            s = s + ' ' + str(APs_train[-1])
#            s = s + ' ' + str(APs_val[-1])
#            s = s + ' ' + str(APs_test[-1])
#            s = s + ' ' + str(APs_t1[-1])
#            s = s + ' ' + str(APs_t2[-1])
#            s = s + ' ' + str(times_per_im_train[-1] )
#
#            out.write( s + '\n')
            
    i = i+1
#FOR TEST        
#of_interest = ['20200312_2118','20200312_1306','20200424_1729','20200424_1752','20200228_0722','20200317_1321','20200411_1733','20200416_2142']
#pareto_x = [0.096, 0.104, 0.146, 0.266, 0.355, 0.465, 0.586]
#pareto_y = [0.572, 0.61, 0.814, 0.867, 0.891, 0.894, 0.9]
    
of_interest = ['20200312_2118','20200420_1741','20200421_1844', '20200314_2313', '20200317_1321','20200411_1733','20200416_2142']
pareto_x = []
pareto_y = []

for key in of_interest:
    pareto_x.append(result_map[key]['times_per_im_train'])
    pareto_y.append(result_map[key]['AP_val'])
    
#Add some more interesting points (not on pareto front)
of_interest = [*of_interest, '20200424_1752', '20200421_1257']

      
plt.figure(figsize=(12,8))
plt.grid()
plt.ylim((0,1.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlim((0,0.7))
plt.xticks(np.arange(0.1, 0.7, 0.1))
plt.xlabel('Inference Time Per Image (s)', fontsize=14, alpha=0.6)
plt.ylabel('Validation Average Precision', fontsize=14, alpha=0.6)



plt.plot(pareto_x, pareto_y,'k--', alpha=0.5 )  
        
for x, y, img_type, rgb_resize_shape, infrared_resize_shape, X, fuse_depth, timestamp, network_depth in zip(times_per_im_train, APs_val, img_types, rgb_resize_shapes, infrared_resize_shapes, Xs, fuse_depths,time_stamps, network_depths):
    plt.scatter(x, y,
                color=color_map[img_type],
                marker='x'if X.lower()=='true'  else 'o', 
                s=12
                )
    if timestamp in of_interest:
        s = 'r'
        s = s + ('x'if X=='TRUE'else '')
        s = s + str(network_depth)        
        s = s + ('_f' + str(fuse_depth) if img_type == 'ensemble' else '')
        #s = s + ('_infrared' + str(infrared_resize_shape) if img_type == 'infrared' else '')
        s = s + ('_rgb' + str(rgb_resize_shape) if img_type != 'infrared' else '')
       
        
        #plt.text(x-0.01, y+0.01,s,  fontsize=8, ha='right', alpha= 0.5,bbox=dict(boxstyle="round", fc="white", ec="gray"))
        plt.annotate(s, xy=(x,y), xytext=(x-0.01, y+0.06), fontsize=7, ha='right', alpha= 0.5,bbox=dict(boxstyle="round", fc="white", ec="gray"),
            arrowprops=dict(arrowstyle="->", facecolor='black', alpha =0.5))


  
   
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Infrared', markerfacecolor=color_map['infrared']),
                   Line2D([0], [0], marker='o', color='w', label='RGB', markerfacecolor=color_map['rgb']),                   
                   Line2D([0], [0], marker='o', color='w', label='RGB+I', markerfacecolor=color_map['ensemble']),
                   Line2D([0], [0], marker='o', color='w', label='ResNet', markerfacecolor='black'),
                   Line2D([0], [0], marker='X', color='w', label='ResNext', markerfacecolor='black'),
                   Line2D([0], [0], label='Pareto Front', linestyle='--', color='black', alpha=0.5)
                   ]   

plt.legend(handles=legend_elements)

