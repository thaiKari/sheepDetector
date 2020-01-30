# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:15:38 2020

@author: karim
"""


import os
import shutil
import cv2
import numpy as np
import matplotlib.patches as patches
from transformations import transform_IR_im_to_vis_coordinate_system, transform_vis_im_to_IR_coordinate_system, transform_vis_pt_list_to_IR_coordinate_system
import matplotlib.pyplot as plt


### GET ALL IMS WITH IR (NOT MSX)
#print('PART 1')
path = 'G:/SAU/Labeled/Val2020'
#all_files = os.listdir(path)
#ims = list(filter(lambda file: '.JPG' in file ,all_files))

IR_path = os.path.join(path, 'IR')
#
#for file in os.listdir(IR_path):
#    if not file in ims:
#        os.remove(os.path.join(IR_path, file))
#
#
Vis_path = os.path.join(path, 'Visual_IR')
#     
#for im in ims:
#    if im in os.listdir(IR_path):
#        src = os.path.join(path, im)
#        dst = os.path.join(Vis_path, im)
#        shutil.copyfile(src, dst)
 

###CONVERT VIS IM AND LABELS TO IR COORDINATE SYSTEM
#print('Part 2a')
#labels = np.load('G:/SAU/Labeled/00_annotations/00_all_labels_20200107.npy', allow_pickle=True).item()
#labels_transformed = {}        
#path_4d =  os.path.join(path, '4d_IR_coord_system')
#w = 640
#h = 480
#
#for file in os.listdir(Vis_path):
##    print(file)
#    if '.JPG' in file:
#        
#        print(file)
#        im_ir = cv2.imread(os.path.join(IR_path, file), cv2.IMREAD_GRAYSCALE) 
#        im_vis = cv2.imread(os.path.join(Vis_path, file))
#        
#        im_vis_transformed = transform_vis_im_to_IR_coordinate_system(im_vis)
#        im_vis_transformed= np.array(im_vis_transformed*255, np.uint8)
#        im_vis_transformed= cv2.cvtColor(im_vis_transformed, cv2.COLOR_BGR2RGB).astype(int)
#        
#        im_4d = np.zeros( (h, w, 4) ).astype(int)
#        im_4d[:,:,:3] = im_vis_transformed
#        im_4d[:,:,-1] = im_ir
#        
##        fig,ax = plt.subplots(1, figsize=(20, 10))
##        ax.imshow(im_4d[:,:,-3:])    
#    
#        
#        label_vis = labels[file]['labels']
#        print(label_vis)
#        
#        new_labels = []
#        
#        for l in label_vis:
#            xmin, ymin = l['geometry'][0]
#            xmax, ymax =  l['geometry'][1]
#            corners = [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]
#            transformed_corners = transform_vis_pt_list_to_IR_coordinate_system(np.asarray(corners, dtype = np.uint))
#            xmin = np.min(transformed_corners[:,0])
#            xmax = np.max(transformed_corners[:,0])
#            ymin = np.min(transformed_corners[:,1])
#            ymax = np.max(transformed_corners[:,1])
#
#            
#            if xmin < 0:
#                xmin = 0
#            if ymin < 0:
#                ymin = 0
#            if xmax > w:
#                xmax = w
#            if ymax > h:
#                ymax = h
#                
#            transformed_pts = [[ int(xmin), int(ymin)], [int(xmax), int(ymax)]]
#                
#            w_box = xmax - xmin
#            h_box = ymax - ymin
#            
#            #Label on visual image is outside scope of IR image
#            if( w_box > 0 and h_box >0):
#                l['geometry'] = transformed_pts
#                new_labels.append(l)
#                
#        new_filename = file[:-4] + '.npy'
#        np.save(os.path.join(path_4d,new_filename), im_4d)
#        im_label_obj  = {}
#        im_label_obj['labels'] = new_labels
#        labels_transformed[new_filename]=im_label_obj
#
#np.save(os.path.join(path_4d,'00_labels.npy'), labels_transformed)                


            
#            rect = patches.Rectangle((xmin, ymin),w,h,linewidth=2,edgecolor='#92d050',facecolor='none')      
#            ax.add_patch(rect)

        
    
        
#
## JOIN VIS AND IR IMAGES TO ONE ARRAY. CROP ARRAY TO OVERLAPING RECT
#print('PART 2b')
path_4d = os.path.join(path, '4d_vis_plus_ir2')
path_vis_cropped = os.path.join(path, 'Visual_IR_cropped2')
#
w = 4056
h = 3040
xmin = 475
ymin = 250
w_new = 3200
h_new = 2400

#for file in os.listdir(IR_path):
#    print(file)
#    if '.JPG' in file:
#
#        im_ir = cv2.imread(os.path.join(IR_path, file), cv2.IMREAD_GRAYSCALE) 
#        im_vis = cv2.imread(os.path.join(Vis_path, file))
#        im_vis_rgb = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB).astype(int)
#        im_ir_transformed = (transform_IR_im_to_vis_coordinate_system(im_ir) * 255).astype(int)
#
## visualize cropped area        
##        fig,ax = plt.subplots(1, figsize=(20, 10))
##        ax.imshow(im_vis)
##        ax.imshow(im_ir_transformed, cmap='gray', alpha = 0.7)
##        rect = patches.Rectangle((xmin,ymin), w_new, h_new ,linewidth=2,edgecolor='r',facecolor='none')
##        ax.add_patch(rect)
#        
#        im_4d = np.zeros( (h, w, 4) ).astype(int)
#        im_4d[:,:,:3] = im_vis_rgb
#        im_4d[:,:,-1] = im_ir_transformed
#        
#        xmax = xmin + w_new
#        ymax = ymin + h_new
#        im_4d_cropped = im_4d[ymin:ymax, xmin:xmax,:]
#        
#        cv2.imwrite(os.path.join(path_vis_cropped,file), im_vis[ymin:ymax, xmin:xmax,:])
#        np.save(os.path.join(path_4d,file[:-4]), im_4d_cropped)
    
#        
## Transform labels to new cropped area
#print('PART 3')
#labels = np.load('G:/SAU/Labeled/00_annotations/00_all_labels_20200107.npy', allow_pickle=True).item()
#new_labels_map = {}
#
#for key in labels.keys():
#    im_labels  = labels[key]['labels']
#    new_labels = []
#    for label in im_labels:
#        geom = label['geometry']
#        geom_xmin, geom_ymin = geom[0]
#        geom_xmax, geom_ymax = geom[1]
#        
#        geom_xmin_new = geom_xmin - xmin
#        geom_xmax_new = geom_xmax - xmin
#        geom_ymin_new = geom_ymin - ymin
#        geom_ymax_new = geom_ymax - ymin
#        
#
#        if geom_xmin_new < 0:
#            geom_xmin_new = 0
#        if geom_ymin_new < 0:
#            geom_ymin_new = 0
#        if geom_xmax_new > w_new:
#            geom_xmax_new = w_new -1
#        if geom_ymax_new > h_new:
#            geom_ymax_new = h_new -1
#                   
#    # Only keep box if w or h greater than threshold
#        T=5
#        if geom_xmax_new - geom_xmin_new > T and geom_ymax_new - geom_ymin_new > T:                
#            new_label = label.copy()
#            new_label['geometry'] = [[geom_xmin_new, geom_ymin_new],[geom_xmax_new, geom_ymax_new]]
#            new_labels.append(new_label)
#            
#    new_im_labels = labels[key].copy()
#    new_im_labels['labels'] = new_labels
#    new_labels_map[key] = new_im_labels
#    
#np.save('G:/SAU/Labeled/00_annotations/00_all_labels_20200107_crop_for_alignment2.npy', new_labels_map)

##Split train val

labels = np.load('G:/SAU/Labeled/00_annotations/00_all_labels_20200107_crop_for_alignment2.npy', allow_pickle=True).item()

new_labels_vis = {}
new_labels = {}

for im_name in os.listdir(path_vis_cropped):
    if '.JPG' in im_name:
        new_im_name = im_name[:-4] + '.npy'
        new_labels_vis[im_name] = labels[im_name]
        new_labels[new_im_name] = labels[im_name]
    
np.save( os.path.join(path_4d, '00_labels.npy'), new_labels )
np.save( os.path.join(path_vis_cropped, '00_labels.npy'), new_labels_vis )
#        
#        
#    
#    