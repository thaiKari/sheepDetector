# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:48:52 2019

@author: karim
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:18:58 2019

@author: karim
"""

import pandas as pd 
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import math
from PIL import Image
from shutil import copyfile

def get_im_details(full_id):
    split = full_id.split('.')
    data  = {}
    data['im_id'] = split[0]
    tmp, x, y = split[1].split('[')
    data['x']= int(x[:-1])
    data['y']= int(y[:-1])
    return data


#Sheep is too close to edge
def is_partial_sheep(xmin, xmax, ymin, ymax, dim):
    D = 35
    
    if(xmax < D):
        return True
    if(ymax < D):
        return True
    if(dim - xmin < D):
        return True
    if(dim - ymin < D):
        return True
    
    return False


def is_edge_sheep(xmin, xmax, ymin, ymax, dim ):
    D=2
    
    if(xmin < D):
        return True
    if(ymin < D):
        return True
    if(dim - xmax < D):
        return True
    if(dim - ymax < D):
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
    

# what portion of bbox is inside box
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
    

def csv_to_label_map(labels):
    im_label_map  = {}
    for i in range(len(labels.index)):

        label = labels.iloc[i]
        im_obj ={}
        im_id = label['External ID']


        if(label['Label'] != 'Skip'):
            boxes = json.loads(label['Label'])
            if 'Sheep' in boxes.keys():
                for b in boxes['Sheep']:
                    geom = list(b['geometry'])
                    xs = list(map( lambda g: g['x'], geom))
                    ys = list(map( lambda g: g['y'], geom))
                    xmin = np.min(xs)
                    xmax = np.max(xs)
                    ymin = np.min(ys)
                    ymax = np.max(ys)
    
                    
                    if 'sheep_color' in b.keys():
                        label_obj = {'sheep_color':b['sheep_color'],
                                     'geometry': [ [xmin, ymin], [xmax, ymax] ]}
                    else:
                        label_obj = {'geometry': [ [xmin, ymin], [xmax, ymax] ]}

                    label_obj['is_lamb'] = ( 'is_lamb' in b.keys() )
    
                    if( 'labels' not in im_obj):
                        im_obj['labels'] = [label_obj]
                    else:
                        im_obj['labels'].append(label_obj)

                
        im_label_map[im_id] = im_obj

    return im_label_map


    
    
#groups labels from same image in dictionary
def build_label_map(labels, dim):
    im_label_map  = {}
    im_label_map_needs_check = {}
    for i in range(len(labels.index)):

        label = labels.iloc[i]
        label_meta  = get_im_details(label['External ID'])
        im_obj ={}
        im_obj_needs_check = {}
        
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
                
                
                new_xmin = xmin - dim/2 + label_meta['x']
                new_xmax = xmax - dim/2 + label_meta['x']
                new_ymin = ymin - dim/2 + label_meta['y']
                new_ymax = ymax - dim/2 + label_meta['y']
                
                label_obj = {'sheep_color':b['sheep_color'],
                             'geometry': [ [new_xmin, new_ymin], [new_xmax, new_ymax]  ]}
                if(not is_edge_sheep(xmin, xmax, ymin, ymax, dim ) and not is_partial_sheep(xmin, xmax, ymin, ymax, dim )):
                    if( 'labels' not in im_obj):
                        im_obj['labels'] = [label_obj]
                    else:
                        im_obj['labels'].append(label_obj)
                    
                else:
                    if( 'labels' not in im_obj_needs_check):
                        im_obj_needs_check['labels'] = [label_obj]
                    else:
                        im_obj_needs_check['labels'].append(label_obj)
                
        im_label_map[label_meta['im_id'] ] = im_obj
        im_label_map_needs_check[label_meta['im_id'] ] = im_obj_needs_check
    return im_label_map, im_label_map_needs_check


def count_labels(label_map):
    n = 0
    for k in label_map.keys(): 
        label_map[k]
        im_data = label_map[k]
    
        if 'labels' in im_data:
            for l in im_data['labels']:
    
                n = n + len(l['geometry'])
    
    return n


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


def show_im_with_pred_and_gt_boxes(im, gt, pred, im_name = None):
    fig,ax = plt.subplots(1, figsize=(30, 30))
    ax.imshow(im)    
    
    for l in gt:
        geom = l['geometry']
        
        xmin, ymin = geom[0]
        xmax, ymax = geom[1]
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=3,edgecolor='#92d050',facecolor='none')
        ax.add_patch(rect)
        
    for l in pred:
        geom = l['geometry']
        
        xmin, ymin = geom[0]
        xmax, ymax = geom[1]
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1.5, linestyle = '--' , edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin,ymin, round(l['confidence'], 3), color='red', fontsize=18 )
    
    plt.axis('off')    
    plt.draw()
    if im_name:
        plt.savefig( 'G:/SAU/Labeled/Train/pred_201912/' + im_name,bbox_inches='tight', transparent="True", pad_inches=0)
    #plt.show()


def show_im_with_boxes_not_saved(im, labels, save_to_filename = False, cmap=None, bbox_edge_color='#92d050'):
    fig,ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(im, cmap=cmap)    
    
    for l in labels:
        geom = l['geometry']
        
        xmin, ymin = geom[0]
        xmax, ymax = geom[1]
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=2,edgecolor=bbox_edge_color,facecolor='none')      
        ax.add_patch(rect)
#        plt.text(xmin,ymin, round(l['confidence'], 3), c='red', fontsize=18 )
    if save_to_filename:
        plt.savefig(save_to_filename)
    plt.draw()
    plt.show()
    


def get_xmin_ymin_xmax_ymax_w_h_from_label(label):
#    print('l', label)
    geom = label['geometry']
        
    xmin, ymin = geom[0]
    xmax, ymax = geom[1]
    w = xmax - xmin
    h = ymax - ymin
    
    return [xmin, ymin, xmax, ymax, w, h]
        

def get_labels_from_crop(xmin, ymin , w_new, h_new, labels):
    
    new_labels = []
    for l in labels:
        geom_xmin, geom_ymin, geom_xmax, geom_ymax, geom_w, geom_h = get_xmin_ymin_xmax_ymax_w_h_from_label(l)
        
        geom_xmin_new = geom_xmin - xmin
        geom_xmax_new = geom_xmax - xmin
        geom_ymin_new = geom_ymin - ymin
        geom_ymax_new = geom_ymax - ymin
        

        if geom_xmin_new < 0:
            geom_xmin_new = 0
        if geom_ymin_new < 0:
            geom_ymin_new = 0
        if geom_xmax_new > w_new:
            geom_xmax_new = w_new -1
        if geom_ymax_new > h_new:
            geom_ymax_new = h_new -1
                   
    # Only keep box if w or h greater than threshold
        T=10
        if geom_xmax_new - geom_xmin_new > T and geom_ymax_new - geom_ymin_new > T:                
            new_label = l.copy()
            new_label['geometry'] = [[geom_xmin_new, geom_ymin_new],[geom_xmax_new, geom_ymax_new]]
            new_labels.append(new_label)
            
    return new_labels


def read_grid_labels(file_path, grid_shape):
    
    f = open(file_path, "r")
    result = {}
    for line in f:
        line_split = line.split()
        im_name = line_split[0]
        grid_vals = np.array(line_split[1:]).reshape(-1,grid_shape[1]).astype(int)
        result[im_name] = grid_vals

    return result


def write_grid_labels_to_txt(path, label_map):
    with open(path, "w") as file:
        file.write('')
        
    for key in label_map.keys():
        grid = label_map[key]
        print(grid)
        print(grid.flatten())
        with open(path, "a") as file:
            file.write(key + ' ' + str(grid.flatten())[1:-1] + '\n' )
            

 

def label_center_in_grid_cell(grid_xmin, grid_ymin, grid_xmax, grid_ymax, labels):
    for l in labels:
        xmin, ymin, xmax, ymax, w, h = get_xmin_ymin_xmax_ymax_w_h_from_label(l)
        
        x_center = (xmin + xmax)/2
        y_center = (ymin + ymax)/2


        if (x_center > grid_xmin and x_center < grid_xmax) and (y_center > grid_ymin and y_center < grid_ymax):
            return True
            
    return False


def grid_cell_has_label_corner(grid_xmin, grid_ymin, grid_xmax, grid_ymax, labels):

    for l in labels:
        xmin, ymin, xmax, ymax, w, h = get_xmin_ymin_xmax_ymax_w_h_from_label(l)
        
        #grid has label if any of the box corners are inside the grid
        corners = [ [xmin, ymin],[xmin, ymax],[xmax, ymin],[xmax, ymax]]

        for c in corners:
            x=c[0]
            y=c[1]

            if (x > grid_xmin and x < grid_xmax) and (y > grid_ymin and y < grid_ymax):
                return True
            
    return False

def grid_cell_has_label(grid_xmin, grid_ymin, grid_xmax, grid_ymax, labels):

    grid_geom = [ [grid_xmin, grid_ymin ], [grid_xmax, grid_ymax ] ]
    intersection_degree
    
    for l in labels:
        label_geom = l['geometry']
        
        if intersection_degree(label_geom, grid_geom) > 0.1:
            return True
        
    return False



def show_im_with_bbox_and_pred_grid(im, labels, pred_grid, Threshold, grid_shape=(6,8), save_to_filename = False, cmap=None, bbox_edge_color='#92d050'):
    fig,ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(im, cmap=cmap)    
    

#        plt.text(xmin,ymin, round(l['confidence'], 3), c='red', fontsize=18 )
    grid_h = math.floor(im.shape[0]/ grid_shape[0])
    grid_w = math.floor(im.shape[1]/ grid_shape[1])
    
    for x in range(grid_shape[1]):
        for y in range(grid_shape[0]):
            if pred_grid[y][x]>Threshold:
                rect = patches.Rectangle((x*grid_w,y*grid_h),grid_w,grid_h,linewidth=2,edgecolor='green',facecolor='green', alpha=0.3)      
#                t = ax.text(x*grid_w + grid_w/2 ,y*grid_h+ grid_h/2,
#                            str(pred_grid[y,x])[:4],
#                           horizontalalignment='center',
#                            verticalalignment='center',
#                            fontsize=20, color='red')
#                t.set_bbox(dict(facecolor='black', alpha=0.3, edgecolor='black'))
            else:
                rect = patches.Rectangle((x*grid_w,y*grid_h),grid_w,grid_h,linewidth=2,edgecolor='white',facecolor='white', alpha = 0.2)      
            t = ax.text(x*grid_w + grid_w/2 ,y*grid_h+ grid_h/2,
                        str(pred_grid[y,x])[:4],
                       horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=16, color='white')
            t.set_bbox(dict(facecolor='black', alpha=0.3, edgecolor='black'))
            ax.add_patch(rect)
        
    
    for l in labels:
        xmin, ymin, xmax, ymax, w, h = get_xmin_ymin_xmax_ymax_w_h_from_label(l)
        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=2,edgecolor=bbox_edge_color,facecolor='none')      
        ax.add_patch(rect)
    
    if save_to_filename:
        plt.savefig(save_to_filename)
    plt.draw()
    plt.show()                                
                                    

def show_im_with_bbox_and_grid(im, labels,grid_shape=(6,8), save_to_filename = False, cmap=None, bbox_edge_color='#92d050'):
    fig,ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(im, cmap=cmap)    
    

#        plt.text(xmin,ymin, round(l['confidence'], 3), c='red', fontsize=18 )
    grid_h = math.floor(im.shape[0]/ grid_shape[0])
    grid_w = math.floor(im.shape[1]/ grid_shape[1])
    
    for x in range(grid_shape[1]):
        for y in range(grid_shape[0]):
            if grid_cell_has_label(x*grid_w, y*grid_h, (x+1)*grid_w,(y+1)*grid_h, labels ):
                rect = patches.Rectangle((x*grid_w,y*grid_h),grid_w,grid_h,linewidth=2,edgecolor='green',facecolor='green', alpha=0.1)      
            else:
                rect = patches.Rectangle((x*grid_w,y*grid_h),grid_w,grid_h,linewidth=2,edgecolor='white',facecolor='none')      
            ax.add_patch(rect)
            
    
    for l in labels:
        xmin, ymin, xmax, ymax, w, h = get_xmin_ymin_xmax_ymax_w_h_from_label(l)
        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=2,edgecolor=bbox_edge_color,facecolor='none')      
        ax.add_patch(rect)
    
    if save_to_filename:
        plt.savefig(save_to_filename)
    plt.draw()
    plt.show()

#def show_im_with_bbox_and_grid(im, labels, grid_shape=(6,8), cmap=None, bbox_edge_color='#92d050'
#    fig,ax = plt.subplots(1, figsize=(20, 20))
#    ax.imshow(im, cmap=cmap)    
#    
#    for l in labels:
#        geom = l['geometry']
#        
#        xmin, ymin = geom[0]
#        xmax, ymax = geom[1]
#        w = xmax - xmin
#        h = ymax - ymin
#        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=2,edgecolor=bbox_edge_color,facecolor='none')      
#        ax.add_patch(rect)
##        plt.text(xmin,ymin, round(l['confidence'], 3), c='red', fontsize=18 )
#    grid_h = math.floor(grid_shape[0])
#    grid_w = math.floor(grid_shape[1])
#    
#    for x in range(grid_shape[1]):
#        for y in range(grid_shape[0]):
#            rect = patches.Rectangle((x,y),grid_w,grid_h,linewidth=2,edgecolor='green',facecolor='none')      
#            ax.add_patch(rect)
#    plt.draw()
#    plt.show()    
                               

   
                               
                               
#
#im_name = 'oct19_103MEDIA_DJI_0584.JPG'
#im = cv2.imread('G:/SAU/Labeled/Val/' + im_name)
#labels = np.load('G:/SAU/Labeled/Val/00_labels.npy', allow_pickle=True).item()
#show_im_with_boxes_not_saved(im, labels[im_name]['labels'])

def show_im_with_boxes(im_name, im_dir, label_map):
    im = cv2.imread(im_dir + im_name )
    labels = label_map[im_name]['labels']
    show_im_with_boxes_not_saved(im, labels)


def map_to_yolo(data, w0, h0):
    data_str = ''
    
    for im_name in data.keys():
        im_data = data[im_name]
        data_str = data_str + im_name + ' '
        
        for label in im_data['labels']:            
            xmin, ymin, xmax, ymax, w, h = get_xmin_ymin_xmax_ymax_w_h_from_label(label)
            data_str = data_str + '{} {} {} {} 0 '.format(xmin, ymin, xmax, ymax) #0 is class id
        
        data_str = data_str + '\n'
    return data_str


def map_to_coco_simple(data, w0=0, h0=0):
    result = {}
    
    for key in data.keys():
        im_labels = data[key]['labels']
        im_labels_new_format = []
        
        for label in im_labels:
            xmin, ymin, xmax, ymax, w, h = get_xmin_ymin_xmax_ymax_w_h_from_label(label)
            im_labels_new_format.append([int(xmin), int(ymin), int(w), int(h)])
        result[key] =  im_labels_new_format
    return result
        
    

def map_to_coco(data, w0=0, h0=0, grid_type=False):
    print('COCOLOCO')
    categories =[{
            'supercategory':'none',
            'name':'sheep',
             'id':0
             }]


    annotations = []
    images = []
    label_n = -1
    w = w0
    h = h0

    
    for im_n in range(len(data.keys())):
        
        k = list(data.keys())[im_n]
        
        d = data[k]
        
        if w0 < 1 or h0 < 1:
            if '[1024][1024]' in k:
                w0 = h0 = 1024
            elif '_Rot45_[2128]' in k:
                w0 = h0 = 2128
            elif '[3040][3040]' in k:
                w0 = h0 = 3040
                
            elif 'CROPPED_[2128]' in k:
                w0 = h0 = 2128
            elif True:
                print(k)

        images.append({'file_name':k,
                       'height': h0,
                       'width':w0,
                       'id': im_n })
        
        if not grid_type:    
            for i in range(len(d['labels'])):
                label_n = label_n +1
                geom = d['labels'][i]['geometry']
                xmin, ymin = geom[0]
                xmax, ymax = geom[1]
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                w = xmax-xmin
                h = ymax-ymin
                
                if w >= 1 and h >=1:
                    annotations.append({
                            'id': label_n,
                            'im_name': k,
                            'bbox': [xmin-1, ymin-1, w, h],
                            'image_id':im_n,
                            'segmentation':[],
                            'ignore':0,
                            'area':w*h,                
                            'iscrowd':0,
                            'category_id':0
                            })
            
        else:
            annotations.append({
                        'id': im_n,
                        'image_id':label_n,
                        'category_id':0,
                        'grid_mask': d
                    })
            
        
    
          
    return {'type':'instances', 'images':images, 'categories':categories, 'annotations':annotations}


#Return split label map but doesnt copy images
def get_split_map(label_map, path, w, h, step, im_size):
    return split(label_map, path, None, w, h, step, im_size)


def split(label_map, path, dst_path, w, h, step, im_size):
    new_label_map = {}
    
    for k in label_map.keys():
        
        if dst_path:
            im = cv2.imread(os.path.join(path, k ))
        labels = label_map[k]['labels']
    
    
        for i in range(math.ceil((w)/step) ):
            for j in range(math.ceil((h)/step)):
                x = i*step
                y = j*step
                
                if x > w - im_size:
                    x =  w - im_size
                if y > h - im_size:
                    y =  h - im_size
                
                split_labels = []
                for l in labels:
                    geom = l['geometry']
                    xmin, ymin = geom[0]
                    xmax, ymax = geom[1]
                    
                    xmin = xmin - x
                    xmax = xmax - x
                    ymin = ymin - y
                    ymax = ymax - y
                    
                    #check if label within box
                    if not (xmin > im_size or ymin > im_size or xmax < 0 or ymax < 0):

                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if xmax > im_size:
                            xmax = im_size
                        if ymax > im_size:
                            ymax = im_size
                            
                        w1 = xmax-xmin
                        h1 = ymax-ymin 
                        
                        #Only save if lbl w and h greater than threshold.
                        T = 20
                        if w1 > T and h1 > T:
                            label_copy = l.copy()
                            label_copy['geometry'] = [[xmin, ymin], [xmax, ymax]]
                            split_labels.append(label_copy)
                        
                if len(split_labels) > 0:
                    if dst_path:
                        split_im = im[y:y+im_size, x:x+im_size, :]                
                    split_imname = k[:-4] + '[' + str(x)+ ']''['+str(y)+']''['+str(im_size)+']' +'.jpg'
                    if dst_path:
                        cv2.imwrite( os.path.join(dst_path, split_imname), split_im )
                    new_label_map[split_imname] = {'labels': split_labels}
                    
    return new_label_map



def rotate_box(geom, M):
    xmin, ymin = geom[0]
    xmax, ymax = geom[1]
    
    bb=[(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax) ]
    
    new_bb = list(bb.copy())
    
    
    
    for i,coord in enumerate(bb):

        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    
    new_bb = np.array(new_bb)

    xmin_new = np.min(new_bb[:,0])
    xmax_new = np.max(new_bb[:,0])
    ymin_new = np.min(new_bb[:,1])
    ymax_new = np.max(new_bb[:,1])
    
    return [[xmin_new, ymin_new],[xmax_new, ymax_new]]

def rotate_im(im, M, h, w):
    # perform the actual rotation and return the image
    return cv2.warpAffine(im, M, (w, h))


def rotate__centercrop(im, deg, w_new, h_new, labels):
    print( w_new, h_new)
    (h, w) = im.shape[:2]
    (cX, cY) = (w // 2, h // 2)


    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    im = rotate_im(im, M, h, w)
    im = im[int(cY - h_new/2) :int(cY + h_new/2), int(cX - w_new/2) :int(cX + w_new/2) ]
    
    new_labels = []
    
    for l in labels['labels']:
        geom = l['geometry'].copy()
        geom = rotate_box(geom, M)
        geom =list(map( lambda p: [p[0] - int(cX - w_new/2)  , p[1] - int(cY - h_new/2)  ] ,geom))
        
        xmin, ymin = geom[0]
        xmax, ymax = geom[1]
        
        #check if label within new crop
        if not (xmin > w_new or ymin > h_new or xmax < 0 or ymax < 0):

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > w_new:
                xmax = w_new
            if ymax > h_new:
                ymax = h_new
                
            w1 = xmax-xmin
            h1 = ymax-ymin 
            
            #Only save if lbl w and h greater than threshold.
            T = 20
            if w1 > T and h1 > T:        
                new_labels.append({
                'sheep_color': l['sheep_color'],
                'geometry': geom
                })
            
#    show_im_with_boxes_not_saved(im, new_labels, save_to_filename = False)
                    
    return im , {'labels':new_labels}

def rotate_crop(im, deg, minxy, w_new, h_new, labels):
    
    (h, w) = im.shape[:2]
    (cX, cY) = (w // 2, h // 2)


    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    im = rotate_im(im, M, h, w)
    im = im[minxy[1]:minxy[1]+h_new,minxy[0]:minxy[0]+w_new ]
    plt.figure()
    plt.imshow(im)
    
    new_labels = []
    
    for l in labels['labels']:
        geom = l['geometry'].copy()
        geom = rotate_box(geom, M)
        geom =list(map( lambda p: [p[0]-minxy[0], p[1]-minxy[1] ] ,geom))
        
        xmin, ymin = geom[0]
        xmax, ymax = geom[1]
        
        #check if label within new crop
        if not (xmin > w_new or ymin > h_new or xmax < 0 or ymax < 0):

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > w_new:
                xmax = w_new
            if ymax > h_new:
                ymax = h_new
                
            w1 = xmax-xmin
            h1 = ymax-ymin 
            
            #Only save if lbl w and h greater than threshold.
            T = 20
            if w1 > T and h1 > T:
                new_labels.append({
                'sheep_color': l['sheep_color'],
                'geometry': [[xmin, ymin],[xmax, ymax]]
                })
                    
    return im , {'labels':new_labels}
    

def get_crop(im, labels, minx, miny, w_new, h_new):
    
    im_new = im[ miny: miny+h_new, minx: minx+w_new]  
    new_labels = []
    
    for l in labels['labels']:
        geom = l['geometry'].copy()
        
        xmin, ymin = geom[0]
        xmax, ymax = geom[1]
        xmin = xmin - minx
        ymin = ymin - miny
        xmax = xmax - minx
        ymax = ymax - miny
        
        #check if label within new crop
        if not (xmin > w_new or ymin > h_new or xmax < 0 or ymax < 0):

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > w_new:
                xmax = w_new
            if ymax > h_new:
                ymax = h_new
                
            w1 = xmax-xmin
            h1 = ymax-ymin 
            
            #Only label if w and h of box greater than threshold.
            T = 5
            if w1 > T and h1 > T:
                new_labels.append({
                'sheep_color': l['sheep_color'],
                'geometry': [[xmin, ymin],[xmax, ymax]]
                })
            
    return im_new, {'labels':new_labels}
    
    

#Rotates by 30 degrees and crops out larges posible image
def rotate_crop_30(im_path, labels):
    return rotate_crop(im_path, 30,(750,500),2553,1900, labels )
    
    
def rotate_crop_60(im_path, labels):
    return rotate_crop(im_path, 60,(975,300),2458,1842, labels )
    


#path = 'G:/SAU/Labeled/00_Annotations'
#labels = pd.read_csv(os.path.join(path, 'grid_boxes_3_4_20200123.csv'))
#label_map = csv_to_label_map(labels)
#np.save(os.path.join(path, '00_grid_boxes_3_4_20200123.npy'), label_map)

##dst_path = 'G:/SAU/Labeled/All_labeled_IR'
#new_label_map = {}
#
#for file in os.listdir(path):
#    
#    if '.csv' in file:
#        print(file)
#        labels = pd.read_csv(os.path.join(path, file))
#        label_map = csv_to_label_map(labels)
#        new_label_map = {**new_label_map, **label_map}
#
#np.save(os.path.join(path, '00_all_labels.npy'), new_label_map)

#labels1 = np.load(os.path.join(path, 'train_labels_REVIEWED.npy'), allow_pickle = True).item()
#labels2 = np.load(os.path.join(path, 'val_labels_REVIEWED.npy'), allow_pickle = True).item()
#labels3 = np.load(os.path.join(path, 'g13-27_labels_REVIEWED.npy'), allow_pickle = True).item()
#
#all_labels = {**labels1, **labels2, **labels3}
#np.save(os.path.join(path, '00_all_labels.npy'), all_labels)
#available_images = os.listdir(path)
#
#
#    
#labels = np.load('G:/SAU/Labeled/00_annotations/00_all_labels-2019-10-31.npy', allow_pickle = True).item()
#path = 'G:/SAU/Labeled/Val'
###src = 'G:/SAU/Labeled/All_labeled'
###dst = 'G:/SAU/Labeled/All_labeled/IR'
#img_list = os.listdir(path)
#train_labels = {}
#
#for file in img_list:
#    if('.JPG' in file):
#        try:
#            train_labels[file] = labels[file]
#        except: print(file)
#    
#np.save( 'G:/SAU/Labeled/00_annotations/00_val_labels.npy', train_labels)


#path = 'G:/SAU/Labeled/Train/'
#no_label = []
#label_map = np.load(os.path.join(path, '00_labels.npy'), allow_pickle = True).item()
##im_name = 'aug19_103MEDIA_DJI_0043.JPG'
#for im_name in ['sep19_101MEDIA_DJI_0218.JPG']:#os.listdir(path)[460:]:
#    if '.JPG' in im_name:       
#        im_path = os.path.join( path, im_name)
#        im = cv2.imread(im_path)
#        if im_name in label_map:
#            im_label = label_map[im_name]
#            print(im_name)
#            show_im_with_boxes_not_saved(im, im_label['labels'])
#        else:
#            print('NO LABEL!!!!!!!!!!')
#            no_label.append(im_name)
##
#wrong = ['aug19_103MEDIA_DJI_0359.JPG','aug19_103MEDIA_DJI_0473.JPG','aug19_103MEDIA_DJI_0475.JPG','aug19_103MEDIA_DJI_0479.JPG']
#for w in wrong:
#    print(w in label_map.keys())
#    del label_map[w]
#    print(w in label_map.keys())
#    
#np.save(os.path.join(path, 'labels_REVIEWED.npy'), label_map)
    
         
#show_im_with_boxes_not_saved(*rotate_crop_60(im, im_label))
#show_im_with_boxes_not_saved(*rotate_crop_30(im, im_label))
#    
#
#rotate_crop(im_path, 0,(0,0),4056,3040, im_label )
#show_im_with_boxes(im_name, path, label_map)