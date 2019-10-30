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
            for b in boxes['Sheep']:
                geom = list(b['geometry'])
                xs = list(map( lambda g: g['x'], geom))
                ys = list(map( lambda g: g['y'], geom))
                xmin = np.min(xs)
                xmax = np.max(xs)
                ymin = np.min(ys)
                ymax = np.max(ys)

                
                label_obj = {'sheep_color':b['sheep_color'],
                             'geometry': [ [xmin, ymin], [xmax, ymax] ]}

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


def show_im_with_boxes_not_saved(im, labels):
    fig,ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(im)    
    
    for l in labels:
        geom = l['geometry']
        
        xmin, ymin = geom[0]
        xmax, ymax = geom[1]
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.draw()
    plt.show()


def show_im_with_boxes(im_name, im_dir, label_map):
    im = cv2.imread(im_dir + im_name )
    labels = label_map[im_name]['labels']
    show_im_with_boxes_not_saved(im, labels)


def map_to_coco(data, w0=0, h0=0):
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
            if 'ROT30' in k:
                w = 2553
                h = 1900
            elif 'ROT60' in k:
                w = 2458
                h = 1842
            elif 'CROPPED' in k:
                split_k = k.split(']')
                w = int(split_k[2][1:])
                h = int(split_k[3][1:])             
            
            elif ']' in k:
                w = h = int(k.split(']')[2][1:])
            else:
                w = 4056
                h = 3040            
        
        if not d['labels'] == 'Skip' and d['labels'] :
            images.append({'file_name':k,
                           'height': h,
                           'width':w,
                           'id': im_n })
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
                            'bbox': [xmin, ymin, w, h],
                            'image_id':im_n,
                            'segmentation':[],
                            'ignore':0,
                            'area':w*h,                
                            'iscrowd':0,
                            'category_id':0
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


def rotate_crop(im, deg, minxy, w_new, h_new, labels):

    
    (h, w) = im.shape[:2]
    (cX, cY) = (w // 2, h // 2)


    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    im = rotate_im(im, M, h, w)
    im = im[minxy[1]:minxy[1]+h_new,minxy[0]:minxy[0]+w_new ]
    
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
            
            #Only save if lbl w and h greater than threshold.
            T = 20
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
    
#labels = np.load('G:/SAU/Labeled/00_annotations/00_all_labels.npy', allow_pickle = True).item()
#path = 'G:/SAU/Labeled/All_labeled'
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
np.save( 'G:/SAU/Labeled/00_annotations/00_all_labels_visual.npy', train_labels)


#path = 'G:/SAU/Labeled/Train/'
#no_label = []
#label_map = np.load(os.path.join(path, 'labels_REVIEWED.npy'), allow_pickle = True).item()
##im_name = 'aug19_103MEDIA_DJI_0043.JPG'
#for im_name in os.listdir(path)[460:]:
#    if '.JPG' in im_name:       
#        im_path = os.path.join( path, im_name)
#        im = cv2.imread(im_path)
#        if im_name in label_map:
#            im_label = label_map[im_name]
#            print(im_name)
#            show_im_with_boxes_not_saved(im, im_label['labels'])
#        else:
#            print('NO LABEL!!!!!!!!!!')
#            print('NO LABEL!!!!!!!!!!')
#            print('NO LABEL!!!!!!!!!!')
#            print('NO LABEL!!!!!!!!!!')
#            print(im_name)
#            print('NO LABEL!!!!!!!!!!')
#            print('NO LABEL!!!!!!!!!!')
#            print('NO LABEL!!!!!!!!!!')
#            no_label.append(im_name)
#
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