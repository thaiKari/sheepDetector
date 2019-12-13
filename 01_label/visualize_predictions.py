# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:19:44 2019

@author: karim
"""

from label_utils import show_im_with_pred_and_gt_boxes, show_im_with_boxes_not_saved
import numpy as np
import os
import cv2

path = 'G:/SAU/Labeled/Val/'
#path = 'interesting_pics'
gt_labels = 'G:/SAU/Labeled/00_annotations/00_all_labels_20191210.npy'

pred_labels = '00_split_libra_20191212_epoch_8.npy'


gt_label_map = np.load(gt_labels, allow_pickle=True).item()

pred_label_map = np.load(os.path.join(path, pred_labels), allow_pickle=True).item()

T = 0.875

for k in pred_label_map.keys():
    labels = pred_label_map[k]['labels']
    pred_label_map[k]['labels'] = list(filter( lambda l: l['confidence'] >= T ,labels ))

test_images = ['aug19_100MEDIA_DJI_0740.JPG',
               'aug19_100MEDIA_DJI_0782.JPG',
               'oct19_103MEDIA_DJI_0586.JPG',
               'oct19_103MEDIA_DJI_0594.JPG',
               'sep19_102MEDIA_DJI_0169.JPG',
               'sep19_102MEDIA_DJI_0195.JPG',
               'sep19_102MEDIA_DJI_0331.JPG',
               'sep19_102MEDIA_DJI_0307.JPG'
               ]

#test_images = os.listdir(path)


impath = path

for filename in test_images: #list(pred_label_map.keys()):
#    print(os.path.join(impath, filename))
    if '.JPG' in filename:
        im = cv2.imread( os.path.join(impath, filename) )
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        gt = gt_label_map[filename]['labels']
        pred = pred_label_map[filename]['labels']
        print(filename)
        show_im_with_pred_and_gt_boxes(im, gt, pred)
#        show_im_with_boxes_not_saved(im, gt)