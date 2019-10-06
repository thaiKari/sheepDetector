# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:04:33 2019

@author: karim
"""

import cv2
import glob
import os

path = ''



images = glob.glob('camera_calibration/new2_1709/termisk/*.jpg')
save_to = 'camera_calibration/new2_1709/termisk/upsized/'
i = 0

for fname in images:
    print(fname)
    img = cv2.imread(fname)
    print(img.shape)
    resized = cv2.resize(img, (1014,750) , interpolation=cv2.INTER_NEAREST)
    print(resized.shape)
    cv2.imwrite( os.path.join(save_to, str(i)+'.jpg'), resized)
    i = i +1

    

print(os.path.join(save_to, fname))