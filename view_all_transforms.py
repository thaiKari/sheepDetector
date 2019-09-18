# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:45:08 2019

@author: karim
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from skimage import transform
from sklearn.linear_model import LinearRegression, RANSACRegressor
import math
from utils import get_metadata, get_next_image,read_filename, read_transformations

image_dir = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/'
images = read_filename('image_list.txt')
transforms = read_transformations('transformations.txt')

plt.figure(figsize=(20,80))

for i in range(len(images)):
    
    optical = imageio.imread(os.path.join(image_dir, images[i]  ))
    thermal = imageio.imread(os.path.join(image_dir, get_next_image(images[i])))
    thermal_transformed = transform.warp(thermal, transform.ProjectiveTransform(transforms[i]), output_shape=optical.shape)
    
    plt.subplot(math.ceil(len(images)/2),  2, i+1)
    plt.imshow(optical)
    plt.imshow(thermal_transformed, alpha=0.7)