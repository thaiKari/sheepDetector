# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:23 2019

@author: karim
"""
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from utils import select_coordinates_from_image, resize_by_scale, get_metadata

optical_im_path = "E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0746.jpg"
thermal_im_path = "E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0747.jpg"

optical = imageio.imread(optical_im_path)
thermal = imageio.imread(thermal_im_path)

#SCALE = 1/4 #Use scale to make entire image fit on screen at same time.
#optical_pts= np.asarray(select_coordinates_from_image(resize_by_scale(SCALE, optical))) / SCALE
#thermal_pts = np.asarray(select_coordinates_from_image(thermal))

thermal_pts = np.array([[ 81,  65],
 [ 99, 127],
 [ 98, 187],
 [135, 173],
 [125, 122],
 [156, 116],
 [149, 87],
 [200,  63],
 [212,  97],
 [289,  66],
 [331, 198],
 [272, 268],
 [214, 293],
 [130, 272]])

optical_pts = np.array([[ 940,  636],
 [1048,  932],
 [1024, 1200],
 [1200, 1132],
 [1148,  912],
 [1296,  888],
 [1252,  748],
 [1512,  640],
 [1548,  776],
 [1892,  652],
 [2116, 1256],
 [1820, 1608],
 [1552, 1720],
 [1176, 1604]])

plt.figure()
plt.imshow(thermal)
plt.figure()
plt.imshow(optical)

tform = transform.ProjectiveTransform() #Or AffineTransform
tform.estimate(optical_pts, thermal_pts)
print(tform.params)
#
#optical_im_path = "E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0960.jpg"
#thermal_im_path = "E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0961.jpg"
##
#optical = imageio.imread(optical_im_path)
#thermal = imageio.imread(thermal_im_path)

warped = transform.warp(thermal, tform, output_shape=optical.shape)
plt.figure()
plt.imshow(warped)

print(thermal_pts.shape)
plt.figure(figsize=(20, 10))
plt.imshow(optical)
plt.imshow(warped, cmap='jet', alpha=0.7)


