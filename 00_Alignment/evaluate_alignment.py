# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:12:17 2019

@author: karim
"""

import numpy as np
from utils import read_pts, undistort_pts, transform_pts
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import  collections as mcollections
import cv2
#print(np.load('./camera_params_thermal/K.npy'))
#
#
#optical_pts_list = read_pts('./Newest_data/optical_key_pts_fakkel.txt')
#
#
#all_optical_pts = []
#
#for i in range(len(optical_pts_list)):
#    all_optical_pts = [*all_optical_pts, *optical_pts_list[i]]
#
#print(len(all_optical_pts))

#REPROJECTION ERROR (RET)
ret = np.load('./camera_params_thermal/ret.npy')

im = cv2.imread('camera_calibration/Fakkel_DD/DJI_0651.JPG')

#DD points
optical_pts = read_pts('./camera_calibration/Fakkel_DD/optical_key_pts_fakkel_DD.txt')
thermal_pts = read_pts('./camera_calibration/Fakkel_DD/thermal_key_pts_fakkel_DD.txt')
thermal_pts_undist = list(map( lambda p: undistort_pts(p), thermal_pts))
thermal_pts_undist_trans = list(map( lambda p: transform_pts(p), thermal_pts_undist ))


x_t = thermal_pts_undist_trans[0][:,0]
y_t = thermal_pts_undist_trans[0][:,1]
x_o = optical_pts[0][:,0]
y_o = optical_pts[0][:,1]

plt.figure(figsize=(8, 6))
plt.imshow(im, alpha= 0.5)
plt.scatter(x_t, y_t,  marker='+', label='visual')
plt.scatter(x_o, y_o,  marker='+', label='infrared')
plt.legend()

all_optical_pts = []
all_thermal_pts = []

for i in range(len(optical_pts)):
    all_optical_pts = [*all_optical_pts, *optical_pts[i]]
    all_thermal_pts = [*all_thermal_pts, *thermal_pts_undist_trans[i]]

all_optical_pts = np.asarray(all_optical_pts)
all_thermal_pts = np.asarray(all_thermal_pts)

#x_t = all_thermal_pts[:,0]
#y_t = all_thermal_pts[:,1]
#x_o = all_optical_pts[:,0]
#y_o = all_optical_pts[:,1]
#
#plt.figure(figsize=(8, 6))
#plt.scatter(x_t, y_t,  marker='+', label='visual')
#plt.scatter(x_o, y_o,  marker='+', label='infrared')
#plt.legend()


def compute_reproj_error(img_points, reprojected_points):
    tot_error=0
    total_points=0
    errors=[]
    for i in range(len(img_points)):
        err = np.sum(np.abs(img_points[i]-reprojected_points[i])**2)
        errors.append(np.sqrt(err/len(img_points[i])))
        tot_error+=err
        total_points+=len(img_points[i])
    
    mean_error=np.sqrt(tot_error/total_points)
    return mean_error, errors

reproj_error_perIm = []
for i in range(len(optical_pts)):
    o = optical_pts[i]
    t = thermal_pts_undist_trans[i]
    reproj_error_perIm.append(compute_reproj_error(o,t)[0])


mean_reproj_error, errors = compute_reproj_error(all_optical_pts, all_thermal_pts)

plt.rcParams.update({'font.size': 14}) 
plt.figure(figsize=(8, 6))
plt.scatter( np.arange(len(reproj_error_perIm)), reproj_error_perIm )
plt.plot([0, len(optical_pts)], [mean_reproj_error, mean_reproj_error], label='mean reprojection error')
plt.legend()
plt.xlabel('Image number')
plt.ylabel('Reprojection error (pixels)')


K = np.load('./camera_params_thermal/K.npy')
reprojected_principle_pt =  transform_pts( undistort_pts(np.asarray([[K[0,2], K[1,2]]]))).reshape(2)
print(reprojected_principle_pt)
xo = reprojected_principle_pt[0]
yo = reprojected_principle_pt[1]



dist_to_origin = []

for p in all_optical_pts:
    x = p[0]
    y = p[1]
    d = np.sqrt( (xo-x)**2 + (yo-y)**2 )
    dist_to_origin.append(d)
    



#error by distance from im_center
plt.figure(figsize=(8, 6))
plt.scatter(dist_to_origin, errors)
plt.plot([np.min(dist_to_origin), np.max(dist_to_origin)], [mean_reproj_error, mean_reproj_error], label='mean reprojection error')
plt.xlabel('Distance to reprojected IR principle point (pixels)')
plt.ylabel('Reprojection error (pixels)')
plt.legend()



print(len(errors))
print('mean', mean_reproj_error)
print('std', np.std(errors))
print('95%', np.percentile(errors, 95))

xo = 4056/2
yo = 3040/2

fig, ax = plt.subplots(figsize=(8, 6))
circle_mean = mpatches.Circle((int(xo),int(yo)), radius =mean_reproj_error, facecolor='r')
circle_95_perc = mpatches.Circle((int(xo),int(yo)), radius =np.percentile(errors, 95))
ax.imshow(im, alpha=0)
ax.add_patch(circle_95_perc)
ax.add_patch(circle_mean)
ax.axis('off')