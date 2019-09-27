# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:48:38 2019

@author: karim
"""
import numpy as np
import cv2
import glob
from utils import read_pts, get_im_num, increment_im, get_next_image, undistort_pts, write_pts
import os
import matplotlib.pyplot as plt
import math




# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('C:/Users/karim/Projects/SAU/camera_calibration/00_allThermal/*.jpg')
#images_optical = glob.glob('camera_calibration/Optical_and_thermal/optical/*.jpg')
#corners_optical = read_pts('Newest_data/Old2/optical_key_pts_check2.txt')
corners_thermal = read_pts('./camera_calibration/00_allThermal/00_corner_coords.txt')


##VISUALIZE
i = 0
plt.figure(figsize=(20,40))
plt.subplots_adjust(wspace=.1, hspace=.1)
Rows = math.ceil(len(images)/6)

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = corners_thermal[i]

    for j in range(corners.shape[0]):
        cv2.circle(img,(int(corners[j,0]),int(corners[j,1])),4,(255,0,0),-1)

    plt.subplot(Rows, 6, i+1)
    plt.axis('off')
    plt.imshow(img)
#    
    corners = corners.reshape(54,1,2)
    i = i + 1
    
    


##CALIBRATION THERMAL
imgpoints = np.asarray(corners_thermal, np.float32)
objpoints = list(map (lambda p: objp , corners_thermal))
objp = np.array([objp])

#ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#np.save("./camera_calibration/00_allThermal/calib_params/ret", ret)
#np.save("./camera_calibration/00_allThermal/calib_params/K", K)
#np.save("./camera_calibration/00_allThermal/calib_params/dist", dist)
#np.save("./camera_calibration/00_allThermal/calib_params/rvecs", rvecs)
#np.save("./camera_calibration/00_allThermal/calib_params/tvecs", tvecs)

## Visualize after undistort:
h,  w = img.shape[:2]
i = 0
plt.figure(figsize=(20,40))

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = corners_thermal[i] #undistored_corners[i]
#    x = corners[:,0]
#    y = corners[:,1]

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
#    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
#    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    
    dst = cv2.undistort(img, K, dist, None, newcameramtx)
    undistorted = cv2.undistortPoints(np.asarray(corners, np.float32).reshape((corners.shape[0],1,2)), K, dist, P=newcameramtx)
    undistorted = undistorted.reshape(undistorted.shape[0], undistorted.shape[2])
#    
#    undistorted = []
#    for pt in corners: #x,y
#        x = pt[0]
#        y = pt[1]
#        undistorted.append( [mapx[y,x], mapy[y,x]])
#        
#    undistorted = np.asarray(undistorted)

    for j in range(corners.shape[0]):
        cv2.circle(dst,(int(undistorted[j,0]),int(undistorted[j,1])),4,(255,0,0),-1)
    
    plt.subplot(Rows, 6, i+1)
    plt.axis('off')
    plt.imshow(dst)
#    
    i = i + 1
    


   

##UNDISTORT TEST THERMAL:
##path = './camera_calibration/test.jpg'
##path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/dji_0881.jpg'
#path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/100MEDIA/dji_0861.jpg'
##path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/101MEDIA/dji_0662.jpg'
##path = 'E:/SAU/Bilder Felles/Sorterte/Flyving_dd_06 09 2019/dji_0652.jpg'
#img = cv2.imread(path)
#print(img.shape)
#h,  w = img.shape[:2]
#print(h,w)
#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
#
#mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
#
#plt.figure()
#plt.imshow(img)
#plt.figure()
#plt.imshow(dst)
#
##Correct all thermal images in dir:
#path_in = './camera_calibration'
#path_out = './camera_calibration/corrected'
#
#first_im = 'DJI_0781.jpg'
#last_im_optical='DJI_0781.jpg'
#cur_im = first_im
#
#K = np.load("./camera_params_thermal/K.npy")
#dist = np.load("./camera_params_thermal/dist.npy")
#
#print(cur_im)
#
#while get_im_num(cur_im) <= get_im_num(last_im_optical):
#    img = cv2.imread(os.path.join(path_in, get_next_image(cur_im)))
#    h,  w = img.shape[:2]
#    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
#    
#    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
#    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
#    print(os.path.join(path_out, str(increment_im(cur_im, 0))))
#    cv2.imwrite( os.path.join(path_out, str(increment_im(cur_im, 1))), dst)
#    
#    cur_im = increment_im(cur_im, 2)


## crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#plt.figure()
#plt.imshow(dst)
#print(dst.shape)



