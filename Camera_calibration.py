# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:48:38 2019

@author: karim
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from utils import read_pts, get_im_num, increment_im, get_next_image
import os



# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('C:/Users/karim/Projects/SAU/camera_calibration/new2_1709/termisk/*.jpg')
corners_optical = read_pts('optical_key_pts_check2.txt')
corners_thermal = read_pts('thermal_key_pts_check2.txt')


##VISUALIZE
i = 0
for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = corners_thermal[i]

    for j in range(corners.shape[0]):
        cv2.circle(img,(int(corners[j,0]),int(corners[j,1])),4,(255,0,0),-1)
#    
    plt.figure()
    plt.imshow(img)
#    
    corners = corners.reshape(54,1,2)
    i = i + 1



##CALIBRATION THERMAL
imgpoints = np.asarray(corners_thermal, np.float32)
objpoints = list(map (lambda p: objp , corners_thermal))
objp = np.array([objp])

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
np.save("./camera_params_thermal/ret", ret)
np.save("./camera_params_thermal/K", K)
np.save("./camera_params_thermal/dist", dist)
np.save("./camera_params_thermal/rvecs", rvecs)
np.save("./camera_params_thermal/tvecs", tvecs)


#UNDISTORT TEST THERMAL:
#path = './camera_calibration/test.jpg'
#path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/dji_0881.jpg'
#path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/100MEDIA/dji_0861.jpg'
path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/101MEDIA/dji_0662.jpg'
path = 'E:/SAU/Bilder Felles/Sorterte/Flyving_dd_06 09 2019/dji_0652.jpg'
img = cv2.imread(path)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(dst)

#Correct all thermal images in dir:
path_in = 'E:/SAU/Bilder Felles/Sorterte/Flyving_dd_06 09 2019'
path_out = 'E:/SAU/Bilder Felles/Sorterte/Flyving_dd_06 09 2019/corrected'

first_im = 'DJI_0651.jpg'
last_im_optical='DJI_0691.jpg'
cur_im = first_im

print(cur_im)

while get_im_num(cur_im) <= get_im_num(last_im_optical):
    img = cv2.imread(os.path.join(path_in, get_next_image(cur_im)))#thermal_im_path[i] ))
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
    
    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    print(os.path.join(path_out, str(increment_im(cur_im, 0))))
    cv2.imwrite( os.path.join(path_out, str(increment_im(cur_im, 1))), dst)
    
    cur_im = increment_im(cur_im, 2)


## crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#plt.figure()
#plt.imshow(dst)
#print(dst.shape)



