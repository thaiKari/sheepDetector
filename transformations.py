# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:48:25 2019

@author: karim
"""

import numpy as np
import cv2
from skimage import transform
import matplotlib.pyplot as plt
from utils import resize_by_scale, get_line_mask, read_pts, select_coordinates_from_image

K = np.load("./parameters/camera_matrix_K.npy")
dist = np.load("./parameters/camera_dist_coeffs.npy")
T_v2IR = np.load("./parameters/Transform_vis_to_IR.npy")
T_IR2v = np.load("./parameters/Transform_IR_to_Vis.npy")
wv, hv = (4056, 3040)
wIR,hIR = (640, 480)

def undistort_IR_im(im):
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))    
    return cv2.undistort(im, K, dist, None, newcameramtx)


def transform_IR_im_to_vis_coordinate_system(im):
    im = undistort_IR_im(im)
    T = transform.AffineTransform(T_v2IR)
    return transform.warp(im, T, output_shape=(hv,wv))


def undistort_IR_pt_list(pts):
    pts =np.asarray(pts, np.float32)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))
    undistorted = cv2.undistortPoints(pts.reshape((pts.shape[0],1,2)), K, dist, P=newcameramtx)
    return undistorted.reshape(undistorted.shape[0], undistorted.shape[2])


def transform_IR_pt_list_to_vis_coordinate_system(pts):
    pts = undistort_IR_pt_list(pts)
    return transform.AffineTransform(T_IR2v)(pts)


def transform_vis_pt_list_to_IR_coordinate_system(pts):
    pts = transform.AffineTransform(T_v2IR)(pts)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))
    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(wIR,hIR),5) 
    pts = list(map( lambda p: [mapx[ int(p[1]), int(p[0])], mapy[int(p[1]), int(p[0])] ] ,pts))
    return np.asarray(pts)


def transform_vis_im_to_IR_coordinate_system(im):
    T = transform.AffineTransform(T_IR2v)
    im = transform.warp(im, T, output_shape=(hIR,wIR))
    imNew = np.zeros_like(im)
    
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))
    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(wIR,hIR),5) 
    SCALE = 1.15
    mapx = cv2.resize(mapx,None,fx=SCALE, fy=SCALE, interpolation = cv2.INTER_LINEAR)
    mapy = cv2.resize(mapy,None,fx=SCALE, fy=SCALE, interpolation = cv2.INTER_LINEAR)
    
    
    
    for x in range(mapx.shape[1]):
        for y in range(mapx.shape[0]):
            try:
                x2 = int(mapx[y,x])
                y2 = int(mapy[y,x])
                imNew[y2,x2] = im[int(y/SCALE), int(x/SCALE)]
            except:
                print('e')
            
    return imNew

    



print(mapx.shape)

####TRANSFORM IR_im TO VIS COORD SYSTEM
im = cv2.imread('00_Alignment/camera_calibration/test/DJI_0692.JPG')
imv = cv2.imread('00_Alignment/camera_calibration/test/DJI_0691.JPG')
#
#plt.figure()
#plt.imshow(transform_IR_im_to_vis_coordinate_system(im))
#plt.imshow(get_line_mask(imv))
#
####TRANSFORM IR pts
IR_pts = read_pts('./00_Alignment/camera_calibration/Fakkel_DD/thermal_key_pts_fakkel_DD.txt')[0]
Vis_pts = read_pts('./00_Alignment/camera_calibration/Fakkel_DD/optical_key_pts_fakkel_DD.txt')[0]

#tformed_pts = transform_IR_pt_list_to_vis_coordinate_system(IR_pts)
#plt.figure()
#plt.scatter(tformed_pts[:,0], tformed_pts[:,1],  marker='+', label='infrared' )
#plt.scatter(Vis_pts[:,0], Vis_pts[:,1],  marker='+', label='visual' )

###TRANSFORM VIS IM TO IR COORD SYSTEM:
#plt.figure()
#plt.imshow(im)
# 
imv_t = transform_vis_im_to_IR_coordinate_system(imv)
print(np.max(imv_t))
imv_t = np.array(imv_t*255, np.uint8)
print(imv_t.shape)
lines = cv2.Canny(imv_t[:,:,0], 100, 200)
plt.figure()
plt.imshow(im)
plt.imshow(imv_t)
#plt.imshow(np.ma.masked_where(lines < 50, lines))

#TRANSFORM VIS PT TO IR COORD SYSTEM

tformed_pts = transform_vis_pt_list_to_IR_coordinate_system(Vis_pts)

plt.figure()
plt.scatter(tformed_pts[:,0], tformed_pts[:,1],  marker='+', label='tformed' )
plt.scatter(IR_pts[:,0], IR_pts[:,1],  marker='+', label='IR' )
plt.legend()
#
#SCALE = 1/4 #Use scale to make entire image fit on screen at same time.
#corners = np.asarray(select_coordinates_from_image(resize_by_scale(SCALE, transform_IR_im_to_vis_coordinate_system(im)))) / SCALE
#print(corners)

#def distort_im(im):
#    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist*-1,(wIR, hIR),1,(wIR, hIR))    
#    return cv2.undistort(im, newcameramtx, dist*-1, None, newcameramtx)
#
#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR)) 
#SCALE =1.15
#NORM_VAL =700   
#mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(wIR,hIR),5) 
#mapx2 = mapy2 = np.zeros_like(mapx)
#
#print(np.max(mapx))
#
#plt.figure()
#plt.imshow(mapx)
#plt.figure()
#plt.imshow(mapy)
#mapx = cv2.resize(mapx,None,fx=SCALE, fy=SCALE, interpolation = cv2.INTER_LINEAR)
#mapy= cv2.resize(mapy,None,fx=SCALE, fy=SCALE, interpolation = cv2.INTER_LINEAR)
#plt.figure()
#plt.imshow(mapx)
#plt.figure()
#plt.imshow(mapy)
#print(mapx.shape)
#print(np.max(mapx))
#print(wIR, hIR)
#
##def find_nearest(array, value):
##    array = np.asarray(array)
##    idx = (np.abs(array - value)).argmin()
##    print(idx)
##    return array[idx]
##
##print( find_nearest(mapx) )
#
#for y in range(mapx.shape[0]):
#    for x in range(mapx.shape[1]):
#        x2 = int(round(mapx[y, x]))
#        y2 = int(round(mapy[y, x]))
#
#        try:
#            if(mapx2[y2,x2]>0):
#                mapx2[y2,x2] = np.mean(x/SCALE,mapx2[y2,x2])
#                mapy2[y2, x2] = np.mean( y/SCALE, mapy2[y2,x2])
#        
#            else:
#                mapx2[y2, x2] = x/SCALE
#                mapy2[y2, x2] = y/SCALE
#        except:
#            print("e")
#
#
#print(np.min(mapx2))       
##for y in range(mapx.shape[0]):
##    for x in range(mapx.shape[1]):
##        if(mapx2[y,x] == 0):
##            try:
##               mapx2[y,x] = np.mean( [mapx2[y-1,x], mapx2[y+1,x], mapx2[y,x-1], mapx2[y,x+1]] )
##               mapy2[y,x]  = np.mean( [mapy2[y-1,x], mapy2[y+1,x], mapy2[y,x-1], mapy2[y,x+1]] )
##            except:
##                print('e')
#
#tformed_im = transform_IR_im_to_vis_coordinate_system(im)
#cropped = transform.warp(tformed_im, transform.AffineTransform( T_IR2v), output_shape=(hIR,wIR))
#plt.figure()
#plt.imshow(cropped)
##plt.figure()
##plt.imshow(transform.warp(tformed_im, transform.AffineTransform( T_IR2v), output_shape=(hIR,wIR)))
#
##scaleX = wIR/ cropped.shape[1]
##print(scaleX)
##scaleY = hIR/ cropped.shape[0]
##print(scaleY)
##cropped = cv2.resize(cropped, (wIR, hIR))
##cropped = distort_im(cropped)
##cropped = cropped[33:420, 57:566]
#cropped =  cv2.remap(cropped, mapx2, mapy2, cv2.INTER_LINEAR)
#
#plt.figure()
#plt.imshow(cropped)
#
#min_corner = [ 396, 2752]
#max_corner = [3748,  176]
#print(cropped.shape)



#corners = select_coordinates_from_image(cropped*255)
#print(corners)

#def distort_point(p):
#
#    cx = K[0,2]
#    cy = K[1,2]
#    fx = K[0,0]
#    fy = K[1,1]
#    k1 = dist[0][0] *-1
#    k2 = dist[0][1] * -1
#    k3 = dist[0][-1] *-1
#    p1 = dist[0][2] * -1
#    p2 = dist[0][3] *-1
#    
#    x = ( p[0]- cx) / fx
#    y = (p[1]- cy) / fy
#
#    
#    r2 = x*x + y*y
#        
#    xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
#    yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
#    
#    xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
#    yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
#    
#    xDistort = xDistort * fx + cx
#    yDistort = yDistort * fy + cy
#    
#    
#    return[xDistort, yDistort]
#
#
#def normalise_p(p):
#    cx = K[0,2]
#    cy = K[1,2]
#    fx = K[0,0]
#    fy = K[1,1]
#    
#    x = ( float(p[0])- float(cx)) / float(fx)
#    y = ( float(p[1])- float(cy)) / float(fy)
#    
#    
#    return np.array([x,y, 1], np.float32)
#
#
#
# 
#
#def transform_vis_points_to_IR(pts):
#    print(T_v2IR)
#
#    T = T_v2IR
#    #scale x:
#    T[0,0] = T[0,0]
#    #scale y:
#    T[1,1] = T[1,1]
#    #trans x:
#    T[0,2] = T[0,2]
#    #trans y:
#    T[1,2] = T[1,2]
#
#    pts = transform.AffineTransform( T )(pts)
#    pts =np.asarray(pts, np.float32)
#    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist*-1,(wIR, hIR),1,(wIR, hIR))
#    undistorted = cv2.undistortPoints(pts.reshape((pts.shape[0],1,2)), K, dist, P=newcameramtx)
#    return undistorted.reshape(undistorted.shape[0], undistorted.shape[2])
#