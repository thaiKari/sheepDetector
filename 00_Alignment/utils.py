# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:39:52 2019

@author: karim
"""
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
from skimage import transform, img_as_ubyte
from datetime import datetime, date
import re
import os
import csv
import pandas as pd
import requests
import xmltodict
import exifread
import urllib.request
import numpy as np
import json
import pytz
from pytz import timezone
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import glob


def select_coordinates_from_image(image_array):
    """Opens a window displaying the image and returns the pixel coordinates of clicked points.
    Close the window to continue running code.
    image_array should be in range [0, 255]... not [0,1]    
    """
    
    coords = []
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)


    img = ImageTk.PhotoImage(image=Image.fromarray(image_array.astype('uint8')))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        coords.append([event.x, event.y])

    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()
        
    return coords



def resize_by_scale(SCALE, img):
    return img_as_ubyte(transform.resize(img, (img.shape[0]*SCALE, img.shape[1]*SCALE),
                       anti_aliasing=True))


def get_regex(deliminators):
    return '|'.join(map(re.escape, deliminators))
    

#returns a pandas df with logdata for the image
# Timestamp format from exif 'YYYY:MM:DD HH:MM:SS'
def get_log_data_for_time(timestamp):
    time_split =  re.split(get_regex([':', ' ']),str(timestamp))
    time_split = [int(n) for n in time_split]
    im_datetime = datetime(*time_split)
    
    log_directory_path = 'G:/SAU/Bilder Felles/Loggdata/Utpakket data'  
    log_directory = os.fsencode(log_directory_path)
    log_filename = None
    log_start_datetime = datetime.fromtimestamp(0) #start of timestamp time (1970)

    #Find the correct log file based on filename
    for file in os.listdir(log_directory):
        current_filename = os.fsdecode(file)
        
        log_time_split =  re.split(get_regex(['-', '_']),current_filename)
        log_time_split[0] = '20%s'%log_time_split[0]
        log_time_split = [int(n) for n in log_time_split[:-1]]
        log_datetime = datetime(*log_time_split)
        
        if(im_datetime < log_datetime):
            break
        else:
            log_filename = current_filename
            log_start_datetime = log_datetime
    
   
#    #Get the number of seconds into the flight that the image was taken
#    im_offsetTime = (im_datetime - log_start_datetime).total_seconds()
#    
    data = pd.read_csv(os.path.join(log_directory_path, log_filename ))
#    # find row with closest offset value
#    index = abs(data['offsetTime'] - im_offsetTime).idxmin()

    #CORRECT... compare gps time. GPS in zulutime vs im_time local. GPS:dateTimeStamp
    def make_utc_datetime(gps_dateTimeStamp):
        time_split =  re.split(get_regex(['-', 'T',':', 'Z']),str(gps_dateTimeStamp))
        time_split = [int(n) for n in time_split[:-1]]
        datetime_obj = datetime(*time_split)
        utc = pytz.utc
        
        return utc.localize(datetime_obj)

    
    amsterdam = timezone('Europe/Amsterdam')

    log_datetimes = [make_utc_datetime(d) for d in data['GPS:dateTimeStamp']]

    time_deltas = [abs(amsterdam.localize(im_datetime)- d) for d in log_datetimes]
    index = time_deltas.index(min(time_deltas))

    return data.loc[index, :]

    
def get_elevation_for_lat_lon(lat, lon):
    
    params = {
        'request': 'Execute',
        'service': 'WPS',
        'version': '1.0.0',
        'identifier': 'elevation',
        'datainputs': 'lat=%f;lon=%f;epsg=4326' % (lat, lon),
    }
    
    response = requests.get("http://openwps.statkart.no/skwms1/wps.elevation2", params=params)
    response_dict = xmltodict.parse(response.content)
    request_outputs = (response_dict['wps:ExecuteResponse']
                    ['wps:ProcessOutputs']
                    ['wps:Output'])
    elevation = (next(item for item in request_outputs if item['ows:Identifier'] =='elevation')
                    ['wps:Data']
                    ['wps:LiteralData']
                    ['#text'])
                     

    return float(elevation)


def get_logdata(filename):
    # get degress from GPS EXIF tag
    def degress(tag):
        d = float(tag.values[0].num) / float(tag.values[0].den)
        m = float(tag.values[1].num) / float(tag.values[1].den)
        s = float(tag.values[2].num) / float(tag.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    
    # read the exif tags
    with open(filename, 'rb') as f:
        tags = exifread.process_file(f)
        
    log_data = get_log_data_for_time( tags['Image DateTime']) 
    return(log_data)
    

def get_metadata(filename):
    
    # get degress from GPS EXIF tag
    def degress(tag):
        d = float(tag.values[0].num) / float(tag.values[0].den)
        m = float(tag.values[1].num) / float(tag.values[1].den)
        s = float(tag.values[2].num) / float(tag.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    
    # read the exif tags
    with open(filename, 'rb') as f:
        tags = exifread.process_file(f)
        
    log_data = get_log_data_for_time( tags['Image DateTime']) 
    
    # get lat/lon
    lat = degress(tags["GPS GPSLatitude"])
    lon = degress(tags["GPS GPSLongitude"])

   
    #get terrain elevation from kartverket
    elevation = get_elevation_for_lat_lon(lat, lon)
    
    
    # get the altitude    
    height_MSL = log_data['GPS(0):heightMSL']
    height_MGL = height_MSL - elevation
    
    # spit it out
#    print(filename)
#    print("Latitude[deg]     : %f, (log %f)" % (lat, log_data['GPS(0):Lat']))
#    print("Longitude[deg]    : %f (log %f)" % (lon, log_data['GPS(0):Long']))
#    print("heightMSL: %f" % height_MSL)
#    print("terrain elevation : %f" % elevation)
#    print("height_MGL : %f" % height_MGL)
#    print('rel_h', log_data['General:relativeHeight'])
#    
#    print()
#    print(lat, lon)
#    print( log_data['GPS(0):Lat'],  log_data['GPS(0):Long'])
    
    return {'GPSlatitude_exif':lat,
            'GPSlongditude_exif':lon,
            'GPSlatitude_log':log_data['GPS(0):Lat'],
            'GPSlongditude_log':log_data['GPS(0):Long'],
            'elevation_terrain': elevation,
            'heightMSL': height_MSL,
            'height_MGL': height_MGL,
            'height_barometric_smooth': log_data['IMU_ATTI(0):barometer:Smooth'], 
            'IMU_ATTI(0):gyro:X': log_data['IMU_ATTI(0):gyro:X'],
            'IMU_ATTI(0):gyro:Y': log_data['IMU_ATTI(0):gyro:Y'],
            'IMU_ATTI(0):gyro:Z': log_data['IMU_ATTI(0):gyro:Z']}
    



def write_pts(filename, pts):
    File_object = open(filename,"a+")
    for pt in pts:
        File_object.write('%d,%d;'%(int(pt[0]),int(pt[1])))
        
    File_object.write('\n')
    File_object.close()
    
def read_pts(filename):
    File_object = open(filename,"r")
    coord_set = File_object.readlines()
    all_pts = []

    
    for coords in coord_set:
        coords_list = list(filter(lambda s: s != '\n' ,  coords.split(';')))
        coords_list = list(map( lambda l: list(map( lambda c: int(c), l.split(',')))  , coords_list));
        all_pts.append(np.asarray(coords_list))
        
    File_object.close()
    return(all_pts)
        

def write_transformation(filename, tform):
    File_object = open(filename,"a+")
    
    numbers_to_write = tform.params.flatten().tolist()
    for n in numbers_to_write:
        File_object.write('%f,'%n)
        
    File_object.write('\n')
    File_object.close()
    
    
def read_transformations(filename):
    File_object = open(filename,"r")
    transformations = File_object.readlines()
    all_transforms = []
    
    for t in transformations:
        t_array = np.asarray(list(filter(lambda s: s != '\n' ,  t.split(',')))).astype(np.float).reshape(3,3)
        all_transforms.append(t_array)
    
    File_object.close()
    return all_transforms


def write_metadata(filename, data):
    File_object = open(filename,"a+")
    File_object.write(json.dumps(data))
    File_object.write('\n')
    File_object.close()
    

def read_metadata(filename):
    File_object = open(filename,"r")
    metadatas = File_object.readlines()
    all_metadata = []
    
    for m in metadatas:
        all_metadata.append(json.loads(m))
        
    File_object.close()
    return(all_metadata)
   
    
def write_filename(filename, im_name):
    File_object = open(filename,"a+")
    File_object.write('%s,'%im_name)
    File_object.close()
    
    
def read_filename(filename):
    File_object = open(filename,"r")
    filenames = list(filter(None,File_object.readline().split(',') ))
            
    File_object.close()
    return filenames


#increment image number by 1
def get_next_image(im_name):
    im_name_part = im_name[-12:]
    dir_part=im_name[:-12]    
    next_im_name = re.sub(r"\d+", str(int(re.search(r"\d+" , im_name_part).group(0)) + 1).zfill(4), im_name_part)
    
    return os.path.join(dir_part, next_im_name)


def get_im_num(im_name):
    return int(re.search(r"\d+" , im_name[-12:]).group(0))


def increment_im(im_name, n):
    im_name_part = im_name[-12:]
    dir_part=im_name[:-12]    
    next_im_name = re.sub(r"\d+", str(int(re.search(r"\d+" , im_name_part).group(0)) + n).zfill(4), im_name_part)
    
    return os.path.join(dir_part, next_im_name)
    

def get_lines(im):    
    if(im.shape[2] == 3):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
  
    im_s = cv2.resize(im, (400,300))
    blur = cv2.GaussianBlur(im_s,(5,5),0)
    im_lines = cv2.Canny(blur,100,200)
    im = cv2.resize(im_lines, (im.shape[1], im.shape[0]))
    
    if(im.shape[0]> 1000):
        kernel = np.ones((9,9),np.uint8)
        im = cv2.erode(im,kernel,iterations = 1)
    
    return im


def get_line_mask(im):
    lines = get_lines(im)
    lines[lines > 50] = 255
    return np.ma.masked_where(lines < 50, lines)

##Image_path = './camera_calibration/test/dji_0691.jpg'
##Image_path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/100MEDIA/DJI_0698.JPG'
#Image_path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/100MEDIA/DJI_0702.JPG' 
##Image_path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0792.JPG'
##Image_path = 'E:/SAU/Bilder Felles/Sorterte/Flyving_dd_06 09 2019/dji_0651.jpg'
#img = cv2.imread(Image_path)
#lines = get_line_mask(img)
#plt.figure(figsize=(20, 10))
#plt.imshow(img, alpha = 0.8)
#plt.imshow(lines, cmap=cm.jet, interpolation='none', alpha = 0.8)

def get_transform_from_mathching_pts(key_pts_input, key_pts_target):
    tform = transform.ProjectiveTransform() #Or AffineTransform
    tform.estimate(key_pts_input, key_pts_target)
    return tform.params
    

def get_transforms_from_matching_pts_list(target_list,input_list):
    transforms_list = []
    
    for i in range(len(input_list)):
        transforms_list.append(get_transform_from_mathching_pts(input_list[i], target_list[i]))
        
    return transforms_list
    

##pts is list of x,y pts in a single image
#def undistort_pts( pts, K, dist, h, w):
#    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
#    undistorted = cv2.undistortPoints(np.asarray(pts, np.float32).reshape((pts.shape[0],1,2)), K, dist, P=newcameramtx)
#    return undistorted.reshape(undistorted.shape[0], undistorted.shape[2])


##Correct all images in dir:
def correct_all_images_in_dir(path_in, path_out):
    images = glob.glob(path_in + '/*.jpg')
    print(path_in + '/*.jpg', images)
    
    
    K = np.load("./camera_params_thermal/K.npy")
    dist = np.load("./camera_params_thermal/dist.npy")
    
    for fname in images:
        print('undistorting' + fname)
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
        
        mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        cv2.imwrite( os.path.join(path_out, fname[-12:]), dst)
    

#undistort thermal_im
def undistort_image(img):
    K = np.load("./camera_params_thermal/K.npy")
    dist = np.load("./camera_params_thermal/dist.npy")
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))    
#    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
    
    return cv2.undistort(img, K, dist, None, newcameramtx)
    

def undistort_pts(pts):
    K = np.load("./camera_params_thermal/K.npy")
    dist = np.load("./camera_params_thermal/dist.npy")
    h,  w = (480, 640)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))  
    undistorted = cv2.undistortPoints(np.asarray(pts, np.float32).reshape((pts.shape[0],1,2)), K, dist, P=newcameramtx)
    return undistorted.reshape(undistorted.shape[0], undistorted.shape[2])


def transform_pts(pts):
    T = np.load('./Newest_data/the_transform.npy')
    return transform.AffineTransform(T).inverse(pts)
    





#Image_path = './camera_calibration/DJI_0007.jpg'
#img = cv2.imread(Image_path)
#print(img.shape)
    
#optical = read_pts('optical_key_pts_check2.txt')
#thermal = read_pts('thermal_key_pts_check2.txt')
#thermal_im_shape = (480, 640)
#h = 480
#w = 640
#
#K = np.load("./camera_params_thermal/K.npy")
#dist = np.load("./camera_params_thermal/dist.npy")
#
#thermal_undistorted = list(map( lambda pts: undistort_pts( pts, K, dist, h, w), thermal))
#
#x_o = thermal[-1][:, 0]
#y_o = thermal[-1][:, 1]
#plt.figure()
#plt.scatter(x_o, y_o)
#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
#
#x = thermal_undistorted[-1][:,0]
#y = thermal_undistorted[-1][:,1]
#x = x.reshape(x.shape[0])
#y = y.reshape(y.shape[0])
#
#plt.figure()
#plt.scatter(x,y)
#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
#
#print(x_o)
#
#transforms = get_transforms_from_matching_pts_list(thermal_undistorted, optical)
#print(transforms)
#
#t2 = read_transformations('transformations_corrected.txt')
#print(t2)
#
#transforms_flattened= list(map( lambda t: t.flatten(), transforms ))
#t2_flattened = list(map( lambda t: t.flatten(), t2 ))
#
#
#
#
#print(len(transforms))
#print(len(t2))
#
#plt.figure(figsize=(20,10))
#for i in range(9):
#
#    y1 = list(map(lambda t: float(t[i]), transforms_flattened))
#    y2 = list(map(lambda t: float(t[i]), t2_flattened))
#
#    plt.subplot(3, 3, i +1)
#    plt.boxplot([y1, y2])

    




#log = get_log_data_for_time('2019:08:21 21:31:22')
#for k in log.keys():
#    print(k)

#print('GPS(0):Lat', log['GPS(0):Lat'])
#print('GPS(0):Long', log['GPS(0):Long'])
#print('GPS(0):heightMSL', log['GPS(0):heightMSL'])
#print('General:relativeHeight', log['General:relativeHeight'])
        
    