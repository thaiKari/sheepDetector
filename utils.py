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
    
    log_directory_path = 'E:/SAU/Bilder Felles/Loggdata/Utpakket data'  
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
    
   
    #Get the number of seconds into the flight that the image was taken
    im_offsetTime = (im_datetime - log_start_datetime).total_seconds()
    
    data = pd.read_csv(os.path.join(log_directory_path, log_filename ))
    # find row with closest offset value
    index = abs(data['offsetTime'] - im_offsetTime).idxmin()
    
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

#log = get_log_data_for_time('2019:08:21 21:31:22')
#print('GPS(0):Lat', log['GPS(0):Lat'])
#print('GPS(0):Long', log['GPS(0):Long'])
#print('GPS(0):heightMSL', log['GPS(0):heightMSL'])
#print('General:relativeHeight', log['General:relativeHeight'])
        
    