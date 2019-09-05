# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:25:31 2019

@author: karim
"""

import pandas as pd
import csv


filename  = 'E:/SAU/Bilder Felles/Loggdata/Utpakket data/19-04-25-13-31-57_FLY002.csv'

with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        print(line)
        
        
data = pd.read_csv(filename) 
# Preview the first 5 lines of the loaded data 
data.head()

# iterating the columns 
for col in data.columns: 
    print(col) 
    
print(data['General:relativeHeight'])#, 'General:absoluteHeight'])

optical_im_path = "E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0746.jpg"
thermal_im_path = "E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0747.jpg"

import PIL.Image
img = PIL.Image.open(thermal_im_path)
exif_data = img._getexif()
print(exif_data)

print(type(exif_data))
print(exif_data.keys())

from PIL.ExifTags import TAGS, GPSTAGS

def get_labeled_exif(exif):
    labeled = {}
    for (key, val) in exif_data.items():
        labeled[TAGS.get(key)] = val

    return labeled
labeled = get_labeled_exif(exif_data)
print(labeled.keys())
print(labeled['GPSInfo'])

def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging



geotags = get_geotagging(exif_data)
print(geotags)

def get_decimal_from_dms(dms, ref):

    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1] / 60.0
    seconds = dms[2][0] / dms[2][1] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)

def get_coordinates(geotags):
    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])

    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    return (lat,lon)


geotags = get_geotagging(exif_data)
print(get_coordinates(geotags))

