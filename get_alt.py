import sys
import exifread
import urllib.request
import json
from utils import get_log_data_for_time, get_elevation_for_lat_lon

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
    print("Latitude[deg]     : %f, (log %f)" % (lat, log_data['GPS(0):Lat']))
    print("Longitude[deg]    : %f (log %f)" % (lon, log_data['GPS(0):Long']))
    print("heightMSL" % height_MSL)
    print("terrain elevation : %f" % elevation)
    print("height_MGL : %f" % height_MGL)
    
    print()
    print(lat, lon)
    print( log_data['GPS(0):Lat'],  log_data['GPS(0):Long'])
    
    return {'GPSlatitude':lat, 'GPSlongditude':lon, 'elevation': elevation}
    

optical_im_path = "E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0746.jpg"
get_metadata(optical_im_path)

