import sys
import exifread
import urllib.request
import json
from get_elevation import get_elevation_for_lat_lon
from utils import get_log_data_for_time

def get_metadata(filename):
    #Thank you borislip: https://mavicpilots.com/threads/altitude-information-from-exif-data-photos.32535/
    
    # get degress from GPS EXIF tag
    def degress(tag):
        d = float(tag.values[0].num) / float(tag.values[0].den)
        m = float(tag.values[1].num) / float(tag.values[1].den)
        s = float(tag.values[2].num) / float(tag.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    
    # read the exif tags
    with open(filename, 'rb') as f:
        tags = exifread.process_file(f)
        

    print(get_log_data_for_time( tags['Image DateTime']) )

    
    # get lat/lon
    lat = degress(tags["GPS GPSLatitude"])
    lon = degress(tags["GPS GPSLongitude"])

   
    #get terrain elevation from kartverket
    elevation = get_elevation_for_lat_lon(lat, lon)
    

    
    # get the altitude    
    alt = tags["GPS GPSAltitude"] #Wrong
    alt = float(alt.values[0].num) / float(alt.values[0].den)
    agl = alt - elevation
    
    # spit it out
    print("Latitude[deg]     : %f" % lat)
    print("Longitude[deg]    : %f" % lon)
    print("Altitude(WRONG) [m, ASL] : %f" % alt)
    print("elevation : %f" % elevation)
    
    return {'GPSlatitude':lat, 'GPSlongditude':lon, 'elevation': elevation}
    

optical_im_path = "E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0746.jpg"
get_metadata(optical_im_path)